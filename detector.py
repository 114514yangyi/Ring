from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Any, Deque, Iterable

import numpy as np

from .features import FrameFeatures, extract_features


class DetectorState(str, Enum):
    AIR_IDLE = "air_idle"
    CONTACT_CANDIDATE = "contact_candidate"
    WRITING = "writing"
    LIFT_CANDIDATE = "lift_candidate"
    AIR_INVALID = "air_invalid"


@dataclass(frozen=True)
class DetectorConfig:
    """Thresholds for the rule-based baseline.

    The baseline intentionally exposes physical thresholds instead of hiding
    them in the implementation. They should be fitted or replaced by a
    learned touch classifier once labeled contact data is available.
    """

    sample_rate: float = 100.0
    history_seconds: float = 0.18
    contact_confirm_seconds: float = 0.10
    lift_confirm_seconds: float = 0.10
    writing_end_seconds: float = 0.45
    air_motion_seconds: float = 0.10
    contact_refractory_seconds: float = 0.15
    min_writing_seconds: float = 0.08
    contact_impact_threshold: float = 0.95
    lift_impact_threshold: float = 1.05
    contact_accel_min: float = 0.18
    writing_motion_min: float = 0.18
    writing_motion_max: float = 5.5
    air_motion_threshold: float = 5.5
    contact_gyro_max: float = 20.0
    allow_implicit_start: bool = True
    emit_air_idle_events: bool = False

    def frames(self, seconds: float, minimum: int = 1) -> int:
        return max(minimum, int(round(seconds * self.sample_rate)))


@dataclass(frozen=True)
class DetectorEvent:
    """A transition or segment event emitted by :class:`WritingStateDetector`."""

    type: str
    timestamp: float
    state: DetectorState
    confidence: float
    valid_operation: bool
    reason: str
    segment_id: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "timestamp": self.timestamp,
            "state": self.state.value,
            "confidence": self.confidence,
            "valid_operation": self.valid_operation,
            "reason": self.reason,
            "segment_id": self.segment_id,
        }


@dataclass
class _Segment:
    segment_id: int
    kind: str
    start_time: float
    end_time: float | None = None
    valid_operation: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "kind": self.kind,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "valid_operation": self.valid_operation,
        }


class WritingStateDetector:
    """Detect writing contact, pen-up, and invalid air movement.

    The detector is causal. A contact or lift is only emitted after enough
    consecutive evidence has accumulated. A pen-up starts an invalid segment;
    a later contact starts a new valid writing segment, which naturally makes
    the same mechanism work for both character boundaries and the final end.
    """

    def __init__(self, config: DetectorConfig | None = None):
        self.config = config or DetectorConfig()
        self.reset()

    def reset(self) -> None:
        self.state = DetectorState.AIR_IDLE
        self.previous: FrameFeatures | None = None
        self.history: Deque[FrameFeatures] = deque(
            maxlen=self.config.frames(self.config.history_seconds, minimum=3)
        )
        self._candidate_frames = 0
        self._candidate_start_time: float | None = None
        self._lift_start_time: float | None = None
        self._quiet_frames = 0
        self._air_motion_frames = 0
        self._writing_frames = 0
        self._ignore_contact_until = float("-inf")
        self._seen_writing = False
        self._segment_counter = 0
        self._active_segment: _Segment | None = None
        self._segments: list[_Segment] = []

    @property
    def segments(self) -> list[dict[str, Any]]:
        return [segment.to_dict() for segment in self._segments]

    @property
    def valid_operation(self) -> bool:
        return self.state == DetectorState.WRITING

    def snapshot(self, timestamp: float | None = None) -> dict[str, Any]:
        """Return the current per-frame gate for a downstream recognizer."""

        if timestamp is None:
            timestamp = self.previous.timestamp if self.previous is not None else 0.0
        return {
            "type": "writing_state_frame",
            "timestamp": float(timestamp),
            "state": self.state.value,
            "valid_operation": self.valid_operation,
            "segment_id": (
                self._active_segment.segment_id
                if self._active_segment is not None
                else None
            ),
        }

    def push(self, sample: Any) -> list[DetectorEvent]:
        fallback = (
            self.previous.timestamp + 1.0 / self.config.sample_rate
            if self.previous is not None
            else 0.0
        )
        current = extract_features(sample, self.previous, fallback_timestamp=fallback)
        self.previous = current
        self.history.append(current)

        if self.state == DetectorState.WRITING:
            return self._update_writing(current)
        if self.state == DetectorState.LIFT_CANDIDATE:
            return self._update_lift_candidate(current)
        if self.state == DetectorState.CONTACT_CANDIDATE:
            return self._update_contact_candidate(current)
        return self._update_air(current)

    def process(self, samples: Iterable[Any]) -> list[DetectorEvent]:
        events: list[DetectorEvent] = []
        for sample in samples:
            events.extend(self.push(sample))
        events.extend(self.flush())
        return events

    def flush(self) -> list[DetectorEvent]:
        """Close open segments at the last observed sample."""

        if self.previous is None:
            return []
        timestamp = self.previous.timestamp
        if self.state in {DetectorState.WRITING, DetectorState.LIFT_CANDIDATE}:
            event = self._finish_writing(timestamp, reason="stream_end")
            return [event] if event is not None else []
        if self._active_segment is not None:
            self._close_segment(timestamp)
        return []

    def _update_air(self, current: FrameFeatures) -> list[DetectorEvent]:
        events: list[DetectorEvent] = []
        contact_evidence = self._contact_evidence(current)
        implicit_start = (
            self.config.allow_implicit_start
            and self.state == DetectorState.AIR_IDLE
            and not self._seen_writing
            and self._writing_like(current)
            and current.timestamp >= self._ignore_contact_until
            and self._recent_mean("motion_score") <= self.config.writing_motion_max
        )
        if contact_evidence or implicit_start:
            if self.state != DetectorState.CONTACT_CANDIDATE:
                self.state = DetectorState.CONTACT_CANDIDATE
                self._candidate_frames = 0
                self._candidate_start_time = current.timestamp
            self._candidate_frames += 1
            if self._candidate_frames >= self.config.frames(
                self.config.contact_confirm_seconds
            ):
                events.append(
                    self._start_writing(
                        current,
                        confidence=self._contact_confidence(current),
                        reason="contact_confirmed" if contact_evidence else "implicit_contact",
                        start_time=self._candidate_start_time,
                    )
                )
            return events

        self._candidate_frames = 0
        air_motion = self._air_like(current)
        if air_motion:
            self._air_motion_frames += 1
            if (
                self.config.emit_air_idle_events
                and self._air_motion_frames
                == self.config.frames(self.config.air_motion_seconds)
            ):
                events.append(
                    DetectorEvent(
                        type="invalid_air_start",
                        timestamp=current.timestamp,
                        state=DetectorState.AIR_INVALID,
                        confidence=self._air_confidence(current),
                        valid_operation=False,
                        reason="air_motion_before_contact",
                    )
                )
            self.state = DetectorState.AIR_INVALID
        else:
            self._air_motion_frames = 0
            if self.state != DetectorState.AIR_INVALID:
                self.state = DetectorState.AIR_IDLE
        return events

    def _update_contact_candidate(self, current: FrameFeatures) -> list[DetectorEvent]:
        if self._contact_evidence(current) or self._writing_like(current):
            if self._candidate_start_time is None:
                self._candidate_start_time = current.timestamp
            self._candidate_frames += 1
        else:
            self._candidate_frames = 0
            self._candidate_start_time = None
            self.state = DetectorState.AIR_IDLE
            return []

        if self._candidate_frames >= self.config.frames(
            self.config.contact_confirm_seconds
        ):
            return [
                self._start_writing(
                    current,
                    confidence=self._contact_confidence(current),
                    reason="contact_confirmed",
                    start_time=self._candidate_start_time,
                )
            ]
        return []

    def _update_writing(self, current: FrameFeatures) -> list[DetectorEvent]:
        events: list[DetectorEvent] = []
        self._writing_frames += 1
        if self._lift_evidence(current):
            self.state = DetectorState.LIFT_CANDIDATE
            self._candidate_frames = 1
            self._lift_start_time = current.timestamp
            self._quiet_frames = 0
            return events

        if self._writing_like(current):
            self._quiet_frames = 0
        else:
            self._quiet_frames += 1
        if (
            self._writing_frames >= self.config.frames(self.config.min_writing_seconds)
            and self._quiet_frames >= self.config.frames(self.config.writing_end_seconds)
        ):
            event = self._finish_writing(current.timestamp, reason="writing_timeout")
            if event is not None:
                events.append(event)
        return events

    def _update_lift_candidate(self, current: FrameFeatures) -> list[DetectorEvent]:
        if self._lift_evidence(current) or self._air_like(current):
            self._candidate_frames += 1
        else:
            self.state = DetectorState.WRITING
            self._candidate_frames = 0
            self._lift_start_time = None
            self._quiet_frames = 0
            return []

        if self._candidate_frames < self.config.frames(self.config.lift_confirm_seconds):
            return []

        event = self._finish_writing(
            current.timestamp,
            reason="pen_up_confirmed",
            segment_end_time=self._lift_start_time,
        )
        return [event] if event is not None else []

    def _start_writing(
        self,
        current: FrameFeatures,
        confidence: float,
        reason: str,
        start_time: float | None = None,
    ) -> DetectorEvent:
        self._segment_counter += 1
        self._active_segment = _Segment(
            segment_id=self._segment_counter,
            kind="writing",
            start_time=(
                current.timestamp
                if start_time is None
                else min(start_time, current.timestamp)
            ),
            valid_operation=True,
        )
        self.state = DetectorState.WRITING
        self._seen_writing = True
        self._candidate_frames = 0
        self._candidate_start_time = None
        self._lift_start_time = None
        self._quiet_frames = 0
        self._air_motion_frames = 0
        self._writing_frames = 0
        return DetectorEvent(
            type="writing_start",
            timestamp=current.timestamp,
            state=self.state,
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            valid_operation=True,
            reason=reason,
            segment_id=self._active_segment.segment_id,
        )

    def _finish_writing(
        self,
        timestamp: float,
        reason: str,
        segment_end_time: float | None = None,
    ) -> DetectorEvent | None:
        if self._active_segment is None:
            self.state = DetectorState.AIR_IDLE
            return None
        segment_id = self._active_segment.segment_id
        self._active_segment.end_time = (
            timestamp if segment_end_time is None else min(segment_end_time, timestamp)
        )
        self._segments.append(self._active_segment)
        self._active_segment = None
        self.state = DetectorState.AIR_INVALID
        self._ignore_contact_until = timestamp + self.config.contact_refractory_seconds
        self._candidate_frames = 0
        self._candidate_start_time = None
        self._lift_start_time = None
        self._quiet_frames = 0
        self._air_motion_frames = 0
        return DetectorEvent(
            type="pen_up",
            timestamp=timestamp,
            state=self.state,
            confidence=1.0,
            valid_operation=False,
            reason=reason,
            segment_id=segment_id,
        )

    def _close_segment(self, timestamp: float) -> None:
        if self._active_segment is None:
            return
        self._active_segment.end_time = timestamp
        self._segments.append(self._active_segment)
        self._active_segment = None

    def _recent_mean(self, attribute: str) -> float:
        if not self.history:
            return 0.0
        return float(np.mean([getattr(item, attribute) for item in self.history]))

    def _contact_evidence(self, current: FrameFeatures) -> bool:
        return (
            current.timestamp >= self._ignore_contact_until
            and
            current.impact_score >= self.config.contact_impact_threshold
            and current.lin_acc_norm >= self.config.contact_accel_min
            and current.gyro_norm <= self.config.contact_gyro_max
            and self._recent_mean("motion_score") <= self.config.writing_motion_max * 1.25
        )

    def _lift_evidence(self, current: FrameFeatures) -> bool:
        return (
            current.impact_score >= self.config.lift_impact_threshold
            or current.motion_score >= self.config.air_motion_threshold
        )

    def _writing_like(self, current: FrameFeatures) -> bool:
        return (
            self.config.writing_motion_min
            <= current.motion_score
            <= self.config.writing_motion_max
        )

    def _air_like(self, current: FrameFeatures) -> bool:
        return current.motion_score >= self.config.air_motion_threshold

    def _contact_confidence(self, current: FrameFeatures) -> float:
        return float(
            np.clip(
                0.35
                + 0.25 * current.impact_score
                + 0.15 * (current.motion_score / max(self.config.writing_motion_max, 1.0)),
                0.0,
                1.0,
            )
        )

    def _air_confidence(self, current: FrameFeatures) -> float:
        return float(
            np.clip(current.motion_score / max(self.config.air_motion_threshold * 2.0, 1.0), 0.0, 1.0)
        )


def detect_samples(
    samples: Iterable[Any], config: DetectorConfig | None = None
) -> tuple[list[DetectorEvent], list[dict[str, Any]]]:
    detector = WritingStateDetector(config=config)
    events = detector.process(samples)
    return events, detector.segments
