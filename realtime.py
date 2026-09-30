from __future__ import annotations

from typing import Any

from .detector import DetectorConfig, DetectorEvent, WritingStateDetector


class RealtimeWritingStateHandler:
    """Bridge raw SmartRing frames to the existing ``ImuBus``."""

    def __init__(self, config: DetectorConfig | None = None):
        self.detector = WritingStateDetector(config=config)

    def on_raw(self, frame: Any, bus: Any) -> None:
        events = self.detector.push(frame)
        bus.publish("writing_state_frame", self.detector.snapshot())
        for event in events:
            bus.publish("writing_state", event.to_dict())

    def flush(self, bus: Any) -> None:
        for event in self.detector.flush():
            bus.publish("writing_state", event.to_dict())
