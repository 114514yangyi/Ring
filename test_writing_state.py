import unittest

import numpy as np

from writing_state.detector import DetectorConfig, DetectorState, WritingStateDetector


def make_sample(timestamp, lin_acc, gyro):
    return {
        "timestamp": timestamp,
        "lin_acc_x": lin_acc,
        "lin_acc_y": 0.0,
        "lin_acc_z": 0.0,
        "gyro_x": gyro,
        "gyro_y": 0.0,
        "gyro_z": 0.0,
    }


def synthetic_sequence(sample_rate=100):
    rows = []
    timestamp = 0.0

    def append(count, lin_acc, gyro):
        nonlocal timestamp
        for _ in range(count):
            rows.append(make_sample(timestamp, lin_acc, gyro))
            timestamp += 1.0 / sample_rate

    append(20, 0.01, 0.0)
    append(3, 0.35, 12.0)  # contact impact
    append(25, 0.08, 8.0)  # first character
    append(4, 0.40, 18.0)  # pen-up burst
    append(15, 0.30, 25.0)  # invalid air movement between characters
    append(3, 0.35, 12.0)  # next contact
    append(22, 0.07, 7.0)  # second character
    append(4, 0.40, 18.0)  # final pen-up
    append(20, 0.02, 0.0)
    return rows


class WritingStateDetectorTests(unittest.TestCase):
    def test_detects_two_writing_segments_and_pen_ups(self):
        detector = WritingStateDetector(
            DetectorConfig(
                sample_rate=100,
                contact_confirm_seconds=0.03,
                lift_confirm_seconds=0.03,
                writing_end_seconds=0.25,
            )
        )
        events = detector.process(synthetic_sequence())
        types = [event.type for event in events]

        self.assertEqual(types.count("writing_start"), 2)
        self.assertEqual(types.count("pen_up"), 2)
        self.assertEqual([segment["kind"] for segment in detector.segments], ["writing", "writing"])
        self.assertTrue(all(segment["valid_operation"] for segment in detector.segments))
        self.assertEqual(detector.state, DetectorState.AIR_INVALID)

    def test_single_frame_spike_does_not_start_writing(self):
        detector = WritingStateDetector(
            DetectorConfig(
                sample_rate=100,
                contact_confirm_seconds=0.10,
                allow_implicit_start=False,
            )
        )
        rows = [
            make_sample(index / 100.0, 0.01 if index != 4 else 0.5, 0.0)
            for index in range(20)
        ]
        events = detector.process(rows)
        self.assertEqual(events, [])
        self.assertEqual(detector.segments, [])

    def test_air_motion_after_pen_up_is_invalid_until_next_contact(self):
        detector = WritingStateDetector(
            DetectorConfig(
                sample_rate=100,
                contact_confirm_seconds=0.03,
                lift_confirm_seconds=0.03,
            )
        )
        rows = [
            make_sample(index / 100.0, 0.07, 7.0)
            for index in range(20)
        ]
        rows += [
            make_sample(0.20 + index / 100.0, 0.40, 25.0)
            for index in range(20)
        ]
        rows += [
            make_sample(0.40 + index / 100.0, 0.02, 0.0)
            for index in range(20)
        ]
        events = detector.process(rows)
        self.assertEqual([event.type for event in events], ["writing_start", "pen_up"])
        self.assertEqual(detector.state, DetectorState.AIR_INVALID)
        self.assertEqual(len(detector.segments), 1)

    def test_realtime_frame_shape_is_supported(self):
        class Vector:
            def __init__(self, x, y, z):
                self.x, self.y, self.z = x, y, z

        class Frame:
            timestamp = 1.0
            lin_acc = Vector(0.07, 0.0, 0.0)
            acc = Vector(0.0, 0.0, 1.0)
            gyro = Vector(7.0, 0.0, 0.0)

        detector = WritingStateDetector(
            DetectorConfig(sample_rate=100, contact_confirm_seconds=0.03)
        )
        for _ in range(5):
            detector.push(Frame())
        self.assertEqual(detector.state, DetectorState.WRITING)
        self.assertTrue(detector.snapshot()["valid_operation"])


if __name__ == "__main__":
    unittest.main()
