import unittest

import numpy as np
import torch

from writing_state.paper_touch import (
    PaperTouchConfig,
    PaperTouchDecoder,
    PaperTouchClassifier,
    TouchState,
    TouchPrediction,
    TouchWindowizer,
    WritingRingTouchResNet,
    build_model_summary,
    load_touch_classifier,
    sample_to_six_axes,
)


def sample(index):
    return {
        "timestamp": index / 200.0,
        "lin_acc_x": 0.01,
        "lin_acc_y": 0.02,
        "lin_acc_z": 0.03,
        "gyro_x": 1.0,
        "gyro_y": 2.0,
        "gyro_z": 3.0,
    }


class PaperTouchTests(unittest.TestCase):
    def test_paper_shape_and_model_forward(self):
        config = PaperTouchConfig()
        windowizer = TouchWindowizer(config)
        window = None
        for index in range(21):
            window = windowizer.push(sample(index))

        self.assertIsNotNone(window)
        self.assertEqual(window.values.shape, (20, 6))

        model = WritingRingTouchResNet()
        output = model(torch.zeros(2, 20, 6))
        self.assertEqual(tuple(output.shape), (2, 4))

    def test_100hz_stream_is_resampled_to_paper_input_shape(self):
        config = PaperTouchConfig(sample_rate=100.0)
        windowizer = TouchWindowizer(config)
        window = None
        for index in range(12):
            row = sample(index)
            row["timestamp"] = index / 100.0
            next_window = windowizer.push(row)
            if index < 10:
                self.assertIsNone(next_window)
            if next_window is not None:
                window = next_window

        self.assertIsNotNone(window)
        self.assertEqual(window.values.shape, (20, 6))
        self.assertAlmostEqual(window.end_timestamp - window.start_timestamp, 0.1)

    def test_model_has_paper_residual_widths(self):
        model = WritingRingTouchResNet()
        self.assertEqual(
            [model.blocks[0].conv2.out_channels,
             model.blocks[1].conv2.out_channels,
             model.blocks[2].conv2.out_channels],
            [8, 16, 32],
        )
        self.assertEqual(build_model_summary()["labels"], ["contact", "air", "lift", "press"])

    def test_decoder_only_emits_press_and_lift_events(self):
        decoder = PaperTouchDecoder()

        def prediction(label):
            return TouchPrediction(
                label=label,
                probabilities=(0.9, 0.03, 0.03, 0.04),
                start_timestamp=0.0,
                end_timestamp=0.1,
                center_timestamp=0.05,
            )

        self.assertEqual(decoder.update(prediction(TouchState.AIR)), [])
        self.assertFalse(decoder.valid_operation)
        events = decoder.update(prediction(TouchState.PRESS))
        self.assertEqual([event.type for event in events], ["writing_start"])
        self.assertTrue(decoder.valid_operation)
        self.assertEqual(decoder.update(prediction(TouchState.CONTACT)), [])
        events = decoder.update(prediction(TouchState.LIFT))
        self.assertEqual([event.type for event in events], ["pen_up"])
        self.assertFalse(decoder.valid_operation)

    def test_classifier_outputs_four_probabilities(self):
        config = PaperTouchConfig()
        classifier = PaperTouchClassifier(WritingRingTouchResNet(), config=config)
        windowizer = TouchWindowizer(config)
        window = None
        for index in range(21):
            window = windowizer.push(sample(index))
        prediction = classifier.predict(window)
        self.assertIn(prediction.label, tuple(TouchState))
        self.assertEqual(len(prediction.probabilities), 4)
        self.assertAlmostEqual(sum(prediction.probabilities), 1.0, places=5)

    def test_zero_linear_acceleration_is_not_replaced_by_raw_acceleration(self):
        values = sample_to_six_axes(
            {
                "lin_acc_x": 0.0,
                "lin_acc_y": 0.0,
                "lin_acc_z": 0.0,
                "acc_x": 0.0,
                "acc_y": 0.0,
                "acc_z": 1.0,
                "gyro_x": 0.0,
                "gyro_y": 0.0,
                "gyro_z": 0.0,
            }
        )
        self.assertTrue(np.allclose(values[:3], [0.0, 0.0, 0.0]))

    def test_trained_checkpoint_can_be_loaded_when_present(self):
        from pathlib import Path

        checkpoint = Path("writing_state/models/paper_touch_resnet.pt")
        if not checkpoint.exists():
            self.skipTest("trained checkpoint not present")
        classifier = load_touch_classifier(checkpoint)
        windowizer = TouchWindowizer(PaperTouchConfig())
        window = None
        for index in range(21):
            window = windowizer.push(sample(index))
        prediction = classifier.predict(window)
        self.assertIn(prediction.label, tuple(TouchState))


if __name__ == "__main__":
    unittest.main()
