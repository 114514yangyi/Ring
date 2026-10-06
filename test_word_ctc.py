import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from writing_state.train_word_ctc import collate_ctc_examples
from writing_state.word_ctc import (
    WordCTC,
    build_word_ctc_summary,
    greedy_ctc_decode,
    levenshtein_distance,
    load_word_ctc,
    snap_to_vocabulary,
)


class CtcDecodeTests(unittest.TestCase):
    def test_levenshtein_distance(self):
        self.assertEqual(levenshtein_distance("the", "the"), 0)
        self.assertEqual(levenshtein_distance("cat", "cats"), 1)
        self.assertEqual(levenshtein_distance("teh", "the"), 2)

    def test_greedy_ctc_decode_collapses_repeats_and_blanks(self):
        logits = np.array(
            [
                [0.0, 9.0, 0.0],
                [0.0, 9.0, 0.0],
                [9.0, 0.0, 0.0],
                [0.0, 0.0, 9.0],
                [0.0, 0.0, 9.0],
            ],
            dtype=np.float32,
        )

        self.assertEqual(greedy_ctc_decode(logits), "ab")

    def test_snap_to_vocabulary(self):
        vocabulary = {"the", "cat", "water"}

        self.assertEqual(snap_to_vocabulary("teh", vocabulary), "the")
        self.assertEqual(snap_to_vocabulary("water", vocabulary), "water")
        self.assertEqual(snap_to_vocabulary("zzz", {"cat"}), "cat")
        self.assertIsNone(snap_to_vocabulary("zzz", {"cat"}, max_distance=1))


class CtcCollateTests(unittest.TestCase):
    def test_collate_pads_targets_and_reports_lengths(self):
        batch = [
            (np.zeros((128, 2), dtype=np.float32), np.array([1, 2], dtype=np.int64)),
            (np.zeros((128, 2), dtype=np.float32), np.array([3], dtype=np.int64)),
        ]

        inputs, targets, input_lengths, target_lengths = collate_ctc_examples(batch)

        self.assertEqual(tuple(inputs.shape), (2, 128, 2))
        np.testing.assert_array_equal(targets.numpy(), np.array([1, 2, 3]))
        np.testing.assert_array_equal(target_lengths.numpy(), np.array([2, 1]))
        np.testing.assert_array_equal(input_lengths.numpy(), np.array([128, 128]))


class WordCtcModelTests(unittest.TestCase):
    def test_forward_outputs_per_frame_class_logits(self):
        model = WordCTC()
        model.eval()

        logits = model(torch.randn(4, 128, 2))

        self.assertEqual(tuple(logits.shape), (128, 4, 27))

    def test_checkpoint_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = WordCTC()
            model.eval()
            path = Path(tmp) / "word_ctc.pt"
            torch.save({"state_dict": model.state_dict()}, path)

            loaded = load_word_ctc(path)
            inputs = torch.randn(2, 128, 2)
            with torch.no_grad():
                np.testing.assert_allclose(
                    model(inputs).detach().numpy(),
                    loaded(inputs).detach().numpy(),
                    atol=1e-6,
                )

    def test_summary_reports_vocabulary_output(self):
        summary = build_word_ctc_summary()

        self.assertEqual(summary["num_classes"], 27)
        self.assertEqual(summary["blank_index"], 0)
        self.assertGreater(summary["parameter_count"], 0)


if __name__ == "__main__":
    unittest.main()
