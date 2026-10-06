import tempfile
import unittest
from pathlib import Path

import numpy as np

from writing_state.word_dataset import (
    WordSample,
    build_word_examples,
    deskew_trajectory,
    discover_word_samples,
    word_group,
)
from writing_state.word_recognition import (
    dtw_distance,
    evaluate_nearest,
    nearest_words,
)


def write_block(root, user, action, sample_id, mask, timestamp, board=None):
    sample_dir = Path(root) / user / str(action)
    sample_dir.mkdir(parents=True, exist_ok=True)
    count = len(mask)
    x = np.zeros((count, 6), dtype=np.float32)
    if board is None:
        board = np.zeros((count, 2), dtype=np.float32)
    np.save(sample_dir / f"{sample_id}_x.npy", x)
    np.save(sample_dir / f"{sample_id}_board.npy", np.asarray(board, dtype=np.float32))
    np.save(sample_dir / f"{sample_id}_mask.npy", np.asarray(mask, dtype=np.float32))
    np.save(sample_dir / f"{sample_id}_y.npy", np.zeros((count, 2), dtype=np.float32))
    np.save(
        sample_dir / f"{sample_id}_timestamp.npy",
        np.asarray(timestamp, dtype=np.float64),
    )


def write_labels(raw_root, user, action, sample_id, labels):
    sample_dir = Path(raw_root) / user / str(action)
    sample_dir.mkdir(parents=True, exist_ok=True)
    with (sample_dir / f"{sample_id}_timestamp.txt").open("w") as file:
        for timestamp, label in labels:
            file.write(f"{timestamp} {label}\n")


class WordDatasetTests(unittest.TestCase):
    def test_deskew_aligns_baseline_with_x_axis(self):
        points = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])

        deskewed = deskew_trajectory(points)

        self.assertAlmostEqual(float(deskewed[-1, 1] - deskewed[0, 1]), 0.0, places=6)
        self.assertGreater(float(deskewed[-1, 0] - deskewed[0, 0]), 0.0)

    def test_word_group_maps_actions(self):
        self.assertEqual(word_group(2), "connected")
        self.assertEqual(word_group(3), "connected")
        self.assertEqual(word_group(4), "unconnected")
        self.assertEqual(word_group(5), "unconnected")

    def test_discover_word_samples_groups_runs_and_filters(self):
        with tempfile.TemporaryDirectory() as tmp:
            clean_root = Path(tmp) / "clean"
            raw_root = Path(tmp) / "raw"
            count = 200
            timestamp = 1_000_000.0 + np.arange(count) * 5000.0
            mask = np.zeros(count, dtype=np.float32)
            mask[10:50] = 1.0
            mask[60:90] = 1.0
            mask[120:180] = 1.0
            write_block(clean_root, "user_0", 2, "0", mask, timestamp)
            write_labels(
                raw_root,
                "user_0",
                2,
                "0",
                [
                    (timestamp[0] - 5000.0, "drop"),
                    (timestamp[2], "wrong"),
                    (timestamp[5], "My"),
                    (timestamp[110], "watch"),
                ],
            )

            samples = discover_word_samples(clean_root, raw_root)

            words = {sample.word: sample.runs for sample in samples}
            self.assertEqual(set(words), {"my", "watch"})
            self.assertEqual(words["my"], ((10, 50), (60, 90)))
            self.assertEqual(words["watch"], ((120, 180),))
            self.assertTrue(all(sample.group == "connected" for sample in samples))

    def test_build_word_examples_shape_and_metadata(self):
        count = 120
        mask = np.zeros(count, dtype=np.float32)
        mask[10:100] = 1.0
        timestamp = 2_000_000.0 + np.arange(count) * 5000.0
        with tempfile.TemporaryDirectory() as tmp:
            clean_root = Path(tmp) / "clean"
            raw_root = Path(tmp) / "raw"
            board = np.stack(
                [np.linspace(0, 0.3, count), np.sin(np.linspace(0, 3, count))], axis=1
            ).astype(np.float32)
            write_block(clean_root, "user_0", 4, "0", mask, timestamp, board)
            write_labels(
                raw_root, "user_0", 4, "0", [(timestamp[0], "the")]
            )
            samples = discover_word_samples(clean_root, raw_root)

            examples = build_word_examples(samples, clean_root, resample_frames=64)

            self.assertEqual(len(examples), 1)
            self.assertEqual(examples[0]["word"], "the")
            self.assertEqual(examples[0]["group"], "unconnected")
            self.assertEqual(examples[0]["trajectory"].shape, (64, 2))
            span = examples[0]["trajectory"].max(0) - examples[0]["trajectory"].min(0)
            self.assertAlmostEqual(float(np.linalg.norm(span)), 1.0, places=6)


class WordRecognitionTests(unittest.TestCase):
    def test_dtw_distance_identity_and_shift(self):
        line = np.stack([np.linspace(0, 1, 40), np.zeros(40)], axis=1)

        self.assertAlmostEqual(dtw_distance(line, line), 0.0, places=6)
        shifted = line + 0.1
        self.assertGreater(dtw_distance(line, shifted), 0.0)

    def test_nearest_words_returns_ranked_labels(self):
        templates = [
            ("a", np.zeros((16, 2))),
            ("b", np.ones((16, 2)) * 10.0),
        ]

        ranked = nearest_words(np.zeros((16, 2)) + 0.1, templates, topk=2)

        self.assertEqual(ranked[0], "a")
        self.assertEqual(ranked[1], "b")

    def test_evaluate_nearest_separable_classes(self):
        rng = np.random.default_rng(0)
        train = [
            {
                "word": "left",
                "trajectory": np.zeros((16, 2), dtype=np.float64),
                "group": "connected",
            },
            {
                "word": "right",
                "trajectory": np.full((16, 2), 5.0),
                "group": "connected",
            },
        ]
        test = [
            {"word": "left", "trajectory": rng.normal(0, 0.01, (16, 2)), "group": "connected"},
            {"word": "right", "trajectory": rng.normal(5, 0.01, (16, 2)), "group": "connected"},
        ]

        metrics = evaluate_nearest(train, test, topk=2)

        self.assertEqual(metrics["overall"]["n"], 2)
        self.assertAlmostEqual(metrics["overall"]["top1"], 1.0)
        self.assertAlmostEqual(metrics["connected"]["top5"], 1.0)


if __name__ == "__main__":
    unittest.main()
