import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from writing_state.character_model import (
    CharacterCNN,
    build_character_summary,
    load_character_model,
)
from writing_state.train_character_classifier import (
    build_recon_examples,
    character_split,
    confusion_matrix,
    predict_run_points,
    topk_accuracy,
)
from writing_state.character_dataset import (
    CharacterSample,
    build_gt_examples,
    contact_runs,
    discover_character_samples,
    extract_gt_trajectory,
    normalize_trajectory,
    resample_trajectory,
)


def write_character_block(root, user, action, sample_id, mask, timestamp, board=None):
    sample_dir = Path(root) / user / str(action)
    sample_dir.mkdir(parents=True, exist_ok=True)
    count = len(mask)
    x = np.zeros((count, 6), dtype=np.float32)
    if board is None:
        board = np.zeros((count, 2), dtype=np.float32)
    y = np.zeros((count, 2), dtype=np.float32)
    np.save(sample_dir / f"{sample_id}_x.npy", x)
    np.save(sample_dir / f"{sample_id}_board.npy", np.asarray(board, dtype=np.float32))
    np.save(sample_dir / f"{sample_id}_mask.npy", np.asarray(mask, dtype=np.float32))
    np.save(sample_dir / f"{sample_id}_y.npy", y)
    np.save(sample_dir / f"{sample_id}_timestamp.npy", np.asarray(timestamp, dtype=np.float64))


def write_label_file(raw_root, user, action, sample_id, labels):
    sample_dir = Path(raw_root) / user / str(action)
    sample_dir.mkdir(parents=True, exist_ok=True)
    with (sample_dir / f"{sample_id}_timestamp.txt").open("w") as file:
        for timestamp, label in labels:
            file.write(f"{timestamp} {label}\n")


class ContactRunTests(unittest.TestCase):
    def test_contact_runs_filters_short_runs(self):
        mask = np.array([0, 1, 1, 1, 0, 1, 0, 1, 1], dtype=np.float32)

        self.assertEqual(contact_runs(mask, min_frames=2), [(1, 4), (7, 9)])
        self.assertEqual(contact_runs(mask, min_frames=1), [(1, 4), (5, 6), (7, 9)])


class TrajectoryFeatureTests(unittest.TestCase):
    def test_resample_trajectory_uniform_along_arc_length(self):
        points = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])

        resampled = resample_trajectory(points, 5)

        np.testing.assert_allclose(resampled[:, 0], np.linspace(0.0, 2.0, 5))
        np.testing.assert_allclose(resampled[:, 1], 0.0)

    def test_resample_trajectory_degenerate_input_repeats_point(self):
        points = np.array([[3.0, 4.0]])

        resampled = resample_trajectory(points, 4)

        np.testing.assert_allclose(resampled, np.tile([3.0, 4.0], (4, 1)))

    def test_normalize_trajectory_unit_bbox_diagonal(self):
        points = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]])

        normalized = normalize_trajectory(points)

        np.testing.assert_allclose(normalized[0], [0.0, 0.0])
        span = normalized.max(axis=0) - normalized.min(axis=0)
        self.assertAlmostEqual(float(np.linalg.norm(span)), 1.0, places=6)


class CharacterDiscoveryTests(unittest.TestCase):
    def test_discover_groups_runs_by_label_interval_and_skips_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            clean_root = Path(tmp) / "clean"
            raw_root = Path(tmp) / "raw"
            count = 100
            timestamp = 1_000_000.0 + np.arange(count) * 5000.0
            mask = np.zeros(count, dtype=np.float32)
            mask[5:15] = 1.0
            mask[30:45] = 1.0
            mask[50:60] = 1.0
            mask[70:90] = 1.0
            write_character_block(clean_root, "user_0", 0, "0", mask, timestamp)
            write_label_file(
                raw_root,
                "user_0",
                0,
                "0",
                [
                    (timestamp[0] - 5000.0, "a"),
                    (timestamp[10], "wrong"),
                    (timestamp[20], "B"),
                    (timestamp[70], "c"),
                ],
            )

            samples = discover_character_samples(clean_root, raw_root)

            labels = {sample.char: sample.runs for sample in samples}
            self.assertEqual(set(labels), {"B", "C"})
            self.assertEqual(labels["B"], ((30, 45), (50, 60)))
            self.assertEqual(labels["C"], ((70, 90),))
            self.assertTrue(all(sample.label == ord(sample.char) - ord("A") for sample in samples))

    def test_extract_gt_trajectory_relative_runs_and_normalized(self):
        count = 80
        board = np.stack(
            [np.arange(count) * 0.01, np.arange(count) * -0.005], axis=1
        ).astype(np.float32)
        sample = CharacterSample(
            user="user_0",
            action=0,
            sample_id="0",
            label=1,
            char="B",
            runs=((10, 30), (40, 60)),
        )

        trajectory = extract_gt_trajectory(sample, board, resample_frames=64)

        self.assertEqual(trajectory.shape, (64, 2))
        np.testing.assert_allclose(trajectory[0], [0.0, 0.0], atol=1e-6)
        span = trajectory.max(axis=0) - trajectory.min(axis=0)
        self.assertAlmostEqual(float(np.linalg.norm(span)), 1.0, places=6)

    def test_build_gt_examples_returns_metadata(self):
        count = 60
        mask = np.zeros(count, dtype=np.float32)
        mask[10:40] = 1.0
        mask[50:60] = 1.0
        timestamp = 2_000_000.0 + np.arange(count) * 5000.0
        with tempfile.TemporaryDirectory() as tmp:
            clean_root = Path(tmp) / "clean"
            raw_root = Path(tmp) / "raw"
            write_character_block(
                clean_root, "user_1", 1, "0", mask, timestamp,
                board=np.stack([np.arange(count) * 0.02, np.zeros(count)], axis=1),
            )
            write_label_file(
                raw_root, "user_1", 1, "0",
                [(timestamp[0], "x"), (timestamp[45], "y")],
            )
            samples = discover_character_samples(clean_root, raw_root)

            examples = build_gt_examples(samples, clean_root, resample_frames=32)

            self.assertEqual(len(examples), 2)
            self.assertEqual({example["char"] for example in examples}, {"X", "Y"})
            self.assertEqual(examples[0]["trajectory"].shape, (32, 2))
            self.assertEqual(examples[0]["user"], "user_1")


class CharacterModelTests(unittest.TestCase):
    def test_forward_outputs_class_logits(self):
        model = CharacterCNN()
        model.eval()

        logits = model(torch.zeros(4, 64, 2))
        probabilities = torch.softmax(logits, dim=-1)

        self.assertEqual(tuple(logits.shape), (4, 26))
        np.testing.assert_allclose(
            probabilities.sum(dim=-1).detach().numpy(), np.ones(4), atol=1e-5
        )

    def test_checkpoint_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = CharacterCNN()
            model.eval()
            path = Path(tmp) / "character.pt"
            torch.save({"state_dict": model.state_dict()}, path)

            loaded = load_character_model(path)
            inputs = torch.randn(3, 64, 2)
            with torch.no_grad():
                expected = model(inputs)
                actual = loaded(inputs)

            np.testing.assert_allclose(
                expected.detach().numpy(), actual.detach().numpy(), atol=1e-6
            )

    def test_summary_reports_architecture(self):
        summary = build_character_summary()

        self.assertEqual(summary["num_classes"], 26)
        self.assertEqual(summary["resample_frames"], 64)
        self.assertGreater(summary["parameter_count"], 0)


class _StubPredictor:
    def predict_deltas(self, x):
        return np.tile(np.array([[0.1, 0.0]], dtype=np.float32), (len(x), 1))


class CharacterTrainingUtilityTests(unittest.TestCase):
    @staticmethod
    def _samples(count):
        return [
            CharacterSample(
                user=f"user_{index % 5}",
                action=0,
                sample_id=str(index),
                label=index % 26,
                char=chr(ord("A") + index % 26),
                runs=((10, 20),),
            )
            for index in range(count)
        ]

    def test_character_split_user_has_no_overlap(self):
        samples = self._samples(50)

        train, val, test, info = character_split(samples, "user")

        train_users = {sample.user for sample in train}
        val_users = {sample.user for sample in val}
        test_users = {sample.user for sample in test}
        self.assertFalse(train_users & val_users)
        self.assertFalse(train_users & test_users)
        self.assertFalse(val_users & test_users)
        self.assertEqual(info["strategy"], "user")

    def test_character_split_random_is_deterministic(self):
        samples = self._samples(100)

        train_a, val_a, test_a, _ = character_split(samples, "random", seed=3)
        train_b, val_b, test_b, _ = character_split(samples, "random", seed=3)

        self.assertEqual((len(train_a), len(val_a), len(test_a)), (70, 10, 20))
        self.assertEqual(
            [sample.sample_id for sample in train_a],
            [sample.sample_id for sample in train_b],
        )
        self.assertEqual(
            [sample.sample_id for sample in test_a],
            [sample.sample_id for sample in test_b],
        )

    def test_topk_accuracy(self):
        logits = np.array(
            [[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.2, 0.3, 0.5]], dtype=np.float32
        )
        labels = np.array([0, 2, 2])

        self.assertAlmostEqual(topk_accuracy(logits, labels, k=1), 2 / 3)
        self.assertAlmostEqual(topk_accuracy(logits, labels, k=2), 2 / 3)

    def test_confusion_matrix_counts(self):
        predictions = np.array([0, 1, 1])
        labels = np.array([0, 1, 2])

        matrix = confusion_matrix(predictions, labels, num_classes=3)

        self.assertEqual(matrix.shape, (3, 3))
        self.assertEqual(int(matrix[0, 0]), 1)
        self.assertEqual(int(matrix[1, 1]), 1)
        self.assertEqual(int(matrix[2, 1]), 1)

    def test_predict_run_points_relative_to_run_start(self):
        mask = np.zeros(30, dtype=np.float32)
        mask[10:20] = 1.0

        run_points = predict_run_points(_StubPredictor(), np.zeros((30, 6)), mask)

        self.assertEqual(set(run_points), {(10, 20)})
        points = run_points[(10, 20)]
        self.assertEqual(points.shape, (10, 2))
        np.testing.assert_allclose(points[0], [0.0, 0.0])
        np.testing.assert_allclose(points[-1], [0.9, 0.0], atol=1e-6)

    def test_build_recon_examples_shapes_and_normalization(self):
        samples = [
            CharacterSample(
                user="user_0", action=0, sample_id="0", label=1, char="B",
                runs=((10, 20), (30, 40)),
            )
        ]
        run_points = {
            ("user_0", 0, "0"): {
                (10, 20): np.stack(
                    [np.arange(10) * 0.1, np.zeros(10)], axis=1
                ).astype(np.float32),
                (30, 40): np.stack(
                    [np.arange(10) * 0.1, np.ones(10) * 0.5], axis=1
                ).astype(np.float32),
            }
        }

        examples = build_recon_examples(samples, run_points, resample_frames=32)

        self.assertEqual(len(examples), 1)
        trajectory = examples[0]["trajectory"]
        self.assertEqual(trajectory.shape, (32, 2))
        np.testing.assert_allclose(trajectory[0], [0.0, 0.0], atol=1e-6)
        span = trajectory.max(axis=0) - trajectory.min(axis=0)
        self.assertAlmostEqual(float(np.linalg.norm(span)), 1.0, places=6)
        self.assertEqual(examples[0]["source"], "recon")


if __name__ == "__main__":
    unittest.main()
