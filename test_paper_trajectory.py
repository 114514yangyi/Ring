import dataclasses
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np
import torch

from writing_state.paper_trajectory import (
    PaperTrajectoryConfig,
    SegmentTrajectory,
    WritingRingTrajectoryNet,
    build_trajectory_summary,
    chunk_slices,
    contact_segments,
    integrate_contact_segments,
    load_trajectory_model,
    load_trajectory_predictor,
    predict_sequence,
    stack_windows,
    summarize_segment_metrics,
    trajectory_metrics,
)
from writing_state.board_scale import fit_scale_from_ranges
from writing_state.build_madgwick_dataset import align_raw_offset, build_sample_arrays
from writing_state.madgwick import (
    MadgwickConfig,
    cross_correlation_offset,
    madgwick_linear_acc,
)
from writing_state.train_paper_trajectory import (
    _segment_rows,
    build_lr_scheduler,
    masked_mse_loss,
    masked_trajectory_loss,
    random_split_samples,
    sample_audit,
    train_one_epoch,
    user_split_samples,
)
from writing_state.paper_trajectory_dataset import (
    PaperTrajectoryDataset,
    SegmentRef,
    TrajectorySample,
    build_segment_refs,
    compute_normalization,
    discover_trajectory_samples,
    load_sample_arrays,
    materialize_segment,
    sample_is_valid,
)


DATA_ROOT = Path("/data/huyang/datasets/WritingRing/clean_data_delete_g/data")


def write_sample(root, user, action, sample_id, x, board=None, mask=None, y=None, timestamp=None):
    sample_dir = Path(root) / user / str(action)
    sample_dir.mkdir(parents=True, exist_ok=True)
    x = np.asarray(x, dtype=np.float32)
    count = len(x)
    if board is None:
        board = np.zeros((count, 2), dtype=np.float32)
    if mask is None:
        mask = np.ones(count, dtype=np.float32)
    if y is None:
        y = np.zeros((count, 2), dtype=np.float32)
    if timestamp is None:
        timestamp = np.arange(count, dtype=np.float64)
    paths = {}
    for suffix, array in (
        ("x", x),
        ("board", board),
        ("mask", mask),
        ("y", y),
        ("timestamp", timestamp),
    ):
        path = sample_dir / f"{sample_id}_{suffix}.npy"
        np.save(path, np.asarray(array))
        paths[suffix] = path
    return TrajectorySample(
        user=user,
        action=action,
        sample_id=sample_id,
        x_path=paths["x"],
        board_path=paths["board"],
        mask_path=paths["mask"],
        y_path=paths["y"],
        timestamp_path=paths["timestamp"],
    )


class TrajectoryDatasetTests(unittest.TestCase):
    def test_materialize_pads_head_for_short_sample(self):
        with tempfile.TemporaryDirectory() as tmp:
            x = np.arange(100 * 6, dtype=np.float32).reshape(100, 6)
            y = np.ones((100, 2), dtype=np.float32)
            mask = np.linspace(0.0, 1.0, 100, dtype=np.float32)
            sample = write_sample(tmp, "user_0", 0, "0", x, mask=mask, y=y)

            segment_x, segment_y, segment_mask, segment_board = materialize_segment(
                sample, SegmentRef(0, 0, 100), 150
            )
            self.assertTrue(np.all(segment_board[:50] == 0.0))
            self.assertTrue(np.all(segment_board[50:] == 0.0))

            self.assertEqual(segment_x.shape, (150, 6))
            self.assertTrue(np.all(segment_x[:50] == 0.0))
            self.assertTrue(np.all(segment_y[:50] == 0.0))
            self.assertTrue(np.all(segment_mask[:50] == 0.0))
            np.testing.assert_allclose(segment_x[50:], x)
            np.testing.assert_allclose(segment_y[50:], y)
            np.testing.assert_allclose(segment_mask[50:], mask)

    def test_materialize_symmetric_trim_for_long_sample(self):
        with tempfile.TemporaryDirectory() as tmp:
            x = np.arange(200 * 6, dtype=np.float32).reshape(200, 6)
            sample = write_sample(tmp, "user_0", 0, "0", x)

            refs = build_segment_refs([sample], 150, strategy="pad_trim")
            self.assertEqual([(ref.start, ref.length) for ref in refs], [(25, 150)])

            segment_x, _, _, _ = materialize_segment(sample, refs[0], 150)
            np.testing.assert_allclose(segment_x, x[25:175])

    def test_build_chunk_refs_covers_all_and_drops_tiny_tail(self):
        with tempfile.TemporaryDirectory() as tmp:
            sample = write_sample(tmp, "user_0", 0, "0", np.zeros((20000, 6), np.float32))
            refs = build_segment_refs([sample], 15000, strategy="chunks", min_chunk=1000)
            self.assertEqual([(ref.start, ref.length) for ref in refs], [(0, 15000), (15000, 5000)])

            sample = write_sample(tmp, "user_0", 0, "1", np.zeros((15999, 6), np.float32))
            refs = build_segment_refs([sample], 15000, strategy="chunks", min_chunk=1000)
            self.assertEqual([(ref.start, ref.length) for ref in refs], [(0, 15000)])

    def test_invalid_length_sample_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            sample = write_sample(
                tmp,
                "user_0",
                0,
                "0",
                np.zeros((100, 6), np.float32),
                board=np.zeros((90, 2), np.float32),
            )
            self.assertFalse(sample_is_valid(sample))
            with self.assertRaises(ValueError):
                load_sample_arrays(sample)

    def test_dataset_returns_normalized_segments(self):
        with tempfile.TemporaryDirectory() as tmp:
            rng = np.random.default_rng(0)
            x = rng.normal(size=(150, 6)).astype(np.float32)
            y = rng.normal(size=(150, 2)).astype(np.float32)
            board = rng.normal(size=(150, 2)).astype(np.float32)
            sample = write_sample(tmp, "user_0", 0, "0", x, y=y, board=board)

            mean, std = compute_normalization([sample])
            dataset = PaperTrajectoryDataset(
                [sample], [SegmentRef(0, 0, 150)], mean, std, 150
            )
            segment_x, segment_y, segment_mask, segment_board = dataset[0]

            self.assertIsInstance(segment_x, torch.Tensor)
            self.assertEqual(tuple(segment_x.shape), (150, 6))
            self.assertEqual(tuple(segment_y.shape), (150, 2))
            self.assertEqual(tuple(segment_mask.shape), (150,))
            self.assertEqual(tuple(segment_board.shape), (150, 2))
            self.assertLess(abs(float(segment_x.mean())), 1e-4)
            self.assertAlmostEqual(float(segment_x.std()), 1.0, places=2)
            np.testing.assert_allclose(segment_board.numpy(), board)

    @unittest.skipUnless(DATA_ROOT.exists(), "WritingRing dataset not present")
    def test_official_data_discovery_and_y_semantics(self):
        samples = discover_trajectory_samples(DATA_ROOT)
        self.assertEqual(len(samples), 566)
        self.assertEqual(len({sample.user for sample in samples}), 21)
        invalid = [sample for sample in samples if not sample_is_valid(sample)]
        self.assertEqual(
            {f"{sample.user}/{sample.action}/{sample.sample_id}" for sample in invalid},
            {"user_0/0/0", "user_7/0/0"},
        )

        sample = next(
            item
            for item in samples
            if item.user == "user_0" and item.action == 1 and item.sample_id == "0"
        )
        _, board, mask, y, _ = load_sample_arrays(sample)
        increments = board[1:] - board[:-1]
        contact = np.flatnonzero((mask[:-1] == 1) & (mask[1:] == 1))

        anchor0_corr = _mean_correlation(increments[contact], y[:-1][contact])

        previous = contact[contact > 0]
        anchor1_corr = _mean_correlation(increments[previous - 1], y[previous])

        print(
            f"y semantics: corr0={anchor0_corr:.4f}, corr1={anchor1_corr:.4f}"
        )
        # Dataset truth (verified on official data): y[i] = board[i+1] - board[i].
        self.assertGreater(anchor0_corr, 0.9)
        self.assertGreater(anchor0_corr, anchor1_corr)

        start, stop = first_contact_run(mask, min_frames=50)
        positioned = np.cumsum(y[start : stop - 1], axis=0)
        ground_truth = board[start + 1 : stop] - board[start]
        integrated_corr = _mean_correlation(positioned, ground_truth)
        print(f"integrated semantics: corr={integrated_corr:.5f}")
        self.assertGreater(integrated_corr, 0.99)


def first_contact_run(mask, min_frames=5):
    padded = np.concatenate([[0], (np.asarray(mask) > 0).astype(np.int8), [0]])
    changes = np.flatnonzero(np.diff(padded))
    for start, stop in zip(changes[::2], changes[1::2]):
        if stop - start >= min_frames:
            return int(start), int(stop)
    raise AssertionError("no contact run long enough")


def _mean_correlation(left, right):
    correlations = []
    for axis in range(left.shape[1]):
        a = left[:, axis]
        b = right[:, axis]
        if np.std(a) < 1e-9 or np.std(b) < 1e-9:
            continue
        correlations.append(float(np.corrcoef(a, b)[0, 1]))
    return float(np.mean(correlations)) if correlations else 0.0


class TrajectoryModelTests(unittest.TestCase):
    def test_stack_windows_alignment(self):
        x = torch.arange(20, dtype=torch.float32).unsqueeze(1).repeat(1, 6)
        windows = stack_windows(x)

        self.assertEqual(tuple(windows.shape), (8, 6, 13))
        for center in range(8):
            expected = torch.arange(center, center + 13, dtype=torch.float32)
            self.assertTrue(torch.allclose(windows[center, 0], expected))

    def test_forward_shape_and_nan_edges(self):
        model = WritingRingTrajectoryNet(PaperTrajectoryConfig())
        model.eval()
        out, _ = model(torch.randn(2, 20, 6))

        self.assertEqual(tuple(out.shape), (2, 20, 2))
        self.assertTrue(torch.isnan(out[:, :6]).all())
        self.assertTrue(torch.isnan(out[:, 14:]).all())
        self.assertFalse(torch.isnan(out[:, 6:14]).any())

    def test_lstm_state_is_carried_across_calls(self):
        torch.manual_seed(0)
        model = WritingRingTrajectoryNet(PaperTrajectoryConfig())
        model.eval()
        x = torch.randn(1, 40, 6)

        with torch.no_grad():
            out_full, _ = model(x)
            out_first, state = model(x[:, :26])
            out_second, _ = model(x[:, 14:], state)

        self.assertTrue(torch.allclose(out_first[:, 6:20], out_full[:, 6:20], atol=1e-5))
        self.assertTrue(
            torch.allclose(out_second[:, 6:20], out_full[:, 20:34], atol=1e-5)
        )

    def test_chunk_slices_remainder(self):
        self.assertEqual(chunk_slices(25, 10, 6), [(0, 16, 6, 10), (4, 25, 10, 19)])
        self.assertEqual(chunk_slices(25, 20, 6), [(0, 25, 6, 19)])
        self.assertEqual(chunk_slices(13, 13, 6), [(0, 13, 6, 7)])
        self.assertEqual(chunk_slices(10, 10, 6), [])

    def test_chunked_inference_matches_full(self):
        torch.manual_seed(0)
        model = WritingRingTrajectoryNet(PaperTrajectoryConfig())
        model.eval()
        x = np.random.default_rng(0).normal(size=(50, 6)).astype(np.float32)

        with torch.no_grad():
            full = model(torch.from_numpy(x)[None])[0][0].numpy()
        chunked = predict_sequence(model, x, chunk_len=13)

        valid = ~np.isnan(full[:, 0])
        self.assertTrue(np.allclose(full[valid], chunked[valid], atol=1e-5))
        self.assertTrue(np.isnan(chunked[~valid]).all())

    def test_checkpoint_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = PaperTrajectoryConfig(dropout=0.0)
            model = WritingRingTrajectoryNet(config)
            model.eval()
            payload = {
                "state_dict": model.state_dict(),
                "mean": [0.0] * 6,
                "std": [1.0] * 6,
                "board_mm_scale": [240.0, 169.5],
                "config": dataclasses.asdict(config),
            }
            path = Path(tmp) / "trajectory.pt"
            torch.save(payload, path)

            loaded = load_trajectory_model(path)
            predictor = load_trajectory_predictor(path)
            x = np.random.default_rng(1).normal(size=(30, 6)).astype(np.float32)

            with torch.no_grad():
                expected = loaded(torch.from_numpy(x)[None])[0][0].numpy()
            got = predictor.predict_deltas(x)

            valid = ~np.isnan(expected[:, 0])
            np.testing.assert_allclose(expected[valid], got[valid], atol=1e-6)

    def test_summary_has_paper_shapes(self):
        summary = build_trajectory_summary()

        self.assertEqual(summary["window_frames"], 13)
        self.assertEqual(summary["window_offset"], 6)
        self.assertEqual(summary["tcn_channels"], [16, 16, 32, 32, 64, 128])
        self.assertEqual(summary["tcn_style"], "valid")
        self.assertEqual(summary["segment_frames"], 15000)
        self.assertGreater(summary["parameter_count"], 0)

    def test_dilated_tcn_forward_shape_and_nan_edges(self):
        config = PaperTrajectoryConfig(tcn_style="dilated", tcn_channels=(32, 64, 128))
        model = WritingRingTrajectoryNet(config)
        model.eval()

        out, _ = model(torch.randn(2, 20, 6))

        self.assertEqual(tuple(out.shape), (2, 20, 2))
        self.assertTrue(torch.isnan(out[:, :6]).all())
        self.assertTrue(torch.isnan(out[:, 14:]).all())
        self.assertFalse(torch.isnan(out[:, 6:14]).any())
        self.assertEqual(model.window_features(torch.randn(3, 13, 6)).shape, (3, 1, 128))

    def test_dilated_tcn_chunked_inference_matches_full(self):
        torch.manual_seed(0)
        config = PaperTrajectoryConfig(tcn_style="dilated", tcn_channels=(32, 64, 128))
        model = WritingRingTrajectoryNet(config)
        model.eval()
        x = np.random.default_rng(3).normal(size=(45, 6)).astype(np.float32)

        with torch.no_grad():
            full = model(torch.from_numpy(x)[None])[0][0].numpy()
        chunked = predict_sequence(model, x, chunk_len=11)

        valid = ~np.isnan(full[:, 0])
        self.assertTrue(np.allclose(full[valid], chunked[valid], atol=1e-5))

    def test_valid_tcn_requires_matching_feature_dim(self):
        with self.assertRaises(ValueError):
            PaperTrajectoryConfig(tcn_channels=(16, 32)).validate()

    def test_two_layer_lstm_forward_and_state(self):
        config = PaperTrajectoryConfig(lstm_layers=2)
        model = WritingRingTrajectoryNet(config)
        model.eval()

        out, state = model(torch.randn(2, 20, 6))

        self.assertEqual(tuple(out.shape), (2, 20, 2))
        self.assertEqual(state[0].shape[0], 2)
        self.assertTrue(torch.isnan(out[:, :6]).all())

    def test_two_layer_lstm_chunked_inference_matches_full(self):
        torch.manual_seed(0)
        config = PaperTrajectoryConfig(lstm_layers=2)
        model = WritingRingTrajectoryNet(config)
        model.eval()
        x = np.random.default_rng(4).normal(size=(40, 6)).astype(np.float32)

        with torch.no_grad():
            full = model(torch.from_numpy(x)[None])[0][0].numpy()
        chunked = predict_sequence(model, x, chunk_len=13)

        valid = ~np.isnan(full[:, 0])
        self.assertTrue(np.allclose(full[valid], chunked[valid], atol=1e-5))

    def test_lstm_layers_must_be_positive(self):
        with self.assertRaises(ValueError):
            PaperTrajectoryConfig(lstm_layers=0).validate()

    def test_bidirectional_forward_shape_and_nan_edges(self):
        config = PaperTrajectoryConfig(bidirectional=True)
        model = WritingRingTrajectoryNet(config)
        model.eval()

        out, _ = model(torch.randn(2, 20, 6))

        self.assertEqual(tuple(out.shape), (2, 20, 2))
        self.assertTrue(torch.isnan(out[:, :6]).all())
        self.assertTrue(torch.isnan(out[:, 14:]).all())
        self.assertFalse(torch.isnan(out[:, 6:14]).any())

    def test_predict_sequence_rejects_chunking_for_bidirectional(self):
        model = WritingRingTrajectoryNet(PaperTrajectoryConfig(bidirectional=True))
        model.eval()

        with self.assertRaises(ValueError):
            predict_sequence(model, np.zeros((40, 6), dtype=np.float32), chunk_len=20)

    def test_bidirectional_full_sequence_prediction_matches_forward(self):
        torch.manual_seed(0)
        model = WritingRingTrajectoryNet(PaperTrajectoryConfig(bidirectional=True))
        model.eval()
        x = np.random.default_rng(6).normal(size=(35, 6)).astype(np.float32)

        with torch.no_grad():
            full = model(torch.from_numpy(x)[None])[0][0].numpy()
        predicted = predict_sequence(model, x)

        valid = ~np.isnan(full[:, 0])
        np.testing.assert_allclose(full[valid], predicted[valid], atol=1e-6)


class TrajectoryMetricTests(unittest.TestCase):
    def test_contact_segments_split_and_min_frames_filter(self):
        mask = np.array([0, 1, 1, 0, 1, 1, 1, 0], dtype=np.float32)

        self.assertEqual(contact_segments(mask, min_frames=2), [(1, 3), (4, 7)])
        self.assertEqual(contact_segments(mask, min_frames=3), [(4, 7)])

    def test_integrate_constant_velocity_is_straight_line(self):
        deltas = np.tile(np.array([[0.1, 0.0]], dtype=np.float32), (20, 1))
        mask = np.ones(20, dtype=np.float32)

        segments = integrate_contact_segments(deltas, mask, min_frames=5)

        self.assertEqual(len(segments), 1)
        segment = segments[0]
        self.assertEqual(segment.start, 0)
        self.assertEqual(segment.end, 20)
        self.assertEqual(segment.xy.shape, (20, 2))
        np.testing.assert_allclose(segment.xy[:, 0], np.arange(20) * 0.1, atol=1e-6)
        self.assertEqual(segment.lost_frames, 0)

    def test_integrate_skips_nan_edges(self):
        deltas = np.tile(np.array([[0.1, 0.0]], dtype=np.float32), (30, 1))
        deltas[:6] = np.nan
        deltas[-6:] = np.nan
        mask = np.ones(30, dtype=np.float32)

        segments = integrate_contact_segments(deltas, mask, min_frames=5)

        self.assertEqual(len(segments), 1)
        segment = segments[0]
        self.assertEqual((segment.start, segment.end), (6, 25))
        self.assertEqual(segment.xy.shape, (19, 2))
        np.testing.assert_allclose(segment.xy[:, 0], np.arange(19) * 0.1, atol=1e-6)
        self.assertEqual(segment.lost_frames, 11)

    def test_normalized_metric_known_error(self):
        gt = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
        pred = gt + np.array([0.1, 0.0], dtype=np.float32)

        metrics = trajectory_metrics(pred, gt, mm_scale=(10.0, 10.0))

        self.assertAlmostEqual(metrics["normalized_mean"], 0.1 / np.sqrt(2), places=6)
        self.assertAlmostEqual(metrics["mm_mean"], 1.0, places=6)
        self.assertEqual(metrics["frac_gt_0.1"], 0.0)
        self.assertEqual(metrics["skipped_degenerate"], 0)
        self.assertEqual(metrics["n_points"], 4)

    def test_degenerate_bbox_is_skipped_not_zero_division(self):
        gt = np.ones((3, 2), dtype=np.float32)
        pred = np.ones((3, 2), dtype=np.float32)

        metrics = trajectory_metrics(pred, gt)

        self.assertEqual(metrics["skipped_degenerate"], 1)
        self.assertTrue(np.isnan(metrics["normalized_mean"]))

    def test_empty_mask_returns_empty_results(self):
        deltas = np.zeros((10, 2), dtype=np.float32)
        mask = np.zeros(10, dtype=np.float32)

        self.assertEqual(integrate_contact_segments(deltas, mask), [])
        summary = summarize_segment_metrics([])
        self.assertEqual(summary["segments"], 0)
        self.assertTrue(np.isnan(summary["normalized_mean"]))

    def test_summarize_segment_metrics_weights_by_points(self):
        rows = [
            {
                "n_points": 100,
                "normalized_mean": 0.1,
                "normalized_p50": 0.1,
                "normalized_p90": 0.2,
                "frac_gt_0.1": 0.5,
                "mm_mean": 2.0,
                "skipped_degenerate": 0,
            },
            {
                "n_points": 10,
                "normalized_mean": 0.2,
                "normalized_p50": 0.2,
                "normalized_p90": 0.3,
                "frac_gt_0.1": 0.1,
                "mm_mean": 4.0,
                "skipped_degenerate": 0,
            },
            {
                "n_points": 3,
                "normalized_mean": float("nan"),
                "skipped_degenerate": 1,
            },
        ]

        summary = summarize_segment_metrics(rows)

        self.assertEqual(summary["segments"], 2)
        self.assertEqual(summary["skipped_degenerate"], 1)
        self.assertAlmostEqual(summary["normalized_mean"], (0.1 * 100 + 0.2 * 10) / 110)
        self.assertAlmostEqual(summary["mm_mean"], (2.0 * 100 + 4.0 * 10) / 110)


class _CountingOptimizer:
    def __init__(self, inner):
        self.inner = inner
        self.steps = 0

    def zero_grad(self, *args, **kwargs):
        return self.inner.zero_grad(*args, **kwargs)

    def step(self, *args, **kwargs):
        self.steps += 1
        return self.inner.step(*args, **kwargs)


class TrainingLoopTests(unittest.TestCase):
    def _loader(self):
        inputs = torch.randn(2, 25, 6)
        targets = torch.zeros(2, 25, 2)
        masks = torch.ones(2, 25)
        boards = torch.zeros(2, 25, 2)
        return torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(inputs, targets, masks, boards), batch_size=2
        )

    def test_train_one_epoch_steps_once_per_batch(self):
        torch.manual_seed(0)
        model = WritingRingTrajectoryNet(PaperTrajectoryConfig())
        optimizer = _CountingOptimizer(torch.optim.Adam(model.parameters(), lr=1e-3))

        train_one_epoch(
            model, self._loader(), optimizer, chunk_len=10, device="cpu",
            update_every="batch",
        )

        self.assertEqual(optimizer.steps, 1)

    def test_train_one_epoch_can_step_every_chunk(self):
        torch.manual_seed(0)
        model = WritingRingTrajectoryNet(PaperTrajectoryConfig())
        optimizer = _CountingOptimizer(torch.optim.Adam(model.parameters(), lr=1e-3))

        train_one_epoch(
            model, self._loader(), optimizer, chunk_len=10, device="cpu",
            update_every="chunk",
        )

        self.assertEqual(optimizer.steps, 2)


class LrSchedulerTests(unittest.TestCase):
    def test_cosine_schedule_decays_towards_eta_min(self):
        parameter = torch.nn.Parameter(torch.zeros(1))
        optimizer = torch.optim.Adam([parameter], lr=1e-3)
        scheduler = build_lr_scheduler(optimizer, epochs=10, schedule="cosine")

        learning_rates = []
        for _ in range(10):
            optimizer.step()
            scheduler.step()
            learning_rates.append(optimizer.param_groups[0]["lr"])

        self.assertLess(learning_rates[-1], 2e-5)
        self.assertGreater(learning_rates[0], 9e-4)

    def test_constant_schedule_returns_none(self):
        optimizer = torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))], lr=1e-3)

        self.assertIsNone(build_lr_scheduler(optimizer, epochs=10, schedule="constant"))


class TrainingUtilityTests(unittest.TestCase):
    @staticmethod
    def _samples(count):
        return [
            types.SimpleNamespace(user=f"user_{i % 5}", action=0, sample_id=str(i))
            for i in range(count)
        ]

    def test_random_split_ratios_and_determinism(self):
        samples = self._samples(100)

        train_a, val_a, test_a, info_a = random_split_samples(samples, seed=7)
        train_b, val_b, test_b, _ = random_split_samples(samples, seed=7)

        self.assertEqual((len(train_a), len(val_a), len(test_a)), (70, 10, 20))
        self.assertEqual(info_a["strategy"], "random")
        self.assertEqual(
            [sample.sample_id for sample in train_a],
            [sample.sample_id for sample in train_b],
        )
        self.assertEqual(
            [sample.sample_id for sample in val_a],
            [sample.sample_id for sample in val_b],
        )
        self.assertEqual(
            [sample.sample_id for sample in test_a],
            [sample.sample_id for sample in test_b],
        )

        _, _, test_other, _ = random_split_samples(samples, seed=8)
        self.assertNotEqual(
            [sample.sample_id for sample in test_a],
            [sample.sample_id for sample in test_other],
        )

    def test_user_split_has_no_user_overlap(self):
        samples = self._samples(50)

        train, val, test, info = user_split_samples(samples)

        train_users = {sample.user for sample in train}
        val_users = {sample.user for sample in val}
        test_users = {sample.user for sample in test}
        self.assertFalse(train_users & val_users)
        self.assertFalse(train_users & test_users)
        self.assertFalse(val_users & test_users)
        self.assertEqual(
            train_users | val_users | test_users,
            {"user_0", "user_1", "user_2", "user_3", "user_4"},
        )
        self.assertEqual(info["strategy"], "user")

    def test_masked_mse_ignores_non_contact_and_nan(self):
        pred = torch.zeros(1, 4, 2)
        target = torch.ones(1, 4, 2)
        mask = torch.zeros(1, 4)
        mask[0, 0] = 1.0

        loss = masked_mse_loss(pred, target, mask)
        self.assertAlmostEqual(float(loss), 1.0, places=6)

        pred[0, 0] = torch.tensor([float("nan"), 0.0])
        loss = masked_mse_loss(pred, target, mask)
        self.assertAlmostEqual(float(loss), 0.0, places=6)

    def test_masked_mse_empty_mask_returns_zero(self):
        pred = torch.ones(2, 5, 2)
        target = torch.zeros(2, 5, 2)
        mask = torch.zeros(2, 5)

        loss = masked_mse_loss(pred, target, mask)

        self.assertEqual(float(loss), 0.0)

    def test_masked_trajectory_loss_zero_for_perfect_prediction(self):
        target = torch.randn(1, 60, 2)
        mask = torch.ones(1, 60)

        loss = masked_trajectory_loss(target.clone(), target, mask, window=20)

        self.assertAlmostEqual(float(loss), 0.0, places=6)

    def test_masked_trajectory_loss_scales_with_bias_squared(self):
        target = torch.zeros(1, 40, 2)
        mask = torch.ones(1, 40)

        small = masked_trajectory_loss(
            torch.full((1, 40, 2), 0.01), target, mask, window=20
        )
        large = masked_trajectory_loss(
            torch.full((1, 40, 2), 0.02), target, mask, window=20
        )

        self.assertGreater(float(small), 0.0)
        self.assertAlmostEqual(float(large) / float(small), 4.0, places=3)

    def test_masked_trajectory_loss_ignores_air_windows(self):
        target = torch.ones(1, 40, 2)
        pred = torch.full((1, 40, 2), 5.0)
        mask = torch.zeros(1, 40)
        mask[0, 10:30] = 1.0
        pred[0, 10:30] = 1.0

        loss = masked_trajectory_loss(pred, target, mask, window=20)

        self.assertEqual(float(loss), 0.0)

    def test_dataset_per_sample_normalization_zero_mean(self):
        with tempfile.TemporaryDirectory() as tmp:
            rng = np.random.default_rng(5)
            x = (rng.normal(size=(150, 6)) * 3 + 7).astype(np.float32)
            sample = write_sample(tmp, "user_0", 0, "0", x)
            mean, std = compute_normalization([sample])
            dataset = PaperTrajectoryDataset(
                [sample],
                [SegmentRef(0, 0, 150)],
                mean,
                std,
                150,
                input_norm="per_sample",
            )

            segment_x, _, _, _ = dataset[0]

            self.assertLess(float(segment_x.mean().abs()), 1e-4)
            self.assertAlmostEqual(float(segment_x.std()), 1.0, places=2)


class SegmentGroundTruthTests(unittest.TestCase):
    @staticmethod
    def _ramp():
        board = np.zeros((20, 2), dtype=np.float32)
        board[:, 1] = 0.5
        board[1:, 0] = np.cumsum(np.full(19, 0.01, dtype=np.float32))
        deltas = np.zeros((20, 2), dtype=np.float32)
        deltas[:-1] = board[1:] - board[:-1]
        return board, deltas

    def test_segment_rows_uses_board_as_ground_truth(self):
        board, deltas = self._ramp()
        rows = _segment_rows(deltas, board, np.ones(20, dtype=np.float32), (1.0, 1.0))

        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["normalized_mean"], 0.0, places=6)

    def test_segment_rows_penalizes_zero_prediction(self):
        board, _ = self._ramp()
        rows = _segment_rows(
            np.zeros((20, 2), dtype=np.float32),
            board,
            np.ones(20, dtype=np.float32),
            None,
        )

        self.assertEqual(len(rows), 1)
        self.assertGreater(rows[0]["normalized_mean"], 0.4)

    def test_sample_audit_counts_invalid_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            good = write_sample(tmp, "user_0", 0, "0", np.zeros((50, 6), np.float32))
            bad = write_sample(
                tmp,
                "user_1",
                0,
                "0",
                np.zeros((50, 6), np.float32),
                board=np.zeros((40, 2), np.float32),
            )

            audit = sample_audit([good, bad])

            self.assertEqual(audit["discovered"], 2)
            self.assertEqual(audit["invalid_count"], 1)
            self.assertEqual(audit["invalid_ids"], ["user_1/0/0"])


class MadgwickTests(unittest.TestCase):
    def test_cross_correlation_recovers_integer_shift(self):
        rng = np.random.default_rng(0)
        clean = rng.normal(size=400)
        raw = np.concatenate([np.zeros(137), clean, np.zeros(61)])

        offset = cross_correlation_offset(clean, raw)

        self.assertEqual(offset, 137)

    def test_static_gravity_is_removed(self):
        count = 3500
        acc = np.tile(np.array([0.0, 0.0, 9.81]), (count, 1))
        gyro = np.zeros((count, 3))

        linear = madgwick_linear_acc(acc, gyro, MadgwickConfig())

        self.assertLess(np.abs(linear[3000:]).max(), 1e-3)

    def test_tilted_static_gravity_is_removed(self):
        count = 3500
        acc = np.tile(np.array([4.905, 0.0, 8.495]), (count, 1))
        gyro = np.zeros((count, 3))

        linear = madgwick_linear_acc(acc, gyro, MadgwickConfig())

        self.assertLess(np.linalg.norm(linear[3000:], axis=1).max(), 0.01)

    def test_align_raw_offset_refines_shift(self):
        rng = np.random.default_rng(1)
        clean = rng.normal(size=(500, 3))
        raw = np.concatenate([np.zeros((211, 3)), clean, np.zeros((90, 3))])

        offset, correlation = align_raw_offset(clean, raw)

        self.assertEqual(offset, 211)
        self.assertGreater(correlation, 0.99)

    def test_build_sample_arrays_replaces_acc_and_keeps_gyro(self):
        rng = np.random.default_rng(2)
        count = 700
        clean = np.empty((count, 6), dtype=np.float32)
        clean[:, :3] = rng.normal(scale=0.1, size=(count, 3)).astype(np.float32)
        clean[:, 3:] = rng.normal(scale=1.0, size=(count, 3)).astype(np.float32)
        raw = np.empty((count + 300, 7), dtype=np.float64)
        raw[:300, :3] = 0.0
        raw[:300, 3:6] = 0.0
        raw[300:, :3] = [0.0, 0.0, 9.81]
        raw[300:, 3:6] = clean[:, 3:]
        raw[:, 6] = 0.0

        built = build_sample_arrays(clean, raw, MadgwickConfig())

        self.assertIsNotNone(built)
        np.testing.assert_allclose(built[:, 3:], clean[:, 3:], atol=1e-6)
        expected_head = madgwick_linear_acc(
            raw[: 300 + count, :3], raw[: 300 + count, 3:6], MadgwickConfig()
        )[300:]
        np.testing.assert_allclose(built[:, :3], expected_head, atol=1e-5)


class BoardScaleTests(unittest.TestCase):
    def test_fit_scale_recovers_known_ratio(self):
        rng = np.random.default_rng(0)
        raw = rng.uniform(0.0, 100.0, size=(500, 2))
        clean = raw * 2.0

        scale_x, scale_y = fit_scale_from_ranges(raw, clean)

        self.assertAlmostEqual(scale_x, 2.0, places=3)
        self.assertAlmostEqual(scale_y, 2.0, places=3)

    def test_fit_scale_degenerate_returns_nan(self):
        raw = np.ones((10, 2), dtype=np.float64)

        scale_x, scale_y = fit_scale_from_ranges(raw, raw)

        self.assertTrue(np.isnan(scale_x))
        self.assertTrue(np.isnan(scale_y))


if __name__ == "__main__":
    unittest.main()
