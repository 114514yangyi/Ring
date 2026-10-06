from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def dtw_distance(
    left: np.ndarray, right: np.ndarray, band_ratio: float = 0.25
) -> float:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if len(a) == 0 or len(b) == 0:
        raise ValueError("dtw requires non-empty trajectories")
    rows, columns = len(a), len(b)
    band = max(abs(rows - columns), int(round(band_ratio * max(rows, columns))), 1)
    infinity = float("inf")
    previous = np.full(columns + 1, infinity)
    previous[0] = 0.0
    for i in range(1, rows + 1):
        current = np.full(columns + 1, infinity)
        low = max(1, i - band)
        high = min(columns, i + band)
        for j in range(low, high + 1):
            cost = float(np.linalg.norm(a[i - 1] - b[j - 1]))
            current[j] = cost + min(previous[j], current[j - 1], previous[j - 1])
        previous = current
    return float(previous[columns])


def _fast_dtw(left: np.ndarray, right: np.ndarray) -> float | None:
    try:
        from dtaidistance import dtw_ndim
    except ImportError:
        return None
    return float(dtw_ndim.distance(left, right, use_c=True))


def nearest_words(
    trajectory: np.ndarray,
    templates: Sequence[tuple[str, np.ndarray]],
    topk: int = 5,
    metric: str = "euclidean",
) -> list[str]:
    trajectory = np.asarray(trajectory, dtype=np.float64)
    scored: list[tuple[float, str]] = []
    for word, template in templates:
        if metric == "dtw":
            distance = dtw_distance(trajectory, template)
        else:
            distance = float(np.linalg.norm(trajectory - template))
        scored.append((distance, word))
    scored.sort(key=lambda item: item[0])
    ranked: list[str] = []
    for _, word in scored:
        if word not in ranked:
            ranked.append(word)
        if len(ranked) >= topk:
            break
    return ranked


def _accuracy(
    order: np.ndarray,
    train_words: Sequence[str],
    test_words: Sequence[str],
    rows: np.ndarray | None = None,
    k: int = 1,
) -> dict[str, Any]:
    if rows is None:
        rows = np.arange(len(test_words))
    if len(rows) == 0:
        return {"n": 0, "top1": float("nan"), "top5": float("nan")}
    hits1 = 0
    hits5 = 0
    for row in rows:
        ranked = [train_words[index] for index in order[row, :5]]
        unique: list[str] = []
        for word in ranked:
            if word not in unique:
                unique.append(word)
        if unique and unique[0] == test_words[row]:
            hits1 += 1
        if test_words[row] in unique:
            hits5 += 1
    return {
        "n": int(len(rows)),
        "top1": hits1 / len(rows),
        "top5": hits5 / len(rows),
    }


def evaluate_nearest(
    train_examples: Sequence[dict],
    test_examples: Sequence[dict],
    topk: int = 5,
    metric: str = "euclidean",
    rerank: int = 50,
) -> dict[str, Any]:
    if not train_examples or not test_examples:
        return {"overall": {"n": 0, "top1": float("nan"), "top5": float("nan")}}
    train_words = [example["word"] for example in train_examples]
    test_words = [example["word"] for example in test_examples]
    train_groups = [example["group"] for example in train_examples]
    test_groups = [example["group"] for example in test_examples]
    matrix_a = np.stack(
        [example["trajectory"].reshape(-1) for example in test_examples]
    ).astype(np.float64)
    matrix_b = np.stack(
        [example["trajectory"].reshape(-1) for example in train_examples]
    ).astype(np.float64)
    norm_a = np.square(matrix_a).sum(axis=1)[:, None]
    norm_b = np.square(matrix_b).sum(axis=1)[None, :]
    distances = norm_a + norm_b - 2.0 * matrix_a @ matrix_b.T
    order = np.argsort(distances, axis=1)
    if metric == "dtw":
        probe = _fast_dtw(matrix_a[0], matrix_b[0])
        if probe is not None:
            candidates = order[:, : min(rerank, order.shape[1])]
            for row in range(len(test_examples)):
                scored = sorted(
                    (_fast_dtw(matrix_a[row], matrix_b[index]), int(index))
                    for index in candidates[row]
                )
                rest = [
                    int(index)
                    for index in order[row]
                    if index not in set(int(value) for value in candidates[row])
                ]
                order[row] = np.asarray(
                    [index for _, index in scored] + rest, dtype=order.dtype
                )
    order = order[:, : max(topk, 5)]

    vocabulary = set(train_words)
    closed_rows = np.asarray(
        [index for index, word in enumerate(test_words) if word in vocabulary],
        dtype=int,
    )
    open_rows = np.asarray(
        [index for index, word in enumerate(test_words) if word not in vocabulary],
        dtype=int,
    )
    metrics: dict[str, Any] = {
        "overall": _accuracy(order, train_words, test_words),
        "closed": _accuracy(order, train_words, test_words, closed_rows),
        "open": _accuracy(order, train_words, test_words, open_rows),
        "vocabulary_size": len(vocabulary),
    }
    for group in ("connected", "unconnected"):
        rows = np.asarray(
            [index for index, value in enumerate(test_groups) if value == group],
            dtype=int,
        )
        metrics[group] = _accuracy(order, train_words, test_words, rows)
    metrics["train_groups"] = {
        group: sum(1 for value in train_groups if value == group)
        for group in ("connected", "unconnected")
    }
    return metrics
