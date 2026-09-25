"""Pure metric helpers shared by calibration and threshold selection."""

from __future__ import annotations


def precision_recall(tp: int, fp: int, fn: int) -> tuple[float, float]:
    """Frame-level precision and recall; 0.0 when the denominators vanish."""
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return precision, recall


def f_beta(precision: float, recall: float, beta: float = 2.0) -> float:
    """Weighted F-beta; 0.0 when both precision and recall are 0."""
    if precision + recall == 0.0:
        return 0.0
    return (1 + beta**2) * precision * recall / (beta**2 * precision + recall)
