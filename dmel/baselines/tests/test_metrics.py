"""Tests for the shared precision/recall/F-beta helpers."""

from __future__ import annotations

import pytest

from dmel.baselines.metrics import f_beta, precision_recall


def test_precision_recall_zero_denominators() -> None:
    assert precision_recall(0, 0, 0) == (0.0, 0.0)
    assert precision_recall(0, 5, 0) == (0.0, 0.0)
    assert precision_recall(0, 0, 5) == (0.0, 0.0)


def test_precision_recall_perfect() -> None:
    assert precision_recall(10, 0, 0) == (1.0, 1.0)


def test_precision_recall_mixed() -> None:
    precision, recall = precision_recall(6, 2, 2)
    assert precision == pytest.approx(0.75)
    assert recall == pytest.approx(0.75)


def test_f_beta_f2_weights_recall() -> None:
    # At equal P/R, F2 == F1; the beta=2 form reduces to 5PR/(4P+R).
    assert f_beta(0.5, 0.5, beta=2.0) == pytest.approx(0.5)
    assert f_beta(1.0, 0.5, beta=2.0) == pytest.approx(5 * 0.5 / (4 + 0.5))


def test_f_beta_zero_when_no_signal() -> None:
    assert f_beta(0.0, 0.0) == 0.0
