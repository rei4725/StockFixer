"""entry_sizing の直のテスト（#1865）: 変動の小さいスキップと新規の株数の決め方。"""

from unittest.mock import MagicMock, patch

import pandas as pd

from src.trading.execution.entry_sizing import (
    MIN_CHANGE_RATIO,
    _sized_entry_qty,
    _skip_small_change,
)
from src.trading.execution.stats import _new_stats

_MOD = "src.trading.execution.entry_sizing"


def _row(diff_ratio: float | None = None) -> pd.Series:
    data: dict[str, object] = {"symbol": "7203", "market": "jp"}
    if diff_ratio is not None:
        data["diff_ratio"] = diff_ratio
    return pd.Series(data)


class TestSkipSmallChange:
    def test_small_change_is_counted_under_key(self) -> None:
        stats = _new_stats()
        assert _skip_small_change(_row(MIN_CHANGE_RATIO * 0.5), "7203", stats, "min_change", "x")
        assert stats["skipped"] == 1
        assert stats["skipped_min_change"] == 1

    def test_missing_diff_ratio_counts_as_small(self) -> None:
        stats = _new_stats()
        assert _skip_small_change(_row(None), "7203", stats, "sell_min_change", "x")
        assert stats["skip_reasons"] == {"sell_min_change": 1}

    def test_large_or_negative_large_change_is_not_skipped(self) -> None:
        stats = _new_stats()
        assert not _skip_small_change(_row(MIN_CHANGE_RATIO * 2), "7203", stats, "min_change", "x")
        assert not _skip_small_change(_row(-MIN_CHANGE_RATIO * 2), "7203", stats, "min_change", "x")
        assert stats["skipped"] == 0

    def test_nan_is_not_skipped_as_before(self) -> None:
        # 前の `abs(nan) < MIN_CHANGE_RATIO` は False で、スキップしなかった
        stats = _new_stats()
        assert not _skip_small_change(_row(float("nan")), "7203", stats, "min_change", "x")
        assert stats["skipped"] == 0


def _risk(qty: int, zero_reason: str = "") -> MagicMock:
    risk = MagicMock()
    risk.calc_position_size.return_value = qty
    risk.zero_reason = zero_reason
    return risk


@patch(f"{_MOD}._resolve_kelly_params", return_value=(0.55, 0.015, 0.008))
class TestSizedEntryQty:
    def test_full_size_passes_kelly_params_and_confidence(self, _kelly: MagicMock) -> None:
        risk, stats = _risk(300), _new_stats()
        qty = _sized_entry_qty(risk, _row(), "7203", 1000.0, 0.9, 1.0, 100, stats, ("a", "b"))
        assert qty == 300
        risk.calc_position_size.assert_called_once_with(
            "7203", 1000.0, win_rate=0.55, avg_win=0.015, avg_loss=0.008, confidence_ratio=0.9
        )
        _kelly.assert_called_once_with("jp", "7203")
        assert stats["skipped"] == 0

    def test_zero_qty_counts_reason_key_and_returns_zero(self, _kelly: MagicMock) -> None:
        risk, stats = _risk(0), _new_stats()
        qty = _sized_entry_qty(risk, _row(), "7203", 1000.0, 0.9, 1.0, 100, stats, ("a", "b"))
        assert qty == 0
        assert stats["skipped"] == 1
        assert stats["skip_reasons"] == {"zero_qty": 1}

    def test_split_ratio_below_one_applies_split(self, _kelly: MagicMock) -> None:
        risk, stats = _risk(400), _new_stats()
        qty = _sized_entry_qty(risk, _row(), "7203", 1000.0, 0.6, 0.5, 100, stats, ("a", "b"))
        assert qty == 200
