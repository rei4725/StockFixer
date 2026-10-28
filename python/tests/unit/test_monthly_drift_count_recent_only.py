"""月次の閾値超過銘柄数が、最後の予想が窓より古い銘柄を数えないことのテスト（#1335）。"""

from datetime import datetime, timedelta, timezone

import pandas as pd

from src.reporting.kpi import _compute_drift_count


def _predicted_at(days_ago: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days_ago)).strftime("%Y%m%d_%H%M%S")


def _row(symbol: str, mae: float, days_ago: int = 1, accuracy: float = 0.55) -> dict:
    return {
        "market": "jp",
        "symbol": symbol,
        "mean_abs_error": mae,
        "direction_accuracy": accuracy,
        "n_samples": 30,
        "last_predicted_at": _predicted_at(days_ago),
    }


def test_stale_symbol_over_threshold_is_not_counted():
    """最後の予想が31日前の銘柄は、MAE が線を超えていても数えない。"""
    rows = [_row(f"S{i}", 0.058) for i in range(8)] + [_row("OLD", 0.15, days_ago=31)]

    assert _compute_drift_count(pd.DataFrame(rows)) == 0


def test_stale_symbols_are_excluded_from_median():
    """古い銘柄の大きな MAE で中央値が上がり、今の銘柄の超過が隠れることがない。"""
    rows = (
        [_row(f"S{i}", 0.058) for i in range(4)]
        + [_row("BAD", 0.10)]
        + [_row(f"OLD{i}", 0.30, days_ago=60) for i in range(6)]
    )

    assert _compute_drift_count(pd.DataFrame(rows)) == 1


def test_recent_symbols_are_counted_as_before():
    """窓の中に予想がある銘柄は、日次と同じ判定で数える。"""
    rows = [_row(f"S{i}", 0.058, days_ago=29) for i in range(8)] + [_row("BAD", 0.15)]

    assert _compute_drift_count(pd.DataFrame(rows)) == 1


def test_table_without_last_predicted_at_counts_as_before():
    """last_predicted_at の列が無い表でも落ちず、全行で数える。"""
    rows = [_row(f"S{i}", 0.058) for i in range(8)] + [_row("BAD", 0.15)]
    df = pd.DataFrame(rows).drop(columns=["last_predicted_at"])

    assert _compute_drift_count(df) == 1
