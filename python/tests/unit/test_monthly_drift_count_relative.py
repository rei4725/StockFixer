"""月次レポートの閾値超過銘柄数が、日次のドリフト検査と同じ判定で数えられることのテスト（#1089）。"""

import pandas as pd

from src.reporting.kpi import _compute_drift_count


def _row(symbol: str, mae: float, n_samples: int = 30) -> dict:
    return {
        "market": "jp",
        "symbol": symbol,
        "mean_abs_error": mae,
        "direction_accuracy": 0.55,
        "n_samples": n_samples,
    }


def test_uniformly_high_mae_counts_nothing():
    """全銘柄の MAE が同じ水準で高い（本番の 0.056〜0.060 相当）なら、超える銘柄は0件。"""
    df = pd.DataFrame([_row(f"S{i}", 0.056 + 0.0005 * i) for i in range(9)])

    assert _compute_drift_count(df) == 0


def test_only_degraded_symbol_is_counted():
    """1銘柄だけ MAE が中央値の1.5倍を超える表では、その銘柄だけを数える。"""
    rows = [_row(f"S{i}", 0.058) for i in range(8)] + [_row("BAD", 0.15)]

    assert _compute_drift_count(pd.DataFrame(rows)) == 1


def test_symbols_below_min_samples_are_not_counted():
    """サンプル数が下限（既定10）より少ない銘柄は、MAE が大きくても数えない。"""
    rows = [_row(f"S{i}", 0.058) for i in range(8)] + [_row("FEW", 0.15, n_samples=5)]

    assert _compute_drift_count(pd.DataFrame(rows)) == 0
