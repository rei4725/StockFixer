"""基準の的中率などの物差しのテスト（#1309）。"""

import pandas as pd
import pytest

from src.prediction.baseline_metrics import BASELINE_METRIC_KEYS, baseline_metrics


def _rows(pred, actual):
    return pd.DataFrame({"predicted_ratio": pred, "actual_ratio": actual})


def test_all_up_predictions_hit_rate_equals_up_rate():
    """全部「上がる」と言う予想では、的中率と実際に上がった割合の差が 0 になる。"""
    rows = _rows([0.06, 0.05, 0.07, 0.10, 0.04], [0.01, -0.02, 0.03, -0.01, 0.0])
    m = baseline_metrics(rows)
    assert m["predicted_up_rate"] == pytest.approx(1.0)
    assert m["actual_up_rate"] == pytest.approx(0.6)
    assert m["hit_rate_minus_up_rate"] == pytest.approx(0.0, abs=1e-12)
    assert m["majority_baseline_accuracy"] == pytest.approx(0.6)
    assert m["hit_rate_minus_majority"] == pytest.approx(0.0, abs=1e-12)


def test_small_table_matches_hand_values():
    """小さな表で、手で引いた値と合う。"""
    # 的中: 1件目(上/上)、2件目(上/下=外れ)、3件目(下/下)、4件目(上/上)
    rows = _rows([0.04, 0.02, -0.01, 0.08], [0.01, -0.03, -0.02, 0.05])
    m = baseline_metrics(rows)
    assert m["n_baseline"] == 4
    assert m["actual_up_rate"] == pytest.approx(0.5)
    assert m["predicted_up_rate"] == pytest.approx(0.75)
    assert m["majority_baseline_accuracy"] == pytest.approx(0.5)
    assert m["hit_rate_minus_up_rate"] == pytest.approx(0.75 - 0.5)
    assert m["hit_rate_minus_majority"] == pytest.approx(0.25)
    # |誤差| = 0.03, 0.05, 0.01, 0.03 → 平均 0.03
    assert m["mae"] == pytest.approx(0.03)
    assert m["mean_abs_predicted_ratio"] == pytest.approx((0.04 + 0.02 + 0.01 + 0.08) / 4)
    assert m["mean_abs_actual_ratio"] == pytest.approx((0.01 + 0.03 + 0.02 + 0.05) / 4)
    # 予想の順位 [3,2,1,4]、実際の順位 [3,1,2,4] → Spearman = 1 - 6*2/(4*15) = 0.8
    assert m["rank_corr"] == pytest.approx(0.8)
    # |予想| の順位 [3,2,1,4]、|実際| の順位 [1,3,2,4] → 1 - 6*6/60 = 0.4
    assert m["abs_rank_corr"] == pytest.approx(0.4)


def test_majority_is_down_when_mostly_down():
    rows = _rows([0.01, 0.01, 0.01, 0.01], [-0.01, -0.02, -0.01, 0.02])
    m = baseline_metrics(rows)
    assert m["actual_up_rate"] == pytest.approx(0.25)
    assert m["majority_baseline_accuracy"] == pytest.approx(0.75)
    assert m["hit_rate_minus_majority"] == pytest.approx(0.25 - 0.75)


def test_constant_predictions_have_no_rank_corr():
    rows = _rows([0.05, 0.05, 0.05], [0.01, -0.01, 0.02])
    assert baseline_metrics(rows)["rank_corr"] is None


def test_missing_values_are_dropped():
    rows = _rows([0.02, None, -0.01], [0.01, 0.03, None])
    m = baseline_metrics(rows)
    assert m["n_baseline"] == 1
    assert m["mae"] == pytest.approx(0.01)
    assert m["rank_corr"] is None


@pytest.mark.parametrize("rows", [None, pd.DataFrame(), pd.DataFrame({"x": [1]})])
def test_empty_rows(rows):
    m = baseline_metrics(rows)
    assert set(m) == set(BASELINE_METRIC_KEYS)
    assert m["n_baseline"] == 0
    assert all(m[k] is None for k in BASELINE_METRIC_KEYS if k != "n_baseline")
