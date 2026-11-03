"""backtest_promotion.judge（勝ちと戻しの判定）のテスト。"""

import numpy as np
import pandas as pd

from src.prediction.backtest_promotion.judge import judge_rollback, judge_win
from src.prediction.baseline_metrics import baseline_metrics

GOOD = {"hit_rate_minus_majority": 0.03, "mae": 0.015, "rank_corr": 0.05, "n_baseline": 1000}
PROD = {"hit_rate_minus_majority": -0.01, "mae": 0.06, "rank_corr": 0.01, "n_baseline": 1000}
LIVE = {"hit_rate_minus_majority": -0.02, "rank_corr": 0.01, "predicted_up_rate": 0.5}
BT_CAND = {"hit_rate_minus_majority": 0.02}
OLD_BETTER = {"hit_rate_minus_majority": 0.1, "rank_corr": 0.1}


def test_always_up_predictions_cannot_win():
    rows = pd.DataFrame(
        {
            "predicted_ratio": [0.06] * 400,
            "actual_ratio": np.r_[np.full(210, 0.01), np.full(190, -0.01)],
        }
    )
    m = baseline_metrics(rows)
    assert abs(m["hit_rate_minus_majority"]) < 1e-9
    assert judge_win(m, m, [0.0] * 20, 400).won is False


def test_win_requires_all_three_metrics():
    d = judge_win(GOOD, PROD, [0.01] * 11 + [-0.01] * 9, 1000)
    assert d.won is True and d.reasons == []
    assert judge_win({**GOOD, "mae": 0.07}, PROD, [0.01] * 20, 1000).won is False
    assert (
        judge_win({**GOOD, "rank_corr": -0.01}, {**PROD, "rank_corr": -0.05}, [0.01] * 20, 1000).won
        is False
    )
    assert (
        judge_win({**GOOD, "hit_rate_minus_majority": -0.02}, PROD, [0.01] * 20, 1000).won is False
    )


def test_win_requires_daily_majority_and_coverage():
    assert (
        judge_win(GOOD, PROD, [0.01] * 10 + [-0.01] * 10, 1000).won is False
    )  # ちょうど半分は負け
    assert judge_win(GOOD, PROD, [], 1000).won is False
    assert judge_win({**GOOD, "n_baseline": 799}, PROD, [0.01] * 20, 1000).won is False
    assert judge_win(GOOD, {**PROD, "n_baseline": 799}, [0.01] * 20, 1000).won is False
    assert judge_win({**GOOD, "n_baseline": 800}, PROD, [0.01] * 20, 1000).won is True


def test_none_metric_counts_as_lost_with_one_reason_each():
    d = judge_win({**GOOD, "mae": None, "rank_corr": None}, PROD, [0.01] * 20, 1000)
    assert d.won is False and len(d.reasons) == 2


def test_rollback_only_when_triggered_and_old_is_better():
    assert (
        judge_rollback(LIVE, BT_CAND, {"hit_rate_minus_majority": 0.0, "rank_corr": 0.02}).rollback
        is True
    )
    d = judge_rollback(LIVE, BT_CAND, {"hit_rate_minus_majority": -0.05, "rank_corr": 0.02})
    assert d.triggered and d.rollback is False and d.old_is_better is False
    assert (
        judge_rollback(
            {**LIVE, "predicted_up_rate": 0.95, "hit_rate_minus_majority": 0.02},
            BT_CAND,
            {"hit_rate_minus_majority": 0.03, "rank_corr": 0.05},
        ).rollback
        is True
    )
    assert (
        judge_rollback(
            {"hit_rate_minus_majority": 0.0, "rank_corr": 0.01, "predicted_up_rate": 0.5},
            BT_CAND,
            {"hit_rate_minus_majority": 0.05, "rank_corr": 0.05},
        ).triggered
        == []
    )


def test_rollback_triggers_individually():
    cand = {"hit_rate_minus_majority": -0.02}
    assert len(judge_rollback({**LIVE, "rank_corr": 0.0}, cand, OLD_BETTER).triggered) == 1
    assert len(judge_rollback({**LIVE, "predicted_up_rate": 0.05}, cand, OLD_BETTER).triggered) == 1
    # ちょうど 0.03 低いのは当たり
    live = {**LIVE, "hit_rate_minus_majority": 0.0}
    assert judge_rollback(live, {"hit_rate_minus_majority": 0.03}, OLD_BETTER).triggered


def test_rollback_none_is_conservative():
    none_live = {"hit_rate_minus_majority": None, "rank_corr": None, "predicted_up_rate": None}
    d = judge_rollback(none_live, BT_CAND, OLD_BETTER)
    assert d.triggered == [] and d.old_is_better is False and d.rollback is False
    d = judge_rollback(LIVE, BT_CAND, {"hit_rate_minus_majority": None, "rank_corr": 0.1})
    assert d.triggered and d.old_is_better is False and d.rollback is False
