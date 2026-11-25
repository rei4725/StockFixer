"""backtest_promotion.judge（勝ちと戻しの判定）のテスト。"""

import numpy as np
import pandas as pd
import pytest

from src.prediction.backtest_promotion.judge import judge_rollback, judge_win
from src.prediction.baseline_metrics import baseline_metrics
from src.reporting.backtest_promotion_trend import build_backtest_promotion_trend

GOOD = {"hit_rate_minus_majority": 0.03, "mae": 0.015, "daily_rank_corr": 0.05, "n_baseline": 1000}
PROD = {"hit_rate_minus_majority": -0.01, "mae": 0.06, "daily_rank_corr": 0.01, "n_baseline": 1000}
LIVE = {"hit_rate_minus_majority": -0.02, "daily_rank_corr": 0.01, "predicted_up_rate": 0.5}
BT_CAND = {"hit_rate_minus_majority": 0.02}
OLD_BETTER = {"hit_rate_minus_majority": 0.1, "daily_rank_corr": 0.1}


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
        judge_win(
            {**GOOD, "daily_rank_corr": -0.01},
            {**PROD, "daily_rank_corr": -0.05},
            [0.01] * 20,
            1000,
        ).won
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
    d = judge_win({**GOOD, "mae": None, "daily_rank_corr": None}, PROD, [0.01] * 20, 1000)
    assert d.won is False and len(d.reasons) == 2


def test_rollback_only_when_triggered_and_old_is_better():
    old = {"hit_rate_minus_majority": 0.0, "daily_rank_corr": 0.02}
    assert judge_rollback(LIVE, BT_CAND, old).rollback is True
    d = judge_rollback(LIVE, BT_CAND, {**old, "hit_rate_minus_majority": -0.05})
    assert d.triggered and d.rollback is False and d.old_is_better is False
    assert (
        judge_rollback(
            {**LIVE, "predicted_up_rate": 0.95, "hit_rate_minus_majority": 0.02},
            BT_CAND,
            {"hit_rate_minus_majority": 0.03, "daily_rank_corr": 0.05},
        ).rollback
        is True
    )
    assert (
        judge_rollback(
            {"hit_rate_minus_majority": 0.0, "daily_rank_corr": 0.01, "predicted_up_rate": 0.5},
            BT_CAND,
            {"hit_rate_minus_majority": 0.05, "daily_rank_corr": 0.05},
        ).triggered
        == []
    )


def test_rollback_triggers_individually():
    cand = {"hit_rate_minus_majority": -0.02}
    assert len(judge_rollback({**LIVE, "daily_rank_corr": 0.0}, cand, OLD_BETTER).triggered) == 1
    assert len(judge_rollback({**LIVE, "predicted_up_rate": 0.05}, cand, OLD_BETTER).triggered) == 1
    # ちょうど 0.03 低いのは当たり
    live = {**LIVE, "hit_rate_minus_majority": 0.0}
    assert judge_rollback(live, {"hit_rate_minus_majority": 0.03}, OLD_BETTER).triggered


def test_rollback_none_is_conservative():
    none_live = {
        "hit_rate_minus_majority": None,
        "daily_rank_corr": None,
        "predicted_up_rate": None,
    }
    d = judge_rollback(none_live, BT_CAND, OLD_BETTER)
    # daily_rank_corr の None は 0 扱いで当たるが、的中率が None なので戻さない
    assert len(d.triggered) == 1 and d.old_is_better is False and d.rollback is False
    d = judge_rollback(LIVE, BT_CAND, {"hit_rate_minus_majority": None, "daily_rank_corr": 0.1})
    assert d.triggered and d.old_is_better is False and d.rollback is False


def test_rollback_compares_old_with_the_new_model_on_the_same_rows():
    # 新しいモデルの5日の全行は当たりに届く。同じ行の新しいモデルは旧 production より良い
    new_same = {"hit_rate_minus_majority": 0.0, "daily_rank_corr": 0.05}
    old_same = {"hit_rate_minus_majority": -0.01, "daily_rank_corr": 0.1}
    d = judge_rollback(LIVE, BT_CAND, old_same, new_same)
    assert d.triggered and d.old_is_better is False and d.rollback is False
    d = judge_rollback(LIVE, BT_CAND, {**old_same, "hit_rate_minus_majority": 0.01}, new_same)
    assert d.old_is_better is True and d.rollback is True


def test_rollback_without_old_production_on_the_same_window_never_rolls_back():
    d = judge_rollback(LIVE, BT_CAND, None, None)
    assert d.triggered and d.old_is_better is False and d.rollback is False


def test_rollback_none_live_rank_corr_is_treated_as_zero():
    # 束ねた rank_corr に値があっても、daily_rank_corr が None なら 0 として扱う
    live = {
        "hit_rate_minus_majority": 0.0,
        "rank_corr": 0.5,
        "daily_rank_corr": None,
        "predicted_up_rate": 0.5,
    }
    d = judge_rollback(live, BT_CAND, {"hit_rate_minus_majority": 0.01, "daily_rank_corr": 0.02})
    assert d.triggered == ["実測の順位相関が0以下（daily_rank_corr 0.0）"]
    assert d.rollback is True
    d = judge_rollback(live, BT_CAND, {"hit_rate_minus_majority": 0.01, "daily_rank_corr": -0.01})
    assert d.rollback is False and d.old_is_better is False


def test_rollback_none_new_rank_corr_on_the_same_rows_is_treated_as_zero():
    new = {"hit_rate_minus_majority": 0.0, "rank_corr": 0.5, "daily_rank_corr": None}
    old = {"hit_rate_minus_majority": 0.01, "rank_corr": 0.1, "daily_rank_corr": 0.02}
    assert judge_rollback(LIVE, BT_CAND, old, new).rollback is True
    d = judge_rollback(LIVE, BT_CAND, {**old, "daily_rank_corr": -0.01}, new)
    assert d.triggered and d.old_is_better is False and d.rollback is False


def test_rollback_none_old_daily_rank_corr_never_rolls_back():
    old = {"hit_rate_minus_majority": 0.1, "rank_corr": 0.9, "daily_rank_corr": None}
    new = {"hit_rate_minus_majority": 0.0, "rank_corr": -0.5, "daily_rank_corr": -0.5}
    d = judge_rollback(LIVE, BT_CAND, old, new)
    assert d.triggered and d.old_is_better is False and d.rollback is False


def test_rollback_when_old_ranks_better_inside_each_market_day():
    """束ねた rank_corr は新が上でも、市場・日の並べで旧が上なら戻す（#2159 の本番の値）。"""
    new = {"hit_rate_minus_majority": -0.02, "rank_corr": 0.145, "daily_rank_corr": -0.037}
    old = {"hit_rate_minus_majority": 0.0, "rank_corr": -0.013, "daily_rank_corr": 0.016}
    d = judge_rollback({**new, "predicted_up_rate": 0.5}, BT_CAND, old, new)
    assert d.triggered and d.old_is_better is True and d.rollback is True
    # 逆（束ねた値だけ旧が上）なら戻さない
    d = judge_rollback(
        {**old, "hit_rate_minus_majority": -0.02, "predicted_up_rate": 0.5},
        BT_CAND,
        {**new, "hit_rate_minus_majority": 0.0},
        {**old, "hit_rate_minus_majority": -0.02},
    )
    assert d.triggered and d.old_is_better is False and d.rollback is False


@pytest.mark.parametrize("pooled, daily, hit", [(-0.056, 0.018, False), (0.018, -0.056, True)])
def test_rollback_trigger_reads_daily_rank_corr(pooled, daily, hit):
    live = {**LIVE, "rank_corr": pooled, "daily_rank_corr": daily}
    d = judge_rollback(live, {"hit_rate_minus_majority": -0.02}, None)
    assert any("順位相関" in t for t in d.triggered) is hit
    assert bool(d.triggered) is hit


def test_rollback_triggers_on_jittered_constant_live_predictions():
    """揺れだけの定数の新 production の実測は、順位相関0以下の見張りに当たる（#1967）。"""
    rows = pd.DataFrame(
        {
            "date": ["2026-11-10"] * 4 + ["2026-11-11"] * 4,
            "predicted_ratio": [0.003 + i * 1e-16 for i in range(4)] * 2,
            "actual_ratio": [-0.02, -0.01, 0.01, 0.03, -0.03, 0.0, 0.02, 0.04],
        }
    )
    live = baseline_metrics(rows)
    d = judge_rollback(live, BT_CAND, OLD_BETTER)
    assert live["daily_rank_corr"] is None
    assert any("実測の順位相関が0以下" in t for t in d.triggered)


def test_win_loses_when_only_pooled_rank_corr_is_better():
    """束ねた rank_corr だけが上で、同じ市場・日の並べが負の候補は負ける（#2157 の本番の値）。"""
    cand = {**GOOD, "rank_corr": 0.145, "daily_rank_corr": -0.037}
    prod = {**PROD, "rank_corr": -0.013, "daily_rank_corr": 0.016}
    d = judge_win(cand, prod, [0.01] * 20, 1000)
    assert d.won is False
    assert len(d.reasons) == 1
    assert "daily_rank_corr" in d.reasons[0]
    assert "-0.037" in d.reasons[0] and "0.016" in d.reasons[0]
    # 負けの理由の流れの表が、順位相関の基準として見分ける
    row = build_backtest_promotion_trend(
        [{"_file": "20261123_220000", "evaluation": {}, "reason": "; ".join(d.reasons)}]
    ).iloc[0]
    assert row["reason_rank_corr"] and not row["reason_other"]


@pytest.mark.parametrize(
    "c_daily, p_daily, won",
    [
        (0.02, 0.01, True),
        (0.01, 0.01, False),
        (0.01, 0.02, False),
        (0.0, -0.01, False),
        (-0.01, -0.02, False),
        (None, 0.01, False),
        (0.02, None, False),
        (None, None, False),
    ],
)
def test_win_compares_daily_rank_corr(c_daily, p_daily, won):
    """daily_rank_corr が production を上回り0より大きいときだけ勝つ。None は負け（#2157）。"""
    cand = {**GOOD, "rank_corr": 0.2, "daily_rank_corr": c_daily}
    prod = {**PROD, "rank_corr": 0.1, "daily_rank_corr": p_daily}
    assert judge_win(cand, prod, [0.01] * 20, 1000).won is won


def test_win_does_not_read_pooled_rank_corr():
    cand = {**GOOD, "rank_corr": -0.1, "daily_rank_corr": 0.05}
    prod = {**PROD, "rank_corr": 0.1, "daily_rank_corr": 0.01}
    assert judge_win(cand, prod, [0.01] * 20, 1000).won is True


def test_win_loses_on_rank_corr_not_above_production_or_tied_hit():
    prod = {**PROD, "daily_rank_corr": 0.08}
    assert judge_win(GOOD, prod, [0.01] * 20, 1000).won is False  # 0 < 0.05 <= 0.08
    tied = {**GOOD, "hit_rate_minus_majority": PROD["hit_rate_minus_majority"]}
    assert judge_win(tied, PROD, [0.01] * 20, 1000).won is False
