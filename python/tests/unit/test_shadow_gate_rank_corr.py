"""影の比べの勝ちと昇格ゲートの基準4が、相場の向きで決まらないテスト（#1985）。

production も challenger も全部「上がる」と言うと、向きの的中率と符号の Sharpe はその窓に
実際に上がった割合だけで決まり、両者で同じになる。勝ち負けは同じ日の銘柄の並べの腕
（daily_rank_corr）で決め、基準4は常に多い方との差（hit_rate_minus_majority）で決める。
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

_DAYS = [f"2026-11-{d:02d}" for d in (10, 11, 12, 13)]
_SYMBOLS = [f"S{i:02d}" for i in range(10)]


def _actual_table(up_rate: float) -> pd.DataFrame:
    """各日に 10 銘柄。上がる銘柄の割合が up_rate になる実際の変化率の表。"""
    n_up = int(round(up_rate * len(_SYMBOLS)))
    rows = []
    for d, day in enumerate(_DAYS):
        for i, symbol in enumerate(_SYMBOLS):
            # 上の n_up 銘柄が上がり、残りが下がる。値は銘柄ごとに違う
            rank = (i + d) % len(_SYMBOLS)
            actual = 0.001 * (len(_SYMBOLS) - rank) if rank < n_up else -0.001 * (rank + 1)
            rows.append({"date": day, "market": "us", "symbol": symbol, "actual_ratio": actual})
    return pd.DataFrame(rows)


def _with_predictions(base: pd.DataFrame, model_name: str, predicted: np.ndarray) -> pd.DataFrame:
    df = base.copy()
    df["model_name"] = model_name
    df["predicted_ratio"] = predicted
    df["direction_match"] = ((df["predicted_ratio"] >= 0) == (df["actual_ratio"] >= 0)).astype(int)
    return df


def _all_up_follow(actual: pd.Series) -> np.ndarray:
    """全部「上がる」で、並べは実際の変化率どおり（順位相関 1）。"""
    return 0.01 + 0.1 * (actual.to_numpy() + 0.02)


def _all_up_reverse(actual: pd.Series) -> np.ndarray:
    """全部「上がる」で、並べは実際の逆（順位相関 -1）。"""
    return 0.01 + 0.1 * (0.02 - actual.to_numpy())


def _evaluate_shadow(tables: dict[str, pd.DataFrame]) -> dict:
    from src.prediction.shadow_evaluation import evaluate_shadow_models

    with (
        patch(
            "src.prediction.shadow_evaluation.load_prediction_accuracy",
            side_effect=lambda **kw: tables[kw["model_name"]],
        ),
        patch("src.prediction.shadow_evaluation._record_ab_result"),
    ):
        return evaluate_shadow_models(production_version="production", challenger_version="chal")


class TestShadowWinsByRankCorr:
    def test_all_up_challenger_with_better_ranking_wins(self):
        """向きの的中率と Sharpe が同じでも、並べの腕の良い challenger が勝つ。"""
        base = _actual_table(0.6)
        tables = {
            "production": _with_predictions(
                base, "production", _all_up_reverse(base["actual_ratio"])
            ),
            "chal": _with_predictions(base, "chal", _all_up_follow(base["actual_ratio"])),
        }
        result = _evaluate_shadow(tables)

        assert result["challenger_hit_rate"] == pytest.approx(result["production_hit_rate"])
        assert result["challenger_sharpe"] == pytest.approx(result["production_sharpe"])
        assert result["challenger_daily_rank_corr"] == pytest.approx(1.0)
        assert result["production_daily_rank_corr"] == pytest.approx(-1.0)
        assert result["challenger_wins"] is True

    def test_all_up_challenger_with_worse_ranking_loses(self):
        base = _actual_table(0.6)
        tables = {
            "production": _with_predictions(
                base, "production", _all_up_follow(base["actual_ratio"])
            ),
            "chal": _with_predictions(base, "chal", _all_up_reverse(base["actual_ratio"])),
        }
        result = _evaluate_shadow(tables)

        assert result["challenger_wins"] is False

    def test_challenger_does_not_win_when_its_rank_corr_is_none(self):
        """日の列が無く順位相関を測れないなら、的中率と Sharpe が上でも勝ちにしない。"""
        base = _actual_table(0.6)
        chal = _with_predictions(base, "chal", base["actual_ratio"].to_numpy())
        tables = {
            "production": _with_predictions(
                base, "production", _all_up_reverse(base["actual_ratio"])
            ),
            "chal": chal.drop(columns=["date"]),
        }
        result = _evaluate_shadow(tables)

        assert result["challenger_hit_rate"] > result["production_hit_rate"]
        assert result["challenger_sharpe"] > result["production_sharpe"]
        assert result["challenger_daily_rank_corr"] is None
        assert result["challenger_wins"] is False

    def test_challenger_does_not_win_when_production_rank_corr_is_none(self):
        """production の順位相関を測れない（日の列が無い）なら勝ちにしない。"""
        base = _actual_table(0.6)
        prod = _with_predictions(base, "production", _all_up_reverse(base["actual_ratio"]))
        tables = {
            "production": prod.drop(columns=["date"]),
            "chal": _with_predictions(base, "chal", base["actual_ratio"].to_numpy()),
        }
        result = _evaluate_shadow(tables)

        assert result["challenger_hit_rate"] > result["production_hit_rate"]
        assert result["production_daily_rank_corr"] is None
        assert result["challenger_wins"] is False

    def test_constant_production_counts_as_zero_rank_corr(self):
        """日ごとに定数の production は並べの腕が無いので、順位相関 0 と比べる。"""
        base = _actual_table(0.6)
        tables = {
            "production": _with_predictions(base, "production", np.full(len(base), 0.01)),
            "chal": _with_predictions(base, "chal", _all_up_follow(base["actual_ratio"])),
        }
        result = _evaluate_shadow(tables)

        assert result["production_daily_rank_corr"] is None
        assert result["challenger_wins"] is True

    def test_result_names_the_win_rule(self):
        base = _actual_table(0.6)
        tables = {
            "production": _with_predictions(
                base, "production", _all_up_reverse(base["actual_ratio"])
            ),
            "chal": _with_predictions(base, "chal", _all_up_follow(base["actual_ratio"])),
        }
        from src.prediction.shadow_evaluation import SHADOW_WIN_RULE

        assert _evaluate_shadow(tables)["win_rule"] == SHADOW_WIN_RULE


class TestWeeklyRecordCarriesRankCorr:
    def test_record_has_rank_corr_and_win_rule(self):
        from src.prediction.shadow_record import apply_shadow_result, new_weekly_shadow_record

        record = new_weekly_shadow_record(days=20)
        apply_shadow_result(
            record,
            {
                "production_daily_rank_corr": -0.1,
                "challenger_daily_rank_corr": 0.2,
                "win_rule": "x",
            },
        )
        assert record["production_daily_rank_corr"] == -0.1
        assert record["challenger_daily_rank_corr"] == 0.2
        assert record["win_rule"] == "x"


def _evaluate_gate(frames: dict[str, pd.DataFrame]):
    from src.prediction.promotion_gate import evaluate_promotion

    wf_df = pd.DataFrame(
        {
            "model_name": ["shadow", "current"],
            "total_return": [0.06, 0.04],
            "max_drawdown": [-0.08, -0.12],
            "sharpe_ratio": [0.7, 0.6],
        }
    )
    with (
        patch(
            "src.prediction.promotion_gate._load_latest_wf_summary",
            return_value=(wf_df, "wf_summary_20261120.csv"),
        ),
        patch(
            "src.prediction.promotion_gate._load_model_window",
            side_effect=lambda name, horizon, window=None: frames.get(name, pd.DataFrame()),
        ),
        patch("src.prediction.promotion_gate._collect_run_ids", return_value=[]),
    ):
        return evaluate_promotion("shadow", "current")


def _same_skill_frames(up_rate: float) -> dict[str, pd.DataFrame]:
    """腕の同じ2つのモデル（どちらも全部「上がる」・同じ並べ）。"""
    base = _actual_table(up_rate)
    predicted = _all_up_follow(base["actual_ratio"])
    return {
        "shadow": _with_predictions(base, "shadow", predicted),
        "current": _with_predictions(base, "current", predicted),
    }


class TestGateHitRateIndependentOfMarketDirection:
    def test_same_skill_models_get_same_verdict_in_up_and_down_windows(self):
        down = _evaluate_gate(_same_skill_frames(0.4)).criteria["hit_rate"]
        up = _evaluate_gate(_same_skill_frames(0.6)).criteria["hit_rate"]

        assert down["shadow"] == pytest.approx(0.4)
        assert up["shadow"] == pytest.approx(0.6)
        assert down["passed"] is True
        assert up["passed"] is True

    def test_shadow_below_current_minus_majority_fails_in_up_window(self):
        """上げの窓で的中率が 0.50 を超えても、常に多い方との差が current より悪ければ落ちる。"""
        base = _actual_table(0.6)
        shadow_pred = _all_up_follow(base["actual_ratio"])
        # 下がった銘柄の1つだけ「下がる」と当てる current
        current_pred = shadow_pred.copy()
        first_down = int(np.flatnonzero(base["actual_ratio"].to_numpy() < 0)[0])
        current_pred[first_down] = -0.01
        frames = {
            "shadow": _with_predictions(base, "shadow", shadow_pred),
            "current": _with_predictions(base, "current", current_pred),
        }
        crit = _evaluate_gate(frames).criteria["hit_rate"]

        assert crit["shadow"] > 0.5
        assert crit["passed"] is False
        assert crit["shadow_minus_majority"] < crit["current_minus_majority"]

    def test_criteria_list_daily_rank_corr_of_both_models(self):
        base = _actual_table(0.6)
        frames = {
            "shadow": _with_predictions(base, "shadow", _all_up_follow(base["actual_ratio"])),
            "current": _with_predictions(base, "current", _all_up_reverse(base["actual_ratio"])),
        }
        crit = _evaluate_gate(frames).criteria["hit_rate"]

        assert crit["shadow_daily_rank_corr"] == pytest.approx(1.0)
        assert crit["current_daily_rank_corr"] == pytest.approx(-1.0)
        assert crit["rule"] == "hit_rate_minus_majority"
