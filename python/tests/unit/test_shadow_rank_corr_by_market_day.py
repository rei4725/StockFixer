"""影の比べの日ごとの順位相関が、市場と取引日ごとに銘柄を束ねるテスト（#2072）。

本番の1回の予想は jp と us を同じ predicted_at で書く。回の時刻で束ねると、市場の間の水準の差を
銘柄の並べの腕と取り違え、朝の回とやり直しの回が割れる。
"""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

# 予想の並べの順（実際は 0..5 の順）。どちらのモデルも市場の中では同じ並べを言う
_ORDER = [1, 0, 3, 2, 5, 4]
_WITHIN_CORR = 1 - 6 * 6 / (6 * 35)


def _row(
    model: str, market: str, i: int, predicted_at: str, predicted: float, actual: float
) -> dict:
    return {
        "market": market,
        "symbol": f"{market}{i}",
        "model_name": model,
        "predicted_at": predicted_at,
        "predicted_ratio": predicted,
        "actual_ratio": actual,
        "direction_match": int((predicted >= 0) == (actual >= 0)),
    }


def _two_market_rows(model: str, jp_level: float, us_level: float) -> pd.DataFrame:
    """jp が上げ us が下げた日が3日。予想は全部「上がる」で、市場の間の水準だけを変える。"""
    rows = []
    for predicted_at in ("20261117_224436", "20261118_224436", "20261119_224436"):
        for market, level, actual_level in (("jp", jp_level, 0.02), ("us", us_level, -0.02)):
            for i, rank in enumerate(_ORDER):
                rows.append(
                    _row(
                        model,
                        market,
                        i,
                        predicted_at,
                        level + 0.001 * rank,
                        actual_level + 0.001 * i,
                    )
                )
    return pd.DataFrame(rows)


def _evaluate(production: pd.DataFrame, challenger: pd.DataFrame) -> dict:
    from src.prediction.shadow_evaluation import evaluate_shadow_models

    tables = {"production": production, "chal": challenger}
    with (
        patch(
            "src.prediction.shadow_evaluation.load_prediction_accuracy",
            side_effect=lambda **kw: tables[kw["model_name"]],
        ),
        patch("src.prediction.shadow_evaluation._record_ab_result"),
    ):
        return evaluate_shadow_models(production_version="production", challenger_version="chal")


def test_level_gap_between_markets_does_not_make_challenger_win():
    result = _evaluate(
        _two_market_rows("production", jp_level=0.01, us_level=0.03),
        _two_market_rows("chal", jp_level=0.03, us_level=0.01),
    )

    assert result["challenger_hit_rate"] == pytest.approx(result["production_hit_rate"])
    assert result["challenger_sharpe"] == pytest.approx(result["production_sharpe"])
    assert result["production_daily_rank_corr"] == pytest.approx(_WITHIN_CORR)
    assert result["challenger_daily_rank_corr"] == pytest.approx(_WITHIN_CORR)
    assert result["challenger_wins"] is False


def test_prediction_std_is_taken_within_market():
    """市場ごとに定数の予想は、市場の間の水準の差があっても定数と見なす。"""
    production = _two_market_rows("production", jp_level=0.01, us_level=0.03)
    challenger = _two_market_rows("chal", jp_level=0.03, us_level=0.01)
    challenger["predicted_ratio"] = challenger["market"].map({"jp": 0.03, "us": 0.01})

    result = _evaluate(production, challenger)

    assert result["challenger_prediction_std"] == pytest.approx(0.0)
    assert result["challenger_daily_rank_corr"] is None
    assert result["challenger_constant"] is True
    assert result["challenger_wins"] is False


@pytest.mark.parametrize("market", ["jp", "us"])
def test_morning_run_and_retry_of_same_trading_day_are_one_bundle(market: str):
    """取引日が同じ朝の回と、UTC の 00:00 を過ぎたやり直しの回は1つの束に入る。"""

    def rows(model: str, retry_level: float) -> pd.DataFrame:
        return pd.DataFrame(
            [
                _row(
                    model,
                    market,
                    i,
                    "20261119_224436" if i < 3 else "20261120_003000",
                    0.01 + 0.001 * i + (0.0 if i < 3 else retry_level),
                    0.001 * (i + 1),
                )
                for i in range(6)
            ]
        )

    # 予想の順位は production が 4,5,6,1,2,3、challenger が 1..6。回ごとに分けるとどちらも 1.0
    production = rows("production", retry_level=-0.006)
    challenger = rows("chal", retry_level=0.0)

    result = _evaluate(production, challenger)

    assert result["production_daily_rank_corr"] == pytest.approx(1 - 6 * 54 / (6 * 35))
    assert result["challenger_daily_rank_corr"] == pytest.approx(1.0)


def test_input_rows_are_not_changed():
    production = _two_market_rows("production", jp_level=0.01, us_level=0.03)
    challenger = _two_market_rows("chal", jp_level=0.03, us_level=0.01)
    before = list(production.columns)

    _evaluate(production, challenger)

    assert list(production.columns) == before
    assert list(challenger.columns) == before
