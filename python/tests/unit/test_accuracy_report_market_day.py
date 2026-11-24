"""採点の JSON の日ごとの物差しを市場と取引日で束ねるテスト（#2141）。"""

import pandas as pd
import pytest

from src.prediction.accuracy_report import build_horizon_report
from src.prediction.baseline_metrics import BASELINE_METRIC_KEYS, baseline_metrics
from src.prediction.shadow_evaluation import _with_market_day

# どちらも jp・us の基準の日が 2026-11-19 になる（UTC の日付は違う）
MORNING = "20261119_220500"
RERUN = "20261120_010000"
NEXT_MORNING = "20261120_220500"

DAILY_KEYS = ("daily_rank_corr", "prediction_std")
JP_PRED = [0.001, 0.002, 0.003, 0.004]
US_PRED = [0.021, 0.023, 0.025, 0.027]
JP_ACTUAL = [0.004, 0.003, 0.002, 0.001]
US_ACTUAL = [0.024, 0.022, 0.023, 0.021]


def _rows(market: str, predicted_at: str, preds: list, actuals: list) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "market": market,
            "symbol": [f"{market}{i}" for i in range(len(preds))],
            "model_name": "production",
            "predicted_at": predicted_at,
            "horizon": 1,
            "predicted_ratio": preds,
            "actual_ratio": actuals,
            "direction_match": [(p >= 0) == (a >= 0) for p, a in zip(preds, actuals)],
        }
    )


def _concat(*frames: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(frames, ignore_index=True)


def _mixed(predicted_at: str = MORNING) -> pd.DataFrame:
    """市場の中はどちらも逆順で、市場の間の水準の差だけで混ぜた順位相関が正になる表。"""
    return _concat(
        _rows("jp", predicted_at, JP_PRED, JP_ACTUAL),
        _rows("us", predicted_at, US_PRED, US_ACTUAL),
    )


def test_daily_rank_corr_is_mean_of_markets():
    rows = _mixed()
    report = build_horizon_report(rows, rows, None)

    # 混ぜた順位相関は正、市場別はどちらも負
    assert report["rank_corr"] > 0
    by_market = {m: v["daily_rank_corr"] for m, v in report["by_market"].items()}
    assert by_market["jp"] == pytest.approx(-1.0)
    assert by_market["us"] == pytest.approx(-0.8)
    for summary in (report, report["four_week"]):
        assert summary["daily_rank_corr"] == pytest.approx(-0.9)
        stds = [v["prediction_std"] for v in report["by_market"].values()]
        assert summary["prediction_std"] == pytest.approx(sum(stds) / 2)


def test_sector_and_magnitude_are_split_by_market():
    rows = _mixed()
    report = build_horizon_report(rows, rows, lambda market, symbol: "Tech")
    assert report["by_sector"]["Tech"]["daily_rank_corr"] == pytest.approx(-0.9)

    # 10行の五分位は2行ずつ。q3 は jp 1行と us 1行なので、同じ束の2行が無い
    rows = _concat(
        _rows("jp", MORNING, JP_PRED + [0.005], JP_ACTUAL + [0.0]),
        _rows("us", MORNING, US_PRED + [0.029], US_ACTUAL + [0.02]),
    )
    by_magnitude = build_horizon_report(rows, rows, None)["by_magnitude"]
    assert by_magnitude["q3"]["n_scored"] == 2
    assert by_magnitude["q3"]["daily_rank_corr"] is None
    assert by_magnitude["q3"]["prediction_std"] is None
    assert by_magnitude["q1"]["daily_rank_corr"] == pytest.approx(-1.0)


def test_rerun_on_same_trading_day_is_one_group():
    window = _concat(
        _rows("jp", MORNING, JP_PRED, JP_ACTUAL),
        _rows("jp", RERUN, US_PRED, US_ACTUAL),
    )
    four_week = build_horizon_report(window.iloc[:4], window, None)["four_week"]

    assert four_week["n_days"] == 1
    # 1つの束なら全行の順位相関・標準偏差と同じ（2つの束なら -0.9 になる）
    assert four_week["rank_corr"] > 0
    assert four_week["daily_rank_corr"] == pytest.approx(four_week["rank_corr"])
    assert four_week["prediction_std"] == pytest.approx(window["predicted_ratio"].std(ddof=1))


@pytest.mark.parametrize("market", ["jp", "us"])
def test_one_market_one_run_per_day_is_unchanged(market):
    window = _concat(
        _rows(market, MORNING, JP_PRED, JP_ACTUAL),
        _rows(market, NEXT_MORNING, US_PRED, US_ACTUAL),
    )
    today = window.iloc[4:]
    report = build_horizon_report(today, window, None)

    # date の列の無い行をそのまま渡すと predicted_at で束ねる（直す前の値）
    for summary, rows in ((report, today), (report["four_week"], window)):
        before = baseline_metrics(rows)
        assert {k: summary[k] for k in BASELINE_METRIC_KEYS} == before


def test_other_values_and_shape_are_unchanged():
    window = _concat(_mixed(MORNING), _mixed(RERUN), _mixed(NEXT_MORNING))
    window["direction_match"] = [i % 8 < 4 or i >= 16 for i in range(len(window))]
    today = window.iloc[16:]
    report = build_horizon_report(today, window, None)
    four_week = report["four_week"]

    common = {"n_scored", "direction_accuracy", "mean_abs_error", *BASELINE_METRIC_KEYS}
    assert set(four_week) == common | {"n_days", "daily_direction_accuracy_mean", "by_market"}
    assert set(report) == common | {
        "four_week",
        "top_misses",
        "by_market",
        "by_magnitude",
        "by_sector",
    }
    for group in (*report["by_market"].values(), *report["by_magnitude"].values()):
        assert set(group) == common

    for summary, rows in ((report, today), (four_week, window)):
        before = baseline_metrics(rows)
        for key in set(BASELINE_METRIC_KEYS) - set(DAILY_KEYS):
            assert summary[key] == before[key], key
    # 日の数は市場で分けず、的中率の日ごとの平均も取引日ごと（(0.5 + 1.0) / 2）のまま
    assert four_week["n_days"] == 2
    assert four_week["daily_direction_accuracy_mean"] == pytest.approx(0.75)
    assert four_week["direction_accuracy"] == pytest.approx(16 / 24)
    # top_misses の項目に date が混ざらない
    assert set(report["top_misses"][0]) == {
        "market",
        "symbol",
        "predicted_at",
        "predicted_ratio",
        "actual_ratio",
        "abs_error",
    }


def test_input_rows_are_not_modified():
    rows = _mixed()
    columns = list(rows.columns)
    build_horizon_report(rows, rows, None)
    assert list(rows.columns) == columns


def test_grouping_matches_shadow_evaluation():
    window = _concat(_mixed(MORNING), _mixed(RERUN), _mixed(NEXT_MORNING))
    four_week = build_horizon_report(window, window, None)["four_week"]
    shadow = baseline_metrics(_with_market_day(window))
    for key in DAILY_KEYS:
        assert four_week[key] == pytest.approx(shadow[key])
