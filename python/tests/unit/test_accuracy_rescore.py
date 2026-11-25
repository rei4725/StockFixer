"""足が直った後に、直近の採点の行を付け直すことのテスト（#2155）。"""

import logging
from contextlib import ExitStack
from unittest.mock import patch

import pandas as pd
import pytest

from src.prediction.db import load_prediction_accuracy, save_prediction_accuracy
from src.utils.db._bulk import bulk_insert
from src.utils.db._connection import _db_connection

_DAILY = "src.orchestration.jobs.daily"
_DISCORD = "src.reporting.discord.discord_utils"
_RESCORE = "src.prediction.accuracy_rescore"
_MODELS = ["production", "challenger"]
# 米国の引けの後（UTC 22:45）の予想。基準の足は 11-17、実績の足は 11-18
_PREDICTED_AT = "20261117_224500"


def _bars(symbol, closes, market="us"):
    """closes（日付 → 終値）を market_data_raw に入れる。"""
    df = pd.DataFrame(
        [
            {
                "market": market,
                "symbol": symbol,
                "ticker": symbol,
                "timeframe": "1d",
                "ts": pd.Timestamp(day),
                "close": close,
            }
            for day, close in closes.items()
        ]
    )
    with _db_connection() as con:
        bulk_insert(con, "market_data_raw", df, columns=list(df.columns))


def _score(symbol, actual, model="production", predicted_at=_PREDICTED_AT, base=100.0, horizon=1):
    """予想 +1% の採点の行を、実績の終値 actual で保存する。"""
    actual_ratio = None if actual is None else (actual - base) / base
    save_prediction_accuracy(
        [
            {
                "market": "us",
                "symbol": symbol,
                "model_name": model,
                "predicted_at": predicted_at,
                "horizon": horizon,
                "predicted_price": base * 1.01,
                "actual_price": actual,
                "predicted_ratio": 0.01,
                "actual_ratio": actual_ratio,
                "direction_match": None if actual is None else actual_ratio >= 0,
            }
        ]
    )


def _row(symbol, model="production", horizon=1):
    df = load_prediction_accuracy(symbol=symbol, horizon=horizon, model_name=model, limit=0)
    assert len(df) == 1
    return df.iloc[0]


def _rescore(model="production"):
    from src.prediction.accuracy_rescore import rescore_fixed_bars

    return rescore_fixed_bars(model_name=model)


def _run_daily(**extra):
    """日次の道を、採点と付け直しは本物のまま、外の依存を差し替えて流す。"""
    from src.orchestration.jobs.daily import run_daily_pipeline

    with ExitStack() as stack:

        def p(target, **kwargs):
            return stack.enter_context(patch(target, **kwargs))

        mocks = {
            "webhook": p(f"{_DISCORD}.send_webhook_text"),
            "completion": p(f"{_DISCORD}.send_daily_pipeline_completion"),
            "predict": p("src.prediction.prediction_pipeline.predict_all_unified", return_value=[]),
        }
        p(f"{_DISCORD}.send_daily_pipeline_error")
        p(f"{_DISCORD}.send_accuracy_summary")
        p("src.prediction.prediction_pipeline.output_top_worst_results")
        p("src.prediction.predict_unified.preload_models", return_value=[])
        p("src.prediction.shadow_evaluation.predict_with_challenger_unified", return_value=[])
        p("src.watchlist.batch_runner.load_target_symbols", return_value=[])
        p("src.market_data.pipeline.run_batch_pipeline")
        p(f"{_DAILY}._raise_if_fetch_mostly_unavailable")
        p(f"{_DAILY}.production_already_predicted_today", return_value=False)
        p(f"{_DAILY}.write_accuracy_json_without_sectors", return_value=None)
        p(f"{_DAILY}.append_accuracy_json_sectors")
        p(f"{_DAILY}.fetch_previous_run_stats", return_value=None)
        p(f"{_DAILY}.evaluate_output_invariants_stage", return_value=([], {}))
        p(f"{_DAILY}.run_daily_drift_check")
        p(f"{_DAILY}.record_health_degraded_streak")
        p(f"{_DAILY}.evaluate_and_notify_alerts")
        p(
            "src.infrastructure.persistence.prediction_result_repository."
            "PostgresPredictionResultRepository"
        )
        for target, kwargs in extra.items():
            mocks[target] = p(f"{_RESCORE}.{target}", **kwargs)
        run_daily_pipeline()
    return mocks


@pytest.mark.parametrize("model", _MODELS)
def test_daily_rescores_row_after_bar_is_fixed(model):
    """予想の表が新しい回に入れ替わった後でも、直った足の終値で付け直すこと"""
    _bars("RSA", {"2026-11-17": 100.0, "2026-11-18": 98.0, "2026-11-19": 97.0})
    _score("RSA", 101.0, model=model)  # 途中の足の終値 101（的中）で採点してある
    new_run = pd.DataFrame(
        [
            {
                "market": "us",
                "symbol": "RSA",
                "predicted_at": "20261119_224500",
                "avg_pred_price": 98.0,
                "current_price": 97.0,
                "model_version": model,
            }
        ]
    )
    with patch(
        "src.prediction.db.load_prediction_results",
        side_effect=lambda model_version=None, **_kw: (
            new_run if model_version == model else pd.DataFrame()
        ),
    ):
        mocks = _run_daily()

    row = _row("RSA", model)
    assert row["actual_price"] == 98.0
    assert row["actual_ratio"] == pytest.approx(-0.02)
    assert not row["direction_match"]
    assert row["predicted_price"] == 101.0
    assert row["predicted_ratio"] == 0.01
    mocks["webhook"].assert_not_called()


@pytest.mark.parametrize("model", _MODELS)
def test_rescore_failure_is_notified_and_later_stages_run(model):
    """片方の付け直しが落ちても、もう片方と後の段は進み、[2/6] の知らせが1回出ること"""

    def fail(*, model_name):
        if model_name == model:
            raise RuntimeError("rescore down")

    mocks = _run_daily(rescore_fixed_bars={"side_effect": fail})

    called = [c.kwargs["model_name"] for c in mocks["rescore_fixed_bars"].call_args_list]
    assert called == _MODELS
    mocks["webhook"].assert_called_once()
    text = mocks["webhook"].call_args.args[0]
    assert "[2/6]" in text and f"({model})" in text and "rescore down" in text
    mocks["predict"].assert_called_once()
    mocks["completion"].assert_called_once()


def test_only_target_rows_are_saved():
    """変わっていない行・基準の合わない行・実績の空の行・期間より古い行・別の horizon は保存しないこと"""
    fixed = {"2026-11-17": 100.0, "2026-11-18": 98.0}
    _bars("SAME", fixed)
    _score("SAME", 98.0 * (1 + 1e-9))  # 浮動小数の誤差だけの違い
    _bars("BASE", {"2026-11-17": 100.01, "2026-11-18": 98.0})  # 基準が 0.01% ずれた
    _score("BASE", 101.0)
    _bars("SPLIT", {"2026-11-17": 50.0, "2026-11-18": 49.0})  # 分割で取り直した
    _score("SPLIT", 101.0)
    _bars("EMPTY", fixed)
    _score("EMPTY", None)
    _bars("H5", fixed)
    _score("H5", 101.0, horizon=5)
    _score("NOBAR", 101.0)
    _bars("OLD", {"2026-10-01": 100.0, "2026-10-02": 98.0})
    _score("OLD", 101.0, predicted_at="20261001_224500")
    for day in pd.bdate_range(end="2026-11-16", periods=9):  # 11-17 と合わせて 10 日分を埋める
        _score("FILL", 101.0, predicted_at=day.strftime("%Y%m%d_224500"))

    with patch(f"{_RESCORE}.save_prediction_accuracy") as save:
        assert _rescore() == 0

    save.assert_not_called()


def test_old_boundary_keeps_ten_days():
    """新しい10日分の一番古い日は付け直し、その前の日は付け直さないこと"""
    days = pd.bdate_range("2026-11-02", periods=11)
    closes = {d.strftime("%Y-%m-%d"): 100.0 for d in days} | {"2026-11-17": 100.0}
    for i, day in enumerate(days):
        symbol = f"D{i:02d}"
        _bars(symbol, closes)
        _score(symbol, 101.0, predicted_at=day.strftime("%Y%m%d_224500"))

    assert _rescore() == 10
    assert _row("D00")["actual_price"] == 101.0
    assert _row("D01")["actual_price"] == 100.0


def test_pending_split_step_is_not_rescored():
    """基準と実績の足の間に記録待ちの分割の段差がある行は付け直さないこと（#2035）"""
    _bars("PEND", {"2026-11-17": 100.0, "2026-11-18": 50.0})
    _score("PEND", 101.0)

    assert _rescore() == 0
    assert _row("PEND")["actual_price"] == 101.0


def test_missing_close_bars_are_skipped_like_scoring():
    """終値の欠けた足は、採点と同じく飛ばして次の足を実績にすること"""
    _bars("GAP", {"2026-11-17": 100.0, "2026-11-18": None, "2026-11-19": 102.0})
    _score("GAP", 99.0)

    assert _rescore() == 1
    row = _row("GAP")
    assert row["actual_price"] == 102.0
    assert row["direction_match"]


def test_base_bar_uses_previous_bar_over_holiday():
    """基準の日に足が無いときは、その日以前の最後の足を基準にすること"""
    _bars("HOL", {"2026-11-13": 100.0, "2026-11-18": 98.0})
    _score("HOL", 101.0)

    assert _rescore() == 1
    assert _row("HOL")["actual_price"] == 98.0


def test_logs_one_info_line_with_counts(caplog):
    """付け直した数と基準の合わない数を1行の INFO で出し、銘柄の名前は出さないこと"""
    for symbol in ("LOGA", "LOGB"):
        _bars(symbol, {"2026-11-17": 100.0, "2026-11-18": 98.0})
        _score(symbol, 101.0)
    _bars("LOGC", {"2026-11-17": 90.0, "2026-11-18": 98.0})
    _score("LOGC", 98.0)  # 実績は同じでも、基準の合わない行として数える

    with patch(f"{_RESCORE}.logger") as log:
        assert _rescore() == 2

    log.info.assert_called_once()
    message = log.info.call_args.args[0] % log.info.call_args.args[1:]
    assert "付け直し=2件" in message
    assert "基準の足が合わない=1件" in message
    assert "LOG" not in message
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_no_rows_returns_zero():
    assert _rescore() == 0


def test_no_bars_in_market_returns_zero():
    """市場の足が1本も読めない回も、実績が全部空の回も落ちないこと"""
    _score("NONE", None)
    assert _rescore() == 0
    _score("NOBAR", 101.0)
    assert _rescore() == 0


@pytest.mark.parametrize("module", ["src.prediction.db.accuracy", "src.utils.db.market_data"])
def test_read_failure_is_raised(module):
    """読み込みの失敗を行なしにせず投げること（採点の行・足のどちらも）"""
    _bars("ERR", {"2026-11-17": 100.0, "2026-11-18": 98.0})
    _score("ERR", 101.0)

    real = pd.read_sql
    table = "prediction_accuracy" if module.endswith("accuracy") else "market_data_raw"

    def read_sql(query, *args, **kwargs):
        if table in query:
            raise RuntimeError("db down")
        return real(query, *args, **kwargs)

    with patch(f"{module}.pd.read_sql", side_effect=read_sql):
        with pytest.raises(RuntimeError, match="db down"):
            _rescore()
