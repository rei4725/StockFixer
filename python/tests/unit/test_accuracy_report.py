"""日次の採点の JSON 書き出し（src/prediction/accuracy_report.py）のテスト。"""

import json
import os
import tempfile
from contextlib import ExitStack
from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

_TODAY_AT = "20261019_223000"
_SECTORS = {
    "7203": "Consumer Cyclical",
    "6758": "Technology",
    "AAPL": "Technology",
    "MSFT": "Technology",
}


def _acc_row(model, market, symbol, predicted_at, pred, act, match):
    return {
        "market": market,
        "symbol": symbol,
        "model_name": model,
        "predicted_at": predicted_at,
        "horizon": 1,
        "predicted_price": 100.0 * (1 + pred),
        "actual_price": 100.0 * (1 + act),
        "predicted_ratio": pred,
        "actual_ratio": act,
        "direction_match": match,
    }


def _accuracy_rows() -> pd.DataFrame:
    rows = [
        # 今日の採点（production）
        _acc_row("production", "jp", "7203", _TODAY_AT, 0.02, 0.01, True),
        _acc_row("production", "jp", "6758", _TODAY_AT, 0.01, -0.03, False),
        _acc_row("production", "us", "AAPL", _TODAY_AT, -0.02, -0.01, True),
        _acc_row("production", "us", "MSFT", _TODAY_AT, 0.00, -0.05, False),
        # 4週の窓の中の前の日
        _acc_row("production", "jp", "7203", "20261010_223000", 0.03, 0.01, True),
        _acc_row("production", "jp", "6758", "20261010_223000", 0.01, 0.01, True),
        # 4週の窓の外
        _acc_row("production", "jp", "7203", "20260901_223000", 0.5, -0.0, False),
    ]
    return pd.DataFrame(rows)


def _snapshot(model_version):
    if model_version != "production":
        return pd.DataFrame()
    return pd.DataFrame(
        [
            {"market": m, "symbol": s, "predicted_at": _TODAY_AT, "model_version": "production"}
            for m, s in [("jp", "7203"), ("jp", "6758"), ("us", "AAPL"), ("us", "MSFT")]
        ]
    )


def _write(tmp_path, accuracy_df, sector_getter):
    from src.prediction.accuracy_report import write_daily_accuracy_report

    with (
        patch("src.prediction.db.load_prediction_accuracy", return_value=accuracy_df) as mock_acc,
        patch(
            "src.prediction.db.load_prediction_results",
            side_effect=lambda **kw: _snapshot(kw.get("model_version")),
        ),
    ):
        path = write_daily_accuracy_report(
            report_date=date(2026, 10, 20),
            output_dir=str(tmp_path),
            sector_getter=sector_getter,
        )
    return path, mock_acc


def test_report_fields_from_scored_rows(tmp_path):
    """作りの採点の行から、件数・的中率・誤差・4週・外れ・内訳が正しく出ること"""
    path, _ = _write(tmp_path, _accuracy_rows(), lambda market, symbol: _SECTORS[symbol])

    assert path == str(tmp_path / "2026-10-20.json")
    with open(path, encoding="utf-8") as f:
        report = json.load(f)

    assert report["date"] == "2026-10-20"
    assert report["window_days"] == 28
    prod = report["models"]["production"]["1"]

    assert prod["n_scored"] == 4
    assert prod["direction_accuracy"] == pytest.approx(0.5)
    assert prod["mean_abs_error"] == pytest.approx(0.0275)

    four_week = prod["four_week"]
    assert four_week["n_scored"] == 6
    assert four_week["n_days"] == 2
    assert four_week["direction_accuracy"] == pytest.approx(4 / 6)
    assert four_week["mean_abs_error"] == pytest.approx(0.13 / 6)
    assert four_week["daily_direction_accuracy_mean"] == pytest.approx(0.75)

    misses = prod["top_misses"]
    assert len(misses) == 4
    assert [m["symbol"] for m in misses[:2]] == ["MSFT", "6758"]
    assert misses[0]["abs_error"] == pytest.approx(0.05)
    assert misses[0]["market"] == "us"
    assert misses[0]["predicted_at"] == _TODAY_AT

    assert prod["by_market"]["jp"]["n_scored"] == 2
    assert prod["by_market"]["jp"]["direction_accuracy"] == pytest.approx(0.5)
    assert prod["by_market"]["jp"]["mean_abs_error"] == pytest.approx(0.025)
    assert prod["by_market"]["us"]["mean_abs_error"] == pytest.approx(0.03)

    tech = prod["by_sector"]["Technology"]
    assert tech["n_scored"] == 3
    assert tech["direction_accuracy"] == pytest.approx(1 / 3)
    assert tech["mean_abs_error"] == pytest.approx(0.1 / 3)
    assert prod["by_sector"]["Consumer Cyclical"]["n_scored"] == 1

    challenger = report["models"]["challenger"]["1"]
    assert challenger["n_scored"] == 0
    assert challenger["four_week"]["n_scored"] == 0


def test_top_misses_capped_at_ten(tmp_path):
    """外れの大きい銘柄は上位10件までであること"""
    rows = [
        _acc_row("production", "jp", f"{1000 + i}", _TODAY_AT, 0.01, 0.01 - i / 100, i == 0)
        for i in range(15)
    ]
    snapshot = pd.DataFrame(
        [{"market": r["market"], "symbol": r["symbol"], "predicted_at": _TODAY_AT} for r in rows]
    )

    from src.prediction.accuracy_report import write_daily_accuracy_report

    with (
        patch("src.prediction.db.load_prediction_accuracy", return_value=pd.DataFrame(rows)),
        patch("src.prediction.db.load_prediction_results", return_value=snapshot),
    ):
        path = write_daily_accuracy_report(
            report_date=date(2026, 10, 20),
            models=("production",),
            output_dir=str(tmp_path),
            sector_getter=lambda market, symbol: "S",
        )

    with open(path, encoding="utf-8") as f:
        prod = json.load(f)["models"]["production"]["1"]
    assert prod["n_scored"] == 15
    assert len(prod["top_misses"]) == 10
    assert prod["top_misses"][0]["symbol"] == "1014"


_LATEST_AT = "20261021_224556"
_OLD_AT = "20261020_224701"


def _mixed_runs_write(tmp_path):
    """スナップショットに今の回と前の回の行が混ざる日の JSON を書く。"""
    from src.prediction.accuracy_report import write_daily_accuracy_report

    rows = [
        _acc_row("production", "jp", "7203", _LATEST_AT, 0.02, 0.01, True),
        _acc_row("production", "us", "AAPL", _LATEST_AT, 0.01, -0.01, False),
        # 前の回で予想が止まった銘柄の古い行（今日の分に入れない）
        _acc_row("production", "us", "PSKY", _OLD_AT, 0.03, -0.30, False),
        _acc_row("production", "jp", "6758", _OLD_AT, 0.01, -0.20, False),
    ]
    snapshot = pd.DataFrame(
        [
            {"market": r["market"], "symbol": r["symbol"], "predicted_at": r["predicted_at"]}
            for r in rows
        ]
    )

    with (
        patch("src.prediction.db.load_prediction_accuracy", return_value=pd.DataFrame(rows)),
        patch("src.prediction.db.load_prediction_results", return_value=snapshot),
        patch("src.prediction.accuracy_report.logger") as mock_logger,
    ):
        path = write_daily_accuracy_report(
            report_date=date(2026, 10, 23),
            models=("production",),
            output_dir=str(tmp_path),
            sector_getter=lambda market, symbol: "S",
        )

    with open(path, encoding="utf-8") as f:
        prod = json.load(f)["models"]["production"]["1"]
    return prod, mock_logger


def test_today_rows_only_from_latest_run(tmp_path):
    """今日の採点は、スナップショットの predicted_at の一番新しい回の行だけから出ること"""
    prod, _ = _mixed_runs_write(tmp_path)

    assert prod["n_scored"] == 2
    assert prod["direction_accuracy"] == pytest.approx(0.5)
    assert [m["symbol"] for m in prod["top_misses"]] == ["AAPL", "7203"]
    assert {m["predicted_at"] for m in prod["top_misses"]} == {_LATEST_AT}
    assert prod["by_market"]["jp"]["n_scored"] == 1
    assert prod["by_market"]["us"]["n_scored"] == 1
    assert prod["by_sector"]["S"]["n_scored"] == 2
    # 4週の窓は前の回の行も含めたまま
    assert prod["four_week"]["n_scored"] == 4


_BACKFILL_AT = "20261020_013000"  # 補完の回（JST 10/20 10:30）
_NEXT_MORNING_AT = "20261020_223000"  # 次の朝の回（JST 10/21 07:30）


def _write_production(tmp_path, rows, report_date):
    from src.prediction.accuracy_report import write_daily_accuracy_report

    snapshot = pd.DataFrame(
        [
            {"market": r["market"], "symbol": r["symbol"], "predicted_at": r["predicted_at"]}
            for r in rows
        ]
    )
    with (
        patch("src.prediction.db.load_prediction_accuracy", return_value=pd.DataFrame(rows)),
        patch("src.prediction.db.load_prediction_results", return_value=snapshot),
    ):
        path = write_daily_accuracy_report(
            report_date=report_date,
            models=("production",),
            output_dir=str(tmp_path),
            sector_getter=lambda market, symbol: "S",
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)["models"]["production"]["1"]


def test_four_week_backfill_and_next_morning_are_separate_days(tmp_path):
    """補完の回と次の朝の回は UTC の日付が同じでも別の日として数えること（#987）"""
    rows = [
        _acc_row("production", "jp", "7203", _BACKFILL_AT, 0.02, 0.01, True),
        _acc_row("production", "us", "AAPL", _BACKFILL_AT, 0.01, -0.01, False),
        _acc_row("production", "jp", "7203", _NEXT_MORNING_AT, 0.02, 0.01, True),
    ]
    prod = _write_production(tmp_path, rows, date(2026, 10, 21))

    four_week = prod["four_week"]
    assert four_week["n_scored"] == 3
    assert four_week["n_days"] == 2
    # 日ごとの的中率 0.5 と 1.0 の平均（行を通した 2/3 ではない）
    assert four_week["daily_direction_accuracy_mean"] == pytest.approx(0.75)


def test_four_week_window_uses_base_bar_date(tmp_path):
    """4週の窓は予想の基準の日で切り、UTC の日付が窓の初日でも基準の日が前なら外すこと（#987）"""
    rows = [
        # 基準の日 9/21（JST 9/22 10:30 の補完の回）は窓（9/22 から）の外
        _acc_row("production", "jp", "7203", "20260922_013000", 0.02, -0.01, False),
        # 基準の日 9/22 のふだんの回は窓の中
        _acc_row("production", "jp", "7203", "20260922_223000", 0.02, 0.01, True),
        _acc_row("production", "jp", "7203", _NEXT_MORNING_AT, 0.02, 0.01, True),
    ]
    prod = _write_production(tmp_path, rows, date(2026, 10, 20))

    assert prod["four_week"]["n_scored"] == 2
    assert prod["four_week"]["n_days"] == 2


def test_old_run_rows_count_logged(tmp_path):
    """スナップショットの前の回の行の数が INFO のログに出ること"""
    _, mock_logger = _mixed_runs_write(tmp_path)

    messages = [c.args[0] % c.args[1:] for c in mock_logger.info.call_args_list]
    assert any("前の回の行" in m and "2件" in m and _LATEST_AT in m for m in messages), messages


def test_zero_scored_day_writes_zero_counts(tmp_path):
    """採点が0件の日も件数0の JSON を書き、例外にならないこと"""
    sector_getter = MagicMock()
    path, _ = _write(tmp_path, pd.DataFrame(), sector_getter)

    with open(path, encoding="utf-8") as f:
        report = json.load(f)

    for model in ("production", "challenger"):
        section = report["models"][model]["1"]
        assert section["n_scored"] == 0
        assert section["direction_accuracy"] is None
        assert section["mean_abs_error"] is None
        assert section["four_week"]["n_scored"] == 0
        assert section["top_misses"] == []
        assert section["by_market"] == {}
        assert section["by_sector"] == {}
    sector_getter.assert_not_called()


def test_sector_lookup_failure_falls_back_to_unknown(tmp_path):
    """業種が引けない銘柄は UNKNOWN に入れ、書き出しを止めないこと"""

    def _broken(market, symbol):
        raise RuntimeError("yfinance down")

    path, _ = _write(tmp_path, _accuracy_rows(), _broken)

    with open(path, encoding="utf-8") as f:
        prod = json.load(f)["models"]["production"]["1"]
    assert prod["by_sector"]["UNKNOWN"]["n_scored"] == 4


def test_reads_scored_rows_from_database(tmp_path):
    """テスト用 DB の prediction_accuracy から読んだ行でも欄が出ること（型の確かめ）"""
    from src.prediction.accuracy_report import write_daily_accuracy_report
    from src.prediction.db import save_prediction_accuracy

    rows = _accuracy_rows()
    save_prediction_accuracy(rows.to_dict("records"))

    with patch(
        "src.prediction.db.load_prediction_results",
        side_effect=lambda **kw: _snapshot(kw.get("model_version")),
    ):
        path = write_daily_accuracy_report(
            report_date=date(2026, 10, 20),
            output_dir=str(tmp_path),
            sector_getter=lambda market, symbol: _SECTORS[symbol],
        )

    with open(path, encoding="utf-8") as f:
        prod = json.load(f)["models"]["production"]["1"]
    assert prod["n_scored"] == 4
    assert prod["direction_accuracy"] == pytest.approx(0.5)
    assert prod["four_week"]["n_scored"] == 6


@patch("src.reporting.discord.discord_utils.send_daily_pipeline_error")
@patch("src.reporting.discord.discord_utils.send_daily_pipeline_completion")
@patch("src.reporting.discord.discord_utils.send_accuracy_summary")
@patch("src.orchestration.jobs.daily.run_daily_drift_check")
@patch("src.prediction.shadow_evaluation.predict_with_challenger_unified", return_value=[])
@patch("src.prediction.predict_unified.preload_models", return_value=[])
@patch("src.orchestration.jobs.daily.fetch_previous_run_stats", return_value=None)
@patch("src.orchestration.jobs.daily.evaluate_output_invariants_stage", return_value=([], {}))
@patch("src.orchestration.jobs.daily.evaluate_and_notify_alerts")
@patch("src.prediction.prediction_pipeline.run_accuracy_check", return_value=pd.DataFrame())
@patch("src.prediction.prediction_pipeline.output_top_worst_results")
@patch("src.prediction.prediction_pipeline.predict_all_unified", return_value=[])
@patch("src.market_data.pipeline.run_batch_pipeline")
@patch("src.watchlist.batch_runner.load_target_symbols", return_value=[])
def test_report_failure_does_not_stop_prediction(
    _symbols,
    _data,
    mock_predict,
    _output,
    mock_accuracy,
    *_rest,
):
    """書き出しが例外でも [3/6] の予測に進むこと"""
    from src.orchestration.jobs.daily import run_daily_pipeline

    with patch(
        "src.prediction.accuracy_report.prepare_daily_accuracy_report",
        side_effect=RuntimeError("disk full"),
    ) as mock_write:
        run_daily_pipeline()

    mock_write.assert_called_once()
    assert mock_accuracy.call_count == 2
    mock_predict.assert_called_once()


@patch("src.reporting.discord.discord_utils.send_daily_pipeline_error")
@patch("src.reporting.discord.discord_utils.send_daily_pipeline_completion")
@patch("src.reporting.discord.discord_utils.send_accuracy_summary")
@patch("src.orchestration.jobs.daily.run_daily_drift_check")
@patch("src.prediction.shadow_evaluation.predict_with_challenger_unified", return_value=[])
@patch("src.prediction.predict_unified.preload_models", return_value=[])
@patch("src.orchestration.jobs.daily.fetch_previous_run_stats", return_value=None)
@patch("src.orchestration.jobs.daily.evaluate_output_invariants_stage", return_value=([], {}))
@patch("src.orchestration.jobs.daily.evaluate_and_notify_alerts")
@patch("src.prediction.prediction_pipeline.run_accuracy_check", return_value=pd.DataFrame())
@patch("src.prediction.prediction_pipeline.output_top_worst_results")
@patch("src.market_data.pipeline.run_batch_pipeline")
@patch("src.watchlist.batch_runner.load_target_symbols", return_value=[])
def test_sector_lookup_runs_after_prediction_not_before(*_mocks):
    """[2/6] から [3/6] の予測の開始までに yfinance の業種を引かず、道の最後に引いて JSON に出すこと"""
    from src.orchestration.jobs.daily import run_daily_pipeline

    events: list[str] = []

    def _sector(market, symbol):
        events.append("sector")
        if symbol == "MSFT":
            raise RuntimeError("429 Too Many Requests")
        return _SECTORS[symbol]

    def _predict(**_kwargs):
        events.append("predict")
        return []

    with (
        tempfile.TemporaryDirectory() as results_dir,
        patch("src.prediction.accuracy_report.get_results_dir", return_value=results_dir),
        patch("src.prediction.db.load_prediction_accuracy", return_value=_accuracy_rows()),
        patch(
            "src.prediction.db.load_prediction_results",
            side_effect=lambda **kw: _snapshot(kw.get("model_version")),
        ),
        patch("src.utils.sector_constraints.get_symbol_sector", side_effect=_sector),
        patch("src.prediction.prediction_pipeline.predict_all_unified", side_effect=_predict),
        patch(
            "src.prediction.accuracy_report.now_jst",
            return_value=pd.Timestamp("2026-10-20 07:00", tz="Asia/Tokyo"),
        ),
    ):
        run_daily_pipeline()

        assert "predict" in events
        assert "sector" not in events[: events.index("predict")]
        assert events.count("sector") == 4

        with open(os.path.join(results_dir, "accuracy", "2026-10-20.json"), encoding="utf-8") as f:
            prod = json.load(f)["models"]["production"]["1"]

    assert prod["n_scored"] == 4
    assert prod["four_week"]["n_scored"] == 6
    assert prod["by_sector"]["Technology"]["n_scored"] == 2
    assert prod["by_sector"]["Consumer Cyclical"]["n_scored"] == 1
    assert prod["by_sector"]["UNKNOWN"]["n_scored"] == 1


def _read_error_patches(stack, module):
    """module の DB の接続を偽にし、pd.read_sql を読み込みの失敗にする。"""
    stack.enter_context(patch(f"{module}._db_connection", MagicMock()))
    stack.enter_context(patch("pandas.read_sql", side_effect=RuntimeError("connection lost")))


def test_accuracy_read_error_raises_and_keeps_previous_json(tmp_path):
    """prediction_accuracy が読めないときは例外にし、当日の JSON を0件で上書きしないこと（#1283）"""
    from src.prediction.accuracy_report import prepare_daily_accuracy_report

    previous = tmp_path / "2026-10-20.json"
    previous.write_text('{"models": {"production": {"1": {"n_scored": 4}}}}', encoding="utf-8")

    with ExitStack() as stack:
        _read_error_patches(stack, "src.prediction.db.accuracy")
        stack.enter_context(
            patch(
                "src.prediction.db.load_prediction_results",
                side_effect=lambda **kw: _snapshot(kw.get("model_version")),
            )
        )
        with pytest.raises(RuntimeError, match="connection lost"):
            prepare_daily_accuracy_report(
                report_date=date(2026, 10, 20), output_dir=str(tmp_path)
            ).write(None)

    assert json.loads(previous.read_text(encoding="utf-8")) == {
        "models": {"production": {"1": {"n_scored": 4}}}
    }


def test_snapshot_read_error_raises_and_writes_no_json(tmp_path):
    """prediction_results が読めないときも例外にし、JSON を書かないこと（#1283）"""
    from src.prediction.accuracy_report import prepare_daily_accuracy_report

    with ExitStack() as stack:
        _read_error_patches(stack, "src.prediction.db.prediction_results")
        stack.enter_context(
            patch("src.prediction.db.load_prediction_accuracy", return_value=_accuracy_rows())
        )
        with pytest.raises(RuntimeError, match="connection lost"):
            prepare_daily_accuracy_report(report_date=date(2026, 10, 20), output_dir=str(tmp_path))

    assert list(tmp_path.glob("*.json")) == []


def test_no_scored_rows_still_writes_zero_json(tmp_path):
    """採点の行が本当に0件の日は、今までどおり n_scored 0 の JSON を書くこと（#1283）"""
    from src.prediction.accuracy_report import prepare_daily_accuracy_report

    with (
        patch("src.prediction.db.accuracy._db_connection", MagicMock()),
        patch("pandas.read_sql", return_value=pd.DataFrame()),
        patch(
            "src.prediction.db.load_prediction_results",
            side_effect=lambda **kw: _snapshot(kw.get("model_version")),
        ),
    ):
        path = prepare_daily_accuracy_report(
            report_date=date(2026, 10, 20), output_dir=str(tmp_path)
        ).write(None)

    with open(path, encoding="utf-8") as f:
        report = json.load(f)
    assert report["models"]["production"]["1"]["n_scored"] == 0


def test_load_prediction_accuracy_default_returns_empty_on_error():
    """raise_on_error を渡さない呼び出し元には、今までどおり空の表を返すこと（#1283）"""
    from src.prediction.db import load_prediction_accuracy

    with (
        patch("src.prediction.db.accuracy._db_connection", MagicMock()),
        patch("pandas.read_sql", side_effect=RuntimeError("connection lost")),
    ):
        assert load_prediction_accuracy(horizon=1).empty
        with pytest.raises(RuntimeError):
            load_prediction_accuracy(horizon=1, raise_on_error=True)
