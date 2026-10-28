"""ドリフトの再学習が、銘柄別のモデルを直近7日以内に更新した銘柄を外すことのテスト（#1331）。"""

import logging
import os
import time
from unittest.mock import patch

import pandas as pd

from src.domain.types import SymbolTask
from src.orchestration.jobs.drift_check import run_daily_drift_check


def _row(symbol: str, mae: float) -> dict:
    return {
        "market": "jp",
        "symbol": symbol,
        "mean_abs_error": mae,
        "direction_accuracy": 0.55,
        "n_samples": 20,
    }


def _write_models(models_dir, symbol: str, days_ago: float) -> None:
    subdir = os.path.join(str(models_dir), f"jp_{symbol}")
    os.makedirs(subdir, exist_ok=True)
    mtime = time.time() - days_ago * 86400
    for name in ("StockXGBoostModel", "StockLightGBMModel"):
        path = os.path.join(subdir, f"{name}.joblib")
        with open(path, "wb") as f:
            f.write(b"x")
        os.utime(path, (mtime, mtime))


@patch("src.orchestration.jobs.drift_check.record_drift_warn_streak")
@patch("src.prediction.training_pipeline.train_models_for_symbol_task")
@patch("src.watchlist.batch_runner.load_target_symbols")
@patch("src.reporting.discord.discord_utils.send_drift_retrain_notification")
@patch("src.prediction.db.load_drift_summary")
def test_recently_retrained_symbol_is_skipped(
    mock_load_summary, mock_notify, mock_load_symbols, mock_train, mock_streak, tmp_path, caplog
):
    """3日前に更新した銘柄は学び直さず、10日前とファイルの無い銘柄は学び直す。"""
    rows = [_row(f"S{i}", 0.058) for i in range(8)] + [
        _row("FRESH", 0.16),
        _row("OLD", 0.15),
        _row("NOFILE", 0.14),
    ]
    mock_load_summary.return_value = pd.DataFrame(rows)
    mock_load_symbols.return_value = [SymbolTask(market="jp", symbol=r["symbol"]) for r in rows]
    mock_train.return_value = {"status": "success"}
    _write_models(tmp_path, "FRESH", days_ago=3)
    _write_models(tmp_path, "OLD", days_ago=10)

    with patch("src.utils.data_path_utils.get_models_dir", return_value=str(tmp_path)):
        with caplog.at_level(logging.INFO):
            run_daily_drift_check()

    trained = [c.args[0].symbol for c in mock_train.call_args_list]
    assert "FRESH" not in trained
    assert sorted(trained) == ["NOFILE", "OLD"]
    mock_notify.assert_called_once()
    notified = [s["symbol"] for s in mock_notify.call_args.args[0]]
    assert "FRESH" in notified
    skip_lines = [r for r in caplog.records if "直近7日以内" in r.getMessage()]
    assert len(skip_lines) == 1
    assert skip_lines[0].levelno == logging.INFO


@patch("src.orchestration.jobs.drift_check.record_drift_warn_streak")
@patch("src.prediction.training_pipeline.train_models_for_symbol_task")
@patch("src.watchlist.batch_runner.load_target_symbols")
@patch("src.reporting.discord.discord_utils.send_drift_retrain_notification")
@patch("src.prediction.db.load_drift_summary")
@patch(
    "src.utils.db.system_config.get_config_value",
    side_effect=lambda key, default=None: "1" if key == "drift.max_retrain_per_run" else default,
)
def test_cap_applies_after_skipping_recent(
    mock_cfg, mock_load_summary, mock_notify, mock_load_symbols, mock_train, mock_streak, tmp_path
):
    """上限は外した後の銘柄に掛かる（上限1件なら、外した次に悪い銘柄を学び直す）。"""
    rows = [_row(f"S{i}", 0.058) for i in range(8)] + [_row("FRESH", 0.16), _row("OLD", 0.15)]
    mock_load_summary.return_value = pd.DataFrame(rows)
    mock_load_symbols.return_value = [SymbolTask(market="jp", symbol=r["symbol"]) for r in rows]
    mock_train.return_value = {"status": "success"}
    _write_models(tmp_path, "FRESH", days_ago=3)

    with patch("src.utils.data_path_utils.get_models_dir", return_value=str(tmp_path)):
        run_daily_drift_check()

    assert [c.args[0].symbol for c in mock_train.call_args_list] == ["OLD"]
    mock_notify.assert_called_once()
