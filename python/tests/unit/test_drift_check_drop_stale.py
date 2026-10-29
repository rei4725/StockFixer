"""日次のドリフト検査が、最後の予想が古い銘柄を判定から外すことのテスト（#1359）。"""

import logging
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pandas as pd

from src.domain.types import SymbolTask
from src.orchestration.jobs.drift_check import run_daily_drift_check


def _stamp(days_ago: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days_ago)).strftime("%Y%m%d_%H%M%S")


def _row(symbol: str, mae: float, hit: float, days_ago: float) -> dict:
    return {
        "market": "jp",
        "symbol": symbol,
        "mean_abs_error": mae,
        "direction_accuracy": hit,
        "n_samples": 20,
        "last_predicted_at": _stamp(days_ago),
    }


@patch("src.orchestration.jobs.drift_check.record_drift_warn_streak")
@patch("src.prediction.training_pipeline.train_models_for_symbol_task")
@patch("src.watchlist.batch_runner.load_target_symbols")
@patch("src.reporting.discord.discord_utils.send_drift_retrain_notification")
@patch("src.prediction.db.load_drift_summary")
def test_stale_symbols_do_not_trigger(
    mock_load_summary, mock_notify, mock_load_symbols, mock_train, mock_streak, caplog
):
    """今の銘柄だけでは超えない日に、古い銘柄だけが悪くても再学習も知らせも出さない。"""
    current = [_row(f"S{i}", 0.010, 0.60, days_ago=1) for i in range(8)]
    stale = [_row(f"OLD{i}", 0.20, 0.10, days_ago=40) for i in range(2)]
    mock_load_summary.return_value = pd.DataFrame(current + stale)
    mock_load_symbols.return_value = [SymbolTask(market="jp", symbol=r["symbol"]) for r in current]

    with caplog.at_level(logging.INFO):
        run_daily_drift_check()

    mock_streak.assert_called_once()
    assert mock_streak.call_args.args[0] is False
    mock_train.assert_not_called()
    mock_notify.assert_not_called()
    lines = [r for r in caplog.records if "古い 2 銘柄" in r.getMessage()]
    assert len(lines) == 1
    assert lines[0].levelno == logging.INFO


@patch("src.orchestration.jobs.drift_check.record_drift_warn_streak")
@patch("src.prediction.training_pipeline.train_models_for_symbol_task")
@patch("src.watchlist.batch_runner.load_target_symbols")
@patch("src.reporting.discord.discord_utils.send_drift_retrain_notification")
@patch("src.prediction.db.load_drift_summary")
def test_stale_symbols_leave_median_and_candidates(
    mock_load_summary, mock_notify, mock_load_symbols, mock_train, mock_streak
):
    """線の中央値・知らせ・再学習の候補は、古い銘柄を外した表から作る。"""
    current = [_row(f"S{i}", 0.030, 0.55, days_ago=1) for i in range(8)] + [
        _row("BAD", 0.060, 0.55, days_ago=1)
    ]
    # 古い銘柄が中央値を引き上げると、BAD は線（中央値×1.5）を超えなくなる
    stale = [_row(f"OLD{i}", 0.20, 0.55, days_ago=40) for i in range(12)]
    mock_load_summary.return_value = pd.DataFrame(current + stale)
    mock_load_symbols.return_value = [
        SymbolTask(market="jp", symbol=r["symbol"]) for r in current + stale
    ]
    mock_train.return_value = {"status": "success"}

    with patch("src.orchestration.jobs.drift_check._is_recently_retrained", return_value=False):
        run_daily_drift_check()

    # 割合の分母も古い銘柄を外した9銘柄で数える（#1369）
    mock_streak.assert_called_once_with(False, 1 / 9, 1, 9)
    assert [c.args[0].symbol for c in mock_train.call_args_list] == ["BAD"]
    mock_notify.assert_called_once()
    notified = [s["symbol"] for s in mock_notify.call_args.args[0]]
    assert notified == ["BAD"]
    mae_threshold = mock_notify.call_args.args[1]
    assert abs(mae_threshold - 0.045) < 1e-9
