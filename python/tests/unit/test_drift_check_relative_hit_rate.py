"""日次ドリフト検査の Hit Rate の線が、全銘柄の分布に比べて傷んだ銘柄だけを拾うことのテスト（#1045）。"""

from unittest.mock import patch

import pandas as pd

from src.domain.types import SymbolTask
from src.orchestration.jobs.drift_check import run_daily_drift_check


def _row(symbol: str, hit_rate: float) -> dict:
    return {
        "market": "jp",
        "symbol": symbol,
        "mean_abs_error": 0.01,
        "direction_accuracy": hit_rate,
        "n_samples": 20,
    }


@patch("src.orchestration.jobs.drift_check.record_drift_warn_streak")
@patch("src.prediction.training_pipeline.train_models_for_symbol_task")
@patch("src.watchlist.batch_runner.load_target_symbols")
@patch("src.reporting.discord.discord_utils.send_drift_retrain_notification")
@patch("src.prediction.db.load_drift_summary")
def test_uniformly_low_hit_rate_triggers_nothing(
    mock_load_summary, mock_notify, mock_load_symbols, mock_train, mock_streak
):
    """全銘柄の Hit Rate が同じ水準で低い（本番の中央値 0.45 前後）なら、超える銘柄は0件。"""
    mock_load_summary.return_value = pd.DataFrame(
        [_row(f"S{i}", 0.40 + 0.01 * i) for i in range(9)]
    )

    run_daily_drift_check()

    mock_notify.assert_not_called()
    mock_train.assert_not_called()
    mock_streak.assert_called_once_with(False)


@patch("src.orchestration.jobs.drift_check.record_drift_warn_streak")
@patch("src.prediction.training_pipeline.train_models_for_symbol_task")
@patch("src.watchlist.batch_runner.load_target_symbols")
@patch("src.reporting.discord.discord_utils.send_drift_retrain_notification")
@patch("src.prediction.db.load_drift_summary")
def test_only_degraded_hit_rate_symbol_triggers(
    mock_load_summary, mock_notify, mock_load_symbols, mock_train, mock_streak
):
    """1銘柄だけ Hit Rate が他より大きく下がった表では、その銘柄だけが超える。"""
    rows = [_row(f"S{i}", 0.45) for i in range(8)] + [_row("BAD", 0.10)]
    mock_load_summary.return_value = pd.DataFrame(rows)
    mock_load_symbols.return_value = [SymbolTask(market="jp", symbol=r["symbol"]) for r in rows]
    mock_train.return_value = {"status": "success"}

    run_daily_drift_check()

    mock_train.assert_called_once()
    assert mock_train.call_args.args[0].symbol == "BAD"
    mock_streak.assert_called_once_with(True)
