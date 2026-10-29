"""日次ドリフト検査の MAE の線が、全銘柄の分布に比べて傷んだ銘柄だけを拾うことのテスト（#1020）。"""

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


@patch("src.orchestration.jobs.drift_check.record_drift_warn_streak")
@patch("src.prediction.training_pipeline.train_models_for_symbol_task")
@patch("src.watchlist.batch_runner.load_target_symbols")
@patch("src.reporting.discord.discord_utils.send_drift_retrain_notification")
@patch("src.prediction.db.load_drift_summary")
def test_uniformly_high_mae_triggers_nothing(
    mock_load_summary, mock_notify, mock_load_symbols, mock_train, mock_streak
):
    """全銘柄の MAE が同じ水準で高い（本番の 0.056〜0.060 相当）なら、超える銘柄は0件。"""
    mock_load_summary.return_value = pd.DataFrame(
        [_row(f"S{i}", 0.056 + 0.0005 * i) for i in range(9)]
    )

    run_daily_drift_check()

    mock_notify.assert_not_called()
    mock_train.assert_not_called()
    mock_streak.assert_called_once()
    assert mock_streak.call_args.args[0] is False


@patch("src.orchestration.jobs.drift_check.record_drift_warn_streak")
@patch("src.prediction.training_pipeline.train_models_for_symbol_task")
@patch("src.watchlist.batch_runner.load_target_symbols")
@patch("src.reporting.discord.discord_utils.send_drift_retrain_notification")
@patch("src.prediction.db.load_drift_summary")
def test_only_degraded_symbol_triggers(
    mock_load_summary, mock_notify, mock_load_symbols, mock_train, mock_streak
):
    """1銘柄だけ誤差が他より大きく悪化した表では、その銘柄だけが超える。"""
    rows = [_row(f"S{i}", 0.058) for i in range(8)] + [_row("BAD", 0.15)]
    mock_load_summary.return_value = pd.DataFrame(rows)
    mock_load_symbols.return_value = [SymbolTask(market="jp", symbol=r["symbol"]) for r in rows]
    mock_train.return_value = {"status": "success"}

    run_daily_drift_check()

    mock_train.assert_called_once()
    assert mock_train.call_args.args[0].symbol == "BAD"
    # 再学習は回すが、9銘柄中1銘柄（11%）は Warn の線（15%）に届かない（#1369）
    mock_streak.assert_called_once_with(False, 1 / 9, 1, 9)
