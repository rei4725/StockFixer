"""銘柄マスタ（symbol_master）を週のジョブで埋める道のテスト（#1080）。取得と DB はモック。"""

import json
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

from src.market_data.symbol_master import SymbolMasterBuildResult
from src.market_data.types import SymbolName


@contextmanager
def _fake_conn():
    con = MagicMock()
    con.execute.return_value.fetchone.return_value = (10 * 1024 * 1024,)
    yield con


def _write_watchlist(tmp_path, jp):
    path = tmp_path / "watchlist.json"
    path.write_text(json.dumps({"jp": jp, "us": ["AAPL"]}), encoding="utf-8")
    return str(path)


@patch("src.utils.db.retention.purge_old_training_logs", return_value={})
@patch("src.reporting.discord.discord_utils.send_db_maintenance_completion")
@patch("src.utils.db._db_connection", side_effect=_fake_conn)
def test_weekly_db_maintenance_fills_symbol_master(_conn, mock_send, _purge, tmp_path):
    from src.orchestration.scheduler import run_weekly_db_maintenance

    record = SymbolName(market="jp", symbol="7203", name_ja="トヨタ自動車", source="jpx")
    built = SymbolMasterBuildResult(records=[record], missing=["1111"])
    path = _write_watchlist(tmp_path, ["7203", "1111"])
    with (
        patch("src.utils.data_path_utils.get_watchlist_path", return_value=path),
        patch(
            "src.market_data.symbol_master.build_jp_symbol_master", return_value=built
        ) as mock_build,
        patch("src.market_data.symbol_master.upsert_symbol_names", return_value=1) as mock_upsert,
    ):
        run_weekly_db_maintenance()

    mock_build.assert_called_once()
    assert mock_build.call_args.args[0] == ["7203", "1111"]
    mock_upsert.assert_called_once_with([record])
    mock_send.assert_called_once()


@patch("src.utils.db.retention.purge_old_training_logs", return_value={})
@patch("src.reporting.discord.discord_utils.send_db_maintenance_completion")
@patch("src.utils.db._db_connection", side_effect=_fake_conn)
def test_symbol_master_fetch_failure_keeps_job_running(_conn, mock_send, mock_purge, tmp_path):
    from src.orchestration.scheduler import run_weekly_db_maintenance

    path = _write_watchlist(tmp_path, ["7203"])
    with (
        patch("src.utils.data_path_utils.get_watchlist_path", return_value=path),
        patch(
            "src.market_data.symbol_master.build_jp_symbol_master",
            side_effect=RuntimeError("JPX 取得失敗"),
        ),
        patch("src.market_data.symbol_master.upsert_symbol_names") as mock_upsert,
        patch("src.orchestration.jobs.symbol_master_job.logger") as mock_logger,
    ):
        run_weekly_db_maintenance()  # 例外が外に出ないこと

    mock_upsert.assert_not_called()
    mock_purge.assert_called_once()
    mock_send.assert_called_once()
    assert mock_send.call_args.kwargs.get("error") is None
    mock_logger.warning.assert_called_once()


def test_symbol_master_is_not_on_daily_path():
    from run_scheduler import SCHEDULE_CONFIG

    config = SCHEDULE_CONFIG["weekly_db_maintenance"]
    assert config["period"] == "weekly"
    assert config["max_executions_per_period"] == 1
    assert not config.get("backfill")
