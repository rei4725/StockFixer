"""ユニットテスト: src.utils.db.system_config"""

import unittest
from unittest.mock import MagicMock, patch


class TestGetConfigValue(unittest.TestCase):
    def _mock_db(self, row):
        mock_con = MagicMock()
        mock_con.execute.return_value.fetchone.return_value = row
        return mock_con

    def test_returns_stored_value(self):
        from src.utils.db.system_config import get_config_value

        mock_con = self._mock_db(("0.05",))
        with patch("src.utils.db.system_config._db_connection") as mock_ctx:
            mock_ctx.return_value.__enter__ = MagicMock(return_value=mock_con)
            mock_ctx.return_value.__exit__ = MagicMock(return_value=False)
            result = get_config_value("drift.mae_threshold", "0.02")
        self.assertEqual(result, "0.05")

    def test_returns_default_when_not_found(self):
        from src.utils.db.system_config import get_config_value

        mock_con = self._mock_db(None)
        with patch("src.utils.db.system_config._db_connection") as mock_ctx:
            mock_ctx.return_value.__enter__ = MagicMock(return_value=mock_con)
            mock_ctx.return_value.__exit__ = MagicMock(return_value=False)
            result = get_config_value("drift.mae_threshold", "0.02")
        self.assertEqual(result, "0.02")

    def test_returns_none_when_no_default(self):
        from src.utils.db.system_config import get_config_value

        mock_con = self._mock_db(None)
        with patch("src.utils.db.system_config._db_connection") as mock_ctx:
            mock_ctx.return_value.__enter__ = MagicMock(return_value=mock_con)
            mock_ctx.return_value.__exit__ = MagicMock(return_value=False)
            result = get_config_value("nonexistent_key")
        self.assertIsNone(result)


class TestSetConfigValue(unittest.TestCase):
    def test_executes_upsert(self):
        from src.utils.db.system_config import set_config_value

        mock_con = MagicMock()
        with patch("src.utils.db.system_config._db_connection") as mock_ctx:
            mock_ctx.return_value.__enter__ = MagicMock(return_value=mock_con)
            mock_ctx.return_value.__exit__ = MagicMock(return_value=False)
            set_config_value("drift.mae_threshold", "0.03")

        mock_con.execute.assert_called_once()
        sql_call = mock_con.execute.call_args
        self.assertIn("INSERT INTO system_config", sql_call[0][0])
        self.assertEqual(sql_call[0][1], ["drift.mae_threshold", "0.03"])


class TestDriftThresholdDriven(unittest.TestCase):
    """scheduler.run_daily_drift_check が DuckDB 設定値を使用することを確認"""

    def test_uses_db_config_value(self):
        get_calls = []

        def mock_get(key, default=None):
            get_calls.append(key)
            if key == "drift.mae_threshold":
                return "0.03"
            if key == "drift.hit_rate_threshold":
                return "0.40"
            return default

        with (
            patch("src.prediction.db.load_drift_summary", return_value=None),
            patch("src.utils.db.system_config.get_config_value", side_effect=mock_get),
        ):
            from src.orchestration import scheduler

            scheduler.run_daily_drift_check()

        self.assertIn("drift.mae_threshold", get_calls)
        self.assertIn("drift.hit_rate_threshold", get_calls)

    def test_uses_db_config_value_for_min_samples_and_max_retrain(self):
        get_calls = []

        def mock_get(key, default=None):
            get_calls.append(key)
            return default

        with (
            patch("src.prediction.db.load_drift_summary", return_value=None),
            patch("src.utils.db.system_config.get_config_value", side_effect=mock_get),
        ):
            from src.orchestration import scheduler

            scheduler.run_daily_drift_check()

        self.assertIn("drift.min_samples", get_calls)
        self.assertIn("drift.max_retrain_per_run", get_calls)
