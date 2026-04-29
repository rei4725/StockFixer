"""scheduler_pipeline モジュールのユニットテスト"""

import os
import tempfile
import unittest
from unittest.mock import patch

import src.utils.data_path_utils as path_utils
import src.utils.db as db_module
from src.orchestration.scheduler import run_daily_pipeline, run_weekly_training


class TestRunDailyPipeline(unittest.TestCase):
    """run_daily_pipeline 関数のテスト"""

    def setUp(self):
        """テスト用一時DBを設定"""
        db_module.close_connection()
        self.tmp_dir = tempfile.mkdtemp()
        self.tmp_db = os.path.join(self.tmp_dir, "test_scheduler.duckdb")
        self._orig_get_db_path = path_utils.get_db_path
        path_utils.get_db_path = lambda: self.tmp_db
        db_module.get_db_path = lambda: self.tmp_db
        db_module._tables_initialized = False

    def tearDown(self):
        db_module.close_connection()
        path_utils.get_db_path = self._orig_get_db_path
        db_module.get_db_path = self._orig_get_db_path
        if os.path.exists(self.tmp_db):
            os.remove(self.tmp_db)
        wal_path = self.tmp_db + ".wal"
        if os.path.exists(wal_path):
            os.remove(wal_path)
        if os.path.exists(self.tmp_dir):
            os.rmdir(self.tmp_dir)

    @patch("src.reporting.discord.discord_utils.send_daily_pipeline_completion")
    @patch("src.prediction.prediction_pipeline.output_top_worst_results")
    @patch("src.prediction.prediction_pipeline.predict_all_unified")
    @patch("src.market_data.pipeline.run_data_batch")
    def test_daily_pipeline_success(
        self,
        mock_run_batch,
        mock_predict,
        mock_output,
        mock_send_completion,
    ):
        """日次パイプラインが正常に完了することを確認"""
        # 実行
        run_daily_pipeline()

        # 各処理が呼び出されたことを確認
        mock_run_batch.assert_called_once()
        mock_predict.assert_called_once()
        mock_output.assert_called_once()
        # 完了通知が送られたことを確認
        mock_send_completion.assert_called_once()

    @patch("src.reporting.discord.discord_utils.send_daily_pipeline_error")
    @patch("src.prediction.prediction_pipeline.output_top_worst_results")
    @patch("src.prediction.prediction_pipeline.predict_all_unified")
    @patch("src.market_data.pipeline.run_data_batch")
    def test_daily_pipeline_data_fetch_error(
        self,
        mock_run_batch,
        mock_predict,
        mock_output,
        mock_send_error,
    ):
        """データ取得エラーが発生した場合を確認"""
        mock_run_batch.side_effect = Exception("データ取得失敗")

        # 例外が発生することを確認
        with self.assertRaises(Exception):  # noqa: B017
            run_daily_pipeline()

        # エラー通知が送られたことを確認
        mock_send_error.assert_called_once()
        # 予測は実行されないはず
        mock_predict.assert_not_called()

    @patch("src.reporting.discord.discord_utils.send_daily_pipeline_error")
    @patch("src.prediction.prediction_pipeline.output_top_worst_results")
    @patch("src.prediction.prediction_pipeline.predict_all_unified")
    @patch("src.market_data.pipeline.run_data_batch")
    def test_daily_pipeline_prediction_error(
        self,
        mock_run_batch,
        mock_predict,
        mock_output,
        mock_send_error,
    ):
        """予測エラーが発生した場合を確認"""
        mock_predict.side_effect = Exception("予測失敗")

        # 例外が発生することを確認
        with self.assertRaises(Exception):  # noqa: B017
            run_daily_pipeline()

        # エラー通知が送られたことを確認
        mock_send_error.assert_called_once()

    @patch("src.reporting.discord.discord_utils.send_daily_pipeline_completion")
    @patch("src.prediction.prediction_pipeline.output_top_worst_results")
    @patch("src.prediction.prediction_pipeline.predict_all_unified")
    @patch("src.market_data.pipeline.run_data_batch")
    def test_daily_pipeline_notification_error(
        self,
        mock_run_batch,
        mock_predict,
        mock_output,
        mock_send_completion,
    ):
        """Discord通知エラーが発生した場合を確認"""
        mock_send_completion.side_effect = Exception("Discord通知失敗")

        # 例外が発生することを確認
        with self.assertRaises(Exception):  # noqa: B017
            run_daily_pipeline()

        # データ取得・予測は正常に完了後、通知で失敗
        mock_run_batch.assert_called_once()
        mock_predict.assert_called_once()


class TestRunWeeklyTraining(unittest.TestCase):
    """run_weekly_training 関数のテスト"""

    @patch("src.prediction.unified_model_pipeline.train_unified_model")
    def test_weekly_training_success(self, mock_train):
        """週次学習が正常に完了することを確認"""
        # 実行
        run_weekly_training()

        # XGBoostとLightGBMの両方でtrain_unified_modelが呼ばれたことを確認
        self.assertEqual(mock_train.call_count, 2)
        calls = mock_train.call_args_list
        # 最初の呼び出しはXGBoost
        self.assertEqual(calls[0][1]["model_type"], "XGBoostModel")
        self.assertEqual(calls[0][1]["model_name"], "UnifiedStockXGBoost")
        # 2番目の呼び出しはLightGBM
        self.assertEqual(calls[1][1]["model_type"], "LightGBMModel")
        self.assertEqual(calls[1][1]["model_name"], "UnifiedStockLightGBM")

    @patch("src.prediction.unified_model_pipeline.train_unified_model")
    def test_weekly_training_xgboost_error(self, mock_train):
        """XGBoost学習エラーが発生した場合を確認"""
        mock_train.side_effect = Exception("学習失敗")

        # 例外が発生することを確認
        with self.assertRaises(Exception):  # noqa: B017
            run_weekly_training()

        # train_unified_modelがエラーで呼ばれたことを確認
        mock_train.assert_called_once()

    @patch("src.prediction.unified_model_pipeline.train_unified_model")
    def test_weekly_training_lightgbm_error(self, mock_train):
        """LightGBM学習エラーが発生した場合を確認"""
        # XGBoostは成功、LightGBMで失敗
        mock_train.side_effect = [None, Exception("学習失敗")]

        # 例外が発生することを確認
        with self.assertRaises(Exception):  # noqa: B017
            run_weekly_training()

        # train_unified_modelが2回呼ばれたことを確認
        self.assertEqual(mock_train.call_count, 2)


if __name__ == "__main__":
    unittest.main()
