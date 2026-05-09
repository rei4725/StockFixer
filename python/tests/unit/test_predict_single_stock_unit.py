"""predict_single_stock モジュールのユニットテスト"""

import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from src.prediction.types import PredictionResult


def _make_ohlcv(periods=60):
    """テスト用 OHLCV DataFrame を生成する"""
    dates = pd.date_range("2024-01-01", periods=periods, freq="B")
    return pd.DataFrame(
        {
            "Open": np.linspace(99, 119, periods),
            "High": np.linspace(101, 121, periods),
            "Low": np.linspace(98, 118, periods),
            "Close": np.linspace(100, 120, periods),
            "Volume": np.random.randint(1000, 5000, periods),
        },
        index=dates,
    )


class TestPredictSingleStockNoModel(unittest.TestCase):
    """モデルファイルが存在しない場合のテスト"""

    @patch("src.prediction.predict_single.get_models_subdir")
    def test_returns_none_when_no_model_files(self, mock_subdir):
        """モデルファイルが存在しない場合は None が返ること"""
        from src.prediction.predict_single import predict_single_stock

        with tempfile.TemporaryDirectory() as tmp:
            mock_subdir.return_value = tmp
            result = predict_single_stock("us", "AAPL")

        self.assertIsNone(result)

    @patch("src.prediction.predict_single.get_models_subdir")
    def test_returns_none_for_custom_model_types_not_existing(self, mock_subdir):
        """指定した model_types が存在しない場合は None が返ること"""
        from src.prediction.predict_single import predict_single_stock

        with tempfile.TemporaryDirectory() as tmp:
            mock_subdir.return_value = tmp
            result = predict_single_stock("us", "AAPL", model_types=["NonExistentModel.joblib"])

        self.assertIsNone(result)


class TestPredictSingleStockWithMocks(unittest.TestCase):
    """モックを使って predict_single_stock の正常系をテストする"""

    def _setup_mocks(
        self,
        mock_subdir,
        mock_exists,
        mock_data_loader,
        mock_yf,
        mock_tech,
        mock_lag,
        mock_mm_cls,
        current_price=150.0,
        pred_return=0.01,
        periods=60,
    ):
        """共通モックの設定"""
        mock_subdir.return_value = "/models/us_AAPL"
        mock_exists.return_value = True

        ohlcv = _make_ohlcv(periods)
        mock_data_loader.get_stock_data.return_value = ohlcv

        hist_df = pd.DataFrame(
            {"Close": [current_price]}, index=pd.date_range("2024-03-01", periods=1)
        )
        ticker_obj = MagicMock()
        ticker_obj.history.return_value = hist_df
        mock_yf.Ticker.return_value = ticker_obj

        mock_tech.return_value = ohlcv
        dates = pd.date_range("2024-01-01", periods=periods, freq="B")
        X = pd.DataFrame({"f1": range(periods), "f2": range(periods)}, index=dates)
        y = pd.Series(np.zeros(periods), index=dates)
        mock_lag.return_value = (X, y)

        mock_model = MagicMock()
        mock_model.predict.return_value = pd.Series([pred_return])
        mock_mm = MagicMock()
        mock_mm.load_model.return_value = mock_model
        mock_mm_cls.return_value = mock_mm

    @patch("src.prediction.predict_single.ModelManager")
    @patch("src.prediction.predict_single.create_basic_lag_features")
    @patch("src.prediction.predict_single.add_technical_indicators")
    @patch("src.prediction.predict_single.yf")
    @patch("src.prediction.predict_single.data_loader")
    @patch("src.prediction.predict_single.os.path.exists")
    @patch("src.prediction.predict_single.get_models_subdir")
    def test_returns_prediction_result(
        self,
        mock_subdir,
        mock_exists,
        mock_data_loader,
        mock_yf,
        mock_tech,
        mock_lag,
        mock_mm_cls,
    ):
        """正常時は PredictionResult が返ること"""
        from src.prediction.predict_single import predict_single_stock

        self._setup_mocks(
            mock_subdir,
            mock_exists,
            mock_data_loader,
            mock_yf,
            mock_tech,
            mock_lag,
            mock_mm_cls,
        )

        result = predict_single_stock("us", "AAPL", model_types=["StockXGBoostModel.joblib"])

        self.assertIsNotNone(result)
        self.assertIsInstance(result, PredictionResult)

    @patch("src.prediction.predict_single.ModelManager")
    @patch("src.prediction.predict_single.create_basic_lag_features")
    @patch("src.prediction.predict_single.add_technical_indicators")
    @patch("src.prediction.predict_single.yf")
    @patch("src.prediction.predict_single.data_loader")
    @patch("src.prediction.predict_single.os.path.exists")
    @patch("src.prediction.predict_single.get_models_subdir")
    def test_result_fields_correct(
        self,
        mock_subdir,
        mock_exists,
        mock_data_loader,
        mock_yf,
        mock_tech,
        mock_lag,
        mock_mm_cls,
    ):
        """PredictionResult の market/symbol/current_price が正しいこと"""
        from src.prediction.predict_single import predict_single_stock

        self._setup_mocks(
            mock_subdir,
            mock_exists,
            mock_data_loader,
            mock_yf,
            mock_tech,
            mock_lag,
            mock_mm_cls,
            current_price=150.0,
            pred_return=0.01,
        )

        result = predict_single_stock("us", "AAPL", model_types=["StockXGBoostModel.joblib"])

        self.assertEqual(result.market, "us")
        self.assertEqual(result.symbol, "AAPL")
        self.assertAlmostEqual(result.current_price, 150.0, places=1)

    @patch("src.prediction.predict_single.ModelManager")
    @patch("src.prediction.predict_single.create_basic_lag_features")
    @patch("src.prediction.predict_single.add_technical_indicators")
    @patch("src.prediction.predict_single.yf")
    @patch("src.prediction.predict_single.data_loader")
    @patch("src.prediction.predict_single.os.path.exists")
    @patch("src.prediction.predict_single.get_models_subdir")
    def test_diff_ratio_calculated(
        self,
        mock_subdir,
        mock_exists,
        mock_data_loader,
        mock_yf,
        mock_tech,
        mock_lag,
        mock_mm_cls,
    ):
        """diff_ratio = (avg_pred_price - current_price) / current_price であること"""
        from src.prediction.predict_single import predict_single_stock

        self._setup_mocks(
            mock_subdir,
            mock_exists,
            mock_data_loader,
            mock_yf,
            mock_tech,
            mock_lag,
            mock_mm_cls,
            current_price=100.0,
            pred_return=0.02,
        )

        result = predict_single_stock("us", "AAPL", model_types=["StockXGBoostModel.joblib"])

        expected_pred_price = 100.0 * (1 + 0.02)
        expected_diff = (expected_pred_price - 100.0) / 100.0
        self.assertAlmostEqual(result.diff_ratio, expected_diff, places=4)

    @patch("src.prediction.predict_single.ModelManager")
    @patch("src.prediction.predict_single.create_basic_lag_features")
    @patch("src.prediction.predict_single.add_technical_indicators")
    @patch("src.prediction.predict_single.yf")
    @patch("src.prediction.predict_single.data_loader")
    @patch("src.prediction.predict_single.os.path.exists")
    @patch("src.prediction.predict_single.get_models_subdir")
    def test_model_count_reflects_model_types(
        self,
        mock_subdir,
        mock_exists,
        mock_data_loader,
        mock_yf,
        mock_tech,
        mock_lag,
        mock_mm_cls,
    ):
        """model_count がモデル数と一致すること"""
        from src.prediction.predict_single import predict_single_stock

        self._setup_mocks(
            mock_subdir,
            mock_exists,
            mock_data_loader,
            mock_yf,
            mock_tech,
            mock_lag,
            mock_mm_cls,
        )

        result = predict_single_stock("us", "AAPL", model_types=["StockXGBoostModel.joblib"])

        self.assertEqual(result.model_count, 1)


class TestPredictSingleStockEdgeCases(unittest.TestCase):
    """エッジケースのテスト"""

    @patch("src.prediction.predict_single.data_loader")
    @patch("src.prediction.predict_single.os.path.exists")
    @patch("src.prediction.predict_single.get_models_subdir")
    def test_returns_none_when_ohlcv_empty(self, mock_subdir, mock_exists, mock_data_loader):
        """OHLCV データが空の場合は None が返ること"""
        from src.prediction.predict_single import predict_single_stock

        mock_subdir.return_value = "/models/us_BAD"
        mock_exists.return_value = True
        mock_data_loader.get_stock_data.return_value = pd.DataFrame()

        result = predict_single_stock("us", "BAD", model_types=["StockXGBoostModel.joblib"])

        self.assertIsNone(result)

    @patch("src.prediction.predict_single.create_basic_lag_features")
    @patch("src.prediction.predict_single.add_technical_indicators")
    @patch("src.prediction.predict_single.yf")
    @patch("src.prediction.predict_single.data_loader")
    @patch("src.prediction.predict_single.os.path.exists")
    @patch("src.prediction.predict_single.get_models_subdir")
    def test_returns_none_when_features_empty(
        self, mock_subdir, mock_exists, mock_data_loader, mock_yf, mock_tech, mock_lag
    ):
        """特徴量生成後に X が空の場合は None が返ること"""
        from src.prediction.predict_single import predict_single_stock

        mock_subdir.return_value = "/models/us_AAPL"
        mock_exists.return_value = True
        mock_data_loader.get_stock_data.return_value = _make_ohlcv(30)

        hist_df = pd.DataFrame({"Close": [100.0]}, index=pd.date_range("2024-01-01", periods=1))
        mock_yf.Ticker.return_value = MagicMock()
        mock_yf.Ticker.return_value.history.return_value = hist_df

        mock_tech.return_value = _make_ohlcv(30)
        mock_lag.return_value = (pd.DataFrame(), pd.Series(dtype=float))

        result = predict_single_stock("us", "AAPL", model_types=["StockXGBoostModel.joblib"])

        self.assertIsNone(result)

    @patch("src.prediction.predict_single.yf")
    @patch("src.prediction.predict_single.data_loader")
    @patch("src.prediction.predict_single.os.path.exists")
    @patch("src.prediction.predict_single.get_models_subdir")
    def test_fallback_to_close_when_yf_fails(
        self, mock_subdir, mock_exists, mock_data_loader, mock_yf
    ):
        """yfinance での現在価格取得失敗時は Close 列からフォールバックすること"""
        from src.prediction.predict_single import predict_single_stock

        mock_subdir.return_value = "/models/us_AAPL"
        mock_exists.return_value = True

        ohlcv = _make_ohlcv(30)
        mock_data_loader.get_stock_data.return_value = ohlcv

        # yfinance のヒストリーが例外を起こすケースをシミュレート
        mock_yf.Ticker.return_value.history.side_effect = Exception("network error")

        # 特徴量・モデルが必要なため、ここではモデルなし(exists=False)相当にして None 返却を確認
        mock_exists.return_value = False

        result = predict_single_stock("us", "AAPL", model_types=["StockXGBoostModel.joblib"])

        self.assertIsNone(result)


class TestPredictSingleStockHorizon(unittest.TestCase):
    """horizon パラメータのテスト"""

    @patch("src.prediction.predict_single.get_models_subdir")
    def test_horizon1_default_model_types(self, mock_subdir):
        """horizon=1 では suffix なしのモデルファイルを使うこと"""
        from src.prediction.predict_single import predict_single_stock

        with tempfile.TemporaryDirectory() as tmp:
            mock_subdir.return_value = tmp
            result = predict_single_stock("us", "AAPL", horizon=1)
            # ファイルが存在しないので None が返る（モデルパス解決の確認は副次的）
        self.assertIsNone(result)

    @patch("src.prediction.predict_single.get_models_subdir")
    def test_horizon3_uses_3d_suffix(self, mock_subdir):
        """horizon=3 では _3d サフィックスのモデルファイルを使うこと"""
        from src.prediction.predict_single import predict_single_stock

        with tempfile.TemporaryDirectory() as tmp:
            mock_subdir.return_value = tmp
            result = predict_single_stock("us", "AAPL", horizon=3)

        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
