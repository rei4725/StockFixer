"""LightGBMModel / XGBoostModel のユニットテスト"""

import sys
import unittest

import numpy as np
import pandas as pd


def _clear_mock_modules():
    """test_model_manager.py が sys.modules にセットしたモックを削除する"""
    from unittest.mock import MagicMock

    for key in ("src.models.lightgbm_model", "src.models.xgboost_model"):
        if key in sys.modules and isinstance(sys.modules[key], MagicMock):
            del sys.modules[key]


def _make_xy(periods=50):
    """テスト用の特徴量行列 X と目的変数 y を生成する"""
    dates = pd.date_range("2024-01-01", periods=periods, freq="D")
    X = pd.DataFrame(
        {
            "f1": np.random.randn(periods),
            "f2": np.linspace(0.0, 1.0, periods),
            "f3": np.random.randint(0, 10, periods).astype(float),
        },
        index=dates,
    )
    y = pd.Series(np.random.randn(periods) * 0.01, index=dates)
    return X, y


class TestLightGBMModel(unittest.TestCase):
    """LightGBMModel のユニットテスト"""

    @classmethod
    def setUpClass(cls):
        _clear_mock_modules()

    def setUp(self):
        self.X, self.y = _make_xy()

    def test_init_default_name(self):
        """デフォルトのモデル名が 'LightGBMModel' であること"""
        from src.models.lightgbm_model import LightGBMModel

        model = LightGBMModel()
        self.assertEqual(model.model_name, "LightGBMModel")

    def test_init_custom_name(self):
        """カスタムモデル名が正しく設定されること"""
        from src.models.lightgbm_model import LightGBMModel

        model = LightGBMModel("MyLGBM")
        self.assertEqual(model.model_name, "MyLGBM")

    def test_train_completes_without_error(self):
        """train が例外なく完了すること"""
        from src.models.lightgbm_model import LightGBMModel

        model = LightGBMModel()
        model.train(self.X, self.y)

    def test_predict_returns_series(self):
        """predict が pd.Series を返すこと"""
        from src.models.lightgbm_model import LightGBMModel

        model = LightGBMModel()
        model.train(self.X, self.y)
        result = model.predict(self.X)

        self.assertIsInstance(result, pd.Series)

    def test_predict_length_matches_input(self):
        """predict の結果長さが入力 X と一致すること"""
        from src.models.lightgbm_model import LightGBMModel

        model = LightGBMModel()
        model.train(self.X, self.y)
        result = model.predict(self.X)

        self.assertEqual(len(result), len(self.X))

    def test_predict_index_matches_input(self):
        """predict の結果インデックスが入力 X と一致すること"""
        from src.models.lightgbm_model import LightGBMModel

        model = LightGBMModel()
        model.train(self.X, self.y)
        result = model.predict(self.X)

        pd.testing.assert_index_equal(result.index, self.X.index)

    def test_predict_before_train_raises_value_error(self):
        """train なしで predict を呼ぶと ValueError が送出されること"""
        from src.models.lightgbm_model import LightGBMModel

        model = LightGBMModel()
        with self.assertRaises(ValueError):
            model.predict(self.X)

    def test_predict_single_row(self):
        """1 行の X に対して predict が機能すること"""
        from src.models.lightgbm_model import LightGBMModel

        model = LightGBMModel()
        model.train(self.X, self.y)
        result = model.predict(self.X.iloc[[-1]])

        self.assertEqual(len(result), 1)


class TestXGBoostModel(unittest.TestCase):
    """XGBoostModel のユニットテスト"""

    @classmethod
    def setUpClass(cls):
        _clear_mock_modules()

    def setUp(self):
        self.X, self.y = _make_xy()

    def test_init_default_name(self):
        """デフォルトのモデル名が 'XGBoostModel' であること"""
        from src.models.xgboost_model import XGBoostModel

        model = XGBoostModel()
        self.assertEqual(model.model_name, "XGBoostModel")

    def test_init_custom_name(self):
        """カスタムモデル名が正しく設定されること"""
        from src.models.xgboost_model import XGBoostModel

        model = XGBoostModel("MyXGB")
        self.assertEqual(model.model_name, "MyXGB")

    def test_train_completes_without_error(self):
        """train が例外なく完了すること"""
        from src.models.xgboost_model import XGBoostModel

        model = XGBoostModel()
        model.train(self.X, self.y)

    def test_predict_returns_series(self):
        """predict が pd.Series を返すこと"""
        from src.models.xgboost_model import XGBoostModel

        model = XGBoostModel()
        model.train(self.X, self.y)
        result = model.predict(self.X)

        self.assertIsInstance(result, pd.Series)

    def test_predict_length_matches_input(self):
        """predict の結果長さが入力 X と一致すること"""
        from src.models.xgboost_model import XGBoostModel

        model = XGBoostModel()
        model.train(self.X, self.y)
        result = model.predict(self.X)

        self.assertEqual(len(result), len(self.X))

    def test_predict_index_matches_input(self):
        """predict の結果インデックスが入力 X と一致すること"""
        from src.models.xgboost_model import XGBoostModel

        model = XGBoostModel()
        model.train(self.X, self.y)
        result = model.predict(self.X)

        pd.testing.assert_index_equal(result.index, self.X.index)

    def test_predict_before_train_raises_value_error(self):
        """train なしで predict を呼ぶと ValueError が送出されること"""
        from src.models.xgboost_model import XGBoostModel

        model = XGBoostModel()
        with self.assertRaises(ValueError):
            model.predict(self.X)

    def test_predict_single_row(self):
        """1 行の X に対して predict が機能すること"""
        from src.models.xgboost_model import XGBoostModel

        model = XGBoostModel()
        model.train(self.X, self.y)
        result = model.predict(self.X.iloc[[-1]])

        self.assertEqual(len(result), 1)


if __name__ == "__main__":
    unittest.main()
