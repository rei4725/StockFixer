"""LightGBMModel / XGBoostModel のユニットテスト"""

import os
import shutil
import sys
import tempfile
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

    def test_train_with_eval_set_completes_without_error(self):
        """eval_set を指定した early stopping 付き学習が例外なく完了すること"""
        from src.models.lightgbm_model import LightGBMModel

        X, y = _make_xy(periods=200)
        X_train, y_train = X.iloc[:160], y.iloc[:160]
        X_val, y_val = X.iloc[160:], y.iloc[160:]

        model = LightGBMModel()
        model.train(X_train, y_train, eval_set=[(X_val, y_val)])
        result = model.predict(X_val)
        self.assertEqual(len(result), len(X_val))


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

    def test_train_with_eval_set_completes_without_error(self):
        """eval_set を指定した early stopping 付き学習が例外なく完了すること"""
        from src.models.xgboost_model import XGBoostModel

        X, y = _make_xy(periods=200)
        X_train, y_train = X.iloc[:160], y.iloc[:160]
        X_val, y_val = X.iloc[160:], y.iloc[160:]

        model = XGBoostModel()
        model.train(X_train, y_train, eval_set=[(X_val, y_val)])
        result = model.predict(X_val)
        self.assertEqual(len(result), len(X_val))


class TestBaseModelSaveLoad(unittest.TestCase):
    """BaseModel.save_model / load_model / get_model_name のテスト（LightGBMModel 経由）"""

    @classmethod
    def setUpClass(cls):
        _clear_mock_modules()

    def setUp(self):
        self.X, self.y = _make_xy()
        from src.models.lightgbm_model import LightGBMModel

        self.model = LightGBMModel("TestLGBM")
        self.model.train(self.X, self.y)
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_get_model_name(self):
        """get_model_name がコンストラクタで設定した名前を返すこと"""
        self.assertEqual(self.model.get_model_name(), "TestLGBM")

    def test_save_and_load_roundtrip(self):
        """save_model で保存し load_model で読み込むことができること"""
        path = os.path.join(self.tmp_dir, "model.joblib")
        self.model.save_model(path)
        self.assertTrue(os.path.exists(path))

        from src.models.lightgbm_model import LightGBMModel

        model2 = LightGBMModel("TestLGBM2")
        model2.load_model(path)
        self.assertIsNotNone(model2.model)

    def test_saved_model_produces_same_predictions(self):
        """保存→読み込み後の予測が元モデルと一致すること"""
        import numpy as np

        path = os.path.join(self.tmp_dir, "model.joblib")
        pred_before = self.model.predict(self.X)
        self.model.save_model(path)

        from src.models.lightgbm_model import LightGBMModel

        model2 = LightGBMModel()
        model2.load_model(path)
        pred_after = model2.predict(self.X)

        np.testing.assert_array_almost_equal(pred_before.values, pred_after.values, decimal=10)

    def test_save_model_raises_on_invalid_path(self):
        """存在しないディレクトリへの保存は例外が伝播すること"""
        with self.assertRaises(OSError):
            self.model.save_model("/nonexistent_dir/model.joblib")

    def test_load_model_raises_on_missing_file(self):
        """存在しないファイルのロードは例外が伝播すること"""
        with self.assertRaises((FileNotFoundError, OSError)):
            self.model.load_model("/nonexistent_dir/model.joblib")


if __name__ == "__main__":
    unittest.main()
