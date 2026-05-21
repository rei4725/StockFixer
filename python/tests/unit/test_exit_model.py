"""ExitModel のユニットテスト (R-402)"""
from __future__ import annotations

import os
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd


def _make_feature_df(n: int = 60) -> pd.DataFrame:
    """ExitModel 用の特徴量 DataFrame を生成する"""
    from src.trading.models.exit_model import FEATURE_COLS

    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    data = {col: rng.uniform(-0.05, 0.05, n) for col in FEATURE_COLS}
    return pd.DataFrame(data, index=dates)


def _make_labels(n: int = 60) -> pd.Series:
    """バイナリラベル Series を生成する"""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.Series(rng.integers(0, 2, n).astype(float), index=dates, name="exit_signal")


class TestExitModelInit(unittest.TestCase):
    def test_default_name(self):
        from src.trading.models.exit_model import ExitModel

        m = ExitModel()
        self.assertEqual(m.model_name, "ExitModel")

    def test_custom_name(self):
        from src.trading.models.exit_model import ExitModel

        m = ExitModel("MyExitModel")
        self.assertEqual(m.model_name, "MyExitModel")

    def test_model_is_xgb_classifier(self):
        import xgboost as xgb

        from src.trading.models.exit_model import ExitModel

        m = ExitModel()
        self.assertIsInstance(m.model, xgb.XGBClassifier)


class TestExitModelTrainPredict(unittest.TestCase):
    def setUp(self):
        self.X = _make_feature_df()
        self.y = _make_labels()

    def test_train_completes(self):
        from src.trading.models.exit_model import ExitModel

        m = ExitModel()
        m.train(self.X, self.y)

    def test_predict_returns_series(self):
        from src.trading.models.exit_model import ExitModel

        m = ExitModel()
        m.train(self.X, self.y)
        result = m.predict(self.X)
        self.assertIsInstance(result, pd.Series)

    def test_predict_length_matches_input(self):
        from src.trading.models.exit_model import ExitModel

        m = ExitModel()
        m.train(self.X, self.y)
        result = m.predict(self.X)
        self.assertEqual(len(result), len(self.X))

    def test_predict_probability_range(self):
        from src.trading.models.exit_model import ExitModel

        m = ExitModel()
        m.train(self.X, self.y)
        proba = m.predict(self.X)
        self.assertTrue((proba >= 0.0).all(), "確率は 0 以上であること")
        self.assertTrue((proba <= 1.0).all(), "確率は 1 以下であること")

    def test_predict_before_train_raises(self):
        from src.trading.models.exit_model import ExitModel

        m = ExitModel()
        with self.assertRaises(ValueError):
            m.predict(self.X)

    def test_predict_index_matches(self):
        from src.trading.models.exit_model import ExitModel

        m = ExitModel()
        m.train(self.X, self.y)
        result = m.predict(self.X)
        pd.testing.assert_index_equal(result.index, self.X.index)

    def test_predict_single_row(self):
        from src.trading.models.exit_model import ExitModel

        m = ExitModel()
        m.train(self.X, self.y)
        result = m.predict(self.X.iloc[[-1]])
        self.assertEqual(len(result), 1)

    def test_train_with_eval_set(self):
        from src.trading.models.exit_model import ExitModel

        X, y = _make_feature_df(120), _make_labels(120)
        X_train, y_train = X.iloc[:90], y.iloc[:90]
        X_val, y_val = X.iloc[90:], y.iloc[90:]
        m = ExitModel()
        m.train(X_train, y_train, eval_set=[(X_val, y_val)])
        result = m.predict(X_val)
        self.assertEqual(len(result), len(X_val))


class TestExitModelMakeLabels(unittest.TestCase):
    def _make_close(self, n: int = 30) -> pd.Series:
        dates = pd.date_range("2024-01-01", periods=n, freq="D")
        prices = 1000.0 + np.arange(n, dtype=float)  # 単調増加
        return pd.Series(prices, index=dates)

    def test_returns_series(self):
        from src.trading.models.exit_model import ExitModel

        close = self._make_close()
        label = ExitModel.make_labels(close)
        self.assertIsInstance(label, pd.Series)

    def test_length_matches_input(self):
        from src.trading.models.exit_model import ExitModel

        close = self._make_close(30)
        label = ExitModel.make_labels(close, lookahead=5)
        self.assertEqual(len(label), len(close))

    def test_tail_is_na(self):
        from src.trading.models.exit_model import ExitModel

        lookahead = 5
        close = self._make_close(30)
        label = ExitModel.make_labels(close, lookahead=lookahead)
        self.assertTrue(
            label.iloc[-lookahead:].isna().all(),
            "末尾 lookahead 行は NaN であること",
        )

    def test_peak_is_labeled_exit(self):
        from src.trading.models.exit_model import ExitModel

        # 価格が途中でピークを迎えるシリーズ
        n = 20
        prices = list(range(1, n + 1)) + list(range(n - 1, 0, -1))
        close = pd.Series(
            prices[:30], index=pd.date_range("2024-01-01", periods=30, freq="D"), dtype=float
        )
        label = ExitModel.make_labels(close, lookahead=5, peak_ratio=0.98)
        # ピーク付近（インデックス 19 前後）は exit=1 になるはず
        non_na = label.dropna()
        self.assertIn(1, non_na.values, "ピーク近傍で exit=1 が発生すること")

    def test_name_is_exit_signal(self):
        from src.trading.models.exit_model import ExitModel

        close = self._make_close()
        label = ExitModel.make_labels(close)
        self.assertEqual(label.name, "exit_signal")


class TestExitModelPrepareFeatures(unittest.TestCase):
    def test_all_feature_cols_present(self):
        from src.trading.models.exit_model import FEATURE_COLS, ExitModel

        df = _make_feature_df()
        result = ExitModel.prepare_features(df)
        for col in FEATURE_COLS:
            self.assertIn(col, result.columns)

    def test_missing_cols_filled_with_zero(self):
        from src.trading.models.exit_model import ExitModel

        empty_df = pd.DataFrame({"other_col": [1, 2, 3]})
        result = ExitModel.prepare_features(empty_df)
        self.assertTrue((result == 0.0).all().all(), "存在しない列は 0.0 で補完されること")

    def test_output_is_float(self):
        from src.trading.models.exit_model import ExitModel

        df = _make_feature_df()
        result = ExitModel.prepare_features(df)
        self.assertTrue(
            all(result[col].dtype in (np.float64, np.float32) for col in result.columns)
        )


class TestExitModelSaveLoad(unittest.TestCase):
    def setUp(self):
        self.X = _make_feature_df()
        self.y = _make_labels()
        from src.trading.models.exit_model import ExitModel

        self.model = ExitModel()
        self.model.train(self.X, self.y)
        self.tmp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_save_and_load_roundtrip(self):
        from src.trading.models.exit_model import ExitModel

        path = os.path.join(self.tmp_dir, "exit_model.joblib")
        self.model.save_model(path)
        self.assertTrue(os.path.exists(path))

        m2 = ExitModel()
        m2.load_model(path)
        pred = m2.predict(self.X)
        self.assertEqual(len(pred), len(self.X))

    def test_loaded_model_same_predictions(self):
        from src.trading.models.exit_model import ExitModel

        path = os.path.join(self.tmp_dir, "exit_model.joblib")
        pred_before = self.model.predict(self.X)
        self.model.save_model(path)

        m2 = ExitModel()
        m2.load_model(path)
        pred_after = m2.predict(self.X)
        np.testing.assert_array_almost_equal(pred_before.values, pred_after.values, decimal=10)


class TestExitModelInInit(unittest.TestCase):
    """__init__.py 経由で ExitModel をインポートできること"""

    def test_importable_from_package(self):
        from src.trading.models import ExitModel

        self.assertTrue(issubclass(ExitModel, object))

    def test_model_type_exit(self):
        from src.prediction.models import ModelType

        self.assertEqual(ModelType.EXIT.value, "ExitModel")


if __name__ == "__main__":
    unittest.main()
