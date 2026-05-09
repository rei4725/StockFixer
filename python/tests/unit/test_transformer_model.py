"""TransformerModel のユニットテスト"""

import os
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd


def _make_xy(periods=80):
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


class TestTransformerModel(unittest.TestCase):
    """TransformerModel のユニットテスト"""

    def setUp(self):
        self.X, self.y = _make_xy()

    def test_init_default_name(self):
        from src.prediction.models.transformer_model import TransformerModel

        model = TransformerModel()
        self.assertEqual(model.model_name, "TransformerModel")

    def test_init_custom_name(self):
        from src.prediction.models.transformer_model import TransformerModel

        model = TransformerModel("MyTransformer")
        self.assertEqual(model.model_name, "MyTransformer")

    def test_train_completes_without_error(self):
        from src.prediction.models.transformer_model import TransformerModel

        model = TransformerModel()
        model.train(self.X, self.y)

    def test_predict_returns_series(self):
        from src.prediction.models.transformer_model import TransformerModel

        model = TransformerModel()
        model.train(self.X, self.y)
        result = model.predict(self.X)

        self.assertIsInstance(result, pd.Series)

    def test_predict_length_matches_input(self):
        from src.prediction.models.transformer_model import TransformerModel

        model = TransformerModel()
        model.train(self.X, self.y)
        result = model.predict(self.X)

        self.assertEqual(len(result), len(self.X))

    def test_predict_index_matches_input(self):
        from src.prediction.models.transformer_model import TransformerModel

        model = TransformerModel()
        model.train(self.X, self.y)
        result = model.predict(self.X)

        pd.testing.assert_index_equal(result.index, self.X.index)

    def test_predict_before_train_raises_value_error(self):
        from src.prediction.models.transformer_model import TransformerModel

        model = TransformerModel()
        with self.assertRaises(ValueError):
            model.predict(self.X)

    def test_predict_single_row(self):
        from src.prediction.models.transformer_model import TransformerModel

        model = TransformerModel()
        model.train(self.X, self.y)
        result = model.predict(self.X.iloc[[-1]])

        self.assertEqual(len(result), 1)

    def test_save_and_load_roundtrip(self):
        from src.prediction.models.transformer_model import TransformerModel

        model = TransformerModel()
        model.train(self.X, self.y)
        tmp_dir = tempfile.mkdtemp()
        try:
            path = os.path.join(tmp_dir, "transformer.joblib")
            model.save_model(path)
            self.assertTrue(os.path.exists(path))

            model2 = TransformerModel()
            model2.load_model(path)
            result = model2.predict(self.X)
            self.assertEqual(len(result), len(self.X))
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_saved_model_produces_same_predictions(self):
        from src.prediction.models.transformer_model import TransformerModel

        model = TransformerModel()
        model.train(self.X, self.y)
        tmp_dir = tempfile.mkdtemp()
        try:
            path = os.path.join(tmp_dir, "transformer.joblib")
            pred_before = model.predict(self.X)
            model.save_model(path)

            model2 = TransformerModel()
            model2.load_model(path)
            pred_after = model2.predict(self.X)

            np.testing.assert_array_almost_equal(pred_before.values, pred_after.values, decimal=10)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


class TestModelTypeEnum(unittest.TestCase):
    """ModelType 列挙型のテスト"""

    def test_model_type_values(self):
        from src.prediction.models import ModelType

        self.assertEqual(ModelType.XGBOOST, "XGBoostModel")
        self.assertEqual(ModelType.LIGHTGBM, "LightGBMModel")
        self.assertEqual(ModelType.TRANSFORMER, "TransformerModel")

    def test_model_type_is_str(self):
        from src.prediction.models import ModelType

        self.assertIsInstance(ModelType.TRANSFORMER, str)


class TestModelManagerTransformerRegistration(unittest.TestCase):
    """ModelManager に TransformerModel が登録されていることを確認するテスト"""

    def test_transformer_model_registered(self):
        import tempfile

        from src.prediction.manager import ModelManager

        with tempfile.TemporaryDirectory() as tmp:
            manager = ModelManager(model_dir=tmp)
            self.assertIn("TransformerModel", manager._registered_models)

    def test_create_transformer_model(self):
        import tempfile

        from src.prediction.manager import ModelManager

        with tempfile.TemporaryDirectory() as tmp:
            manager = ModelManager(model_dir=tmp)
            model = manager.create_model("TransformerModel", "test_transformer")
            self.assertEqual(model.get_model_name(), "test_transformer")


if __name__ == "__main__":
    unittest.main()
