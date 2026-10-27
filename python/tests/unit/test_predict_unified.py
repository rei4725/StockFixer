"""predict_unified モジュールのユニットテスト"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

# shap が未インストールの場合はモック
if "shap" not in sys.modules:
    sys.modules["shap"] = MagicMock()


class TestLoadFeatureData(unittest.TestCase):
    """load_feature_data のテスト"""

    @patch("src.prediction.predict_unified.load_stock_features")
    def test_returns_dataframe_on_success(self, mock_load):
        """正常時に DataFrame が返ること"""
        from src.prediction.predict_unified import load_feature_data

        df = pd.DataFrame({"close_lag1": [1.0, 2.0, 3.0]})
        mock_load.return_value = df

        result = load_feature_data("jp", "7203")
        self.assertIsNotNone(result)
        self.assertIsInstance(result, pd.DataFrame)

    @patch("src.prediction.predict_unified.load_stock_features")
    def test_raises_on_exception(self, mock_load):
        """読み込みの例外は None にせず投げること（#1272）"""
        from src.prediction.predict_unified import load_feature_data

        mock_load.side_effect = Exception("特徴量取得エラー")
        with self.assertRaises(Exception):
            load_feature_data("jp", "7203")

    @patch("src.prediction.predict_unified.load_stock_features")
    def test_returns_dataframe_or_none_on_empty(self, mock_load):
        """空 DataFrame 時は DataFrame または None が返ること（実装依存）"""
        from src.prediction.predict_unified import load_feature_data

        mock_load.return_value = pd.DataFrame()
        result = load_feature_data("jp", "7203")
        # 空 DataFrame はそのまま返される（例外はなし）
        self.assertTrue(result is None or isinstance(result, pd.DataFrame))

    @patch("src.prediction.predict_unified.load_stock_features")
    def test_passes_market_and_symbol(self, mock_load):
        """market と symbol が load_stock_features に渡されること"""
        from src.prediction.predict_unified import load_feature_data

        mock_load.return_value = pd.DataFrame({"x": [1]})
        load_feature_data("us", "AAPL")
        mock_load.assert_called_once_with("us", "AAPL", raise_on_error=True)


class TestGetCachedModel(unittest.TestCase):
    """get_cached_model のテスト"""

    def setUp(self):
        """テスト前にキャッシュをクリア"""
        import src.prediction.predict_unified as mod

        mod._model_cache.clear()

    def tearDown(self):
        """テスト後もキャッシュをクリア"""
        import src.prediction.predict_unified as mod

        mod._model_cache.clear()

    @patch("src.prediction.unified_model_pipeline.load_unified_model")
    def test_loads_model_on_first_call(self, mock_load):
        """初回呼び出しでモデルがロードされること"""
        from src.prediction.predict_unified import get_cached_model

        mock_model = MagicMock()
        mock_load.return_value = mock_model

        result = get_cached_model("XGBoostModel")
        self.assertIsNotNone(result)
        mock_load.assert_called_once_with("XGBoostModel")

    @patch("src.prediction.unified_model_pipeline.load_unified_model")
    def test_returns_cached_model_on_second_call(self, mock_load):
        """2回目以降の呼び出しでキャッシュが使われること（load は1回のみ）"""
        from src.prediction.predict_unified import get_cached_model

        mock_model = MagicMock()
        mock_load.return_value = mock_model

        get_cached_model("XGBoostModel")
        get_cached_model("XGBoostModel")
        # モデルロードは1回のみ（2回目はキャッシュから返す）
        mock_load.assert_called_once()

    @patch("src.prediction.unified_model_pipeline.load_unified_model")
    def test_returns_none_when_load_fails(self, mock_load):
        """モデルロード失敗時（FileNotFoundError）は None が返ること"""
        from src.prediction.predict_unified import get_cached_model

        mock_load.side_effect = FileNotFoundError("モデルファイルが見つかりません")
        result = get_cached_model("NonExistentModel")
        self.assertIsNone(result)

    @patch("src.prediction.unified_model_pipeline.load_unified_model")
    def test_does_not_cache_none_on_load_failure(self, mock_load):
        """ロード失敗時は None をキャッシュせず、次の呼び出しで再びロードを試みること"""
        from src.prediction.predict_unified import get_cached_model

        mock_load.side_effect = FileNotFoundError("見つかりません")
        get_cached_model("MissingModel")
        get_cached_model("MissingModel")
        self.assertEqual(mock_load.call_count, 2)


class TestGetCachedModelReload(unittest.TestCase):
    """get_cached_model がディスク上のファイルの変化に追従すること（#887）"""

    def setUp(self):
        import tempfile

        import src.prediction.predict_unified as mod

        mod._model_cache.clear()
        self._tmp = tempfile.TemporaryDirectory()
        self.unified_dir = os.path.join(self._tmp.name, "unified")
        os.makedirs(self.unified_dir)
        dir_patcher = patch(
            "src.prediction.predict_unified.get_models_dir", return_value=self._tmp.name
        )
        dir_patcher.start()
        self.addCleanup(dir_patcher.stop)

    def tearDown(self):
        import src.prediction.predict_unified as mod

        mod._model_cache.clear()
        self._tmp.cleanup()

    def _write_model(self, name: str, mtime: float) -> None:
        path = os.path.join(self.unified_dir, f"{name}.joblib")
        with open(path, "wb") as f:
            f.write(b"model")
        os.utime(path, (mtime, mtime))

    @patch("src.prediction.unified_model_pipeline.load_unified_model")
    def test_loads_after_file_appears(self, mock_load):
        """ファイルが無くて None を返した後にファイルができたら、次の呼び出しで読み込むこと"""
        from src.prediction.predict_unified import get_cached_model

        new_model = MagicMock(name="new")
        mock_load.side_effect = [FileNotFoundError("見つかりません"), new_model]

        self.assertIsNone(get_cached_model("LateModel"))
        self._write_model("LateModel", 1_700_000_000)
        self.assertIs(get_cached_model("LateModel"), new_model)

    @patch("src.prediction.unified_model_pipeline.load_unified_model")
    def test_reloads_when_file_replaced(self, mock_load):
        """キャッシュ後にファイルを置き替えたら（昇格・学習し直し）、新しいモデルを返すこと"""
        from src.prediction.predict_unified import get_cached_model

        old_model = MagicMock(name="old")
        new_model = MagicMock(name="new")
        mock_load.side_effect = [old_model, new_model]

        self._write_model("SwapModel", 1_700_000_000)
        self.assertIs(get_cached_model("SwapModel"), old_model)
        self._write_model("SwapModel", 1_700_000_100)
        self.assertIs(get_cached_model("SwapModel"), new_model)
        self.assertEqual(mock_load.call_count, 2)

    @patch("src.prediction.unified_model_pipeline.load_unified_model")
    def test_does_not_reload_while_file_unchanged(self, mock_load):
        """ファイルが変わらない間は読み直さないこと（銘柄ごとに読み込まない）"""
        from src.prediction.predict_unified import get_cached_model

        model = MagicMock(name="model")
        mock_load.return_value = model

        self._write_model("StableModel", 1_700_000_000)
        for _ in range(5):
            self.assertIs(get_cached_model("StableModel"), model)
        mock_load.assert_called_once_with("StableModel")


class TestPreloadModels(unittest.TestCase):
    """preload_models のテスト"""

    def setUp(self):
        import src.prediction.predict_unified as mod

        mod._model_cache.clear()

    def tearDown(self):
        import src.prediction.predict_unified as mod

        mod._model_cache.clear()

    @patch("src.prediction.predict_unified.get_cached_model")
    def test_preloads_default_models(self, mock_get):
        """デフォルトモデルが事前ロードされること"""
        from src.prediction.predict_unified import preload_models

        mock_get.return_value = MagicMock()
        preload_models()
        # UnifiedStockXGBoost と UnifiedStockLightGBM が呼ばれること
        called_names = [call.args[0] for call in mock_get.call_args_list]
        self.assertIn("UnifiedStockXGBoost", called_names)
        self.assertIn("UnifiedStockLightGBM", called_names)

    @patch("src.prediction.predict_unified.get_cached_model")
    def test_preloads_specified_models(self, mock_get):
        """指定したモデル名のみがロードされること"""
        from src.prediction.predict_unified import preload_models

        mock_get.return_value = MagicMock()
        preload_models(model_types=["ModelA", "ModelB"])
        called_names = [call.args[0] for call in mock_get.call_args_list]
        self.assertEqual(called_names, ["ModelA", "ModelB"])

    @patch("src.prediction.predict_unified.get_cached_model")
    def test_calls_get_cached_model_per_model_type(self, mock_get):
        """各モデルタイプごとに get_cached_model が呼ばれること"""
        from src.prediction.predict_unified import preload_models

        mock_get.return_value = None
        preload_models(model_types=["ModelA"])
        mock_get.assert_called_once_with("ModelA")

    @patch("src.prediction.predict_unified.get_cached_model")
    def test_preloads_all_returns_none_model(self, mock_get):
        """get_cached_model が None を返す場合もクラッシュしないこと"""
        from src.prediction.predict_unified import preload_models

        mock_get.return_value = None
        preload_models()  # 例外なし
        self.assertEqual(mock_get.call_count, 2)  # デフォルト2モデル


if __name__ == "__main__":
    unittest.main()
