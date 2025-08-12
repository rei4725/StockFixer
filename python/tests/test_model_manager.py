import unittest
import pandas as pd
import os
import shutil
from unittest.mock import MagicMock, patch
import sys

# ModelManagerが依存するモジュールをモック
# これにより、xgboostやlightgbmがインストールされていなくてもModelManagerをインポートできるようになる
sys.modules['python.src.models.xgboost_model'] = MagicMock()
sys.modules['python.src.models.lightgbm_model'] = MagicMock()

# モック後にModelManagerをインポート
from python.src.models.model_manager import ModelManager
from python.src.models.base_model import BaseModel

# BaseModelのモッククラス
class MockBaseModel(BaseModel):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.trained = False
        self.predicted_data = None
        self.saved_path = None
        self.loaded_path = None

    def train(self, X: pd.DataFrame, y: pd.Series):
        self.trained = True
        print(f"MockBaseModel {self.model_name} trained.")

    def predict(self, X: pd.DataFrame) -> pd.Series:
        self.predicted_data = pd.Series([0.5] * len(X), index=X.index)
        print(f"MockBaseModel {self.model_name} predicted.")
        return self.predicted_data

    def save_model(self, path: str):
        self.saved_path = path
        print(f"MockBaseModel {self.model_name} saved to {path}.")

    def load_model(self, path: str):
        self.loaded_path = path
        self.trained = True # ロードされたら学習済みとみなす
        print(f"MockBaseModel {self.model_name} loaded from {path}.")

class TestModelManager(unittest.TestCase):

    def setUp(self):
        self.test_model_dir = "python/test_models_dir"
        # 既存のテストディレクトリがあれば削除
        if os.path.exists(self.test_model_dir):
            shutil.rmtree(self.test_model_dir)
        
        # ModelManagerの初期化
        self.model_manager = ModelManager(model_dir=self.test_model_dir)
        
        # ModelManagerの_registered_modelsをモックされたクラスに置き換える
        # これにより、ModelManagerが内部でXGBoostModelやLightGBMModelをインスタンス化しようとした際に
        # 実際のクラスではなくMockBaseModelが使われるようになる
        self.model_manager._registered_models = {
            "XGBoostModel": MockBaseModel,
            "LightGBMModel": MockBaseModel,
        }

        # テスト用のダミーデータ
        self.dummy_X = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
        self.dummy_y = pd.Series([0, 1, 0])

    def tearDown(self):
        # テスト後に作成されたディレクトリを削除
        if os.path.exists(self.test_model_dir):
            shutil.rmtree(self.test_model_dir)

    def test_initialization(self):
        """
        ModelManagerの初期化とモデルディレクトリの作成を確認します。
        """
        self.assertTrue(os.path.exists(self.test_model_dir))
        self.assertIsInstance(self.model_manager.models, dict)
        # ModelManagerの初期化時に登録されるデフォルトモデルがモックされていることを確認
        self.assertEqual(len(self.model_manager._registered_models), 2) 
        self.assertEqual(self.model_manager._registered_models["XGBoostModel"], MockBaseModel)
        self.assertEqual(self.model_manager._registered_models["LightGBMModel"], MockBaseModel)

    def test_register_model_type(self):
        """
        新しいモデルタイプが正しく登録されることを確認します。
        """
        class NewModel(BaseModel):
            def __init__(self, model_name: str):
                super().__init__(model_name)
            def train(self, X, y): pass
            def predict(self, X): return pd.Series()
            def save_model(self, path): pass
            def load_model(self, path): pass

        self.model_manager.register_model_type("NewModel", NewModel)
        self.assertIn("NewModel", self.model_manager._registered_models)
        self.assertEqual(self.model_manager._registered_models["NewModel"], NewModel)

    def test_register_model_type_invalid_class(self):
        """
        BaseModelを継承していないクラスを登録しようとした場合にValueErrorが発生することを確認します。
        """
        class InvalidModel:
            pass
        with self.assertRaises(ValueError):
            self.model_manager.register_model_type("InvalidModel", InvalidModel)

    def test_create_model(self):
        """
        モデルが正しく作成され、内部辞書に追加されることを確認します。
        """
        model_instance = self.model_manager.create_model("XGBoostModel", "test_xgb_model")
        self.assertIsInstance(model_instance, MockBaseModel)
        self.assertIn("test_xgb_model", self.model_manager.models)
        self.assertEqual(self.model_manager.models["test_xgb_model"], model_instance)

        model_instance_lgbm = self.model_manager.create_model("LightGBMModel", "test_lgbm_model")
        self.assertIsInstance(model_instance_lgbm, MockBaseModel)
        self.assertIn("test_lgbm_model", self.model_manager.models)

    def test_create_model_unregistered_type(self):
        """
        未登録のモデルタイプを指定した場合にValueErrorが発生することを確認します。
        """
        with self.assertRaises(ValueError):
            self.model_manager.create_model("NonExistentModel", "my_model")

    def test_get_model(self):
        """
        既存のモデルが正しく取得できることを確認します。
        """
        model_instance = self.model_manager.create_model("XGBoostModel", "get_test_model")
        retrieved_model = self.model_manager.get_model("get_test_model")
        self.assertEqual(model_instance, retrieved_model)

    def test_get_model_not_found(self):
        """
        存在しないモデル名を指定した場合にValueErrorが発生することを確認します。
        """
        with self.assertRaises(ValueError):
            self.model_manager.get_model("non_existent_model")

    def test_train_model(self):
        """
        モデルの学習メソッドが呼び出され、モデルが保存されることを確認します。
        """
        model_name = "train_test_model"
        model_instance = self.model_manager.create_model("XGBoostModel", model_name)
        
        # trainとsave_modelが呼び出されることを確認するためにモックを置き換える
        # side_effectを設定して、元のMockBaseModelのメソッドも実行されるようにする
        original_train = model_instance.train
        original_save_model = model_instance.save_model
        model_instance.train = MagicMock(side_effect=original_train)
        model_instance.save_model = MagicMock(side_effect=original_save_model)

        self.model_manager.train_model(model_name, self.dummy_X, self.dummy_y)
        
        model_instance.train.assert_called_once_with(self.dummy_X, self.dummy_y)
        model_instance.save_model.assert_called_once()
        self.assertTrue(model_instance.trained) # MockBaseModelのtrainedフラグも確認

    def test_predict_with_model(self):
        """
        モデルの予測メソッドが呼び出され、結果が返されることを確認します。
        """
        model_name = "predict_test_model"
        model_instance = self.model_manager.create_model("XGBoostModel", model_name)
        
        # predictが呼び出されることを確認するためにモックを置き換える
        model_instance.predict = MagicMock(return_value=pd.Series([0.1, 0.2, 0.3]))

        predictions = self.model_manager.predict_with_model(model_name, self.dummy_X)
        
        model_instance.predict.assert_called_once_with(self.dummy_X)
        self.assertIsInstance(predictions, pd.Series)
        self.assertEqual(len(predictions), len(self.dummy_X))

    def test_save_model(self):
        """
        モデルが指定されたパスに保存されることを確認します。
        """
        model_name = "save_test_model"
        model_instance = self.model_manager.create_model("XGBoostModel", model_name)
        
        # save_modelが呼び出されることを確認するためにモックを置き換える
        # side_effectを設定して、元のMockBaseModelのメソッドも実行されるようにする
        original_save_model = model_instance.save_model
        model_instance.save_model = MagicMock(side_effect=original_save_model)

        self.model_manager.save_model(model_name)
        
        expected_path = os.path.join(self.test_model_dir, f"{model_name}.joblib")
        model_instance.save_model.assert_called_once_with(expected_path)
        self.assertEqual(model_instance.saved_path, expected_path) # MockBaseModelのsaved_pathも確認

    def test_load_model_existing_instance(self):
        """
        既存のモデルインスタンスが正しくロードされることを確認します。
        """
        model_name = "load_test_model_existing"
        model_instance = self.model_manager.create_model("XGBoostModel", model_name)
        
        # load_modelが呼び出されることを確認するためにモックを置き換える
        # side_effectを設定して、元のMockBaseModelのメソッドも実行されるようにする
        original_load_model = model_instance.load_model
        model_instance.load_model = MagicMock(side_effect=original_load_model)

        # ダミーのモデルファイルを事前に作成
        dummy_model_path = os.path.join(self.test_model_dir, f"{model_name}.joblib")
        with open(dummy_model_path, 'w') as f:
            f.write("dummy model content")

        self.model_manager.load_model(model_name)
        
        model_instance.load_model.assert_called_once_with(dummy_model_path)
        self.assertTrue(model_instance.trained) # MockBaseModelのtrainedフラグも確認

    def test_load_model_new_instance_with_type(self):
        """
        新しいモデルインスタンスがタイプ指定で正しくロードされることを確認します。
        """
        model_name = "load_test_model_new"
        model_type = "XGBoostModel"
        
        # ダミーのモデルファイルを事前に作成
        dummy_model_path = os.path.join(self.test_model_dir, f"{model_name}.joblib")
        with open(dummy_model_path, 'w') as f:
            f.write("dummy model content")

        # create_modelが呼び出されることを確認するためにモックを置き換える
        with patch.object(self.model_manager, 'create_model', wraps=self.model_manager.create_model) as mock_create_model:
            self.model_manager.load_model(model_name, model_type=model_type)
            mock_create_model.assert_called_once_with(model_type, model_name)
            
            loaded_model = self.model_manager.get_model(model_name)
            self.assertIsInstance(loaded_model, MockBaseModel)
            self.assertTrue(loaded_model.trained) # MockBaseModelのtrainedフラグも確認
            self.assertEqual(loaded_model.loaded_path, dummy_model_path) # MockBaseModelのloaded_pathも確認

    def test_load_model_new_instance_no_type_error(self):
        """
        新しいモデルインスタンスをロードする際にmodel_typeが指定されていない場合にValueErrorが発生することを確認します。
        """
        model_name = "load_test_model_no_type"
        with self.assertRaises(ValueError) as cm:
            self.model_manager.load_model(model_name)
        self.assertIn("model_typeを指定してください", str(cm.exception))

if __name__ == '__main__':
    unittest.main()
