import pandas as pd
from src.models.model_manager import ModelManager
from src.features.technical_analysis import create_basic_lag_features
from src.data.data_loader import get_stock_data
import os

def train_and_test_model(model_manager, model_type, model_name, X, y):
    try:
        print(f"モデル '{model_name}' ({model_type}) を作成中...")
        model = model_manager.create_model(model_type, model_name)
        print(f"モデル '{model_name}' の作成に成功しました。")

        print(f"モデル '{model_name}' を学習中...")
        model_manager.train_model(model_name, X, y)
        print(f"モデル '{model_name}' の学習と保存に成功しました。")

        model_path = os.path.join(model_manager.model_dir, f"{model_name}.joblib")
        if os.path.exists(model_path):
            print(f"モデルファイルが '{model_path}' に存在することを確認しました。")
        else:
            print(f"エラー: モデルファイル '{model_path}' が見つかりません。")

        print(f"モデル '{model_name}' をロードしてテスト中...")
        loaded_model = model_manager.load_model(model_name, model_type)
        print(f"モデル '{model_name}' のロードに成功しました。")
        
        if not X.empty:
            predict_data = X.iloc[[-1]]
            prediction = loaded_model.predict(predict_data)
            print(f"ロードしたモデルでの予測結果: {prediction.tolist()}")
        else:
            print("予測用データがありません。")

    except ValueError as e:
        print(f"エラーが発生しました: {e}")
    except Exception as e:
        print(f"予期せぬエラーが発生しました: {e}")

def run_model_creation():
    """
    必須情報のみで特徴量を作成し、モデルを学習・保存するサンプルスクリプト。
    """
    print("モデル作成と保存のスクリプトを開始します。")

    # python/src/data 配下の全CSVを読み込んで結合
    import glob
    import re
    from src.utils.csv_io import load_dataframe_from_csv

    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not csv_files:
        print("データフォルダにCSVファイルがありません。")
        return

    print(f"データフォルダ内のCSVファイル: {csv_files}")
    df_list = []
    for f in csv_files:
        df = load_dataframe_from_csv(f)
        if df is not None:
            df_list.append(df)
    if not df_list:
        print("有効なデータがありません。")
        return

    all_data = pd.concat(df_list, ignore_index=True)
    print(f"全データ結合後のshape: {all_data.shape}")

    # y列は「最後の1列」と仮定
    X = all_data.iloc[:, :-1]
    y = all_data.iloc[:, -1]

    # 特徴量名の確認
    print("特徴量名一覧:", list(X.columns))

    # 特殊文字を含む特徴量名を正規化（LightGBM対策）
    def normalize_col(col):
        return re.sub(r'[^0-9a-zA-Z_]', '_', col)
    X.columns = [normalize_col(str(c)) for c in X.columns]

    # ModelManagerのインスタンスを作成
    model_manager = ModelManager()

    # XGBoostモデル
    train_and_test_model(model_manager, "XGBoostModel", "StockXGBoostModel", X, y)

    # LightGBMモデル
    train_and_test_model(model_manager, "LightGBMModel", "StockLightGBMModel", X, y)

    print("スクリプトを終了します。")

if __name__ == "__main__":
    run_model_creation()
