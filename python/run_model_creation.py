
from src.models.model_manager import ModelManager
from src.features.technical_analysis import create_basic_lag_features
from src.data.data_loader import get_stock_data
import os

def train_and_test_model(model_manager, model_type, model_name, X, y, market=None, symbol=None):
    try:
        print(f"モデル '{model_name}' ({model_type}) を作成中...")
        model = model_manager.create_model(model_type, model_name)
        print(f"モデル '{model_name}' の作成に成功しました。")

        print(f"モデル '{model_name}' を学習中...")
        model_manager.train_model(model_name, X, y, market=market, symbol=symbol)
        print(f"モデル '{model_name}' の学習と保存に成功しました。")

        # モデルファイルパスをmarket, symbol付きで確認
        if market and symbol:
            model_path = os.path.join(model_manager.model_dir, f"{market}_{symbol}", f"{model_name}.joblib")
        else:
            model_path = os.path.join(model_manager.model_dir, f"{model_name}.joblib")
        if os.path.exists(model_path):
            print(f"モデルファイルが '{model_path}' に存在することを確認しました。")
        else:
            print(f"エラー: モデルファイル '{model_path}' が見つかりません。")

        print(f"モデル '{model_name}' をロードしてテスト中...")
        loaded_model = model_manager.load_model(model_name, model_type, market=market, symbol=symbol)
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

    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(script_dir, "data"))
    csv_files = glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True)

    if not csv_files:
        print("データフォルダにCSVファイルがありません。")
        return

    print(f"データフォルダ内のCSVファイル: {csv_files}")
    model_manager = ModelManager()

    def extract_market_symbol(filename):
        # サブディレクトリ名（例: us_DIS）からmarket, symbolを取得
        dir_name = os.path.basename(os.path.dirname(filename))
        if "_" in dir_name:
            market, symbol = dir_name.split("_", 1)
            return market, symbol
        # フォールバック: ファイル名からも試みる
        base = os.path.basename(filename)
        parts = base.split('_')
        if len(parts) >= 2:
            return parts[0], parts[1]
        return "unknown", "unknown"

    for f in csv_files:
        df = load_dataframe_from_csv(f)
        if df is None or df.empty:
            print(f"{f} のデータが無効です。")
            continue

        market, symbol = extract_market_symbol(f)
        print(f"処理中: market={market}, symbol={symbol}, file={f}")

        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]

        print("特徴量名一覧:", list(X.columns))

        def normalize_col(col):
            return re.sub(r'[^0-9a-zA-Z_]', '_', col)
        X.columns = [normalize_col(str(c)) for c in X.columns]

        # XGBoostモデル
        train_and_test_model(model_manager, "XGBoostModel", "StockXGBoostModel", X, y, market, symbol)

        # LightGBMモデル
        train_and_test_model(model_manager, "LightGBMModel", "StockLightGBMModel", X, y, market, symbol)

    print("スクリプトを終了します。")

if __name__ == "__main__":
    run_model_creation()
