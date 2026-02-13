"""
バッチデータ作成・モデル学習スクリプト（並列処理対応）

データ取得とモデル作成を銘柄別に並列で処理する
"""
import csv
import os
import re
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from src.utils.data_path_utils import get_watchlist_path, get_data_dir, get_data_subdir

# 並列数の設定（yfinance API制限を考慮）
MAX_DATA_WORKERS = 5  # データ取得の並列数
MAX_MODEL_WORKERS = 4  # モデル作成の並列数（CPU数に応じて調整）


def fetch_stock_data(market: str, symbol: str) -> dict:
    """
    単一銘柄のデータ取得（並列処理用）
    """
    from src.services.data_pipeline import save_stock_data_with_features
    
    try:
        print(f"[データ取得開始] {market}/{symbol}")
        save_stock_data_with_features(
            market=market,
            symbol=symbol,
            start_date=None,
            end_date=None,
            out_dir=None  # data_path_utilsのデフォルトを使用
        )
        print(f"[データ取得完了] {market}/{symbol}")
        return {"market": market, "symbol": symbol, "status": "success"}
    except Exception as e:
        print(f"[データ取得エラー] {market}/{symbol}: {e}")
        return {"market": market, "symbol": symbol, "status": "error", "error": str(e)}


def train_model_for_symbol(market: str, symbol: str) -> dict:
    """
    単一銘柄のモデル作成（並列処理用）
    """
    import glob
    from src.models.model_manager import ModelManager
    from src.utils.csv_io import load_dataframe_from_csv
    from src.utils.data_path_utils import get_data_subdir
    
    try:
        print(f"[モデル作成開始] {market}/{symbol}")
        
        # 該当銘柄のCSVファイルを取得
        data_dir = get_data_subdir(market, symbol)
        csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
        
        if not csv_files:
            return {"market": market, "symbol": symbol, "status": "skip", "reason": "CSVなし"}
        
        # 最初のCSVファイルを使用
        df = load_dataframe_from_csv(csv_files[0])
        if df is None or df.empty:
            return {"market": market, "symbol": symbol, "status": "skip", "reason": "データ無効"}
        
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        
        # 特徴量名の正規化
        def normalize_col(col):
            return re.sub(r'[^0-9a-zA-Z_]', '_', str(col))
        X.columns = [normalize_col(c) for c in X.columns]
        
        # ModelManagerは各プロセスで新規作成
        model_manager = ModelManager()
        
        # XGBoostモデル
        xgb_model = model_manager.create_model("XGBoostModel", "StockXGBoostModel")
        model_manager.train_model("StockXGBoostModel", X, y, market=market, symbol=symbol)
        
        # LightGBMモデル
        lgb_model = model_manager.create_model("LightGBMModel", "StockLightGBMModel")
        model_manager.train_model("StockLightGBMModel", X, y, market=market, symbol=symbol)
        
        print(f"[モデル作成完了] {market}/{symbol}")
        return {"market": market, "symbol": symbol, "status": "success"}
    except Exception as e:
        print(f"[モデル作成エラー] {market}/{symbol}: {e}")
        return {"market": market, "symbol": symbol, "status": "error", "error": str(e)}


def load_target_symbols() -> list:
    """
    CSVから対象銘柄リストを読み込む
    """
    symbols = []
    csv_path = get_watchlist_path()
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            symbols.append({
                "market": row["市場"],
                "symbol": row["銘柄コード"]
            })
    return symbols


def batch_fetch_data(symbols: list) -> list:
    """
    データ取得を並列実行
    """
    print(f"\n{'='*50}")
    print(f"データ取得開始（並列数: {MAX_DATA_WORKERS}）")
    print(f"対象銘柄数: {len(symbols)}")
    print(f"{'='*50}\n")
    
    results = []
    with ThreadPoolExecutor(max_workers=MAX_DATA_WORKERS) as executor:
        futures = {
            executor.submit(fetch_stock_data, s["market"], s["symbol"]): s
            for s in symbols
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
    
    return results


def batch_train_models(symbols: list) -> list:
    """
    モデル作成を並列実行
    """
    print(f"\n{'='*50}")
    print(f"モデル作成開始（並列数: {MAX_MODEL_WORKERS}）")
    print(f"対象銘柄数: {len(symbols)}")
    print(f"{'='*50}\n")
    
    results = []
    with ProcessPoolExecutor(max_workers=MAX_MODEL_WORKERS) as executor:
        futures = {
            executor.submit(train_model_for_symbol, s["market"], s["symbol"]): s
            for s in symbols
        }
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
    
    return results


def print_summary(phase: str, results: list):
    """
    結果サマリーを出力
    """
    success = [r for r in results if r["status"] == "success"]
    errors = [r for r in results if r["status"] == "error"]
    skipped = [r for r in results if r["status"] == "skip"]
    
    print(f"\n{'='*50}")
    print(f"{phase} 結果サマリー")
    print(f"{'='*50}")
    print(f"成功: {len(success)}")
    print(f"スキップ: {len(skipped)}")
    print(f"エラー: {len(errors)}")
    
    if errors:
        print("\nエラー詳細:")
        for e in errors:
            print(f"  - {e['market']}/{e['symbol']}: {e.get('error', 'unknown')}")
    print()


def main():
    # 対象銘柄を読み込み
    symbols = load_target_symbols()
    
    if not symbols:
        print("対象銘柄がありません。")
        return
    
    # Step 1: データ取得（並列）
    data_results = batch_fetch_data(symbols)
    print_summary("データ取得", data_results)
    
    # Step 2: モデル作成（並列）
    # データ取得が成功した銘柄のみ対象
    successful_symbols = [
        {"market": r["market"], "symbol": r["symbol"]}
        for r in data_results
        if r["status"] == "success"
    ]
    
    if successful_symbols:
        model_results = batch_train_models(successful_symbols)
        print_summary("モデル作成", model_results)
    else:
        print("データ取得が成功した銘柄がないため、モデル作成をスキップします。")
    
    print("全処理完了")


if __name__ == "__main__":
    main()
