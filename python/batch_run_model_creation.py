"""
バッチモデル作成スクリプト（並列処理対応）

モデル作成を銘柄別に並列で処理する
"""
import csv
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from src.utils.data_path_utils import get_watchlist_path

# 並列数の設定（CPU数に応じて調整）
MAX_MODEL_WORKERS = 4


def train_model_for_symbol(market: str, symbol: str) -> dict:
    """
    単一銘柄のモデル作成（並列処理用）
    """
    from src.models.model_manager import ModelManager
    from src.utils.db import load_stock_features
    
    try:
        print(f"[モデル作成開始] {market}/{symbol}")
        
        # DBから特徴量データを取得
        df = load_stock_features(market, symbol)
        
        if df is None or df.empty:
            return {"market": market, "symbol": symbol, "status": "skip", "reason": "データなし"}
        
        # 文字列列とターゲット列を除外
        exclude_cols = ["y", "market", "symbol"]
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        X = df[feature_cols]
        y = df["y"]
        
        # 特徴量名の正規化
        def normalize_col(col):
            return re.sub(r'[^0-9a-zA-Z_]', '_', str(col))
        X.columns = [normalize_col(c) for c in X.columns]
        
        # ModelManagerは各プロセスで新規作成
        model_manager = ModelManager()
        
        # XGBoostモデル
        model_manager.create_model("XGBoostModel", "StockXGBoostModel")
        model_manager.train_model("StockXGBoostModel", X, y, market=market, symbol=symbol)
        
        # LightGBMモデル
        model_manager.create_model("LightGBMModel", "StockLightGBMModel")
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


def print_summary(results: list):
    """
    結果サマリーを出力
    """
    success = [r for r in results if r["status"] == "success"]
    errors = [r for r in results if r["status"] == "error"]
    skipped = [r for r in results if r["status"] == "skip"]
    
    print(f"\n{'='*50}")
    print(f"モデル作成 結果サマリー")
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
    
    # モデル作成（並列）
    model_results = batch_train_models(symbols)
    print_summary(model_results)
    
    print("モデル作成完了")


if __name__ == "__main__":
    main()
