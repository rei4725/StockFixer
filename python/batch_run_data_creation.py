"""
バッチデータ取得スクリプト（並列処理対応）

データ取得を銘柄別に並列で処理する
"""
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.utils.data_path_utils import get_watchlist_path

# 並列数の設定（yfinance API制限を考慮）
MAX_DATA_WORKERS = 5  # データ取得の並列数


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
    
    # データ取得（並列）
    data_results = batch_fetch_data(symbols)
    print_summary("データ取得", data_results)
    
    print("データ取得完了")


if __name__ == "__main__":
    main()
