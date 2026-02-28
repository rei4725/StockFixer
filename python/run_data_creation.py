"""
株価データ取得・特徴量生成・保存スクリプト

単一銘柄: py run_data_creation.py --market us --symbol AAPL
バッチ:   py run_data_creation.py --batch
"""

import argparse
from src.services.data_pipeline import (
    save_stock_data_with_features,
    fetch_stock_data_with_features,
    save_features_to_db,
)


# バッチ取得の並列数（yfinance API制限を考慮）
MAX_DATA_WORKERS = 5


def _fetch_only(task: dict) -> dict:
    """バッチランナー用: データ取得＋特徴量生成のみ（DB書き込みなし）"""
    market, symbol = task["market"], task["symbol"]
    try:
        print(f"[データ取得開始] {market}/{symbol}")
        result = fetch_stock_data_with_features(market=market, symbol=symbol)
        if result is None:
            print(f"[データ取得スキップ] {market}/{symbol}")
            return {"market": market, "symbol": symbol, "status": "skip"}
        print(f"[データ取得完了] {market}/{symbol}")
        return {"market": market, "symbol": symbol, "status": "success", "data": result}
    except Exception as e:
        print(f"[データ取得エラー] {market}/{symbol}: {e}")
        return {"market": market, "symbol": symbol, "status": "error", "error": str(e)}


def run_single(market: str, symbol: str, start_date=None, end_date=None):
    """単一銘柄のデータを取得・保存する"""
    try:
        save_stock_data_with_features(
            market=market,
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
        )
    except Exception as e:
        print(f"エラーが発生しました: {e}")


def run_batch():
    """
    ウォッチリストの全銘柄をバッチ取得する。
    
    フェーズ1: データ取得＋特徴量生成（並列） - I/O boundなAPI呼び出し
    フェーズ2: DB書き込み（逐次） - DuckDBの排他ロック制約を回避
    """
    from src.services.batch_runner import load_target_symbols, run_parallel, print_summary

    symbols = load_target_symbols()
    if not symbols:
        print("対象銘柄がありません。")
        return

    # フェーズ1: データ取得＋特徴量生成（並列）
    fetch_results = run_parallel(
        func=_fetch_only,
        tasks=symbols,
        max_workers=MAX_DATA_WORKERS,
        label="データ取得",
    )

    # フェーズ2: DB書き込み（逐次） - DuckDB排他ロック制約のため直列実行
    success_data = [r for r in fetch_results if r.get("status") == "success" and r.get("data")]
    print(f"\n{'='*50}")
    print(f"DB書き込み開始（逐次）")
    print(f"対象件数: {len(success_data)}")
    print(f"{'='*50}\n")

    db_results = []
    for i, r in enumerate(success_data, 1):
        market, symbol, data = r["data"]
        try:
            save_features_to_db(market, symbol, data)
            db_results.append({"market": market, "symbol": symbol, "status": "success"})
        except Exception as e:
            print(f"[DB書き込みエラー] {market}/{symbol}: {e}")
            db_results.append({"market": market, "symbol": symbol, "status": "error", "error": str(e)})
        if i % 50 == 0:
            print(f"  ... {i}/{len(success_data)} 件完了")

    # 最終サマリー（取得フェーズのエラー/スキップ + DB書き込み結果を統合）
    final_results = db_results.copy()
    final_results += [r for r in fetch_results if r.get("status") in ("error", "skip")]
    print_summary("データ更新", final_results)


def main():
    parser = argparse.ArgumentParser(description="株価データ取得・特徴量生成・保存スクリプト")
    parser.add_argument("--batch", action="store_true", help="ウォッチリストの全銘柄を並列取得する")
    parser.add_argument("--market", type=str, help="市場名（例: us, jp）")
    parser.add_argument("--symbol", type=str, help="銘柄コード（例: AAPL, 7203）")
    parser.add_argument("--start_date", type=str, default=None, help="開始日（YYYY-MM-DD）")
    parser.add_argument("--end_date", type=str, default=None, help="終了日（YYYY-MM-DD）")
    args = parser.parse_args()

    if args.batch:
        run_batch()
    else:
        if not args.market or not args.symbol:
            parser.error("--market と --symbol は必須です（--batch 指定時を除く）")
        run_single(args.market, args.symbol, args.start_date, args.end_date)


if __name__ == "__main__":
    main()
