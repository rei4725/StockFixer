"""
株価データ取得・特徴量生成・保存スクリプト

単一銘柄: py run_data_creation.py --market us --symbol AAPL
バッチ:   py run_data_creation.py --batch
"""

import argparse
from src.services.data_pipeline import save_stock_data_with_features


# バッチ取得の並列数（yfinance API制限を考慮）
MAX_DATA_WORKERS = 5


def _fetch_single(task: dict) -> dict:
    """バッチランナー用: 単一銘柄のデータ取得"""
    market, symbol = task["market"], task["symbol"]
    try:
        print(f"[データ取得開始] {market}/{symbol}")
        save_stock_data_with_features(market=market, symbol=symbol)
        print(f"[データ取得完了] {market}/{symbol}")
        return {"market": market, "symbol": symbol, "status": "success"}
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
    """ウォッチリストの全銘柄を並列でデータ取得する"""
    from src.services.batch_runner import load_target_symbols, run_parallel, print_summary

    symbols = load_target_symbols()
    if not symbols:
        print("対象銘柄がありません。")
        return

    results = run_parallel(
        func=_fetch_single,
        tasks=symbols,
        max_workers=MAX_DATA_WORKERS,
        label="データ取得",
    )
    print_summary("データ取得", results)


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
