"""
株価データ取得・特徴量生成・保存スクリプト

単一銘柄: py run_data_creation.py --market us --symbol AAPL
バッチ:   py run_data_creation.py --batch
取得のみ: py run_data_creation.py --market us --symbol AAPL --fetch-only
"""

import argparse
from src.services.data_pipeline import (
    fetch_stock_data_with_features,
    save_stock_data_with_features,
    save_features_to_db,
    run_data_batch,
)


def run_single(market: str, symbol: str, start_date=None, end_date=None, fetch_only=False):
    """単一銘柄のデータを取得・保存する"""
    try:
        if fetch_only:
            # データ取得と特徴量生成のみ（DB保存しない）
            result = fetch_stock_data_with_features(
                market=market,
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
            )
            if result is None:
                print(f"[取得失敗] {market}/{symbol}")
                return
            market, symbol, data = result
            print(f"[取得完了] {market}/{symbol} - {len(data)}行")
        else:
            # データ取得 → 特徴量生成 → DB保存
            save_stock_data_with_features(
                market=market,
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
            )
    except Exception as e:
        print(f"エラーが発生しました: {e}")


def main():
    parser = argparse.ArgumentParser(description="株価データ取得・特徴量生成・保存スクリプト")
    parser.add_argument("--batch", action="store_true", help="ウォッチリストの全銘柄を並列取得する")
    parser.add_argument("--market", type=str, help="市場名（例: us, jp）")
    parser.add_argument("--symbol", type=str, help="銘柄コード（例: AAPL, 7203）")
    parser.add_argument("--start_date", type=str, default=None, help="開始日（YYYY-MM-DD）")
    parser.add_argument("--end_date", type=str, default=None, help="終了日（YYYY-MM-DD）")
    parser.add_argument("--fetch-only", action="store_true", help="データ取得と特徴量生成のみ（DB保存しない）")
    args = parser.parse_args()

    if args.batch:
        run_data_batch(fetch_only=args.fetch_only)
    else:
        if not args.market or not args.symbol:
            parser.error("--market と --symbol は必須です（--batch 指定時を除く）")
        run_single(args.market, args.symbol, args.start_date, args.end_date, fetch_only=args.fetch_only)


if __name__ == "__main__":
    main()
