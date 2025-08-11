import argparse
from src.data.data_saver import save_stock_data_with_features

def main():
    parser = argparse.ArgumentParser(description="株価データ取得・特徴量生成・保存スクリプト")
    parser.add_argument("--market", type=str, required=True, help="市場名（例: us, jp）")
    parser.add_argument("--symbol", type=str, required=True, help="銘柄コード（例: AAPL, 7203）")
    parser.add_argument("--start_date", type=str, required=True, help="開始日（YYYY-MM-DD）")
    parser.add_argument("--end_date", type=str, required=True, help="終了日（YYYY-MM-DD）")
    parser.add_argument("--out_dir", type=str, default="python/data", help="保存先ディレクトリ")
    args = parser.parse_args()

    try:
        save_stock_data_with_features(
            market=args.market,
            symbol=args.symbol,
            start_date=args.start_date,
            end_date=args.end_date,
            out_dir=args.out_dir
        )
    except Exception as e:
        print(f"エラーが発生しました: {e}")

if __name__ == "__main__":
    main()
