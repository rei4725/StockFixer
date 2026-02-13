import argparse
from src.services.data_pipeline import save_stock_data_with_features
from src.utils.data_path_utils import get_data_dir


def main():
    parser = argparse.ArgumentParser(description="株価データ取得・特徴量生成・保存スクリプト")
    parser.add_argument("--market", type=str, required=True, help="市場名（例: us, jp）")
    parser.add_argument("--symbol", type=str, required=True, help="銘柄コード（例: AAPL, 7203）")
    parser.add_argument("--start_date", type=str, required=False, help="開始日（YYYY-MM-DD）")
    parser.add_argument("--end_date", type=str, required=False, help="終了日（YYYY-MM-DD）")
    parser.add_argument("--out_dir", type=str, default=None, help="保存先ディレクトリ（省略時はデフォルト）")
    args = parser.parse_args()
    
    # out_dirが指定されていなければdata_path_utilsから取得
    out_dir = args.out_dir if args.out_dir else get_data_dir()

    try:
        save_stock_data_with_features(
            market=args.market,
            symbol=args.symbol,
            start_date=args.start_date,
            end_date=args.end_date,
            out_dir=out_dir
        )
    except Exception as e:
        print(f"エラーが発生しました: {e}")

if __name__ == "__main__":
    main()
