import argparse
import pprint
from src.models.predict_single_stock import predict_single_stock

def main():
    parser = argparse.ArgumentParser(description="指定したmarket,symbolの予想株価を案分して表示")
    parser.add_argument("--market", required=True, help="マーケット名 (例: us)")
    parser.add_argument("--symbol", required=True, help="ティッカーシンボル (例: AAPL)")
    args = parser.parse_args()

    result = predict_single_stock(args.market, args.symbol)
    if result is None:
        print("予測に失敗しました。モデルまたはデータが存在しない可能性があります。")
    else:
        pprint.pprint(result, sort_dicts=False, width=120)

if __name__ == "__main__":
    main()
