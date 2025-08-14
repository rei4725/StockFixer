import csv
import pprint
from src.models.predict_single_stock import predict_single_stock

WATCHLIST_PATH = "python/監視対象.csv"

def main():
    results = []
    with open(WATCHLIST_PATH, encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            market, symbol = row[0], row[1]
            company = row[2] if len(row) > 2 else ""
            result = predict_single_stock(market, symbol)
            if result is None:
                print(f"[警告] {market},{symbol} ({company}) の予測に失敗しました。")
            else:
                print(f"{market},{symbol} ({company}) の予想株価:")
                pprint.pprint(result, sort_dicts=False, width=120)
                print("-" * 40)

if __name__ == "__main__":
    main()
