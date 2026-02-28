"""
予測スクリプト（統合版）

単一銘柄:     py run_predict.py --mode single --market us --symbol AAPL
ウォッチリスト: py run_predict.py --mode watchlist
Top10/Worst10: py run_predict.py --mode top10 [--individual]
"""

import argparse
import csv
import pprint
from datetime import datetime

import pandas as pd

from src.models.predict_single_stock import predict_single_stock
from src.utils.data_path_utils import get_monitor_list_path
from src.utils.db import save_prediction_results


def run_single(market: str, symbol: str):
    """単一銘柄の予測結果を表示・DB保存する"""
    result = predict_single_stock(market, symbol)
    if result is None:
        print("予測に失敗しました。モデルまたはデータが存在しない可能性があります。")
    else:
        pprint.pprint(result.to_dict("records")[0], sort_dicts=False, width=120)
        now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_prediction_results(now_str, result)


def run_watchlist():
    """監視リストの全銘柄を予測・DB保存する"""
    watchlist_path = get_monitor_list_path()
    output_rows = []
    with open(watchlist_path, encoding="utf-8") as f:
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
                pprint.pprint(result.to_dict("records")[0], sort_dicts=False, width=120)
                print("-" * 40)
                output_rows.append(result)

    if output_rows:
        now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        df_all = pd.concat(output_rows, ignore_index=True)
        save_prediction_results(now_str, df_all)
        print(f"\n結果保存完了: run_timestamp={now_str}")


def run_top10(use_individual: bool = False):
    """Top10/Worst10銘柄を予測・表示・DB保存する"""
    from src.services.prediction_pipeline import (
        predict_all_individual,
        predict_all_unified,
        output_top_worst_results,
    )

    if use_individual:
        output_rows = predict_all_individual()
        output_top_worst_results(output_rows, mode="individual")
    else:
        output_rows = predict_all_unified()
        output_top_worst_results(output_rows, mode="unified")


def main():
    parser = argparse.ArgumentParser(description="株価予測スクリプト")
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["single", "watchlist", "top10"],
        help="実行モード: single=単一銘柄, watchlist=監視リスト, top10=差異割合ランキング",
    )
    parser.add_argument("--market", type=str, help="市場名（singleモード時に必須）")
    parser.add_argument("--symbol", type=str, help="銘柄コード（singleモード時に必須）")
    parser.add_argument(
        "--individual",
        action="store_true",
        help="銘柄別モデルを使用する（top10モード時。デフォルトは統合モデル）",
    )
    args = parser.parse_args()

    if args.mode == "single":
        if not args.market or not args.symbol:
            parser.error("singleモードでは --market と --symbol が必須です")
        run_single(args.market, args.symbol)
    elif args.mode == "watchlist":
        run_watchlist()
    elif args.mode == "top10":
        run_top10(use_individual=args.individual)


if __name__ == "__main__":
    main()
