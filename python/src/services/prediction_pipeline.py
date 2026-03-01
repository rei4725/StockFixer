"""
予測パイプラインサービス

銘柄別モデル・統合モデルでの全銘柄予測、Top10/Worst10集計、DB保存を行う
"""

import os
import glob
import json
import pandas as pd
import warnings
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.models.predict_single_stock import predict_single_stock
from src.utils.df_to_string import df_to_pretty_string
from src.utils.data_path_utils import get_models_dir
from src.utils.db import save_prediction_results, get_all_symbols

# yfinanceの警告を抑制
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

# 並列実行時のワーカー数（デフォルト=1で同期実行、スレッド間競合を回避）
# ガイドライン参照: 並列処理は競合バグの原因となるため同期集計が安定
MAX_WORKERS = 1


def get_optimal_params(market: str, symbol: str) -> dict:
    """
    保存された最適パラメータをJSONから読み込む。

    Args:
        market: マーケット識別子
        symbol: 銘柄シンボル

    Returns:
        最適パラメータ辞書、見つからない場合は空辞書
    """
    # python/src/services/prediction_pipeline.py -> python/config
    config_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config")
    json_path = os.path.join(config_dir, "optimal_params.json")

    if not os.path.exists(json_path):
        return {}

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            all_params = json.load(f)
        key = f"{market}_{symbol}"
        params = all_params.get(key, {})
        if params:
            print(f"[{market}_{symbol}] 最適パラメータを読み込みました: 閾値={params.get('threshold')}, SharpeRatio={params.get('metrics', {}).get('sharpe_ratio')}")
        return params
    except Exception as e:
        print(f"[{market}_{symbol}] 最適パラメータ読み込みエラー: {e}")
        return {}


def find_model_files(model_root=None, model_name="StockXGBoostModel.joblib"):
    """
    モデルディレクトリを再帰的に探索し、(market, symbol, model_path)のリストを返す
    """
    if model_root is None:
        model_root = get_models_dir()
    pattern = os.path.join(model_root, "*_*", model_name)
    files = glob.glob(pattern)
    result = []
    for path in files:
        parts = path.replace("\\", "/").split("/")
        if len(parts) >= 3:
            market_symbol = parts[-2]
            if "_" in market_symbol:
                market, symbol = market_symbol.split("_", 1)
                result.append((market, symbol, path))
    return result


def predict_all_individual(max_workers=MAX_WORKERS):
    """
    銘柄別モデルで全銘柄の予測を実行する

    Returns:
        list[pd.DataFrame]: 各銘柄の予測結果
    """
    model_types = ["StockXGBoostModel.joblib", "StockLightGBMModel.joblib"]
    all_keys = set()
    for model_type in model_types:
        model_files = find_model_files(model_name=model_type)
        for market, symbol, _ in model_files:
            all_keys.add((market, symbol))

    tasks = [(market, symbol, model_types) for market, symbol in all_keys]
    output_rows = []
    print(f"並列予測開始（銘柄別モデル）: {len(tasks)}銘柄, ワーカー数: {max_workers}")

    def wrapper(args):
        market, symbol, mtypes = args
        try:
            return predict_single_stock(market, symbol, model_types=mtypes)
        except Exception as e:
            print(f"[{market}_{symbol}] 予測エラー: {e}")
            return None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(wrapper, task): task for task in tasks}
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                output_rows.append(result)

    return output_rows


def predict_all_unified(max_workers=MAX_WORKERS):
    """
    統合モデルで全銘柄の予測を実行する

    Returns:
        list[pd.DataFrame]: 各銘柄の予測結果
    """
    from src.models.predict_unified import predict_with_unified_model, preload_models

    model_types = ["UnifiedStockXGBoost", "UnifiedStockLightGBM"]

    # モデルを事前ロード（並列実行前に1回だけロード）
    preload_models(model_types)

    all_keys = get_all_symbols()

    def wrapper(args):
        market, symbol = args
        try:
            return predict_with_unified_model(market, symbol, model_types=model_types)
        except Exception as e:
            print(f"[{market}_{symbol}] 予測エラー: {e}")
            return None

    output_rows = []
    print(f"並列予測開始（統合モデル）: {len(all_keys)}銘柄, ワーカー数: {max_workers}")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(wrapper, task): task for task in all_keys}
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                output_rows.append(result)

    return output_rows


def output_top_worst_results(output_rows, mode="individual"):
    """
    予測結果からTop10/Worst10を出力し、全結果をDBに保存する

    Args:
        output_rows: predict_all_*の戻り値
        mode: "individual" or "unified"
    """
    if not output_rows:
        print("有効な結果がありませんでした。")
        return

    df_result = pd.concat(output_rows, ignore_index=True)
    display_columns = ["market", "symbol", "current_price", "avg_pred_price", "diff_ratio", "model_count"]
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 全予測結果をDBに保存（Delete-Insert）
    save_prediction_results(now_str, df_result[display_columns])

    for market, df_market in df_result.groupby("market"):
        # Top10
        df_top10 = df_market.sort_values("diff_ratio", ascending=False).head(10)
        df_top10_display = df_top10[display_columns]
        print(df_to_pretty_string(
            df_top10_display,
            header=f"=== {market} 差異割合上位10銘柄 ({mode}) === 実行日時: {now_str}"
        ))

        # ワースト10
        df_worst10 = df_market.sort_values("diff_ratio", ascending=True).head(10)
        df_worst10_display = df_worst10[display_columns]
        print(df_to_pretty_string(
            df_worst10_display,
            header=f"=== {market} 差異割合ワースト10銘柄 ({mode}) === 実行日時: {now_str}"
        ))

    print(f"\n結果保存完了: predicted_at={now_str}")


def run_predict_single(market: str, symbol: str):
    """
    単一銘柄の予測結果を表示・DB保存する
    
    Args:
        market: マーケット識別子
        symbol: 銘柄シンボル
    """
    import pprint
    result = predict_single_stock(market, symbol)
    if result is None:
        print("予測に失敗しました。モデルまたはデータが存在しない可能性があります。")
    else:
        pprint.pprint(result.to_dict("records")[0], sort_dicts=False, width=120)
        now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_prediction_results(now_str, result)


def run_predict_watchlist():
    """
    監視リストの全銘柄を予測・DB保存する
    """
    import csv
    from src.utils.data_path_utils import get_monitor_list_path
    
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
                import pprint
                pprint.pprint(result.to_dict("records")[0], sort_dicts=False, width=120)
                print("-" * 40)
                output_rows.append(result)

    if output_rows:
        now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        df_all = pd.concat(output_rows, ignore_index=True)
        save_prediction_results(now_str, df_all)
        print(f"\n結果保存完了: run_timestamp={now_str}")
