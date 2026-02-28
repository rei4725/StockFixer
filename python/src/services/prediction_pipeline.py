"""
予測パイプラインサービス

銘柄別モデル・統合モデルでの全銘柄予測、Top10/Worst10集計、DB保存を行う
"""

import os
import glob
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

# 並列実行時のワーカー数（I/Oバウンドのため多めに設定可能）
MAX_WORKERS = 10


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
