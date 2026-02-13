import os
import glob
import pandas as pd
import warnings
import logging
import argparse
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.models.predict_single_stock import predict_single_stock
from src.utils.df_to_string import df_to_pretty_string
from src.utils.data_path_utils import get_models_dir, get_data_dir, get_results_dir, get_results_subdir, ensure_dir

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
        # 例: python/models/us_AAPL/StockXGBoostModel.joblib
        parts = path.replace("\\", "/").split("/")
        if len(parts) >= 3:
            market_symbol = parts[-2]
            if "_" in market_symbol:
                market, symbol = market_symbol.split("_", 1)
                result.append((market, symbol, path))
    return result


def predict_wrapper(args):
    """並列実行用ラッパー関数"""
    market, symbol, model_types = args
    try:
        return predict_single_stock(market, symbol, model_types=model_types)
    except Exception as e:
        print(f"[{market}_{symbol}] 予測エラー: {e}")
        return None


def get_all_symbols_from_data():
    """データディレクトリから全銘柄を取得"""
    data_dir = get_data_dir()
    all_keys = set()
    pattern = os.path.join(data_dir, "*_*")
    for dir_path in glob.glob(pattern):
        if not os.path.isdir(dir_path):
            continue
        dir_name = os.path.basename(dir_path)
        if "_" in dir_name:
            market, symbol = dir_name.split("_", 1)
            all_keys.add((market, symbol))
    return all_keys


def run_with_individual_models():
    """銘柄別モデルで予測を実行"""
    model_types = ["StockXGBoostModel.joblib", "StockLightGBMModel.joblib"]
    all_keys = set()
    for model_type in model_types:
        model_files = find_model_files(model_name=model_type)
        for market, symbol, _ in model_files:
            all_keys.add((market, symbol))

    tasks = [(market, symbol, model_types) for market, symbol in all_keys]
    output_rows = []
    print(f"並列予測開始（銘柄別モデル）: {len(tasks)}銘柄, ワーカー数: {MAX_WORKERS}")
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(predict_wrapper, task): task for task in tasks}
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                output_rows.append(result)
    
    return output_rows


def run_with_unified_model():
    """統合モデルで予測を実行"""
    from src.models.predict_unified import predict_with_unified_model, preload_models
    
    model_types = ["UnifiedStockXGBoost", "UnifiedStockLightGBM"]
    
    # モデルを事前ロード（並列実行前に1回だけロード）
    preload_models(model_types)
    
    all_keys = get_all_symbols_from_data()
    
    def unified_wrapper(args):
        market, symbol = args
        try:
            return predict_with_unified_model(market, symbol, model_types=model_types)
        except Exception as e:
            print(f"[{market}_{symbol}] 予測エラー: {e}")
            return None
    
    tasks = list(all_keys)
    output_rows = []
    print(f"並列予測開始（統合モデル）: {len(tasks)}銘柄, ワーカー数: {MAX_WORKERS}")
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(unified_wrapper, task): task for task in tasks}
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                output_rows.append(result)
    
    return output_rows


def output_results(output_rows, mode="individual"):
    """結果を出力・保存"""
    if output_rows:
        df_result = pd.concat(output_rows, ignore_index=True)
        display_columns = ["market", "symbol", "current_price", "avg_pred_price", "diff_ratio", "model_count"]

        now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_dir = get_results_subdir(now_str)
        ensure_dir(result_dir)

        for market, df_market in df_result.groupby("market"):
            # Top10
            df_top10 = df_market.sort_values("diff_ratio", ascending=False).head(10)
            df_top10_display = df_top10[display_columns]
            top10_path = os.path.join(result_dir, f"{market}_top10_diff_stocks.csv")
            print(df_to_pretty_string(
                df_top10_display,
                header=f"=== {market} 差異割合上位10銘柄 ({mode}) === 実行日時: {now_str}"
            ))
            df_top10_display.to_csv(top10_path, index=False)

            # ワースト10
            df_worst10 = df_market.sort_values("diff_ratio", ascending=True).head(10)
            df_worst10_display = df_worst10[display_columns]
            worst10_path = os.path.join(result_dir, f"{market}_worst10_diff_stocks.csv")
            print(df_to_pretty_string(
                df_worst10_display,
                header=f"=== {market} 差異割合ワースト10銘柄 ({mode}) === 実行日時: {now_str}"
            ))
            df_worst10_display.to_csv(worst10_path, index=False)
        
        print(f"\n結果保存先: {result_dir}")
    else:
        print("有効な結果がありませんでした。")


def main():
    parser = argparse.ArgumentParser(description="Top10/ワースト10銘柄を予測")
    parser.add_argument(
        "--individual",
        action="store_true",
        help="銘柄別モデルを使用する（デフォルトは統合モデル）"
    )
    args = parser.parse_args()

    if args.individual:
        output_rows = run_with_individual_models()
        output_results(output_rows, mode="individual")
    else:
        output_rows = run_with_unified_model()
        output_results(output_rows, mode="unified")

if __name__ == "__main__":
    main()
