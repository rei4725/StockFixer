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
    """DBから全銘柄を取得"""
    return set(get_all_symbols())


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
    """結果を出力・DB保存"""
    if output_rows:
        df_result = pd.concat(output_rows, ignore_index=True)
        display_columns = ["market", "symbol", "current_price", "avg_pred_price", "diff_ratio", "model_count"]

        now_str = datetime.now().strftime("%Y%m%d_%H%M%S")

        for market, df_market in df_result.groupby("market"):
            # Top10
            df_top10 = df_market.sort_values("diff_ratio", ascending=False).head(10)
            df_top10_display = df_top10[display_columns]
            print(df_to_pretty_string(
                df_top10_display,
                header=f"=== {market} 差異割合上位10銘柄 ({mode}) === 実行日時: {now_str}"
            ))
            save_prediction_results(now_str, df_top10_display, rank_type="top10")

            # ワースト10
            df_worst10 = df_market.sort_values("diff_ratio", ascending=True).head(10)
            df_worst10_display = df_worst10[display_columns]
            print(df_to_pretty_string(
                df_worst10_display,
                header=f"=== {market} 差異割合ワースト10銘柄 ({mode}) === 実行日時: {now_str}"
            ))
            save_prediction_results(now_str, df_worst10_display, rank_type="worst10")
        
        print(f"\n結果保存完了: run_timestamp={now_str}")
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
