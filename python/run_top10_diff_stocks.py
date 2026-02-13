import os
import glob
import pandas as pd
from datetime import datetime
from src.models.predict_single_stock import predict_single_stock
from src.utils.df_to_string import df_to_pretty_string
from src.utils.data_path_utils import get_models_dir, get_results_dir, get_results_subdir, ensure_dir

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

def main():
    # すべてのモデルファイルからmarket,symbolの組み合わせを抽出
    model_types = ["StockXGBoostModel.joblib", "StockLightGBMModel.joblib"]
    all_keys = set()
    for model_type in model_types:
        model_files = find_model_files(model_name=model_type)
        for market, symbol, _ in model_files:
            all_keys.add((market, symbol))

    output_rows = []
    for market, symbol in all_keys:
        result = predict_single_stock(market, symbol, model_types=model_types)
        if result is not None:
            output_rows.append(result)

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
                header=f"=== {market} 差異割合上位10銘柄 === 実行日時: {now_str} / ファイル: {top10_path}"
            ))
            df_top10_display.to_csv(top10_path, index=False)

            # ワースト10
            df_worst10 = df_market.sort_values("diff_ratio", ascending=True).head(10)
            df_worst10_display = df_worst10[display_columns]
            worst10_path = os.path.join(result_dir, f"{market}_worst10_diff_stocks.csv")
            print(df_to_pretty_string(
                df_worst10_display,
                header=f"=== {market} 差異割合ワースト10銘柄 === 実行日時: {now_str} / ファイル: {worst10_path}"
            ))
            df_worst10_display.to_csv(worst10_path, index=False)
    else:
        print("有効な結果がありませんでした。")

if __name__ == "__main__":
    main()
