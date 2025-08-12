import os
import glob
import pandas as pd
from datetime import datetime
from src.models.predict_single_stock import predict_single_stock
from src.utils.df_to_string import df_to_pretty_string

def find_model_files(model_root="python/models", model_name="StockXGBoostModel.joblib"):
    """
    モデルディレクトリを再帰的に探索し、(market, symbol, model_path)のリストを返す
    """
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
        df_result = df_result.sort_values("diff_ratio", ascending=False).head(10)
        # 共通で使うカラムのみ抽出
        display_columns = ["market", "symbol", "current_price", "avg_pred_price", "diff_ratio", "model_count"]
        df_display = df_result[display_columns]

        now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"python/results/{now_str}_top10_diff_stocks.csv"

        print(df_to_pretty_string(
            df_display,
            header=f"=== 差異割合上位10銘柄（モデル案分）=== 実行日時: {now_str} / ファイル: {output_path}"
        ))
        # 出力先を python/results フォルダに変更
        os.makedirs("python/results", exist_ok=True)
        df_display.to_csv(output_path, index=False)
    else:
        print("有効な結果がありませんでした。")

if __name__ == "__main__":
    main()
