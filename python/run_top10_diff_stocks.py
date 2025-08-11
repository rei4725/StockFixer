import os
import glob
import pandas as pd
import traceback
from src.models.model_manager import ModelManager
from src.data.data_loader import get_stock_data
from src.features.technical_analysis import add_technical_indicators, create_basic_lag_features

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
    model_types = ["StockXGBoostModel.joblib", "StockLightGBMModel.joblib"]
    symbol_results = {}
    for model_type in model_types:
        model_files = find_model_files(model_name=model_type)
        for market, symbol, model_path in model_files:
            try:
                df = get_stock_data(symbol, pd.Timestamp.today() - pd.Timedelta(days=90), pd.Timestamp.today())
                if df.empty or "Close" not in df.columns:
                    print(f"[{symbol}] 株価データ取得失敗")
                    continue
                current_price = df["Close"].iloc[-1]
                df_feat = add_technical_indicators(df)
                X, y = create_basic_lag_features(df_feat)
                if X.empty:
                    print(f"[{symbol}] 特徴量生成失敗")
                    continue
                latest_X = X.iloc[[-1]]
                mm = ModelManager()
                model_name = os.path.splitext(os.path.basename(model_path))[0]
                model = mm.load_model(model_name, market=market, symbol=symbol)
                pred = model.predict(latest_X)
                if isinstance(pred, (pd.Series, list, tuple)):
                    pred_price = float(pred[-1])
                else:
                    pred_price = float(pred)
                key = (market, symbol)
                if key not in symbol_results:
                    symbol_results[key] = {
                        "current_price": current_price,
                        "pred_prices": [],
                    }
                # 既にcurrent_priceがセットされている場合は上書きしない
                symbol_results[key]["pred_prices"].append(pred_price)
            except Exception as e:
                print(f"[{symbol}] エラー: {e}")
                traceback.print_exc()
                continue

    # symbolごとに予想株価を案分（平均）し、差異割合を計算
    output_rows = []
    for (market, symbol), vals in symbol_results.items():
        pred_prices = vals["pred_prices"]
        if not pred_prices:
            continue
        avg_pred_price = sum(pred_prices) / len(pred_prices)
        current_price = vals["current_price"]
        diff_ratio = (avg_pred_price - current_price) / current_price
        output_rows.append({
            "market": market,
            "symbol": symbol,
            "current_price": current_price,
            "avg_pred_price": avg_pred_price,
            "diff_ratio": diff_ratio,
            "model_count": len(pred_prices)
        })

    df_result = pd.DataFrame(output_rows)
    if not df_result.empty:
        df_result = df_result.sort_values("diff_ratio", ascending=False).head(10)
        print("=== 差異割合上位10銘柄（モデル案分）===")
        print(df_result[["market", "symbol", "current_price", "avg_pred_price", "diff_ratio", "model_count"]])
        df_result[["market", "symbol", "current_price", "avg_pred_price", "diff_ratio", "model_count"]].to_csv("top10_diff_stocks.csv", index=False)
    else:
        print("有効な結果がありませんでした。")

if __name__ == "__main__":
    main()
