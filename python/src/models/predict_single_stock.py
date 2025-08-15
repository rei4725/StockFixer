import os
import pandas as pd
import traceback
from src.models.model_manager import ModelManager
from src.data import data_loader
from src.features.technical_analysis import add_technical_indicators, create_basic_lag_features
from src.utils.data_path_utils import get_models_subdir, get_data_dir

def predict_single_stock(market: str, symbol: str, model_types=None, lookback_days=90):
    """
    指定したmarket, symbolについて、複数モデルで予測値を案分（平均）し、現値・予測値・差異割合を返す
    """
    if model_types is None:
        model_types = ["StockXGBoostModel.joblib", "StockLightGBMModel.joblib"]

    pred_prices = []
    current_price = None
    for model_type in model_types:
        model_path = os.path.join(get_models_subdir(market, symbol), model_type)
        if not os.path.exists(model_path):
            # モデルが存在しない場合はデータ取得・特徴量生成・CSV保存・モデル作成・学習・保存を自動実行
            from src.data.data_saver import save_stock_data_with_features
            save_stock_data_with_features(market, symbol, out_dir=get_data_dir())
            df = data_loader.get_stock_data(market, symbol, pd.Timestamp.today() - pd.Timedelta(days=lookback_days), pd.Timestamp.today())
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
            mm = ModelManager(model_dir="python/models")
            model_name = os.path.splitext(os.path.basename(model_path))[0]
            # モデル新規作成・学習・保存
            if "XGBoost" in model_name:
                model_type_name = "XGBoostModel"
            elif "LightGBM" in model_name:
                model_type_name = "LightGBMModel"
            else:
                print(f"[{symbol}] 未知のモデルタイプ: {model_name}")
                continue
            model = mm.create_model(model_type_name, model_name)
            mm.train_model(model_name, X, y, market=market, symbol=symbol)
            # その後ロードして予測
            model = mm.load_model(model_name, market=market, symbol=symbol)
        try:
            df = data_loader.get_stock_data(market, symbol, pd.Timestamp.today() - pd.Timedelta(days=lookback_days), pd.Timestamp.today())
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
            pred_prices.append(pred_price)
        except Exception as e:
            print(f"[{symbol}] エラー: {e}")
            traceback.print_exc()
            continue

    if not pred_prices or current_price is None:
        return None

    avg_pred_price = sum(pred_prices) / len(pred_prices)
    diff_ratio = (avg_pred_price - current_price) / current_price
    # DataFrame形式で返す（run_top10_diff_stocks.pyと同じカラム構成）
    return pd.DataFrame([{
        "market": market,
        "symbol": symbol,
        "current_price": current_price,
        "avg_pred_price": avg_pred_price,
        "diff_ratio": diff_ratio,
        "model_count": len(pred_prices)
    }])
