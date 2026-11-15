"""1銘柄の直近の予想を SHAP で説明する（predict_single から切り出した, #1832）。

予想の値を作る道（predict_single_stock）からは呼ばれない。循環を作らないため、
このモジュールは predict_single から _is_stale_bars を import し、
predict_single はこのモジュールを import しない。
"""

import os

import pandas as pd

from src.prediction.manager import ModelManager
from src.prediction.ports import get_market_data_port
from src.prediction.predict_single import _is_stale_bars
from src.utils.data_path_utils import get_models_subdir, normalize_col
from src.utils.logger import get_logger

logger = get_logger(__name__)


def explain_prediction_shap(
    market: str,
    symbol: str,
    horizon: int = 1,
    top_n: int = 10,
    model_type: str = "StockXGBoostModel",
) -> "dict | None":
    """
    SHAP を使って直近の予測に対する特徴量寄与度を計算する。

    Args:
        market: 市場名
        symbol: 銘柄コード
        horizon: 予測ホライズン（1 = 翌日）
        top_n: 上位何件の特徴量を返すか
        model_type: ベースモデル名（suffixなし）

    Returns:
        dict または None。キー:
            "symbol": str,
            "horizon": int,
            "direction": "UP" | "DOWN",
            "top_features": [{"feature": str, "shap_value": float}, ...] (top_n 件)
        shap ライブラリが未インストールの場合は None を返す。
    """
    try:
        import shap
    except ImportError:
        print(
            "shap ライブラリがインストールされていません。pip install shap>=0.46 で追加してください。"
        )
        return None

    suffix = f"_{horizon}d" if horizon > 1 else ""
    model_filename = f"{model_type}{suffix}.joblib"
    model_path = os.path.join(get_models_subdir(market, symbol), model_filename)

    if not os.path.exists(model_path):
        print(f"[{symbol}] モデルが存在しません: {model_path}")
        return None

    try:
        _mds = get_market_data_port()
        df = _mds.get_stock_data(
            market,
            symbol,
            pd.Timestamp.today() - pd.Timedelta(days=90),
            pd.Timestamp.today(),
        )
        if df.empty or "Close" not in df.columns:
            return None
        if _is_stale_bars(market, symbol, df):
            return None

        df_feat = _mds.add_technical_indicators(df)
        X, _ = _mds.create_basic_lag_features(
            df_feat, target_horizon=horizon, drop_missing_target=False
        )
        if X.empty:
            return None
        # 学習パイプラインと同じ特徴量名正規化・market_encoded 付与
        X = X.copy()
        X.columns = [normalize_col(c) for c in X.columns]
        market_codes = {"us": 0, "jp": 1}
        X["market_encoded"] = market_codes.get(market, -1)
        latest_X = X.iloc[[-1]]

        mm = ModelManager()
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        model = mm.load_model(model_name, market=market, symbol=symbol)

        # SHAP Explainer（XGBoost/LightGBM はツリーベースなので TreeExplainer が最速）
        raw_model = model.model if hasattr(model, "model") else model
        explainer = shap.TreeExplainer(raw_model)
        shap_values = explainer.shap_values(latest_X)

        if hasattr(shap_values, "values"):
            vals = shap_values.values[0]
        elif isinstance(shap_values, list):
            vals = shap_values[0][0]
        else:
            vals = shap_values[0]

        feature_names = list(latest_X.columns)
        pairs = sorted(zip(feature_names, vals), key=lambda x: abs(x[1]), reverse=True)

        pred_return = float(model.predict(latest_X)[0])
        direction = "UP" if pred_return >= 0 else "DOWN"

        return {
            "symbol": symbol,
            "horizon": horizon,
            "direction": direction,
            "top_features": [
                {"feature": name, "shap_value": float(val)} for name, val in pairs[:top_n]
            ],
        }
    except Exception:
        logger.error("[%s] SHAP計算エラー: horizon=%s", symbol, horizon, exc_info=True)
        return None
