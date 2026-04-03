import logging
import os
import traceback
import warnings

import numpy as np
import pandas as pd
import yfinance as yf

from src.data import data_loader
from src.domain.types import PredictionResult
from src.features.technical_analysis import add_technical_indicators, create_basic_lag_features
from src.models.model_manager import ModelManager
from src.utils.data_path_utils import get_models_subdir, get_ticker

# yfinanceの警告を抑制
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("yfinance").setLevel(logging.CRITICAL)


def _resolve_model_types(horizon: int, model_types=None) -> list:
    """
    ホライズンに応じたモデルファイル名リストを解決する（後方互換）。

    Args:
        horizon: 予測ホライズン（営業日）
        model_types: 明示指定がある場合はそのまま返す

    Returns:
        モデルファイル名のリスト
    """
    if model_types is not None:
        return list(model_types)
    if horizon == 1:
        return ["StockXGBoostModel.joblib", "StockLightGBMModel.joblib"]
    return [
        f"StockXGBoostModel_{horizon}d.joblib",
        f"StockLightGBMModel_{horizon}d.joblib",
    ]


def _fetch_current_price(market: str, symbol: str, df: pd.DataFrame) -> "float | None":
    """
    yfinanceでリアルタイム価格を取得し、失敗時はdfの末尾Close値を使用する。

    Args:
        market: マーケット識別子
        symbol: 銘柄コード
        df: 株価DataFrame（fallback用）

    Returns:
        現在価格（float）。取得不能な場合は None。
    """
    try:
        yf_ticker = get_ticker(market, symbol)
        ticker_obj = yf.Ticker(yf_ticker)
        hist = ticker_obj.history(period="1d")
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
    except Exception:
        pass
    # yfinance失敗時はdfの末尾を使用（fallback）
    try:
        close_col = df["Close"]
        if isinstance(close_col, pd.DataFrame):
            close_col = close_col.iloc[:, 0]
        return float(close_col.iloc[-1])
    except Exception:
        return None


def _run_single_model_prediction(
    model_path: str,
    market: str,
    symbol: str,
    df: pd.DataFrame,
    current_price: float,
) -> "tuple[float, float] | None":
    """
    1つのモデルで予測を行い (pred_price, pred_return) を返す。

    Args:
        model_path: モデルファイルパス
        market: マーケット識別子
        symbol: 銘柄コード
        df: 株価DataFrame
        current_price: 現在価格（絶対価格計算に使用）

    Returns:
        (pred_price, pred_return) のタプル。失敗時は None。
    """
    df_feat = add_technical_indicators(df)
    X, _ = create_basic_lag_features(df_feat)
    if X.empty:
        return None
    latest_X = X.iloc[[-1]]
    mm = ModelManager()
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    model = mm.load_model(model_name, market=market, symbol=symbol)
    pred = model.predict(latest_X)
    if isinstance(pred, pd.Series):
        pred_return = float(pred.iloc[-1])
    elif isinstance(pred, (list, tuple)):
        pred_return = float(pred[-1])
    else:
        # numpy配列など: flat[-1] で末尾要素をスカラー変換（ndim>0 の場合の警告を回避）
        pred_return = float(np.asarray(pred).flat[-1])
    pred_price = current_price * (1 + pred_return)
    return pred_price, pred_return


def _build_prediction_result(
    market: str,
    symbol: str,
    current_price: "float | None",
    pred_prices: list,
    pred_returns: list,
) -> "PredictionResult | None":
    """
    複数モデルの予測を集計して PredictionResult を構築する（純粋関数）。

    Args:
        market: マーケット識別子
        symbol: 銘柄コード
        current_price: 現在価格
        pred_prices: 各モデルの予測絶対価格リスト
        pred_returns: 各モデルの予測変化率リスト

    Returns:
        集計済み PredictionResult。pred_prices が空の場合は None。
    """
    if not pred_prices or current_price is None:
        return None
    avg_pred_price = sum(pred_prices) / len(pred_prices)
    diff_ratio = (avg_pred_price - current_price) / current_price
    model_std = float(np.std(pred_returns)) if len(pred_returns) > 1 else 0.0
    confidence_ratio = 1.0 / (1.0 + model_std)
    return PredictionResult(
        market=market,
        symbol=symbol,
        current_price=float(current_price),
        avg_pred_price=float(avg_pred_price),
        diff_ratio=float(diff_ratio),
        model_count=int(len(pred_prices)),
        confidence_ratio=confidence_ratio,
    )


def predict_single_stock(
    market: str, symbol: str, model_types=None, lookback_days=90, horizon: int = 1
) -> "PredictionResult | None":
    """
    指定したmarket, symbolについて、複数モデルで予測値を案分（平均）し、現値・予測値・差異割合を返す

    Args:
        market: マーケット掲別子
        symbol: 銃柔コード
        model_types: 使用するモデルファイル名のリスト（デフォルト: None → 自動設定）
        lookback_days: データ取得日数
        horizon: 予測ホライズン（営業日）。1=翌日, 3=3日後, 5=5日後, 10=10日後
    """
    resolved_model_types = _resolve_model_types(horizon, model_types)

    pred_prices = []
    pred_returns = []
    current_price = None
    for model_type in resolved_model_types:
        model_path = os.path.join(get_models_subdir(market, symbol), model_type)
        if not os.path.exists(model_path):
            # モデルが存在しない場合はスキップ（並列実行時のDB書き込みロック回避）
            # モデル作成は run_model_creation.py で事前に実行すること
            print(f"[{symbol}] モデルが存在しません: {model_path}")
            continue
        try:
            df = data_loader.get_stock_data(
                market,
                symbol,
                pd.Timestamp.today() - pd.Timedelta(days=lookback_days),
                pd.Timestamp.today(),
            )
            if df.empty or "Close" not in df.columns:
                print(f"[{symbol}] 株価データ取得失敗")
                continue
            current_price = _fetch_current_price(market, symbol, df)
            if current_price is None:
                print(f"[{symbol}] 価格取得失敗")
                continue
            result = _run_single_model_prediction(model_path, market, symbol, df, current_price)
            if result is None:
                print(f"[{symbol}] 特徴量生成失敗")
                continue
            pred_price, pred_return = result
            pred_prices.append(pred_price)
            pred_returns.append(pred_return)
        except Exception as e:
            print(f"[{symbol}] エラー: {e}")
            traceback.print_exc()
            continue

    return _build_prediction_result(market, symbol, current_price, pred_prices, pred_returns)


def predict_single_stock_multi_horizon(
    market: str, symbol: str, horizons: list = None
) -> "PredictionResult | None":
    """
    複数ホライズンで予測し、各ホライズンの値を横に結合して返す。

    Args:
        market: マーケット識別子
        symbol: 銘柄コード
        horizons: 予測ホライズンのリスト。デフォルト: [1, 3, 5, 10]

    Returns:
        PredictionResult：各ホライズンの予測値を含む。全ホライズン失敗時は None。
    """
    if horizons is None:
        horizons = [1, 3, 5, 10]

    results: dict = {}
    for h in horizons:
        r = predict_single_stock(market, symbol, horizon=h)
        if r is not None:
            results[h] = r

    if not results:
        return None

    base_h = 1 if 1 in results else min(results.keys())
    base = results[base_h]

    base_dir = base.diff_ratio > 0
    return PredictionResult(
        market=base.market,
        symbol=base.symbol,
        current_price=base.current_price,
        avg_pred_price=base.avg_pred_price,
        diff_ratio=base.diff_ratio,
        model_count=base.model_count,
        avg_pred_price_3d=results[3].avg_pred_price if 3 in results else None,
        avg_pred_price_5d=results[5].avg_pred_price if 5 in results else None,
        avg_pred_price_10d=results[10].avg_pred_price if 10 in results else None,
        diff_ratio_3d=results[3].diff_ratio if 3 in results else None,
        diff_ratio_5d=results[5].diff_ratio if 5 in results else None,
        diff_ratio_10d=results[10].diff_ratio if 10 in results else None,
        confluence_score=sum(
            1 for h in horizons if h in results and (results[h].diff_ratio > 0) == base_dir
        ),
    )


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
        print("shap ライブラリがインストールされていません。pip install shap>=0.46 で追加してください。")
        return None

    suffix = f"_{horizon}d" if horizon > 1 else ""
    model_filename = f"{model_type}{suffix}.joblib"
    model_path = os.path.join(get_models_subdir(market, symbol), model_filename)

    if not os.path.exists(model_path):
        print(f"[{symbol}] モデルが存在しません: {model_path}")
        return None

    try:
        df = data_loader.get_stock_data(
            market,
            symbol,
            pd.Timestamp.today() - pd.Timedelta(days=90),
            pd.Timestamp.today(),
        )
        if df.empty or "Close" not in df.columns:
            return None

        df_feat = add_technical_indicators(df)
        X, _ = create_basic_lag_features(df_feat, target_horizon=horizon)
        if X.empty:
            return None

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
    except Exception as e:
        print(f"[{symbol}] SHAP計算エラー: {e}")
        traceback.print_exc()
        return None
