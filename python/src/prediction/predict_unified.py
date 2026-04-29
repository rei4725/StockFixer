"""
統合モデル用予測処理

統合モデルを使用して全銘柄の予測を行う
"""

import logging
import warnings
from threading import Lock
from typing import Any, Dict, List, Optional

import pandas as pd
import yfinance as yf

from src.prediction.types import PredictionResult
from src.utils.data_path_utils import get_ticker
from src.utils.db import get_all_symbols, load_model_weights, load_stock_features
from src.utils.logger import get_logger

logger = get_logger(__name__)

# yfinanceの警告を抑制
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("yfinance").setLevel(logging.CRITICAL)

# モデルキャッシュ（スレッドセーフ）
_model_cache: Dict[str, Any] = {}
_model_cache_lock = Lock()


def load_feature_data(market: str, symbol: str) -> Optional[pd.DataFrame]:
    """
    特徴量データをDBから読み込む

    Args:
        market: 市場名
        symbol: 銘柄コード

    Returns:
        特徴量DataFrame または None
    """
    try:
        df = load_stock_features(market, symbol)
        return df
    except Exception:
        logger.warning("特徴量データ読み込み失敗: market=%s symbol=%s", market, symbol, exc_info=True)
        return None


def get_cached_model(model_name: str):
    """
    キャッシュからモデルを取得（なければロード）
    スレッドセーフな実装
    """
    from src.services.unified_model_pipeline import load_unified_model

    with _model_cache_lock:
        if model_name not in _model_cache:
            try:
                _model_cache[model_name] = load_unified_model(model_name)
            except FileNotFoundError:
                _model_cache[model_name] = None
        return _model_cache[model_name]


def preload_models(model_types: List[str] = None):
    """
    モデルを事前にロードしてキャッシュする
    並列処理の前に呼び出すことで、スレッド間でモデルを共有できる
    """
    if model_types is None:
        model_types = ["UnifiedStockXGBoost", "UnifiedStockLightGBM"]

    logger.info("モデルを事前ロード中: %s", model_types)
    for model_name in model_types:
        model = get_cached_model(model_name)
        if model is not None:
            logger.info("  - %s: ロード完了", model_name)
        else:
            logger.info("  - %s: 見つかりません", model_name)
    logger.info("モデルの事前ロード完了")


def predict_with_unified_model(
    market: str,
    symbol: str,
    model_types: List[str] = None,
    lookback_days: int = 90,
    horizon: int = 1,
) -> Optional[PredictionResult]:
    """
    統合モデルを使用して1銘柄の予測を行う

    Args:
        market: 市場名
        symbol: 銘柄コード
        model_types: 使用するモデル名のリスト
        lookback_days: データ取得日数（未使用、互換性のため残す）
        horizon: 予測ホライズン（営業日）。1=翌日, 3=3日後, 5=5日後, 10=10日後。

    Returns:
        予測結果のDataFrame または None
    """
    if model_types is None:
        if horizon == 1:
            model_types = ["UnifiedStockXGBoost", "UnifiedStockLightGBM"]
        else:
            model_types = [
                f"UnifiedStockXGBoost_{horizon}d",
                f"UnifiedStockLightGBM_{horizon}d",
            ]

    # 特徴量データ読み込み
    df = load_feature_data(market, symbol)
    if df is None or df.empty:
        return None

    # y列（予測ターゲット）と特徴量を分離
    if "y" not in df.columns:
        return None

    # 特徴量列（文字列列とyを除外）
    exclude_cols = ["y", "market", "symbol"]
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    X = df[feature_cols]

    # 最新行を取得（必ずコピーを作成）
    latest_X = X.iloc[[-1]].copy()

    # 現在価格をyfinanceからリアルタイムで取得
    try:
        yf_ticker = get_ticker(market, symbol)
        ticker_obj = yf.Ticker(yf_ticker)
        hist = ticker_obj.history(period="1d")
        if not hist.empty:
            current_price = float(hist["Close"].iloc[-1])
        else:
            # フォールバック: CSVのClose_lag1を使用
            if "Close_lag1" in df.columns:
                current_price = float(df["Close_lag1"].iloc[-1])
            else:
                current_price = float(df["y"].iloc[-2]) if len(df) > 1 else float(df["y"].iloc[-1])
    except Exception:
        logger.warning("現在価格取得失敗（フォールバック使用）: market=%s symbol=%s", market, symbol, exc_info=True)
        # エラー時はフォールバック
        if "Close_lag1" in df.columns:
            current_price = float(df["Close_lag1"].iloc[-1])
        else:
            current_price = float(df["y"].iloc[-2]) if len(df) > 1 else float(df["y"].iloc[-1])

    # market_encoded列がない場合のみ追加（後方互換性）
    if "market_encoded" not in latest_X.columns:
        market_codes = {"us": 0, "jp": 1}
        latest_X["market_encoded"] = market_codes.get(market, 0)

    # 各モデルで予測（キャッシュされたモデルを使用）
    pred_prices = []
    succeeded_model_names = []
    for model_name in model_types:
        try:
            model = get_cached_model(model_name)
            if model is None:
                continue

            # モデルの特徴量と入力特徴量を揃える（コピーを作成して変更）
            if hasattr(model, "model") and hasattr(model.model, "feature_names_in_"):
                expected_features = list(model.model.feature_names_in_)
                latest_X_aligned = latest_X.copy()
                # 不足している特徴量は0で埋める
                for feat in expected_features:
                    if feat not in latest_X_aligned.columns:
                        latest_X_aligned[feat] = 0
                # 期待される特徴量のみ選択
                latest_X_aligned = latest_X_aligned[expected_features]
            else:
                latest_X_aligned = latest_X.copy()

            pred = model.predict(latest_X_aligned)
            if isinstance(pred, pd.Series):
                pred_return = float(pred.iloc[-1])
            elif isinstance(pred, (list, tuple)):
                pred_return = float(pred[-1])
            else:
                pred_return = float(pred)
            # 変化率から絶対価格を計算
            pred_price = current_price * (1 + pred_return)
            pred_prices.append(pred_price)
            succeeded_model_names.append(model_name)
        except Exception:
            logger.warning(
                "モデル予測スキップ: model=%s market=%s symbol=%s", model_name, market, symbol, exc_info=True
            )
            continue

    if not pred_prices:
        return None

    # model_metrics の directional_accuracy をソフトマックス重みでアンサンブル（R-202）
    weights = load_model_weights(market, symbol, succeeded_model_names)
    avg_pred_price = float(sum(p * w for p, w in zip(pred_prices, weights)))
    diff_ratio = (avg_pred_price - current_price) / current_price

    return PredictionResult(
        market=market,
        symbol=symbol,
        current_price=float(current_price),
        avg_pred_price=float(avg_pred_price),
        diff_ratio=float(diff_ratio),
        model_count=int(len(pred_prices)),
    )


def predict_all_with_unified_model(
    model_types: List[str] = None, data_dir: str = None
) -> pd.DataFrame:
    """
    全銘柄について統合モデルで予測を行う

    Args:
        model_types: 使用するモデル名のリスト
        data_dir: データディレクトリ

    Returns:
        全銘柄の予測結果DataFrame
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if model_types is None:
        model_types = ["UnifiedStockXGBoost", "UnifiedStockLightGBM"]

    # モデルを事前にロード
    preload_models(model_types)

    if data_dir is None:
        pass  # DBから直接取得するため不要

    # 全銘柄をDBから取得
    all_keys = get_all_symbols()

    print(f"予測対象: {len(all_keys)}銘柄")

    # 並列予測
    def predict_wrapper(args):
        market, symbol = args
        try:
            return predict_with_unified_model(market, symbol, model_types=model_types)
        except Exception:
            logger.warning("銘柄予測失敗: market=%s symbol=%s", market, symbol, exc_info=True)
            return None

    results = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(predict_wrapper, key): key for key in all_keys}
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                results.append(result)

    if results:
        return pd.concat(results, ignore_index=True)
    return pd.DataFrame()
