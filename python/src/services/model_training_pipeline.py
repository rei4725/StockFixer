"""
銘柄別モデル学習パイプラインサービス

指定した銘柄のデータを使い、XGBoost・LightGBMモデルを学習・保存する
"""

import re
from src.models.model_manager import ModelManager
from src.utils.db import load_stock_features


def train_models_for_symbol(market: str, symbol: str) -> dict:
    """
    単一銘柄に対してXGBoost・LightGBMモデルを学習・保存する

    Args:
        market: 市場名（例: "us", "jp"）
        symbol: 銘柄コード（例: "AAPL", "7203"）

    Returns:
        dict: {"market", "symbol", "status", ...}
    """
    try:
        print(f"[モデル作成開始] {market}/{symbol}")

        # DBから特徴量データを取得
        df = load_stock_features(market, symbol)

        if df is None or df.empty:
            return {"market": market, "symbol": symbol, "status": "skip", "reason": "データなし"}

        # 文字列列とターゲット列を除外
        exclude_cols = ["y", "market", "symbol"]
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        X = df[feature_cols]
        y = df["y"]

        # 特徴量名の正規化
        def normalize_col(col):
            return re.sub(r'[^0-9a-zA-Z_]', '_', str(col))
        X.columns = [normalize_col(c) for c in X.columns]

        # ModelManagerは各呼び出しで新規作成
        model_manager = ModelManager()

        # XGBoostモデル
        model_manager.create_model("XGBoostModel", "StockXGBoostModel")
        model_manager.train_model("StockXGBoostModel", X, y, market=market, symbol=symbol)

        # LightGBMモデル
        model_manager.create_model("LightGBMModel", "StockLightGBMModel")
        model_manager.train_model("StockLightGBMModel", X, y, market=market, symbol=symbol)

        print(f"[モデル作成完了] {market}/{symbol}")
        return {"market": market, "symbol": symbol, "status": "success"}
    except Exception as e:
        print(f"[モデル作成エラー] {market}/{symbol}: {e}")
        return {"market": market, "symbol": symbol, "status": "error", "error": str(e)}


def train_models_for_symbol_task(task: dict) -> dict:
    """
    バッチランナー用ラッパー（dict引数を展開して呼び出す）

    Args:
        task: {"market": str, "symbol": str}

    Returns:
        dict: train_models_for_symbolの戻り値
    """
    return train_models_for_symbol(task["market"], task["symbol"])
