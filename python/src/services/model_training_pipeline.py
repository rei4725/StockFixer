"""
銘柄別モデル学習パイプラインサービス

指定した銘柄のデータを使い、XGBoost・LightGBMモデルを学習・保存する
"""

import re

from src.models.model_manager import ModelManager
from src.utils.db import load_stock_features
from src.utils.logger import get_logger

logger = get_logger(__name__)


def load_features_for_training(market: str, symbol: str) -> dict:
    """
    学習用の特徴量データをDBから読み込む（DB書き込みなし、並列安全）。

    Args:
        market: 市場名（例: "us", "jp"）
        symbol: 銘柄コード（例: "AAPL", "7203"）

    Returns:
        dict: {"market", "symbol", "status", "X", "y"}
    """
    try:
        logger.info(f"[データ読み込み] {market}/{symbol}")
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
            return re.sub(r"[^0-9a-zA-Z_]", "_", str(col))

        X.columns = [normalize_col(c) for c in X.columns]

        return {"market": market, "symbol": symbol, "status": "success", "X": X, "y": y}
    except Exception as e:
        logger.error(f"[データ読み込みエラー] {market}/{symbol}: {e}", exc_info=True)
        return {"market": market, "symbol": symbol, "status": "error", "error": str(e)}


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
        logger.info(f"[モデル作成開始] {market}/{symbol}")

        # DBから特徴量データを取得
        loaded = load_features_for_training(market, symbol)
        if loaded["status"] != "success":
            return loaded

        X, y = loaded["X"], loaded["y"]

        # ModelManagerは各呼び出しで新規作成
        model_manager = ModelManager()

        # XGBoostモデル
        model_manager.create_model("XGBoostModel", "StockXGBoostModel")
        model_manager.train_model("StockXGBoostModel", X, y, market=market, symbol=symbol)

        # LightGBMモデル
        model_manager.create_model("LightGBMModel", "StockLightGBMModel")
        model_manager.train_model("StockLightGBMModel", X, y, market=market, symbol=symbol)

        logger.info(f"[モデル作成完了] {market}/{symbol}")
        return {"market": market, "symbol": symbol, "status": "success"}
    except Exception as e:
        logger.error(f"[モデル作成エラー] {market}/{symbol}: {e}", exc_info=True)
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


def run_model_batch():
    """
    ウォッチリストの全銘柄のモデルを作成する。

    フェーズ1: DB読み込み（並列） - DuckDB読み取りはスレッド並列で安全
    フェーズ2: モデル学習・保存（逐次） - ファイルI/Oの競合を回避
    """
    from src.services.batch_runner import load_target_symbols, print_summary, run_parallel

    # バッチ作成の並列数（CPU数に応じて調整）
    MAX_MODEL_WORKERS = 3

    def _load_features_task(task: dict) -> dict:
        """バッチランナー用: DB読み込みのみ（並列安全）"""
        return load_features_for_training(task["market"], task["symbol"])

    symbols = load_target_symbols()
    if not symbols:
        logger.warning("対象銘柄がありません。")
        return

    # フェーズ1: データ読み込み（並列）
    load_results = run_parallel(
        func=_load_features_task,
        tasks=symbols,
        max_workers=MAX_MODEL_WORKERS,
        label="データ読み込み",
    )

    # フェーズ2: モデル学習・保存（逐次）
    success_data = [r for r in load_results if r.get("status") == "success" and "X" in r]
    print(f"\n{'='*50}")
    print("モデル学習開始（逐次）")
    print(f"対象件数: {len(success_data)}")
    print(f"{'='*50}\n")

    train_results = []
    for i, r in enumerate(success_data, 1):
        market, symbol = r["market"], r["symbol"]
        try:
            print(f"[モデル作成開始] {market}/{symbol}")
            model_manager = ModelManager()
            model_manager.create_model("XGBoostModel", "StockXGBoostModel")
            model_manager.train_model(
                "StockXGBoostModel", r["X"], r["y"], market=market, symbol=symbol
            )
            model_manager.create_model("LightGBMModel", "StockLightGBMModel")
            model_manager.train_model(
                "StockLightGBMModel", r["X"], r["y"], market=market, symbol=symbol
            )
            print(f"[モデル作成完了] {market}/{symbol}")
            train_results.append({"market": market, "symbol": symbol, "status": "success"})
        except Exception as e:
            print(f"[モデル作成エラー] {market}/{symbol}: {e}")
            train_results.append(
                {"market": market, "symbol": symbol, "status": "error", "error": str(e)}
            )
        if i % 50 == 0:
            print(f"  ... {i}/{len(success_data)} 件完了")

    # 最終サマリー
    final_results = train_results.copy()
    final_results += [r for r in load_results if r.get("status") in ("error", "skip")]
    print_summary("モデル作成", final_results)
