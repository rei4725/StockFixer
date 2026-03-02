"""
統合モデルパイプラインサービス

全銘柄のデータを結合して1つのモデルを学習するサービス
"""

import os
from typing import Tuple

import pandas as pd
from src.utils.data_path_utils import ensure_dir, get_models_dir
from src.utils.db import load_all_stock_features


def load_all_stock_data(data_dir: str = None) -> pd.DataFrame:
    """
    全銘柄のデータをDBから読み込み、結合して返す

    Args:
        data_dir: 後方互換のため残置（未使用、DBから直接取得）

    Returns:
        pd.DataFrame: 全銘柄の結合データ（market, symbol列付き）
    """
    df = load_all_stock_features()
    if df.empty:
        return pd.DataFrame()

    # 銘柄数カウント
    if "market" in df.columns and "symbol" in df.columns:
        n_symbols = df.groupby(["market", "symbol"]).ngroups
        print(f"合計: {len(df)}行, {n_symbols}銘柄")

    return df


def prepare_unified_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    統合モデル用の特徴量を準備する

    Args:
        df: 全銘柄の結合データ

    Returns:
        X: 特徴量DataFrame
        y: ターゲットSeries
    """
    # market_encoded列がない場合のみ作成（後方互換性）
    if "market_encoded" not in df.columns and "market" in df.columns:
        market_codes = {"us": 0, "jp": 1}
        df["market_encoded"] = df["market"].map(market_codes).fillna(-1).astype(int)

    # y列があるか確認
    if "y" in df.columns:
        y = df["y"]
        # 特徴量から除外する列（文字列列とターゲット）
        exclude_cols = ["market", "symbol", "y"]
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        X = df[feature_cols]
    else:
        # y列がない場合は作成（翌日終値）
        raise ValueError("y列がデータに含まれていません。特徴量生成済みのCSVを使用してください。")

    # NaNを除去
    valid_idx = X.notna().all(axis=1) & y.notna()
    X = X[valid_idx]
    y = y[valid_idx]

    return X, y


def train_unified_model(
    model_type: str = "XGBoostModel",
    model_name: str = "UnifiedStockModel",
    data_dir: str = None,
    save_dir: str = None,
) -> None:
    """
    全銘柄のデータを使って統合モデルを学習・保存する

    Args:
        model_type: モデルタイプ ("XGBoostModel" or "LightGBMModel")
        model_name: 保存するモデル名
        data_dir: データディレクトリ
        save_dir: モデル保存ディレクトリ
    """
    from src.models.model_manager import ModelManager

    print("=== 統合モデル学習開始 ===")

    # 全データ読み込み
    print("\n1. データ読み込み中...")
    df = load_all_stock_data(data_dir)
    if df.empty:
        print("データが見つかりません。")
        return

    # 特徴量準備
    print("\n2. 特徴量準備中...")
    X, y = prepare_unified_features(df)
    print(f"特徴量数: {len(X.columns)}, サンプル数: {len(X)}")

    # モデル作成・学習
    print(f"\n3. モデル学習中 ({model_type})...")
    mm = ModelManager(model_dir=save_dir)
    model = mm.create_model(model_type, model_name)
    model.train(X, y)

    # 保存（unified/ディレクトリに保存）
    print("\n4. モデル保存中...")
    if save_dir is None:
        save_dir = os.path.join(get_models_dir(), "unified")
    ensure_dir(save_dir)

    save_path = os.path.join(save_dir, f"{model_name}.joblib")
    model.save_model(save_path)
    print(f"保存完了: {save_path}")

    print("\n=== 統合モデル学習完了 ===")


def load_unified_model(model_name: str = "UnifiedStockModel", model_dir: str = None):
    """
    統合モデルをロードする

    Args:
        model_name: モデル名
        model_dir: モデルディレクトリ

    Returns:
        ロードされたモデル
    """
    from src.models.model_manager import ModelManager

    if model_dir is None:
        model_dir = os.path.join(get_models_dir(), "unified")

    mm = ModelManager(model_dir=model_dir)

    # XGBoostかLightGBMかを判定
    xgb_path = os.path.join(model_dir, f"{model_name}.joblib")
    if os.path.exists(xgb_path):
        # モデルタイプを判定するためファイル名から推測
        if "LightGBM" in model_name:
            model = mm.create_model("LightGBMModel", model_name)
        else:
            model = mm.create_model("XGBoostModel", model_name)
        model.load_model(xgb_path)
        return model

    raise FileNotFoundError(f"統合モデルが見つかりません: {xgb_path}")
