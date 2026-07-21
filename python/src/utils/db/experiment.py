"""
experiment_runs テーブルの CRUD 操作（R-211 実験トラッキング基盤）

学習ごとに run_id・パラメータ・メトリクス・特徴量来歴を自動記録する。
R-206（Optuna）・R-207（A/B テスト）のインフラ共有テーブルとして機能する。
"""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime
from typing import Optional

import pandas as pd

from src.utils.db._connection import _db_connection
from src.utils.logger import get_logger

logger = get_logger(__name__)

_ALLOWED_METRICS = frozenset({"directional_accuracy", "rmse"})


def generate_run_id() -> str:
    """一意の run_id を生成する（UUID v4）。"""
    return str(uuid.uuid4())


def _feature_hash(feature_names: list[str]) -> str:
    """特徴量セットの来歴を追跡するためのハッシュを返す（SHA-256 先頭16文字）。"""
    joined = ",".join(sorted(feature_names))
    return hashlib.sha256(joined.encode()).hexdigest()[:16]


def save_experiment_run(
    run_id: str,
    market: str,
    symbol: str,
    model_name: str,
    trained_at: str,
    horizon: int = 1,
    rmse: Optional[float] = None,
    directional_accuracy: Optional[float] = None,
    n_samples: Optional[int] = None,
    feature_names: Optional[list[str]] = None,
    params: Optional[dict] = None,
) -> None:
    """
    実験ランを experiment_runs テーブルに保存する。

    Args:
        run_id: 一意の実験ID（generate_run_id() で生成）
        market: マーケット識別子
        symbol: 銘柄シンボル
        model_name: モデル名
        trained_at: 学習日時文字列 (例: "20260314_120000")
        horizon: 予測ホライズン（営業日）
        rmse: 学習データでの RMSE
        directional_accuracy: 方向正解率
        n_samples: 学習サンプル数
        feature_names: 使用した特徴量名リスト
        params: モデルハイパーパラメータ等の任意辞書
    """
    n_features = len(feature_names) if feature_names else None
    feat_hash = _feature_hash(feature_names) if feature_names else None
    params_json = json.dumps(params, ensure_ascii=False) if params else None

    with _db_connection() as con:
        con.execute(
            """
            INSERT INTO experiment_runs
                (run_id, market, symbol, model_name, trained_at, horizon,
                 rmse, directional_accuracy, n_samples, n_features,
                 feature_hash, params_json, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (run_id) DO UPDATE SET
                market = EXCLUDED.market,
                symbol = EXCLUDED.symbol,
                model_name = EXCLUDED.model_name,
                trained_at = EXCLUDED.trained_at,
                horizon = EXCLUDED.horizon,
                rmse = EXCLUDED.rmse,
                directional_accuracy = EXCLUDED.directional_accuracy,
                n_samples = EXCLUDED.n_samples,
                n_features = EXCLUDED.n_features,
                feature_hash = EXCLUDED.feature_hash,
                params_json = EXCLUDED.params_json
            """,
            [
                run_id,
                market,
                symbol,
                model_name,
                trained_at,
                horizon,
                rmse,
                directional_accuracy,
                n_samples,
                n_features,
                feat_hash,
                params_json,
                datetime.now(),
            ],
        )
    logger.debug(
        f"experiment_run 保存: run_id={run_id} [{market}/{symbol}/{model_name}] "
        f"RMSE={rmse} DA={directional_accuracy} n_feat={n_features} hash={feat_hash}"
    )


def load_experiment_runs(
    market: Optional[str] = None,
    symbol: Optional[str] = None,
    model_name: Optional[str] = None,
    limit: int = 100,
) -> pd.DataFrame:
    """
    experiment_runs テーブルから実験ランを取得する。

    Args:
        market: マーケットフィルタ（None なら全マーケット）
        symbol: 銘柄フィルタ（None なら全銘柄）
        model_name: モデル名フィルタ（None なら全モデル）
        limit: 取得件数上限（created_at 降順）

    Returns:
        pd.DataFrame
    """
    query = "SELECT * FROM experiment_runs WHERE 1=1"
    params: list = []
    if market is not None:
        query += " AND market = %s"
        params.append(market)
    if symbol is not None:
        query += " AND symbol = %s"
        params.append(symbol)
    if model_name is not None:
        query += " AND model_name = %s"
        params.append(model_name)
    query += f" ORDER BY created_at DESC LIMIT {int(limit)}"

    with _db_connection() as con:
        try:
            return pd.read_sql(query, con, params=params)
        except Exception as e:
            logger.error(f"experiment_runs 読み込み失敗: {e}", exc_info=True)
            return pd.DataFrame()


def load_best_run(
    market: str,
    symbol: str,
    model_name: str,
    metric: str = "directional_accuracy",
) -> Optional[dict]:
    """
    指定銘柄・モデルの最良ランを返す。

    Args:
        market: マーケット識別子
        symbol: 銘柄シンボル
        model_name: モデル名
        metric: 比較指標（"directional_accuracy" または "rmse"）
                directional_accuracy は大きい方が良い / rmse は小さい方が良い

    Returns:
        最良ランの dict、なければ None
    """
    if metric not in _ALLOWED_METRICS:
        raise ValueError(f"metric must be one of {sorted(_ALLOWED_METRICS)}, got: {metric!r}")
    order = "DESC" if metric == "directional_accuracy" else "ASC"
    with _db_connection() as con:
        try:
            row = pd.read_sql(
                f"""
                SELECT * FROM experiment_runs
                WHERE market = %s AND symbol = %s AND model_name = %s
                  AND {metric} IS NOT NULL
                ORDER BY {metric} {order}
                LIMIT 1
                """,
                con,
                params=[market, symbol, model_name],
            )
            if row.empty:
                return None
            return row.iloc[0].to_dict()
        except Exception as e:
            logger.error(f"load_best_run 失敗 [{market}/{symbol}/{model_name}]: {e}", exc_info=True)
            return None
