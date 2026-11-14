"""model_metrics テーブル: モデル精度指標の保存とアンサンブル重み算出。"""

import threading

from src.prediction.types import TrainingMetrics
from src.utils.db._connection import _db_connection
from src.utils.logger import get_logger

logger = get_logger(__name__)

# 統合モデルの名前（UnifiedStockXGBoost・UnifiedStockLightGBM と _{h}d つき）の頭。
# 統合モデルの学習は model_metrics に書かず、prediction_accuracy の model_name は
# production・challenger・unified だけなので、この名前で引いても当たらない（#1806）。
_UNIFIED_MODEL_PREFIX = "UnifiedStock"

# 等分を INFO で出したモデル名の組（プロセスで1回だけ出す。銘柄ごとには出さない）
_EQUAL_WEIGHTS_LOGGED: set[tuple[str, ...]] = set()
_EQUAL_WEIGHTS_LOCK = threading.Lock()


def _unified_equal_weights(model_names: list[str]) -> list[float]:
    """統合モデルの重みを等分で返し、その名前の組で初めてのときだけ INFO を出す。"""
    key = tuple(model_names)
    with _EQUAL_WEIGHTS_LOCK:
        first = key not in _EQUAL_WEIGHTS_LOGGED
        _EQUAL_WEIGHTS_LOGGED.add(key)
    if first:
        logger.info(
            "統合モデルのアンサンブルの重みは等分（DB に部品のモデルごとの誤差が無いため）: "
            "models=%s",
            ",".join(model_names),
        )
    n = len(model_names)
    return [1.0 / n] * n


def save_model_metrics(
    market: str,
    symbol: str,
    model_name: str,
    trained_at: str,
    metrics: TrainingMetrics,
) -> None:
    """
    モデル学習後の精度指標を model_metrics テーブルに保存する。

    Args:
        market: マーケット識別子
        symbol: 銘柄シンボル
        model_name: モデル名 (ex: "StockXGBoostModel")
        trained_at: 学習日時文字列 (ex: "20260314_120000")
        metrics: TrainingMetrics（rmse, directional_accuracy, n_samples）
    """
    with _db_connection() as con:
        con.execute(
            """
            INSERT INTO model_metrics
                (market, symbol, model_name, trained_at, rmse, directional_accuracy, n_samples)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (market, symbol, model_name, trained_at) DO UPDATE SET
                rmse = EXCLUDED.rmse,
                directional_accuracy = EXCLUDED.directional_accuracy,
                n_samples = EXCLUDED.n_samples
            """,
            [
                market,
                symbol,
                model_name,
                trained_at,
                metrics.rmse,
                metrics.directional_accuracy,
                metrics.n_samples,
            ],
        )
    logger.debug(
        f"model_metrics 保存: [{market}_{symbol}/{model_name}] "
        f"RMSE={metrics.rmse:.6f}, "
        f"方向正解率={metrics.directional_accuracy:.2%}"
    )


def load_model_weights(
    market: str,
    symbol: str,
    model_names: list[str],
    recent_n: int = 20,
) -> list[float]:
    """
    model_metrics テーブルからモデルごとの最新 directional_accuracy を取得し、
    ソフトマックス正規化した重みリストを返す。

    データが不足している場合は均等重みを返す。統合モデル（名前が UnifiedStock で
    始まる）は、どちらの表にも行が無いので、問い合わせを流さずに等分を返す（#1806）。

    Args:
        market: マーケット識別子
        symbol: 銘柄シンボル
        model_names: 重みを求めるモデル名リスト（predict_single_stock の model_types と対応）

    Returns:
        len(model_names) と同じ長さのfloatリスト（合計1.0）
    """
    import math

    if model_names and all(name.startswith(_UNIFIED_MODEL_PREFIX) for name in model_names):
        return _unified_equal_weights(model_names)

    n = len(model_names)
    equal_weights = [1.0 / n] * n

    # --- 1. Walk-Forward MAE 逆数重み (R-202) ---
    maes: list[float | None] = []
    with _db_connection() as con:
        for name in model_names:
            rows = con.execute(
                "SELECT predicted_ratio, actual_ratio FROM prediction_accuracy "
                "WHERE market = %s AND symbol = %s AND model_name = %s "
                "AND predicted_ratio IS NOT NULL AND actual_ratio IS NOT NULL "
                "ORDER BY predicted_at DESC LIMIT %s",
                [market, symbol, name, recent_n],
            ).fetchall()
            if rows:
                mae = sum(abs(r[0] - r[1]) for r in rows) / len(rows)
                maes.append(mae)
            else:
                maes.append(None)

    if all(m is not None for m in maes):
        valid_maes = [m for m in maes if m is not None]
        floored = [max(m, 1e-9) for m in valid_maes]
        inv = [1.0 / m for m in floored]
        total = sum(inv)
        weights = [v / total for v in inv]
        if all(abs(w - weights[0]) < 1e-9 for w in weights):
            return equal_weights
        return weights

    # --- 2. directional_accuracy ソフトマックス重み ---
    accs: list[float] = []
    with _db_connection() as con:
        for name in model_names:
            row = con.execute(
                "SELECT directional_accuracy FROM model_metrics "
                "WHERE market = %s AND symbol = %s AND model_name = %s "
                "ORDER BY trained_at DESC LIMIT 1",
                [market, symbol, name],
            ).fetchone()
            accs.append(float(row[0]) if row and row[0] is not None else 0.5)

    if not accs or all(a == accs[0] for a in accs):
        return equal_weights

    exps = [math.exp(a * 10) for a in accs]
    total = sum(exps)
    return [e / total for e in exps]
