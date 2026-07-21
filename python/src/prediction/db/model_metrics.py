"""model_metrics テーブル: モデル精度指標の保存とアンサンブル重み算出。"""

from src.prediction.types import TrainingMetrics
from src.utils.db._connection import _db_connection
from src.utils.logger import get_logger

logger = get_logger(__name__)


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
            INSERT OR REPLACE INTO model_metrics
                (market, symbol, model_name, trained_at, rmse, directional_accuracy, n_samples)
            VALUES (?, ?, ?, ?, ?, ?, ?)
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

    データが不足している場合は均等重みを返す。

    Args:
        market: マーケット識別子
        symbol: 銘柄シンボル
        model_names: 重みを求めるモデル名リスト（predict_single_stock の model_types と対応）

    Returns:
        len(model_names) と同じ長さのfloatリスト（合計1.0）
    """
    import math

    n = len(model_names)
    equal_weights = [1.0 / n] * n

    # --- 1. Walk-Forward MAE 逆数重み (R-202) ---
    maes: list[float | None] = []
    with _db_connection() as con:
        for name in model_names:
            rows = con.execute(
                "SELECT predicted_ratio, actual_ratio FROM prediction_accuracy "
                "WHERE market = ? AND symbol = ? AND model_name = ? "
                "AND predicted_ratio IS NOT NULL AND actual_ratio IS NOT NULL "
                "ORDER BY predicted_at DESC LIMIT ?",
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
                "WHERE market = ? AND symbol = ? AND model_name = ? "
                "ORDER BY trained_at DESC LIMIT 1",
                [market, symbol, name],
            ).fetchone()
            accs.append(float(row[0]) if row and row[0] is not None else 0.5)

    if not accs or all(a == accs[0] for a in accs):
        return equal_weights

    exps = [math.exp(a * 10) for a in accs]
    total = sum(exps)
    return [e / total for e in exps]
