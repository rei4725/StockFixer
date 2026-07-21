"""prediction_accuracy / accuracy_weekly_snapshots テーブル: 予測精度追跡とドリフト集計。"""

from datetime import datetime, timedelta

import pandas as pd

from src.utils.db._connection import _db_connection
from src.utils.logger import get_logger

logger = get_logger(__name__)


def save_prediction_accuracy(rows: list[dict]) -> int:
    """
    予測精度データを prediction_accuracy テーブルへ保存（UPSERT）する。

    Args:
        rows: 各行は market, symbol, model_name, predicted_at, horizon 等を持つ dict

    Returns:
        挿入/更新件数
    """
    if not rows:
        return 0

    inserted = 0
    with _db_connection() as con:
        for row in rows:
            try:
                con.execute(
                    """
                    INSERT OR REPLACE INTO prediction_accuracy
                        (market, symbol, model_name, predicted_at, horizon,
                         predicted_price, actual_price, predicted_ratio, actual_ratio,
                         direction_match, checked_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    """,
                    [
                        row["market"],
                        row["symbol"],
                        row["model_name"],
                        row["predicted_at"],
                        row.get("horizon", 1),
                        row.get("predicted_price"),
                        row.get("actual_price"),
                        row.get("predicted_ratio"),
                        row.get("actual_ratio"),
                        row.get("direction_match"),
                    ],
                )
                inserted += 1
            except Exception as e:
                logger.error(
                    f"prediction_accuracy 保存失敗 [{row.get('market')}_{row.get('symbol')}]: {e}",
                    exc_info=True,
                )
    logger.info(f"prediction_accuracy 保存完了: {inserted}件")
    return inserted


def load_prediction_accuracy(
    market: str = None,
    symbol: str = None,
    horizon: int = 1,
    limit: int = 500,
) -> pd.DataFrame:
    """
    prediction_accuracy テーブルからデータを取得する。

    Args:
        market: フィルタ対象の市場名（None なら全市場）
        symbol: フィルタ対象の銘柄コード（None なら全銘柄）
        horizon: フィルタ対象のホライズン（デフォルト 1）
        limit: 返却行数の上限

    Returns:
        pd.DataFrame（結果なし時は空 DataFrame）
    """
    query = "SELECT * FROM prediction_accuracy WHERE horizon = ?"
    params: list = [horizon]
    if market is not None:
        query += " AND market = ?"
        params.append(market)
    if symbol is not None:
        query += " AND symbol = ?"
        params.append(symbol)
    query += " ORDER BY predicted_at DESC"
    if limit:
        query += f" LIMIT {int(limit)}"

    with _db_connection() as con:
        try:
            return con.execute(query, params).fetchdf()
        except Exception as e:
            logger.error(f"prediction_accuracy 読み込み失敗: {e}", exc_info=True)
            return pd.DataFrame()


def load_top_prediction_misses(
    horizon: int = 1,
    top_n: int = 10,
    since_days: int = 30,
) -> pd.DataFrame:
    """
    直近 since_days 日の予測について外れ幅が大きい上位 top_n 件を返す。

    外れ幅 = |predicted_ratio - actual_ratio|

    Args:
        horizon: 対象ホライズン
        top_n: 返却件数上限
        since_days: 対象期間（日数）

    Returns:
        pd.DataFrame: [market, symbol, model_name, predicted_at, horizon,
                       predicted_ratio, actual_ratio, abs_error]
        abs_error 降順でソート済み。actual_ratio / predicted_ratio が NULL のレコードは除外。
    """
    since = (datetime.now() - timedelta(days=since_days)).strftime("%Y%m%d")
    with _db_connection() as con:
        try:
            return con.execute(
                f"""
                SELECT
                    market, symbol, model_name, predicted_at, horizon,
                    predicted_ratio, actual_ratio,
                    ABS(predicted_ratio - actual_ratio) AS abs_error
                FROM prediction_accuracy
                WHERE horizon = ?
                  AND actual_ratio IS NOT NULL
                  AND predicted_ratio IS NOT NULL
                  AND predicted_at >= ?
                ORDER BY abs_error DESC
                LIMIT {int(top_n)}
                """,
                [horizon, since],
            ).fetchdf()
        except Exception as e:
            logger.error(f"load_top_prediction_misses 失敗: {e}", exc_info=True)
            return pd.DataFrame()


def load_drift_summary(horizon: int = 1, recent_n: int = 30) -> pd.DataFrame:
    """
    直近 recent_n 件の予測について銘柄ごとの方向正解率・平均誤差を集計する。

    Args:
        horizon: 対象ホライズン
        recent_n: 直近何件を対象にするか（銘柄ごとに最新 N 件）

    Returns:
        pd.DataFrame: [market, symbol, direction_accuracy, mean_abs_error, n_samples]
    """
    with _db_connection() as con:
        try:
            return con.execute(
                f"""
                WITH ranked AS (
                    SELECT *,
                        ROW_NUMBER() OVER (
                            PARTITION BY market, symbol ORDER BY predicted_at DESC
                        ) AS rn
                    FROM prediction_accuracy
                    WHERE horizon = ?
                )
                SELECT
                    market,
                    symbol,
                    AVG(CAST(direction_match AS INTEGER)) AS direction_accuracy,
                    AVG(ABS(predicted_ratio - actual_ratio))  AS mean_abs_error,
                    COUNT(*) AS n_samples
                FROM ranked
                WHERE rn <= {int(recent_n)}
                GROUP BY market, symbol
                ORDER BY direction_accuracy ASC
                """,
                [horizon],
            ).fetchdf()
        except Exception as e:
            logger.error(f"load_drift_summary 失敗: {e}", exc_info=True)
            return pd.DataFrame()


def save_weekly_accuracy_snapshot(week_start: str, df: pd.DataFrame) -> None:
    """
    週次精度スナップショットを accuracy_weekly_snapshots テーブルに保存する（UPSERT）。

    Args:
        week_start: ISO 形式の週開始日（月曜日, 例: "2026-05-11"）
        df: load_drift_summary() が返す DataFrame
            列: market, symbol, direction_accuracy, mean_abs_error, n_samples
    """
    required = {"market", "symbol", "direction_accuracy", "mean_abs_error", "n_samples"}
    if df is None or df.empty or not required.issubset(df.columns):
        return

    snap = df[list(required)].copy()
    snap["week_start"] = week_start

    with _db_connection() as con:
        try:
            con.execute(
                "DELETE FROM accuracy_weekly_snapshots WHERE week_start = ?",
                [week_start],
            )
            con.execute("""
                INSERT INTO accuracy_weekly_snapshots
                    (week_start, market, symbol, direction_accuracy, mean_abs_error, n_samples)
                SELECT week_start, market, symbol, direction_accuracy, mean_abs_error, n_samples
                FROM snap
                """)
            logger.debug(f"accuracy_weekly_snapshots 保存: week_start={week_start}, {len(snap)}件")
        except Exception as e:
            logger.error(f"save_weekly_accuracy_snapshot 失敗: {e}", exc_info=True)


def load_weekly_accuracy_snapshots(n_weeks: int = 4) -> pd.DataFrame:
    """
    直近 n_weeks 週分の精度スナップショットを取得する。

    Args:
        n_weeks: 取得する週数（デフォルト 4）

    Returns:
        pd.DataFrame: [week_start, market, symbol, direction_accuracy, mean_abs_error, n_samples]
        week_start 降順でソート済み
    """
    with _db_connection() as con:
        try:
            weeks_subq = (
                f"SELECT DISTINCT week_start FROM accuracy_weekly_snapshots "
                f"ORDER BY week_start DESC LIMIT {int(n_weeks)}"
            )
            return con.execute(f"""
                SELECT week_start, market, symbol, direction_accuracy, mean_abs_error, n_samples
                FROM accuracy_weekly_snapshots
                WHERE week_start IN ({weeks_subq})
                ORDER BY week_start DESC, market, symbol
                """).fetchdf()
        except Exception as e:
            logger.error(f"load_weekly_accuracy_snapshots 失敗: {e}", exc_info=True)
            return pd.DataFrame()
