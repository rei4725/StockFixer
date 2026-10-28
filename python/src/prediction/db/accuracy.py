"""prediction_accuracy / accuracy_weekly_snapshots テーブル: 予測精度追跡とドリフト集計。"""

import re
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from src.utils.db._bulk import bulk_insert
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
                    INSERT INTO prediction_accuracy
                        (market, symbol, model_name, predicted_at, horizon,
                         predicted_price, actual_price, predicted_ratio, actual_ratio,
                         direction_match, checked_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, CURRENT_TIMESTAMP)
                    ON CONFLICT (market, symbol, model_name, predicted_at, horizon) DO UPDATE SET
                        predicted_price = EXCLUDED.predicted_price,
                        actual_price = EXCLUDED.actual_price,
                        predicted_ratio = EXCLUDED.predicted_ratio,
                        actual_ratio = EXCLUDED.actual_ratio,
                        direction_match = EXCLUDED.direction_match,
                        checked_at = EXCLUDED.checked_at
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
    model_name: Optional[str] = None,
    days: Optional[int] = None,
    raise_on_error: bool = False,
) -> pd.DataFrame:
    """
    prediction_accuracy テーブルからデータを取得する。

    Args:
        market: フィルタ対象の市場名（None なら全市場）
        symbol: フィルタ対象の銘柄コード（None なら全銘柄）
        horizon: フィルタ対象のホライズン（デフォルト 1）
        limit: 返却行数の上限（0 なら上限なし）
        model_name: フィルタ対象のモデル名ラベル（None なら全モデル）
        days: 指定時は、上のフィルタに合う行の predicted_at の日付（YYYYMMDD）の
            新しい方から days 日分だけを返す（predicted_at の下限をその日にする。#958）
        raise_on_error: True なら読み込みの例外を空の表にせず投げ、行が無いことと分ける
            （日次の採点の JSON を0件で上書きしないため。#1283）

    Returns:
        pd.DataFrame（結果なし時は空 DataFrame）
    """
    where = "horizon = %s"
    filter_params: list = [horizon]
    if market is not None:
        where += " AND market = %s"
        filter_params.append(market)
    if symbol is not None:
        where += " AND symbol = %s"
        filter_params.append(symbol)
    if model_name is not None:
        where += " AND model_name = %s"
        filter_params.append(model_name)

    query = f"SELECT * FROM prediction_accuracy WHERE {where}"
    params: list = list(filter_params)
    if days is not None:
        query += (
            " AND predicted_at >= ("
            "SELECT MIN(d) FROM ("
            "SELECT DISTINCT SUBSTRING(predicted_at, 1, 8) AS d "
            f"FROM prediction_accuracy WHERE {where} "
            f"ORDER BY d DESC LIMIT {int(days)}) AS window_days)"
        )
        params += filter_params
    query += " ORDER BY predicted_at DESC"
    if limit:
        query += f" LIMIT {int(limit)}"

    with _db_connection() as con:
        try:
            return pd.read_sql(query, con, params=params)
        except Exception as e:
            logger.error(f"prediction_accuracy 読み込み失敗: {e}", exc_info=True)
            if raise_on_error:
                raise
            return pd.DataFrame()


def load_top_prediction_misses(
    horizon: int = 1,
    top_n: int = 10,
    since_days: int = 30,
    model_name: str = "production",
) -> pd.DataFrame:
    """
    直近 since_days 日の予測について外れ幅が大きい上位 top_n 件を返す。

    外れ幅 = |predicted_ratio - actual_ratio|

    Args:
        horizon: 対象ホライズン
        top_n: 返却件数上限
        since_days: 対象期間（日数）
        model_name: 対象のモデル名ラベル（challenger の採点を混ぜない。#877）

    Returns:
        pd.DataFrame: [market, symbol, model_name, predicted_at, horizon,
                       predicted_ratio, actual_ratio, abs_error]
        abs_error 降順でソート済み。actual_ratio / predicted_ratio が NULL のレコードは除外。
    """
    since = (datetime.now() - timedelta(days=since_days)).strftime("%Y%m%d")
    with _db_connection() as con:
        try:
            return pd.read_sql(
                f"""
                SELECT
                    market, symbol, model_name, predicted_at, horizon,
                    predicted_ratio, actual_ratio,
                    ABS(predicted_ratio - actual_ratio) AS abs_error
                FROM prediction_accuracy
                WHERE horizon = %s
                  AND model_name = %s
                  AND actual_ratio IS NOT NULL
                  AND predicted_ratio IS NOT NULL
                  AND predicted_at >= %s
                ORDER BY abs_error DESC
                LIMIT {int(top_n)}
                """,
                con,
                params=[horizon, model_name, since],
            )
        except Exception as e:
            logger.error(f"load_top_prediction_misses 失敗: {e}", exc_info=True)
            return pd.DataFrame()


def load_drift_summary(
    horizon: int = 1,
    recent_n: int = 30,
    model_name: str = "production",
    raise_on_error: bool = False,
) -> pd.DataFrame:
    """
    直近 recent_n 件の予測について銘柄ごとの方向正解率・平均誤差を集計する。

    Args:
        horizon: 対象ホライズン
        recent_n: 直近何件を対象にするか（銘柄ごとに最新 N 件）
        model_name: 対象のモデル名ラベル（challenger の採点を混ぜない。#877）
        raise_on_error: True なら読み込みの失敗を空の表にせず投げ直す
            （日次ドリフト検査が「精度データなし」と分けるため。#1271）

    Returns:
        pd.DataFrame: [market, symbol, direction_accuracy, mean_abs_error, n_samples,
            last_predicted_at]（last_predicted_at は銘柄ごとの最後の予想の時刻。
            月次レポートが今は予想していない銘柄を外すために使う。#1335）
    """
    with _db_connection() as con:
        try:
            return pd.read_sql(
                f"""
                WITH ranked AS (
                    SELECT *,
                        ROW_NUMBER() OVER (
                            PARTITION BY market, symbol ORDER BY predicted_at DESC
                        ) AS rn
                    FROM prediction_accuracy
                    WHERE horizon = %s AND model_name = %s
                )
                SELECT
                    market,
                    symbol,
                    AVG(CAST(direction_match AS INTEGER)) AS direction_accuracy,
                    AVG(ABS(predicted_ratio - actual_ratio))  AS mean_abs_error,
                    COUNT(*) AS n_samples,
                    MAX(predicted_at) AS last_predicted_at
                FROM ranked
                WHERE rn <= {int(recent_n)}
                GROUP BY market, symbol
                ORDER BY direction_accuracy ASC
                """,
                con,
                params=[horizon, model_name],
            )
        except Exception as e:
            logger.error(f"load_drift_summary 失敗: {e}", exc_info=True)
            if raise_on_error:
                raise
            return pd.DataFrame()


# predicted_at（'YYYYMMDD_HHMMSS' の文字列）の日付部分を 'YYYYMMDD' で取り出す SQL 式
PREDICTED_DATE_SQL = "LEFT(REGEXP_REPLACE(predicted_at, '[^0-9]', '', 'g'), 8)"
_JST_OFFSET = timedelta(hours=9)


def jst_day(predicted_at: str) -> str:
    """predicted_at（UTC の 'YYYYMMDD_HHMMSS'）の JST の日付（YYYYMMDD）を返す。

    時刻が読めないときは数字の先頭8字を返す。
    """
    digits = re.sub(r"[^0-9]", "", predicted_at)
    try:
        utc = datetime.strptime(digits[:14], "%Y%m%d%H%M%S")
    except ValueError:
        return digits[:8]
    return (utc + _JST_OFFSET).strftime("%Y%m%d")


def load_weekly_accuracy_summary(
    week_start: str,
    horizon: int = 1,
    model_name: str = "production",
    raise_on_error: bool = False,
) -> pd.DataFrame:
    """
    week_start からの7日（JST の日付）に予想した採点の行だけで、銘柄ごとの方向正解率・平均誤差を集計する。

    load_drift_summary の銘柄ごと直近 N 件の窓は週をまたいで重なり、当週の的中率の
    下がりを薄めるため、週の記録はこちらで作る（#1023）。predicted_at は UTC で、
    JST 月曜 07:30 の回は UTC では日曜になるため、JST の日付で週に切る（#1229）。

    Args:
        week_start: ISO 形式の週開始日（月曜日, 例: "2026-05-11"）
        horizon: 対象ホライズン
        model_name: 対象のモデル名ラベル（challenger の採点を混ぜない。#877）
        raise_on_error: True なら読み込みの失敗を空の表にせず投げ直す
            （週次レポートが失敗の週を黙って欠かないため。#1305）

    Returns:
        pd.DataFrame: [market, symbol, direction_accuracy, mean_abs_error, n_samples]
    """
    start = datetime.fromisoformat(week_start)
    since = start.strftime("%Y%m%d")
    until = (start + timedelta(days=7)).strftime("%Y%m%d")
    # JST の日付は UTC の日付と同じか1日後なので、UTC で1日前から読んで JST で切る
    utc_since = (start - timedelta(days=1)).strftime("%Y%m%d")
    columns = ["market", "symbol", "direction_accuracy", "mean_abs_error", "n_samples"]
    with _db_connection() as con:
        try:
            rows = pd.read_sql(
                f"""
                SELECT
                    market,
                    symbol,
                    predicted_at,
                    CAST(direction_match AS INTEGER) AS hit,
                    ABS(predicted_ratio - actual_ratio) AS abs_error
                FROM prediction_accuracy
                WHERE horizon = %s AND model_name = %s
                  AND direction_match IS NOT NULL
                  AND {PREDICTED_DATE_SQL} >= %s
                  AND {PREDICTED_DATE_SQL} < %s
                """,
                con,
                params=[horizon, model_name, utc_since, until],
            )
        except Exception as e:
            logger.error(f"load_weekly_accuracy_summary 失敗: {e}", exc_info=True)
            if raise_on_error:
                raise
            return pd.DataFrame()
    if rows.empty:
        return pd.DataFrame(columns=columns)
    days = rows["predicted_at"].astype(str).map(jst_day)
    rows = rows[(days >= since) & (days < until)]
    summary = (
        rows.groupby(["market", "symbol"], as_index=False)
        .agg(
            direction_accuracy=("hit", "mean"),
            mean_abs_error=("abs_error", "mean"),
            n_samples=("hit", "size"),
        )
        .sort_values("direction_accuracy", kind="stable")
        .reset_index(drop=True)
    )
    return summary[columns]


def save_weekly_accuracy_snapshot(week_start: str, df: pd.DataFrame) -> None:
    """
    週次精度スナップショットを accuracy_weekly_snapshots テーブルに保存する（UPSERT）。

    Args:
        week_start: ISO 形式の週開始日（月曜日, 例: "2026-05-11"）
        df: load_drift_summary() が返す DataFrame
            列: market, symbol, direction_accuracy, mean_abs_error, n_samples

    Raises:
        Exception: DELETE か書き込みが失敗したとき（その週の既存の行は残る）
    """
    required = {"market", "symbol", "direction_accuracy", "mean_abs_error", "n_samples"}
    if df is None or df.empty or not required.issubset(df.columns):
        return

    snap = df[list(required)].copy()
    snap["week_start"] = week_start

    # DELETE と書き込みを1つのトランザクションにし、書き込みが落ちたときに
    # その週の既存の行が消えたまま残らないようにする（#1328）。
    # 失敗は投げ直し、週次レポートの失敗の知らせに届ける。
    with _db_connection() as con:
        try:
            with con.transaction():
                con.execute(
                    "DELETE FROM accuracy_weekly_snapshots WHERE week_start = %s",
                    [week_start],
                )
                bulk_insert(
                    con,
                    "accuracy_weekly_snapshots",
                    snap,
                    columns=[
                        "week_start",
                        "market",
                        "symbol",
                        "direction_accuracy",
                        "mean_abs_error",
                        "n_samples",
                    ],
                )
            logger.debug(f"accuracy_weekly_snapshots 保存: week_start={week_start}, {len(snap)}件")
        except Exception as e:
            logger.error(f"save_weekly_accuracy_snapshot 失敗: {e}", exc_info=True)
            raise


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
            return pd.read_sql(
                f"""
                SELECT week_start, market, symbol, direction_accuracy, mean_abs_error, n_samples
                FROM accuracy_weekly_snapshots
                WHERE week_start IN ({weeks_subq})
                ORDER BY week_start DESC, market, symbol
                """,
                con,
            )
        except Exception as e:
            logger.error(f"load_weekly_accuracy_snapshots 失敗: {e}", exc_info=True)
            return pd.DataFrame()
