"""prediction_results テーブルの CRUD 操作（shadow 比較含む）。"""

from typing import Optional

import pandas as pd

from src.prediction.types import PredictionResult
from src.utils.db._bulk import bulk_insert
from src.utils.db._connection import _db_connection
from src.utils.logger import get_logger
from src.utils.run_context import get_run_id

logger = get_logger(__name__)


def save_prediction_results(predicted_at: str, results: list[PredictionResult]) -> None:
    """
    予測結果を DB に保存する（Delete-Insert 方式）。
    対象銘柄・モデルバージョンの既存データを削除してから挿入する。

    シャドーモード利用時は model_version フィールドを設定することで、
    production / challenger 両バージョンを同一テーブルに共存させられる。

    Args:
        predicted_at: 予測実行日時 (例: "20260213_142903")
        results: 予測結果のリスト
    """
    save_df = PredictionResult.to_dataframe(results)
    save_df["predicted_at"] = predicted_at

    # model_version が設定されていない行は 'production' をデフォルト値として使う
    if "model_version" not in save_df.columns:
        save_df["model_version"] = "production"
    else:
        save_df["model_version"] = save_df["model_version"].fillna("production")

    save_df["run_id"] = get_run_id()

    base_cols = [
        "market",
        "symbol",
        "predicted_at",
        "model_version",
        "run_id",
        "current_price",
        "avg_pred_price",
        "diff_ratio",
        "model_count",
        "confidence_ratio",
    ]
    multi_horizon_cols = [
        "avg_pred_price_3d",
        "avg_pred_price_5d",
        "avg_pred_price_10d",
        "diff_ratio_3d",
        "diff_ratio_5d",
        "diff_ratio_10d",
        "confluence_score",
    ]
    extra_cols = [c for c in multi_horizon_cols if c in save_df.columns]
    cols = base_cols + extra_cols

    save_df = save_df[[c for c in cols if c in save_df.columns]]
    for c in cols:
        if c not in save_df.columns:
            save_df[c] = None

    with _db_connection() as con:
        pairs = save_df[["market", "symbol", "model_version"]].drop_duplicates()
        for _, row in pairs.iterrows():
            con.execute(
                "DELETE FROM prediction_results WHERE market = %s AND symbol = %s"
                " AND model_version = %s",
                [row["market"], row["symbol"], row["model_version"]],
            )
        bulk_insert(con, "prediction_results", save_df, columns=cols)
    logger.info(f"DB保存完了: prediction_results [{predicted_at}] ({len(save_df)}行)")


def load_latest_prediction_timestamp() -> Optional[str]:
    """最新の predicted_at タイムスタンプを返す。なければ None。"""
    with _db_connection() as con:
        try:
            result = con.execute(
                "SELECT DISTINCT predicted_at FROM prediction_results "
                "ORDER BY predicted_at DESC LIMIT 1"
            ).fetchone()
            return result[0] if result else None
        except Exception as e:
            logger.error(f"最新予測タイムスタンプ取得失敗: {e}", exc_info=True)
            return None


def load_previous_run_stats(
    exclude_predicted_at: str, model_version: str = "production"
) -> Optional[tuple[list[int], list[float]]]:
    """今回を除く直近ランの model_count / diff_ratio を返す。

    出力 invariant の急変チェック（B-1/B-2/B-3）が使う。集計は行わず生の値を
    返す。中央値・標準偏差の計算は Python 側で行い、SQL 方言差を持ち込まない。

    Args:
        exclude_predicted_at: 今回ランの predicted_at（これより前を対象にする）
        model_version: 対象のモデルバージョン

    Returns:
        (model_counts, diff_ratios) のタプル。前回ランが無ければ None。
    """
    try:
        with _db_connection() as con:
            row = con.execute(
                "SELECT predicted_at FROM prediction_results "
                "WHERE model_version = %s AND predicted_at < %s "
                "ORDER BY predicted_at DESC LIMIT 1",
                (model_version, exclude_predicted_at),
            ).fetchone()
            if not row:
                logger.info("前回ラン統計なし（比較をスキップ）")
                return None

            previous_at = row[0]
            rows = con.execute(
                "SELECT model_count, diff_ratio FROM prediction_results "
                "WHERE predicted_at = %s AND model_version = %s",
                (previous_at, model_version),
            ).fetchall()

        model_counts = [int(r[0]) for r in rows if r[0] is not None]
        diff_ratios = [float(r[1]) for r in rows if r[1] is not None]
        logger.info("前回ラン統計を取得: predicted_at=%s 件数=%d", previous_at, len(model_counts))
        return model_counts, diff_ratios
    except Exception as e:
        logger.error(f"前回ラン統計の取得失敗: {e}", exc_info=True)
        return None


def load_prediction_results(
    predicted_at: str = None,
    market: str = None,
    top_n: int = None,
    worst_n: int = None,
    limit: int = None,
    model_version: str = None,
) -> pd.DataFrame:
    """
    予測結果を DB から取得する。

    Args:
        predicted_at: 予測日時（None なら最新）。limit 指定時は全タイムスタンプから取得
        market: マーケットフィルタ（None なら全マーケット）
        top_n: 上位 N 件のみ取得（diff_ratio 降順）
        worst_n: 下位 N 件のみ取得（diff_ratio 昇順）
        limit: 取得件数上限。predicted_at=None のときは全タイムスタンプ対象で N 件取得
        model_version: モデルバージョンフィルタ（None なら全バージョン）

    Returns:
        予測結果 DataFrame
    """
    if limit is not None and predicted_at is None:
        # 全タイムスタンプから直近 N 件を取得（精度チェック用）
        query = "SELECT * FROM prediction_results"
        params: list = []
        conditions = []
        if market is not None:
            conditions.append("market = %s")
            params.append(market)
        if model_version is not None:
            conditions.append("model_version = %s")
            params.append(model_version)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY predicted_at DESC"
        query += f" LIMIT {int(limit)}"
        with _db_connection() as con:
            try:
                return pd.read_sql(query, con, params=params)
            except Exception as e:
                logger.error(f"prediction_results 読み込み失敗: {e}", exc_info=True)
                return pd.DataFrame()

    if predicted_at is None:
        predicted_at = load_latest_prediction_timestamp()
        if predicted_at is None:
            return pd.DataFrame()

    query = "SELECT * FROM prediction_results WHERE predicted_at = %s"
    params = [predicted_at]

    if market is not None:
        query += " AND market = %s"
        params.append(market)

    if model_version is not None:
        query += " AND model_version = %s"
        params.append(model_version)

    if worst_n is not None:
        query += " ORDER BY diff_ratio ASC LIMIT %s"
        params.append(worst_n)
    elif top_n is not None:
        query += " ORDER BY diff_ratio DESC LIMIT %s"
        params.append(top_n)
    else:
        query += " ORDER BY diff_ratio DESC"

    with _db_connection() as con:
        try:
            return pd.read_sql(query, con, params=params)
        except Exception as e:
            logger.error(f"prediction_results 読み込み失敗: {e}", exc_info=True)
            return pd.DataFrame()


def load_prediction_markets(predicted_at: str = None) -> list:
    """
    指定タイムスタンプの予測結果に含まれるマーケット一覧を返す。

    Args:
        predicted_at: None なら最新

    Returns:
        マーケット名のリスト
    """
    if predicted_at is None:
        predicted_at = load_latest_prediction_timestamp()
        if predicted_at is None:
            return []

    with _db_connection() as con:
        try:
            result = con.execute(
                "SELECT DISTINCT market FROM prediction_results "
                "WHERE predicted_at = %s ORDER BY market",
                [predicted_at],
            ).fetchall()
            return [row[0] for row in result]
        except Exception as e:
            logger.error(f"予測マーケット一覧取得失敗: {e}", exc_info=True)
            return []


def load_shadow_comparison(
    market: str = None,
    symbol: str = None,
    production_version: str = "production",
    challenger_version: str = "challenger",
    limit: int = 1000,
) -> pd.DataFrame:
    """
    シャドーモードの production / challenger 両バージョンの予測結果を取得する。

    prediction_accuracy テーブルに実績が記録されていない場合は prediction_results から
    最新のスナップショットを返す。

    Args:
        market: マーケットフィルタ（None なら全マーケット）
        symbol: 銘柄フィルタ（None なら全銘柄）
        production_version: 本番バージョンのラベル（デフォルト "production"）
        challenger_version: チャレンジャーバージョンのラベル（デフォルト "challenger"）
        limit: 取得件数上限

    Returns:
        pd.DataFrame: 全列 + model_version 列を含む予測結果
    """
    query = """
        SELECT * FROM prediction_results
        WHERE model_version IN (%s, %s)
    """
    params: list = [production_version, challenger_version]

    if market is not None:
        query += " AND market = %s"
        params.append(market)
    if symbol is not None:
        query += " AND symbol = %s"
        params.append(symbol)

    query += " ORDER BY predicted_at DESC LIMIT %s"
    params.append(int(limit))

    with _db_connection() as con:
        try:
            return pd.read_sql(query, con, params=params)
        except Exception as e:
            logger.error(f"load_shadow_comparison 失敗: {e}", exc_info=True)
            return pd.DataFrame()
