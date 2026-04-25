"""
prediction_results / model_metrics / prediction_accuracy / paper_real_diff テーブルの CRUD 操作

予測結果・モデル精度指標・予測精度追跡データを管理する。
"""

from datetime import datetime, timedelta
from typing import Any, Optional

import pandas as pd

from src.domain.types import PredictionResult, TrainingMetrics
from src.utils.db._connection import _db_connection
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# prediction_results
# ---------------------------------------------------------------------------


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

    base_cols = [
        "market",
        "symbol",
        "predicted_at",
        "model_version",
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
                "DELETE FROM prediction_results WHERE market = ? AND symbol = ? AND model_version = ?",
                [row["market"], row["symbol"], row["model_version"]],
            )
        col_str = ", ".join(cols)
        con.register("_save_df_temp", save_df)
        con.execute(
            f"INSERT INTO prediction_results ({col_str}) SELECT {col_str} FROM _save_df_temp"
        )
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


def load_prediction_results(
    predicted_at: str = None,
    market: str = None,
    top_n: int = None,
    worst_n: int = None,
    limit: int = None,
) -> pd.DataFrame:
    """
    予測結果を DB から取得する。

    Args:
        predicted_at: 予測日時（None なら最新）。limit 指定時は全タイムスタンプから取得
        market: マーケットフィルタ（None なら全マーケット）
        top_n: 上位 N 件のみ取得（diff_ratio 降順）
        worst_n: 下位 N 件のみ取得（diff_ratio 昇順）
        limit: 取得件数上限。predicted_at=None のときは全タイムスタンプ対象で N 件取得

    Returns:
        予測結果 DataFrame
    """
    if limit is not None and predicted_at is None:
        # 全タイムスタンプから直近 N 件を取得（精度チェック用）
        query = "SELECT * FROM prediction_results"
        params: list = []
        if market is not None:
            query += " WHERE market = ?"
            params.append(market)
        query += " ORDER BY predicted_at DESC"
        query += f" LIMIT {int(limit)}"
        with _db_connection() as con:
            try:
                return con.execute(query, params).fetchdf()
            except Exception as e:
                logger.error(f"prediction_results 読み込み失敗: {e}", exc_info=True)
                return pd.DataFrame()

    if predicted_at is None:
        predicted_at = load_latest_prediction_timestamp()
        if predicted_at is None:
            return pd.DataFrame()

    query = "SELECT * FROM prediction_results WHERE predicted_at = ?"
    params = [predicted_at]

    if market is not None:
        query += " AND market = ?"
        params.append(market)

    if worst_n is not None:
        query += " ORDER BY diff_ratio ASC LIMIT ?"
        params.append(worst_n)
    elif top_n is not None:
        query += " ORDER BY diff_ratio DESC LIMIT ?"
        params.append(top_n)
    else:
        query += " ORDER BY diff_ratio DESC"

    with _db_connection() as con:
        try:
            return con.execute(query, params).fetchdf()
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
                "WHERE predicted_at = ? ORDER BY market",
                [predicted_at],
            ).fetchall()
            return [row[0] for row in result]
        except Exception as e:
            logger.error(f"予測マーケット一覧取得失敗: {e}", exc_info=True)
            return []


def upsert_paper_real_diff(
    market: str,
    symbol: str,
    predicted_at: str,
    side: int,
    signal_price: float,
    mode: str,
    order_id: str,
    actual_price: float | None = None,
    checked_at: datetime | None = None,
) -> None:
    """paper / real の価格差追跡テーブルを更新する。"""
    checked_at = checked_at or datetime.now()
    with _db_connection() as con:
        row = con.execute(
            """
            SELECT signal_price, paper_order_id, real_order_id, paper_price, real_price,
                   paper_slippage, real_slippage, paper_filled_at, real_checked_at, created_at
            FROM paper_real_diff
            WHERE market = ? AND symbol = ? AND predicted_at = ? AND side = ?
            """,
            [market, symbol, predicted_at, side],
        ).fetchone()

        merged: dict[str, Any] = {
            "signal_price": signal_price,
            "paper_order_id": None,
            "real_order_id": None,
            "paper_price": None,
            "real_price": None,
            "paper_slippage": None,
            "real_slippage": None,
            "paper_filled_at": None,
            "real_checked_at": None,
            "created_at": checked_at,
        }
        if row:
            merged.update(
                {
                    "signal_price": (float(row[0]) if row[0] is not None else signal_price),
                    "paper_order_id": row[1],
                    "real_order_id": row[2],
                    "paper_price": row[3],
                    "real_price": row[4],
                    "paper_slippage": row[5],
                    "real_slippage": row[6],
                    "paper_filled_at": row[7],
                    "real_checked_at": row[8],
                    "created_at": row[9] or checked_at,
                }
            )

        merged["signal_price"] = signal_price
        if mode == "paper":
            merged["paper_order_id"] = order_id
            if actual_price is not None:
                merged["paper_price"] = actual_price
                merged["paper_slippage"] = (
                    (actual_price - signal_price) / signal_price if signal_price else None
                )
                merged["paper_filled_at"] = checked_at
        else:
            merged["real_order_id"] = order_id
            if actual_price is not None:
                merged["real_price"] = actual_price
                merged["real_slippage"] = (
                    (actual_price - signal_price) / signal_price if signal_price else None
                )
                merged["real_checked_at"] = checked_at

        price_diff = None
        if merged["paper_price"] is not None and merged["real_price"] is not None:
            price_diff = float(merged["real_price"]) - float(merged["paper_price"])

        con.execute(
            "DELETE FROM paper_real_diff WHERE market = ? AND symbol = ? "
            "AND predicted_at = ? AND side = ?",
            [market, symbol, predicted_at, side],
        )
        con.execute(
            """
            INSERT INTO paper_real_diff (
                market, symbol, predicted_at, side, signal_price,
                paper_order_id, real_order_id, paper_price, real_price,
                paper_slippage, real_slippage, price_diff,
                paper_filled_at, real_checked_at, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """,
            [
                market,
                symbol,
                predicted_at,
                side,
                merged["signal_price"],
                merged["paper_order_id"],
                merged["real_order_id"],
                merged["paper_price"],
                merged["real_price"],
                merged["paper_slippage"],
                merged["real_slippage"],
                price_diff,
                merged["paper_filled_at"],
                merged["real_checked_at"],
                merged["created_at"],
            ],
        )


def load_paper_real_diff_summary(recent_days: int = 7) -> dict:
    """直近期間の paper / real 乖離サマリーを返す。"""
    since = datetime.now() - timedelta(days=recent_days)
    with _db_connection() as con:
        row = con.execute(
            """
            SELECT
                COUNT(*) AS tracked_count,
                COUNT(*) FILTER (
                    WHERE paper_price IS NOT NULL AND real_price IS NOT NULL
                ) AS comparable_count,
                AVG(paper_slippage) AS avg_paper_slippage,
                AVG(real_slippage) AS avg_real_slippage,
                AVG(ABS(price_diff)) AS avg_abs_price_diff,
                AVG(ABS(price_diff / NULLIF(signal_price, 0))) AS avg_abs_diff_ratio,
                MAX(ABS(price_diff)) AS max_abs_price_diff
            FROM paper_real_diff
            WHERE COALESCE(paper_filled_at, real_checked_at, created_at) >= ?
            """,
            [since],
        ).fetchone()

    return {
        "tracked_count": int(row[0] or 0),
        "comparable_count": int(row[1] or 0),
        "avg_paper_slippage": float(row[2] or 0.0),
        "avg_real_slippage": float(row[3] or 0.0),
        "avg_abs_price_diff": float(row[4] or 0.0),
        "avg_abs_diff_ratio": float(row[5] or 0.0),
        "max_abs_price_diff": float(row[6] or 0.0),
    }


# ---------------------------------------------------------------------------
# model_metrics
# ---------------------------------------------------------------------------


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
        n = len(accs) if accs else len(model_names)
        return [1.0 / n] * n

    import math

    exps = [math.exp(a * 10) for a in accs]
    total = sum(exps)
    return [e / total for e in exps]


# ---------------------------------------------------------------------------
# prediction_accuracy
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# shap_values
# ---------------------------------------------------------------------------


def save_shap_values(
    market: str,
    symbol: str,
    model_name: str,
    trained_at: str,
    shap_df: pd.DataFrame,
) -> None:
    """
    SHAP特徴量寄与をshap_valuesテーブルに保存する（Delete-Insert方式）。

    Args:
        market: マーケット識別子
        symbol: 銘柄シンボル
        model_name: モデル名
        trained_at: 学習日時文字列 (ex: "20260314_120000")
        shap_df: 列 [feature, shap_mean, shap_rank] を持つ DataFrame
    """
    save_df = shap_df.copy()
    save_df["market"] = market
    save_df["symbol"] = symbol
    save_df["model_name"] = model_name
    save_df["trained_at"] = trained_at

    with _db_connection() as con:
        con.execute(
            "DELETE FROM shap_values "
            "WHERE market = ? AND symbol = ? AND model_name = ? AND trained_at = ?",
            [market, symbol, model_name, trained_at],
        )
        con.execute(
            """
            INSERT INTO shap_values
                (market, symbol, model_name, trained_at, feature, shap_mean, shap_rank)
            SELECT market, symbol, model_name, trained_at, feature, shap_mean, shap_rank
            FROM save_df
            """
        )
    logger.debug(f"shap_values 保存: [{market}_{symbol}/{model_name}] {len(save_df)}特徴量")


def load_shap_latest(
    market: str,
    symbol: str,
    model_name: str,
    top_n: int = 10,
    bottom_n: int = 10,
) -> pd.DataFrame:
    """
    指定銘柄・モデルの最新SHAP値を取得する。

    Args:
        market: マーケット識別子
        symbol: 銘柄シンボル
        model_name: モデル名
        top_n: 上位N特徴量
        bottom_n: 下位N特徴量

    Returns:
        pd.DataFrame: [feature, shap_mean, shap_rank, trained_at]
    """
    with _db_connection() as con:
        try:
            latest = con.execute(
                "SELECT MAX(trained_at) FROM shap_values "
                "WHERE market = ? AND symbol = ? AND model_name = ?",
                [market, symbol, model_name],
            ).fetchone()[0]
            if latest is None:
                return pd.DataFrame()
            top_df = con.execute(
                f"""
                SELECT feature, shap_mean, shap_rank, trained_at
                FROM shap_values
                WHERE market = ? AND symbol = ? AND model_name = ? AND trained_at = ?
                ORDER BY shap_rank ASC
                LIMIT {int(top_n)}
                """,
                [market, symbol, model_name, latest],
            ).fetchdf()
            bottom_df = con.execute(
                f"""
                SELECT feature, shap_mean, shap_rank, trained_at
                FROM shap_values
                WHERE market = ? AND symbol = ? AND model_name = ? AND trained_at = ?
                ORDER BY shap_rank DESC
                LIMIT {int(bottom_n)}
                """,
                [market, symbol, model_name, latest],
            ).fetchdf()
            return pd.concat([top_df, bottom_df], ignore_index=True).drop_duplicates(
                subset=["feature"]
            )
        except Exception as e:
            logger.error(
                f"load_shap_latest 失敗 [{market}_{symbol}/{model_name}]: {e}",
                exc_info=True,
            )
            return pd.DataFrame()


def save_feature_selection(
    market: str,
    symbol: str,
    model_name: str,
    trained_at: str,
    selection_df: pd.DataFrame,
) -> None:
    """Permutation Importance に基づく特徴量選択結果を保存する。"""
    save_df = selection_df.copy()
    save_df["market"] = market
    save_df["symbol"] = symbol
    save_df["model_name"] = model_name
    save_df["trained_at"] = trained_at

    with _db_connection() as con:
        con.execute(
            "DELETE FROM feature_selection_log "
            "WHERE market = ? AND symbol = ? AND model_name = ? AND trained_at = ?",
            [market, symbol, model_name, trained_at],
        )
        con.execute(
            """
            INSERT INTO feature_selection_log (
                market, symbol, model_name, trained_at, feature,
                importance_mean, importance_std, importance_rank,
                is_excluded, protected_by_shap
            )
            SELECT market, symbol, model_name, trained_at, feature,
                   importance_mean, importance_std, importance_rank,
                   is_excluded, protected_by_shap
            FROM save_df
            """
        )


def load_excluded_features(
    market: str,
    symbol: str,
    require_all_models: bool = True,
) -> list[str]:
    """直近の特徴量選択結果から次回学習で除外する特徴量名を返す。"""
    with _db_connection() as con:
        latest = con.execute(
            "SELECT MAX(trained_at) FROM feature_selection_log WHERE market = ? AND symbol = ?",
            [market, symbol],
        ).fetchone()[0]
        if latest is None:
            return []

        rows = con.execute(
            """
            SELECT feature, is_excluded, protected_by_shap
            FROM feature_selection_log
            WHERE market = ? AND symbol = ? AND trained_at = ?
            """,
            [market, symbol, latest],
        ).fetchall()

    by_feature: dict[str, list[tuple[bool, bool]]] = {}
    for feature, is_excluded, protected_by_shap in rows:
        by_feature.setdefault(str(feature), []).append((bool(is_excluded), bool(protected_by_shap)))

    excluded_features: list[str] = []
    for feature, values in by_feature.items():
        if any(protected for _excluded, protected in values):
            continue
        if require_all_models:
            should_exclude = all(excluded for excluded, _protected in values)
        else:
            should_exclude = any(excluded for excluded, _protected in values)
        if should_exclude:
            excluded_features.append(feature)

    return sorted(excluded_features)


# ---------------------------------------------------------------------------
# order_run_summary (R-214)
# ---------------------------------------------------------------------------


def save_order_run_summary(
    run_id: str,
    market: str,
    mode: str,
    buy_orders: int,
    sell_orders: int,
    short_orders: int,
    skipped: int,
    skipped_min_change: int,
    total_turnover: float,
    min_change_ratio: float,
) -> None:
    """発注実行サマリーを order_run_summary テーブルに保存する。"""
    with _db_connection() as con:
        con.execute(
            """
            INSERT INTO order_run_summary
                (run_id, market, mode, run_at, buy_orders, sell_orders, short_orders,
                 skipped, skipped_min_change, total_turnover, min_change_ratio)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                run_id,
                market,
                mode,
                buy_orders,
                sell_orders,
                short_orders,
                skipped,
                skipped_min_change,
                total_turnover,
                min_change_ratio,
            ],
        )
    logger.info(
        f"order_run_summary 保存: run_id={run_id} market={market} mode={mode} "
        f"buy={buy_orders} sell={sell_orders} short={short_orders} "
        f"skipped={skipped}(min_change={skipped_min_change}) turnover={total_turnover:.0f}"
    )


def load_turnover_comparison(market: str, limit: int = 30) -> pd.DataFrame:
    """
    order_run_summary から直近の売買代金推移を取得する。

    Args:
        market: マーケット識別子
        limit: 取得件数上限

    Returns:
        pd.DataFrame: [run_id, market, mode, run_at, total_turnover, buy_orders, sell_orders,
                        short_orders, skipped_min_change]
    """
    with _db_connection() as con:
        try:
            return con.execute(
                """
                SELECT run_id, market, mode, run_at, total_turnover,
                       buy_orders, sell_orders, short_orders, skipped_min_change
                FROM order_run_summary
                WHERE market = ?
                ORDER BY run_at DESC
                LIMIT ?
                """,
                [market, limit],
            ).fetchdf()
        except Exception as e:
            logger.error(f"load_turnover_comparison 失敗: {e}", exc_info=True)
            return pd.DataFrame()


# ---------------------------------------------------------------------------
# shadow_mode (R-207 A/B テスト)
# ---------------------------------------------------------------------------


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
        WHERE model_version IN (?, ?)
    """
    params: list = [production_version, challenger_version]

    if market is not None:
        query += " AND market = ?"
        params.append(market)
    if symbol is not None:
        query += " AND symbol = ?"
        params.append(symbol)

    query += " ORDER BY predicted_at DESC"
    query += f" LIMIT {int(limit)}"

    with _db_connection() as con:
        try:
            return con.execute(query, params).fetchdf()
        except Exception as e:
            logger.error(f"load_shadow_comparison 失敗: {e}", exc_info=True)
            return pd.DataFrame()
