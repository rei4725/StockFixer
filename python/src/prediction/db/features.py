"""shap_values / feature_selection_log テーブル: SHAP寄与と特徴量選択結果。"""

import pandas as pd

from src.utils.db._bulk import bulk_insert
from src.utils.db._connection import _db_connection
from src.utils.logger import get_logger

logger = get_logger(__name__)


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
            "WHERE market = %s AND symbol = %s AND model_name = %s AND trained_at = %s",
            [market, symbol, model_name, trained_at],
        )
        bulk_insert(
            con,
            "shap_values",
            save_df,
            columns=[
                "market",
                "symbol",
                "model_name",
                "trained_at",
                "feature",
                "shap_mean",
                "shap_rank",
            ],
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
                "WHERE market = %s AND symbol = %s AND model_name = %s",
                [market, symbol, model_name],
            ).fetchone()[0]
            if latest is None:
                return pd.DataFrame()
            top_df = pd.read_sql(
                f"""
                SELECT feature, shap_mean, shap_rank, trained_at
                FROM shap_values
                WHERE market = %s AND symbol = %s AND model_name = %s AND trained_at = %s
                ORDER BY shap_rank ASC
                LIMIT {int(top_n)}
                """,
                con,
                params=[market, symbol, model_name, latest],
            )
            bottom_df = pd.read_sql(
                f"""
                SELECT feature, shap_mean, shap_rank, trained_at
                FROM shap_values
                WHERE market = %s AND symbol = %s AND model_name = %s AND trained_at = %s
                ORDER BY shap_rank DESC
                LIMIT {int(bottom_n)}
                """,
                con,
                params=[market, symbol, model_name, latest],
            )
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
            "WHERE market = %s AND symbol = %s AND model_name = %s AND trained_at = %s",
            [market, symbol, model_name, trained_at],
        )
        bulk_insert(
            con,
            "feature_selection_log",
            save_df,
            columns=[
                "market",
                "symbol",
                "model_name",
                "trained_at",
                "feature",
                "importance_mean",
                "importance_std",
                "importance_rank",
                "is_excluded",
                "protected_by_shap",
            ],
        )


def load_feature_exclusion_candidates(market: str, symbol: str) -> pd.DataFrame:
    """最新の Permutation Importance 結果から除外候補特徴量を返す。

    Returns:
        feature, importance_mean, importance_rank 列を持つ DataFrame (除外候補のみ)。
        データなしのときは空 DataFrame。
    """
    with _db_connection() as con:
        latest = con.execute(
            "SELECT MAX(trained_at) FROM feature_selection_log WHERE market = %s AND symbol = %s",
            [market, symbol],
        ).fetchone()[0]
        if latest is None:
            return pd.DataFrame()
        rows = con.execute(
            """
            SELECT feature,
                   AVG(importance_mean) AS importance_mean,
                   CAST(AVG(importance_rank) AS INTEGER) AS importance_rank
            FROM feature_selection_log
            WHERE market = %s AND symbol = %s AND trained_at = %s
              AND is_excluded = TRUE AND protected_by_shap = FALSE
            GROUP BY feature
            ORDER BY importance_rank DESC
            """,
            [market, symbol, latest],
        ).fetchall()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows, columns=["feature", "importance_mean", "importance_rank"])


def load_excluded_features(
    market: str,
    symbol: str,
    require_all_models: bool = True,
) -> list[str]:
    """直近の特徴量選択結果から次回学習で除外する特徴量名を返す。"""
    with _db_connection() as con:
        latest = con.execute(
            "SELECT MAX(trained_at) FROM feature_selection_log WHERE market = %s AND symbol = %s",
            [market, symbol],
        ).fetchone()[0]
        if latest is None:
            return []

        rows = con.execute(
            """
            SELECT feature, is_excluded, protected_by_shap
            FROM feature_selection_log
            WHERE market = %s AND symbol = %s AND trained_at = %s
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
