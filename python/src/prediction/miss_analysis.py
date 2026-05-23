"""
予測外れ原因のフィードバック分析（外れ日 SHAP 分析）

外れ幅上位レコードに対し stock_features の特徴量値と shap_values を突き合わせ、
外れの原因となった特徴量を特定する。繰り返し外れ要因には警告ログを出力する。
"""

from collections import Counter
from dataclasses import dataclass

import pandas as pd

from src.prediction.db import load_shap_latest
from src.utils.db.stock_features import load_stock_features
from src.utils.logger import get_logger

logger = get_logger(__name__)

_ANOMALY_Z_THRESHOLD = 1.5  # この絶対Zスコア超過で「異常値」とみなす


@dataclass
class MissCauseFeature:
    """外れ原因として特定された特徴量"""

    feature: str
    shap_rank: int  # 学習時の SHAPランク（小さいほど重要）
    shap_mean: float  # 学習時の SHAP 平均値
    miss_count: int  # 外れ日に異常値を示した回数


def analyze_miss_causes(
    market: str,
    symbol: str,
    miss_df: pd.DataFrame,
    model_name: str = "UnifiedStockXGBoost",
    top_n: int = 5,
    repeat_threshold: int = 3,
) -> list[MissCauseFeature]:
    """
    外れ日の特徴量値と SHAP 値を突き合わせ、外れ原因特徴量を返す。

    Args:
        market: マーケット識別子
        symbol: 銘柄シンボル
        miss_df: load_top_prediction_misses() の結果（当該銘柄行）
        model_name: 参照する SHAP モデル名
        top_n: 返す特徴量数
        repeat_threshold: この回数以上の異常値出現で警告ログを出す

    Returns:
        MissCauseFeature のリスト（shap_rank 昇順）
    """
    if miss_df.empty:
        return []

    shap_df = load_shap_latest(market, symbol, model_name, top_n=top_n, bottom_n=0)
    if shap_df.empty:
        logger.warning("SHAP値なし: %s/%s モデル=%s", market, symbol, model_name)
        return []

    miss_dates = _extract_dates(miss_df["predicted_at"].tolist())
    if not miss_dates:
        return []

    features_df = load_stock_features(market, symbol)
    if features_df is None or features_df.empty or "date" not in features_df.columns:
        logger.warning("stock_features なし: %s/%s", market, symbol)
        return []

    features_df = features_df.copy()
    features_df["_date_str"] = features_df["date"].astype(str).str[:10]
    miss_rows = features_df[features_df["_date_str"].isin(miss_dates)]
    if miss_rows.empty:
        logger.debug("外れ日に対応する stock_features なし: %s/%s", market, symbol)
        return []

    result: list[MissCauseFeature] = []
    top_features = shap_df.sort_values("shap_rank").head(top_n)

    for _, shap_row in top_features.iterrows():
        feat = str(shap_row["feature"])
        if feat not in features_df.columns:
            continue

        all_values = pd.to_numeric(features_df[feat], errors="coerce").dropna()
        if len(all_values) < 2:
            continue
        std = all_values.std()
        if std == 0:
            continue
        mean = all_values.mean()

        miss_values = pd.to_numeric(miss_rows[feat], errors="coerce").dropna()
        if miss_values.empty:
            continue

        z_scores = (miss_values - mean) / std
        extreme_count = int((z_scores.abs() > _ANOMALY_Z_THRESHOLD).sum())
        if extreme_count == 0:
            continue

        if extreme_count >= repeat_threshold:
            logger.warning(
                "繰り返し外れ要因検知: %s/%s 特徴量=%s (%d回)",
                market,
                symbol,
                feat,
                extreme_count,
            )

        result.append(
            MissCauseFeature(
                feature=feat,
                shap_rank=int(shap_row["shap_rank"]),
                shap_mean=float(shap_row["shap_mean"]),
                miss_count=extreme_count,
            )
        )

    return sorted(result, key=lambda x: (x.shap_rank, -x.miss_count))


def run_miss_analysis_batch(
    miss_df: pd.DataFrame,
    model_name: str = "UnifiedStockXGBoost",
    top_n: int = 5,
    repeat_threshold: int = 3,
) -> dict[tuple[str, str], list[MissCauseFeature]]:
    """
    外れ上位レコードを銘柄ごとに分析し、全銘柄横断で繰り返し外れ要因を警告する。

    Args:
        miss_df: load_top_prediction_misses() の結果（全銘柄）
        model_name: 参照する SHAP モデル名
        top_n: 銘柄ごとに返す特徴量数
        repeat_threshold: 繰り返し警告の閾値

    Returns:
        {(market, symbol): [MissCauseFeature, ...]} の辞書
    """
    if miss_df.empty:
        return {}

    results: dict[tuple[str, str], list[MissCauseFeature]] = {}
    for (market, symbol), group in miss_df.groupby(["market", "symbol"]):
        causes = analyze_miss_causes(
            market=str(market),
            symbol=str(symbol),
            miss_df=group,
            model_name=model_name,
            top_n=top_n,
            repeat_threshold=repeat_threshold,
        )
        if causes:
            results[(str(market), str(symbol))] = causes

    _warn_repeated_features(results, repeat_threshold)
    return results


def _extract_dates(predicted_at_list: list[str]) -> set[str]:
    """予測日時文字列から YYYY-MM-DD 形式の日付セットを抽出する。"""
    dates: set[str] = set()
    for pa in predicted_at_list:
        try:
            date_part = str(pa).split("_")[0]
            if len(date_part) == 8 and date_part.isdigit():
                dates.add(f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:8]}")
        except Exception:
            pass
    return dates


def _warn_repeated_features(
    results: dict[tuple[str, str], list[MissCauseFeature]],
    threshold: int,
) -> None:
    """全銘柄横断で繰り返し外れ要因となっている特徴量を警告ログに出力する。"""
    feature_counts: Counter = Counter()
    for causes in results.values():
        for cause in causes:
            feature_counts[cause.feature] += 1

    for feature, count in feature_counts.items():
        if count >= threshold:
            logger.warning(
                "繰り返し外れ要因（全銘柄横断）: 特徴量=%s が %d 銘柄で外れ要因として検知",
                feature,
                count,
            )
