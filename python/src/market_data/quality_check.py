"""データ品質チェックモジュール

market_data パイプラインが DB 保存前に呼び出すデータ品質チェック群。
DB 書き込みを行わないため並列実行フェーズでも使用できる。
"""

from dataclasses import dataclass, field
from typing import List

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)

# デフォルト閾値
NAN_RATE_THRESHOLD: float = 0.05
CONSECUTIVE_MISSING_DAYS: int = 5
ZERO_VOLUME_DAYS: int = 5


@dataclass
class QualityIssue:
    check: str
    level: str  # "warning" | "error"
    detail: str


@dataclass
class QualityCheckResult:
    market: str
    symbol: str
    passed: bool  # error が 1 件もなければ True（warning は許容）
    issues: List[QualityIssue] = field(default_factory=list)


def run_quality_checks(
    market: str,
    symbol: str,
    df: pd.DataFrame,
    nan_threshold: float = NAN_RATE_THRESHOLD,
    consecutive_missing_days: int = CONSECUTIVE_MISSING_DAYS,
    zero_volume_days: int = ZERO_VOLUME_DAYS,
) -> QualityCheckResult:
    """全データ品質チェックを実行して結果を返す。

    テクニカル指標追加・NaN 補完前の生 OHLCV DataFrame を渡すこと。
    DB 書き込みは行わないため並列実行フェーズから呼び出し可能。
    """
    issues: List[QualityIssue] = []
    issues.extend(_check_nan_rate(df, nan_threshold))
    issues.extend(_check_consecutive_missing(df, consecutive_missing_days))
    issues.extend(_check_zero_volume(df, zero_volume_days))

    passed = not any(i.level == "error" for i in issues)
    result = QualityCheckResult(market=market, symbol=symbol, passed=passed, issues=issues)

    for issue in issues:
        msg = f"[品質チェック] {market}/{symbol}: {issue.detail}"
        if issue.level == "error":
            logger.error(msg)
        else:
            logger.warning(msg)

    return result


def _check_nan_rate(df: pd.DataFrame, threshold: float) -> List[QualityIssue]:
    """各列の NaN 率 > threshold なら warning を返す。"""
    issues = []
    for col in df.columns:
        nan_rate = float(df[col].isna().mean())
        if nan_rate > threshold:
            issues.append(
                QualityIssue(
                    check="nan_rate",
                    level="warning",
                    detail=f"列 '{col}' NaN率 {nan_rate:.1%}（閾値: {threshold:.1%}）",
                )
            )
    return issues


def _check_consecutive_missing(df: pd.DataFrame, n_days: int) -> List[QualityIssue]:
    """Close 列の連続 NaN が n_days 以上なら error を返す。"""
    close_col = next((c for c in df.columns if c.lower() == "close"), None)
    if close_col is None:
        return []
    max_consec = _max_consecutive_true(df[close_col].isna())
    if max_consec >= n_days:
        return [
            QualityIssue(
                check="consecutive_missing",
                level="error",
                detail=f"Close 列が {max_consec} 日連続欠損（閾値: {n_days} 日）",
            )
        ]
    return []


def _check_zero_volume(df: pd.DataFrame, n_days: int) -> List[QualityIssue]:
    """Volume 列の連続 0 が n_days 以上なら warning を返す（上場廃止疑い）。"""
    vol_col = next((c for c in df.columns if c.lower() == "volume"), None)
    if vol_col is None:
        return []
    max_zero = _max_consecutive_true(df[vol_col].fillna(0) == 0)
    if max_zero >= n_days:
        return [
            QualityIssue(
                check="zero_volume",
                level="warning",
                detail=f"出来高 0 が {max_zero} 日連続（上場廃止疑い、閾値: {n_days} 日）",
            )
        ]
    return []


def _max_consecutive_true(series: pd.Series) -> int:
    """True が最大何回連続するかを返す。"""
    max_count = 0
    count = 0
    for v in series:
        if v:
            count += 1
            if count > max_count:
                max_count = count
        else:
            count = 0
    return max_count
