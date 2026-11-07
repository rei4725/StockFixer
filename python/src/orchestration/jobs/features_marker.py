"""特徴量の版の印（夜間の仮の入れ替えが、今のコードの版の特徴量かを見るため）。

日次の道は印を書けなくても止めない。
"""

from typing import TYPE_CHECKING

from src.utils.logger import get_logger

if TYPE_CHECKING:
    from src.domain.types import BatchResult

logger = get_logger(__name__)

# 作れなかった銘柄の割合をこの市場だけでも見る（取得の線と同じ jp、#1710）
_ORDER_MARKET = "jp"


def current_code_version() -> str | None:
    """予想のコードの版。出せなければ None（日次の道は止めない）。"""
    from src.prediction.backtest_promotion import prediction_code_version

    try:
        return prediction_code_version()
    except Exception:
        logger.error("予想のコードの版を出せなかった（特徴量の版の印は書かない）", exc_info=True)
        return None


def record_features_version(version: str | None) -> None:
    """stock_features を作り直したコードの版の印を書く。失敗しても日次の道は止めない。"""
    if version is None:
        return
    from src.prediction.backtest_promotion import save_features_version

    try:
        save_features_version(version)
        logger.info("特徴量の版の印を書いた: %s", version)
    except Exception:
        logger.error("特徴量の版の印を書けなかった", exc_info=True)


def _coverage_gap(
    total: int, batch_result: "BatchResult", market_totals: dict[str, int]
) -> str | None:
    """作れなかった銘柄の割合が全体か jp で 1 - MIN_COVERAGE を超えたら件数の文を返す。

    数え方は取得の線（unavailable_gates）と同じスキップ+エラーで、分母も同じ対象の数。
    """
    from src.orchestration.jobs.unavailable_gates import _skipped_market
    from src.prediction.backtest_promotion.judge import MIN_COVERAGE

    line = 1 - MIN_COVERAGE
    n_skipped = len(batch_result.skipped)
    n_failed = len(batch_result.failed)
    if total > 0 and (n_skipped + n_failed) / total > line:
        return f"スキップ {n_skipped} + エラー {n_failed} / 対象 {total}（線 {line:.0%}）"
    market = _ORDER_MARKET
    market_total = market_totals.get(market, 0)
    if market_total <= 0:
        return None
    m_skipped = sum(1 for s in batch_result.skipped if _skipped_market(s) == market)
    m_failed = sum(1 for f in batch_result.failed if f.market == market)
    if (m_skipped + m_failed) / market_total > line:
        return (
            f"{market} のスキップ {m_skipped} + エラー {m_failed}"
            f" / {market} の対象 {market_total}（線 {line:.0%}）"
        )
    return None


def record_features_version_if_covered(
    version: str | None,
    total: int,
    batch_result: "BatchResult",
    market_totals: dict[str, int],
) -> None:
    """作れなかった銘柄が線の内なら印を書く。超えた回は書かずに1行残す（#1710）。

    印が前の版のままの夜は、夜間の入れ替えが「特徴量がまだ今のコードで作り直されていない」で
    飛ばし、版を記録しない。日次の道は止めない。
    """
    if version is None:
        return
    try:
        gap = _coverage_gap(total, batch_result, market_totals)
    except Exception:
        logger.error("特徴量の作れた割合を数えられなかった（印は書かない）", exc_info=True)
        return
    if gap is not None:
        logger.warning("[1/6] 特徴量を作れなかった銘柄が多いため特徴量の版の印を書かない: %s", gap)
        return
    record_features_version(version)
