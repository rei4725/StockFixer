"""
特徴量の DB 保存と品質の知らせ

pipeline.py から移した（#1831）。pipeline.py は同じ名前で import して使い続ける。
"""

from src.domain.ports import AlertLevel, NotificationPort
from src.market_data.quality_check import QualityCheckResult
from src.utils.db import upsert_stock_features
from src.utils.db.symbol_master import symbol_name_suffix
from src.utils.logger import get_logger

logger = get_logger(__name__)


def save_features_to_db(
    market: str,
    symbol: str,
    data,
    quality_result: QualityCheckResult = None,
    notification_port: "NotificationPort | None" = None,
) -> None:
    """
    特徴量DataFrameをDBに保存する（逐次実行用）。

    Args:
        market: マーケット識別子
        symbol: 銘柄シンボル
        data: 特徴量DataFrame
        quality_result: データ品質チェック結果（Noneの場合はスキップ）
        notification_port: 通知ポート（Noneの場合は Discord 通知なし）
    """
    upsert_stock_features(market, symbol, data)
    logger.info(f"DB保存完了: {market}_{symbol}")
    if quality_result is not None and quality_result.issues:
        _notify_quality_issues(market, symbol, quality_result, notification_port)


def _notify_quality_issues(
    market: str,
    symbol: str,
    quality_result: QualityCheckResult,
    notification_port: "NotificationPort | None" = None,
) -> None:
    """品質チェック結果をDBに記録し、通知が必要なものを notification_port 経由で送る。"""
    from src.utils.db.quality_log import insert_quality_log

    issues_dicts = [
        {"check": i.check, "level": i.level, "detail": i.detail} for i in quality_result.issues
    ]
    try:
        insert_quality_log(market, symbol, issues_dicts)
    except Exception as e:
        logger.error(f"品質ログ記録エラー: {e}", exc_info=True)

    # エラーまたは上場廃止疑い（zero_volume）のみ Discord 通知
    notify_issues = [
        i for i in quality_result.issues if i.level == "error" or i.check == "zero_volume"
    ]
    if not notify_issues or notification_port is None:
        return

    level = (
        AlertLevel.ERROR if any(i.level == "error" for i in notify_issues) else AlertLevel.WARNING
    )
    title = f"[データ品質] {market}/{symbol}{symbol_name_suffix(market, symbol)}"
    message = "\n".join(f"・{i.detail}" for i in notify_issues)
    try:
        notification_port.send_alert(title=title, message=message, level=level)
    except Exception as e:
        logger.error(f"品質チェック Discord通知エラー: {e}", exc_info=True)
