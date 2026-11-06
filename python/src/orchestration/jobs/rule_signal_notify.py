"""日次ルールシグナルの知らせ。

daily.py の行数の上限（600 行）を超えないよう、run_daily_rule_signals から分けた。
"""

from src.utils.logger import get_logger

logger = get_logger(__name__)


def notify_rule_signals(signals: list, market: str, trade_stats: dict, total: int) -> None:
    """シグナルと発注の結果を、ルールの実績を添えて知らせる。失敗はログに残して続ける。"""
    try:
        from src.reporting.discord.discord_utils import send_rule_daily_signals
        from src.scoring.rule_report import load_rule_track_record

        send_rule_daily_signals(
            signals=signals,
            market=market,
            buy_orders=trade_stats["buy_orders"],
            sell_orders=trade_stats["sell_orders"],
            total=total,
            track_record=load_rule_track_record(market),
        )
    except Exception as e:
        logger.error("ルールシグナル通知失敗: %s", e, exc_info=True)
