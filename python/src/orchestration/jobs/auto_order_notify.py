"""自動発注の後の知らせ（完了の知らせと相関リスクの警告）。

run_daily_auto_order から行数の上限を離すために移した（#2096）。動きは移す前と同じ。
"""

from src.utils.logger import get_logger

logger = get_logger(__name__)


def notify_auto_order_completion(stats, mode) -> None:
    """自動発注の完了を Discord に知らせる。送れなくてもログに残すだけで投げない。"""
    try:
        from src.orchestration.jobs.trade_lines import order_trade_lines
        from src.reporting.discord.discord_utils import send_daily_order_completion

        send_daily_order_completion(
            buy_orders=stats["buy_orders"],
            sell_orders=stats["sell_orders"],
            mode=mode,
            trading_stopped=stats.get("trading_stopped", False),
            stop_reason=stats.get("stop_reason"),
            daily_loss=stats.get("daily_loss"),
            daily_loss_limit=stats.get("daily_loss_limit"),
            errors=stats.get("errors", 0),
            sl_tp_unevaluable_symbols=stats.get("sl_tp_unevaluable_symbols"),
            skipped=stats.get("skipped", 0),
            skip_reasons=stats.get("skip_reasons"),
            details=order_trade_lines(stats),
        )
    except Exception as e:
        logger.error("自動発注完了通知失敗: %s", e, exc_info=True)


def notify_correlation_blocked(stats) -> None:
    """相関リスクで止めた日だけ警告を送る。送れなくてもログに残すだけで投げない。"""
    if stats.get("correlation_blocked"):
        try:
            from config.settings import CORRELATION_ENC_THRESHOLD
            from src.reporting.discord.discord_utils import send_correlation_alert

            send_correlation_alert(
                enc=stats.get("enc", 0.0),
                enc_threshold=CORRELATION_ENC_THRESHOLD,
                avg_correlation=stats.get("avg_correlation", 0.0),
                n_symbols=stats.get("n_held_symbols", 0),
                symbols=stats.get("held_symbols_list", []),
            )
        except Exception as e:
            logger.error("相関リスク警告通知失敗: %s", e, exc_info=True)
