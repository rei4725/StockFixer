"""実口座の約定の後に、約定値を paper_real_diff へ同期する（#1758）。

発注の直後（8:50）は成行の注文がまだ約定しておらず、同期は何も書かない。
そこで寄りの後（9:05 の約定処理）と引けの後（15:30 のレポート）にもう一度呼ぶ。
同じ注文の約定値は、行にもう実口座の約定値があれば書き直さない
（trading.execution.recording の _sync_live_execution_diffs）。
"""

from src.trading.execution import _sync_live_execution_diffs
from src.utils.logger import get_logger

logger = get_logger(__name__)


def sync_live_fills_after_fill(skipped: str) -> int:
    """紙の処理 skipped を飛ばす実口座の回で、約定値を同期する。紙の約定の件数として 0 を返す。

    同期の失敗はジョブを止めず、ERROR に出す（注文は出し終えている）。
    """
    logger.info("live モードのため %s をスキップし、約定値を同期する", skipped)
    from src.infrastructure.persistence.trade_diff_repository import PostgresTradeDiffSink
    from src.trading.brokers.kabu.kabu_client import KabuBroker

    try:
        _sync_live_execution_diffs(KabuBroker(), PostgresTradeDiffSink())
    except Exception:
        logger.error("[exec] 約定の後の約定値の同期に失敗", exc_info=True)
    return 0
