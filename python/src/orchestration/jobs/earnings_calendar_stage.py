"""日次パイプラインの決算日の取り直しの段（#2201）。

earnings_calendar を書く道がどのジョブにも入っておらず、表が0行のままだった。
日次パイプラインの最後に、対象の jp 銘柄の決算日を取り直して書く。
予想・学習・発注の値は変えない（それらは今までどおり loader.get_earnings_dates を使う）。
"""

from collections.abc import Iterable
from typing import Any

from src.utils.logger import get_logger

logger = get_logger(__name__)


def refresh_earnings_calendar_stage(tasks: Iterable[Any]) -> None:
    """決算日を取り直して earnings_calendar に書く。失敗しても投げず warning を出す。"""
    logger.info("[決算日] earnings_calendar 更新開始")
    try:
        from src.market_data.event_calendar import refresh_event_calendars

        saved = refresh_event_calendars(tasks)
        logger.info("[決算日] earnings_calendar 更新完了: %d 銘柄", saved)
    except Exception as e:
        logger.warning("[決算日] earnings_calendar 更新失敗: %s", e, exc_info=True)
