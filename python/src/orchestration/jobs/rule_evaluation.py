"""週次ルール評価の期間の決め方と後始末（知らせと失敗の判定）。

weekly.py の行数の上限（600 行）を超えないよう、run_weekly_rule_evaluation から分けた。
"""

from datetime import date

import pandas as pd

from src.orchestration.job_errors import NonRetryableJobError
from src.utils.logger import get_logger

logger = get_logger(__name__)


def rule_evaluation_window(
    today: date, years: int, start: str | None = None, end: str | None = None
) -> tuple[str, str]:
    """評価の期間（start, end）を返す。end は yfinance と同じく含まない日。

    既定は today を終わりとする直近 years 年。期間の終わりを固定していると、
    それより後に上場した銘柄が毎週失敗に数えられ、最近の値動きも評価に入らない。
    start / end を渡したとき（環境変数で固定したとき）はそちらを使う。
    """
    window_end = pd.Timestamp(end) if end else pd.Timestamp(today)
    window_start = pd.Timestamp(start) if start else window_end - pd.DateOffset(years=years)
    return window_start.strftime("%Y-%m-%d"), window_end.strftime("%Y-%m-%d")


def finish_rule_evaluation(summary: dict[str, int], market: str) -> None:
    """評価の結果を Discord へ知らせ、評価の失敗があればジョブを失敗にする（#1087）。

    失敗した銘柄は rule_best_by_symbol が前の週の行のまま（または行が無いまま）
    16:30 のルールの発注に使われるため、知らせを送った後に NonRetryableJobError を投げる。
    やり直しても同じ銘柄が同じく落ち、同じ知らせが重なるだけなのでリトライはさせない。
    """
    errors = summary.get("errors", 0)
    try:
        from src.reporting.discord.discord_utils import send_rule_evaluation_completion

        send_rule_evaluation_completion(
            evaluated=summary["evaluated"],
            effective=summary["effective"],
            skipped=summary["skipped"],
            market=market,
            errors=errors,
        )
    except Exception as e:
        logger.error("ルール評価通知失敗: %s", e, exc_info=True)

    if errors > 0:
        logger.error("週次ルール評価で評価の失敗が %d 件あった", errors)
        raise NonRetryableJobError(f"週次ルール評価で評価の失敗が {errors} 件あった")
