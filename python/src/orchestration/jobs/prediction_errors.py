"""日次の [3/6] 予測で、例外で予想を出せなかった銘柄を知らせる（#1118）。

保有中の銘柄の予想が欠けると売りシグナルの決済が止まるため、予想の保存と後の段は
止めずに、件数と先頭の銘柄を1回知らせる。
"""

from src.orchestration.jobs.common import _notify_stage_failure
from src.utils.logger import get_logger

logger = get_logger(__name__)

# 知らせに並べる銘柄の上限
_NOTICE_SYMBOL_LIMIT = 5


def notify_prediction_errors(
    error_keys: list[tuple[str, str]], stage: str = "[3/6] 予測 (production)"
) -> None:
    """例外で予想を出せなかった銘柄が1件以上なら、件数と先頭の銘柄を知らせる。

    stage は知らせとログに出す段の名前（[3.5/6] の challenger でも使う #1139）。
    """
    if not error_keys:
        return
    names = [f"{market}_{symbol}" for market, symbol in sorted(error_keys)]
    head = ", ".join(names[:_NOTICE_SYMBOL_LIMIT])
    if len(names) > _NOTICE_SYMBOL_LIMIT:
        head += " ほか"
    message = f"例外で予想を出せなかった銘柄が {len(names)} 件あった: {head}"
    logger.warning("%s: %s", stage, message)
    _notify_stage_failure(f"{stage} の一部の銘柄", RuntimeError(message))
