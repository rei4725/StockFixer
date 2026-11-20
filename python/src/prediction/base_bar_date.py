"""
予想の採点の基準の日と、分割の倍率を求める純粋な関数

prediction_pipeline.py の精度チェックと、採点の基準の日を使うモジュールが読む。
"""

import math
from datetime import datetime, time, timedelta, timezone

from src.domain.market_close import MARKET_CLOSE as _MARKET_CLOSE

# 保存した現在値と raw の基準の終値の比がこの幅に収まれば、分割による調整の違いは無いとみなす
_SPLIT_GAP_TOLERANCE = 1.3
# 比を分割の比（0.5 刻み）に丸めたときに許す、丸めとの相対の差
_SPLIT_RATIO_TOLERANCE = 0.1


def _split_adjust_scale(current_price: float, base_close: float) -> float | None:
    """
    保存した現在値を raw の足の調整の水準に合わせる倍率を返す。

    比が _SPLIT_GAP_TOLERANCE の内なら分割は無いとみなして 1.0 を返す。
    外なら 0.5 刻みの分割の比に丸め、丸めとの差が小さいときだけその倍率を返し、
    分割の比に合わないときや、値が有限でないときは None を返す。
    """
    if not (math.isfinite(base_close) and math.isfinite(current_price)):
        return None
    if base_close <= 0 or current_price <= 0:
        return None
    gap = base_close / current_price
    ratio = max(gap, 1 / gap)
    if ratio <= _SPLIT_GAP_TOLERANCE:
        return 1.0
    split_ratio = round(ratio * 2) / 2
    if abs(ratio / split_ratio - 1) > _SPLIT_RATIO_TOLERANCE:
        return None
    return split_ratio if gap > 1 else 1 / split_ratio


def _base_bar_date(predicted_at_str: str, market: str) -> datetime:
    """
    予想の時刻（UTC の YYYYMMDD_HHMMSS）に引けていた最後の足の日を返す。

    市場の現地の時刻が引けより前なら、現地の前の日を返す。休日はその日以前の
    足を採るので、そのまま返してよい。時刻が読めないときや市場が不明なときは、
    UTC の日付を返す。

    Raises:
        ValueError: 先頭8字が日付として読めないとき
    """
    pred_date = datetime.strptime(predicted_at_str[:8], "%Y%m%d")
    if market not in _MARKET_CLOSE:
        return pred_date
    try:
        predicted_at = datetime.strptime(predicted_at_str[:15], "%Y%m%d_%H%M%S")
    except ValueError:
        return pred_date
    tz, close_time = _MARKET_CLOSE[market]
    local = predicted_at.replace(tzinfo=timezone.utc).astimezone(tz)
    base_day = local.date() if local.time() >= close_time else local.date() - timedelta(days=1)
    return datetime.combine(base_day, time())
