"""optimal_params.json の stop_loss_pct・take_profit_pct を読む小さな検査（#1241）。

NaN・無限・0以下は比べがいつも False になり SL/TP が黙って外れるため、None とみなす。
"""

import math
from typing import Any, Mapping, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


def positive_pct_param(params: Mapping[str, Any], key: str, label: str) -> Optional[float]:
    """params[key] が有限の正の数なら float で返し、そうでなければ None を返す。

    null・欠けは今までどおり黙って None、NaN・無限・0以下は WARNING を出して None にする。
    """
    value = params.get(key)
    if value is None:
        return None
    pct = float(value)
    if math.isfinite(pct) and pct > 0:
        return pct
    logger.warning(f"[{label}] {key}={pct} は有限の正の数でないため null とみなす")
    return None
