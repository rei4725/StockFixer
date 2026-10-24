"""保有ポジションの時価評価（#844）。

現在値が取れない（NaN の）保有を、総資産の計算でどう数えるかをまとめる。
"""

import logging
import math
from typing import Any, Iterable, Mapping


def positions_market_value(positions: Iterable[Mapping[str, Any]], logger: logging.Logger) -> float:
    """保有ポジションの時価評価額の合計を返す。

    評価できない保有は時価0として数える（#844）。NaN で総資産ごと NaN になると
    DD が 0.0 になりゲートが外れるため、止める側（DD を大きく見る側）に倒す。
    WARNING は呼び出し側の logger に出す。
    """
    market_value = 0.0
    for p in positions:
        price = float(p.get("current_price", 0.0))
        if math.isnan(price):
            logger.warning(f"[risk] {p.get('symbol')}: 現在値が取れず時価0として総資産に数える")
            continue
        market_value += float(p.get("qty", 0)) * price
    return market_value
