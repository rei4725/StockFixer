"""予測変化率を ATR ベースの妥当なレンジへクリップする。

モデルの点予測は、特徴量欠損時の外挿や薄商いの影響で、その銘柄が実際には
動かないような極端な変化率を出すことがある。ATR（Average True Range）は
「その銘柄が実際に1日でどれだけ動くか」を直接反映する指標なので、これを
基準に予測変化率が超えてよい上限を決め、はみ出た分は丸める。

DB もネットワークも触れない純関数として実装する。
"""

from __future__ import annotations

import math
from typing import Optional

# ATR の何倍までを「あり得るレンジ」とみなすか。
# backtest 側のストップロスで使われる 1〜2倍よりかなり緩め（通常の予測は
# 一切潰さず、明らかな外れ値だけを丸めるのが目的）。
DEFAULT_ATR_CLIP_MULTIPLIER = 3.0


def clip_diff_ratio_to_atr_range(
    diff_ratio: float,
    atr: Optional[float],
    current_price: float,
    horizon: int = 1,
    atr_multiplier: float = DEFAULT_ATR_CLIP_MULTIPLIER,
) -> float:
    """予測変化率を ATR ベースの妥当なレンジへクリップする。

    Args:
        diff_ratio: モデルが出した予測変化率
        atr: 直近の ATR（価格単位）。None または 0 以下なら判定不能として
            クリップしない
        current_price: 現在価格。0 以下ならクリップしない
        horizon: 予測ホライズン（営業日）。ランダムウォーク近似で
            sqrt(horizon) 倍してレンジを広げる
        atr_multiplier: ATR の何倍まで許容するか

    Returns:
        [-max_ratio, max_ratio] にクリップした変化率。判定不能な入力の
        場合は diff_ratio をそのまま返す。
    """
    if atr is None or atr <= 0 or current_price <= 0:
        return diff_ratio

    max_ratio = atr_multiplier * (atr / current_price) * math.sqrt(max(horizon, 1))
    if max_ratio <= 0:
        return diff_ratio

    return max(-max_ratio, min(max_ratio, diff_ratio))
