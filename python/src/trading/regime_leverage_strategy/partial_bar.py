"""日足が途中の足(引け後に届いたが、値がまだ確定していない足)かを出来高で見分ける(#2144)。"""

from __future__ import annotations

from typing import Optional

import pandas as pd

# 直前の足の出来高の中央値を取る本数(判定する足は含まない)と、判定に要る最少の本数
_BASELINE_BARS = 20
_MIN_BASELINE_BARS = 10
# 出来高が直前の中央値のこの割合に満たない足を、途中の足とみなす。us の 2026-11-18 の途中の足は
# 中央値で約15%だった(#2140)。半日立会いの日は取引の時間が短いだけで出来高は40〜60%ほど
# あるので、その下限より低い 0.3 なら当たらない
_PARTIAL_VOLUME_RATIO = 0.3


def partial_bar_volume_ratio(df: pd.DataFrame, bar_at: pd.Timestamp) -> Optional[float]:
    """bar_at の足が途中の足なら、直前の足の中央値に対する出来高の比を返す。違えば None。

    bar_at より後の足(まだ引けていない当日の足)は使わない。出来高の列が無い・その足の出来高が
    欠けている・比べる足が足りないときは、見分けられないので途中の足とみなさない。
    """
    if "Volume" not in df.columns:
        return None
    volumes = pd.to_numeric(df["Volume"], errors="coerce")
    volume = float(volumes[df.index <= bar_at].iloc[-1])
    previous = volumes[df.index < bar_at].dropna()
    previous = previous[previous > 0].tail(_BASELINE_BARS)
    if pd.isna(volume) or len(previous) < _MIN_BASELINE_BARS:
        return None
    ratio = volume / float(previous.median())
    return ratio if ratio < _PARTIAL_VOLUME_RATIO else None
