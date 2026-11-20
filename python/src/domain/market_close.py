"""市場ごとの引けの時刻の表（#2054）。

足を取り込む側（market_data/bar_close.py）と、予想の採点の基準の日を決める側
（prediction/base_bar_date.py）が同じ表を読む。
"""

from datetime import time
from zoneinfo import ZoneInfo

# 市場ごとの (タイムゾーン, 引けの時刻)。jp は東証 15:30 JST、us は米国東部 16:00
MARKET_CLOSE: dict[str, tuple[ZoneInfo, time]] = {
    "jp": (ZoneInfo("Asia/Tokyo"), time(15, 30)),
    "us": (ZoneInfo("America/New_York"), time(16, 0)),
}
