"""
1回の発注の回の中で VIX を1度だけ読むためのキャッシュ（#1046）

calc_position_size は銘柄ごとに呼ばれるため、外の取得（yfinance の ^VIX）を候補の数だけ
待たないよう、RiskManager の1つのインスタンス（= 1回の発注）の中で最初に読んだ値を使い回す。
長く動くプロセスで前の日の VIX を使わないよう、UTC の日付が変われば読み直す。
"""

from datetime import date, datetime, timezone
from typing import Callable, Optional


def _today() -> date:
    return datetime.now(timezone.utc).date()


class VixRunCache:
    """読み込みの関数を UTC の日付ごとに1度だけ呼び、結果（None も含む）を使い回す。"""

    def __init__(self, fetch: Callable[[], Optional[float]]) -> None:
        self._fetch = fetch
        self._day: Optional[date] = None
        self._value: Optional[float] = None

    def get(self) -> Optional[float]:
        today = _today()
        if self._day != today:
            self._value = self._fetch()
            self._day = today
        return self._value
