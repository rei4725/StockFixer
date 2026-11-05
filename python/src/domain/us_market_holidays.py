"""米国の取引所(NYSE)の休場日の判定と、営業日の判定(#1608)

定まった規則の休場だけを計算で持つ。年ごとの表にしないのは、Good Friday を含む規則が
年で変わらず、表の足し忘れで週次の判定が止まるのを防ぐため。大統領の国葬のような臨時の
休場は入らない(その日は足が届かないのと同じに扱われ、判定の呼び出し側が失敗を知らせる)。
"""

from datetime import date, timedelta

_MONDAY = 0
_THURSDAY = 3
_SATURDAY = 5
_SUNDAY = 6


def _nth_weekday(year: int, month: int, weekday: int, nth: int) -> date:
    """その月の nth 番目の weekday の日を返す。"""
    first = date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return first + timedelta(days=offset + 7 * (nth - 1))


def _last_monday_of_may(year: int) -> date:
    may_31 = date(year, 5, 31)
    return may_31 - timedelta(days=may_31.weekday())


def _observed(day: date) -> date:
    """土曜の祝日は前の金曜、日曜の祝日は次の月曜に休む。"""
    if day.weekday() == _SATURDAY:
        return day - timedelta(days=1)
    if day.weekday() == _SUNDAY:
        return day + timedelta(days=1)
    return day


def _easter_sunday(year: int) -> date:
    """グレゴリオ暦の復活祭の日を返す(Anonymous Gregorian の計算)。"""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    m = (32 + 2 * e + 2 * i - h - k) % 7
    n = (a + 11 * h + 22 * m) // 451
    month, day = divmod(h + m - 7 * n + 114, 31)
    return date(year, month, day + 1)


def _holidays(year: int) -> frozenset[date]:
    new_year = date(year, 1, 1)
    days = {
        _nth_weekday(year, 1, _MONDAY, 3),  # キング牧師記念日
        _nth_weekday(year, 2, _MONDAY, 3),  # 大統領の日
        _easter_sunday(year) - timedelta(days=2),  # Good Friday
        _last_monday_of_may(year),  # 戦没者追悼記念日
        _observed(date(year, 6, 19)),  # ジューンティーンス
        _observed(date(year, 7, 4)),  # 独立記念日
        _nth_weekday(year, 9, _MONDAY, 1),  # 労働者の日
        _nth_weekday(year, 11, _THURSDAY, 4),  # 感謝祭
        _observed(date(year, 12, 25)),  # クリスマス
    }
    # 元日が土曜の年は、前の金曜(前の年の 12/31)を休まない(NYSE の規則)
    if new_year.weekday() != _SATURDAY:
        days.add(_observed(new_year))
    return frozenset(days)


def is_us_market_holiday(day: date) -> bool:
    """土日以外で米国の取引所が休む日なら True。"""
    return day in _holidays(day.year)


def is_us_business_day(day: date) -> bool:
    """米国の取引所の営業日(土日と休場日を除く)なら True。"""
    return day.weekday() < _SATURDAY and not is_us_market_holiday(day)
