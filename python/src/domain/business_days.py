"""最後の足からの営業日の遅れの数え方（#2163）

取得（market_data）・予想（prediction）・ルール（rule_engine）の古い足の判定が使う。
"""

from datetime import date, timedelta

from src.domain.jp_market_holidays import is_jp_business_day


def count_business_days_behind(last_day: date, base_day: date, market: str | None = None) -> int:
    """last_day の翌日から base_day までの営業日の数を返す。last_day が base_day 以後なら 0。

    jp は東証の休場日も除き、ほかの市場と None は土日だけを除く（祝日は数え分けない）。
    base_day が営業日でなくても、直近の営業日まで戻して数えるのと同じ数になる。
    """
    days = (last_day + timedelta(days=i) for i in range(1, (base_day - last_day).days + 1))
    if market == "jp":
        return sum(1 for day in days if is_jp_business_day(day))
    return sum(1 for day in days if day.weekday() < 5)
