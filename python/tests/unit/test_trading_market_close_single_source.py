"""trading が持つ引けの時刻とタイムゾーンが、domain/market_close.py の表から読まれているか（#2121）"""

from src.domain.market_close import MARKET_CLOSE
from src.trading.brokers.paper import bar_timing
from src.trading.regime_leverage_strategy import service


def test_paper_bar_timing_reads_jp_close_from_table() -> None:
    tz, close_time = MARKET_CLOSE["jp"]
    assert bar_timing._JP_CLOSE == close_time
    # 同じ値を手で持つのではなく、表の値そのものを読む
    assert bar_timing._JP_CLOSE is close_time
    assert bar_timing._JST == tz.key


def test_regime_leverage_reads_us_close_from_table() -> None:
    tz, close_time = MARKET_CLOSE["us"]
    assert service._US_MARKET_CLOSE == close_time
    assert service._US_MARKET_CLOSE is close_time
    assert service._US_MARKET_TZ == tz
    assert service._US_MARKET_TZ is tz
