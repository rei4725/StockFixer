"""
ユニットテスト: 1回の発注の中で ^VIX を1度だけ読む（#1046）

calc_position_size は銘柄ごとに呼ばれるため、同じ回の中では最初に読んだ VIX を使い回し、
別の日（または別の回の RiskManager）では読み直すことを固定する。
"""

from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd

from src.trading.brokers.base import BrokerBase
from src.trading.risk_manager import RiskManager

_KELLY = dict(win_rate=0.6, avg_win=0.02, avg_loss=0.01)


def _make_risk() -> RiskManager:
    broker = MagicMock(spec=BrokerBase)
    broker.broker_name = "paper"
    broker.get_balance.return_value = 10_000_000.0
    broker.get_positions.return_value = []
    risk = RiskManager(broker, market="us")
    risk.get_current_dd_ratio = MagicMock(return_value=0.0)  # type: ignore[method-assign]
    return risk


def _vix_ticker(value: float, end: date) -> MagicMock:
    idx = pd.bdate_range(end=end, periods=1)
    ticker = MagicMock()
    ticker.history.return_value = pd.DataFrame({"Close": [value]}, index=idx)
    return ticker


def test_vix_fetched_once_for_three_symbols_in_one_run():
    """1回の発注で3銘柄の calc_position_size を呼んでも ^VIX の取得は1回だけ。"""
    ticker = _vix_ticker(20.0, date.today())
    risk = _make_risk()
    with patch("yfinance.Ticker", return_value=ticker) as ticker_cls:
        for symbol in ("AAPL", "MSFT", "NVDA"):
            risk.calc_position_size(symbol, price=100.0, **_KELLY)

    assert ticker_cls.call_count == 1
    assert ticker.history.call_count == 1


def test_cached_vix_still_scales_every_symbol():
    """使い回した VIX でも、閾値以上なら同じ回のすべての銘柄で縮小が掛かる。"""
    from src.trading.risk_manager import VIX_POSITION_SCALE

    base = _make_risk()
    with patch("src.trading.risk_manager.fetch_latest_vix", return_value=None):
        qty_base = base.calc_position_size("AAPL", price=100.0, **_KELLY)

    risk = _make_risk()
    with patch("yfinance.Ticker", return_value=_vix_ticker(35.0, date.today())):
        qtys = [risk.calc_position_size(s, price=100.0, **_KELLY) for s in ("AAPL", "MSFT")]

    assert qty_base > 0
    assert qtys == [int(qty_base * VIX_POSITION_SCALE)] * 2


def test_vix_refetched_on_next_day():
    """長く動くプロセスで同じ RiskManager を使っても、日付が変われば読み直す。"""
    ticker = _vix_ticker(20.0, date.today())
    risk = _make_risk()
    today = date(2026, 10, 26)
    with (
        patch("yfinance.Ticker", return_value=ticker),
        patch(
            "src.trading.vix_run_cache._today",
            side_effect=[today, today, today + timedelta(days=1)],
        ),
    ):
        for symbol in ("AAPL", "MSFT", "NVDA"):
            risk.calc_position_size(symbol, price=100.0, **_KELLY)

    assert ticker.history.call_count == 2


def test_new_run_refetches_vix():
    """別の回（新しい RiskManager）では読み直す。"""
    ticker = _vix_ticker(20.0, date.today())
    with patch("yfinance.Ticker", return_value=ticker):
        _make_risk().calc_position_size("AAPL", price=100.0, **_KELLY)
        _make_risk().calc_position_size("AAPL", price=100.0, **_KELLY)

    assert ticker.history.call_count == 2
