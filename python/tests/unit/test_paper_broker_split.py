"""ユニットテスト: ペーパーの保有を株式分割で直す（#898）

実 Postgres（conftest の `_isolate_db` でロールバック分離）を使う。株価と分割の記録はモックで、ネットにつながない。
"""

import math
from datetime import date
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.domain.ports import MarketDataPort
from src.infrastructure.in_memory import InMemoryTradeDiffSink
from src.trading.brokers.paper import paper_broker
from src.trading.brokers.paper.paper_broker import PaperBroker
from src.trading.risk_manager import RiskManager
from src.utils.db._connection import _db_connection

_TODAY = date(2026, 3, 15)


@pytest.fixture(autouse=True)
def _setup(monkeypatch):
    monkeypatch.setattr(paper_broker, "_today_jst", lambda: _TODAY)
    with _db_connection() as con:
        con.execute("DELETE FROM paper_positions")
        con.execute(
            "INSERT INTO paper_positions (symbol, qty, avg_price, updated_at) "
            "VALUES ('7203', 100, 2000.0, '2026-03-02 01:00:00')"
        )


def _broker(close: float, splits: object) -> PaperBroker:
    market_data = MagicMock(spec=MarketDataPort)
    market_data.get_ohlcv.return_value = pd.DataFrame(
        {"Open": [close], "High": [close], "Low": [close], "Close": [close]},
        index=pd.to_datetime(["2026-03-15"]),
    )
    market_data.get_splits.return_value = splits
    return PaperBroker(market_data_port=market_data, trade_diff_sink=InMemoryTradeDiffSink())


def _splits(*items: tuple[str, float]) -> pd.Series:
    index = pd.DatetimeIndex([d for d, _ in items]).tz_localize("Asia/Tokyo")
    return pd.Series([r for _, r in items], index=index, name="Stock Splits")


def _sl_tp(pos: dict) -> tuple[bool, bool]:
    risk = RiskManager(MagicMock(), "jp", pos["symbol"])
    sl, tp, _ = risk.check_sl_tp(pos["avg_price"], pos["current_price"])
    return sl, tp


def test_split_after_holding_adjusts_qty_and_avg_and_sl_not_triggered():
    broker = _broker(990.0, _splits(("2026-03-10", 2.0)))
    [pos] = broker.get_positions()
    assert pos["qty"] == 200
    assert pos["avg_price"] == pytest.approx(1000.0)
    assert pos["unrealized_pnl"] == pytest.approx((990.0 - 1000.0) * 200)
    assert _sl_tp(pos) == (False, False)
    # 直した値を書き戻し、次の回に二重に直さない
    [again] = broker.get_positions()
    assert again["qty"] == 200
    assert again["avg_price"] == pytest.approx(1000.0)


def test_split_before_holding_is_not_applied():
    broker = _broker(1990.0, _splits(("2026-02-20", 2.0)))
    [pos] = broker.get_positions()
    assert pos["qty"] == 100
    assert pos["avg_price"] == pytest.approx(2000.0)


def test_unknown_splits_with_split_like_gap_stops_evaluation(caplog):
    broker = _broker(990.0, None)
    with caplog.at_level("WARNING"):
        [pos] = broker.get_positions()
    assert pos["qty"] == 100
    assert math.isnan(pos["current_price"])
    assert any("分割" in r.getMessage() for r in caplog.records)


def test_unknown_splits_without_gap_keeps_sl_tp():
    broker = _broker(1800.0, None)
    [pos] = broker.get_positions()
    assert pos["current_price"] == pytest.approx(1800.0)
    assert pos["qty"] == 100


def test_known_no_split_keeps_sl_tp_on_large_drop():
    broker = _broker(990.0, _splits())
    [pos] = broker.get_positions()
    assert pos["current_price"] == pytest.approx(990.0)
    assert pos["qty"] == 100
    assert _sl_tp(pos)[0] is True
