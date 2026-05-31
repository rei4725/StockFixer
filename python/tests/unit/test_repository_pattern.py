"""
ユニットテスチE Repository パターン (#282)

InMemoryPredictionRepository を使って execution.py の run_daily_orders めE
DuckDB なしでテストできることを検証する
"""

from unittest.mock import patch

import pandas as pd
import pytest

from src.domain.ports import PredictionResultRepository
from src.infrastructure.in_memory import InMemoryBrokerAdapter, InMemoryPredictionRepository
from src.trading.brokers.base import OrderType
from src.trading.execution import run_daily_orders
from src.trading.types import TradingGateStatus

# ---------------------------------------------------------------------------
# InMemoryPredictionRepository.get_latest_by_market
# ---------------------------------------------------------------------------


class TestInMemoryPredictionRepositoryGetLatest:
    def _make_result(self, market: str, symbol: str, diff_ratio: float, predicted_at: str):
        from dataclasses import dataclass

        @dataclass
        class _R:
            market: str
            symbol: str
            diff_ratio: float
            predicted_at: str
            current_price: float = 1000.0
            confidence_ratio: float = 1.0

        return _R(
            market=market,
            symbol=symbol,
            diff_ratio=diff_ratio,
            predicted_at=predicted_at,
        )

    def test_returns_latest_per_symbol(self):
        repo = InMemoryPredictionRepository()
        repo.save("20260501_090000", [self._make_result("jp", "7203", 0.01, "20260501_090000")])
        repo.save("20260502_090000", [self._make_result("jp", "7203", 0.02, "20260502_090000")])
        df = repo.get_latest_by_market("jp")
        assert len(df) == 1
        assert float(df.iloc[0]["diff_ratio"]) == pytest.approx(0.02)

    def test_filters_by_market(self):
        repo = InMemoryPredictionRepository()
        repo.save("20260501_090000", [self._make_result("jp", "7203", 0.01, "20260501_090000")])
        repo.save("20260501_090000", [self._make_result("us", "AAPL", 0.02, "20260501_090000")])
        df = repo.get_latest_by_market("jp")
        assert all(df["market"] == "jp")
        assert len(df) == 1

    def test_sorted_by_diff_ratio_desc(self):
        repo = InMemoryPredictionRepository()
        for sym, ratio in [("A", 0.01), ("B", 0.03), ("C", 0.02)]:
            repo.save("20260501_090000", [self._make_result("jp", sym, ratio, "20260501_090000")])
        df = repo.get_latest_by_market("jp")
        assert list(df["symbol"]) == ["B", "C", "A"]

    def test_empty_when_no_data(self):
        repo = InMemoryPredictionRepository()
        df = repo.get_latest_by_market("jp")
        assert df.empty

    def test_implements_port(self):
        repo = InMemoryPredictionRepository()
        assert isinstance(repo, PredictionResultRepository)


# ---------------------------------------------------------------------------
# run_daily_orders の prediction_repo 経由でテスト（DB なし）
# ---------------------------------------------------------------------------

_GATE_OK = TradingGateStatus(
    is_allowed=True,
    stop_active=False,
    reason_code=None,
    reason=None,
    daily_loss=0.0,
    daily_loss_limit=None,
)


def _make_repo_with_predictions(n: int = 3) -> InMemoryPredictionRepository:
    from dataclasses import dataclass

    @dataclass
    class _R:
        market: str
        symbol: str
        diff_ratio: float
        current_price: float
        confidence_ratio: float
        predicted_at: str

    repo = InMemoryPredictionRepository()
    for i in range(n):
        r = _R(
            market="jp",
            symbol=f"720{i}",
            diff_ratio=0.02 + 0.001 * i,
            current_price=1000.0 + i * 10,
            confidence_ratio=1.0,
            predicted_at="20260518_090000",
        )
        repo.save("20260518_090000", [r])
    return repo


_COMMON_PATCHES = [
    patch("src.trading.execution._record_order"),
    patch(
        "src.trading.execution._choose_order_params",
        return_value=(OrderType.MARKET, 0.0, "market", "open"),
    ),
    patch(
        "src.trading.execution.RiskManager.evaluate_trading_gate",
        return_value=_GATE_OK,
    ),
    patch("src.trading.execution.RiskManager.calc_position_size", return_value=100),
    patch("src.trading.risk_manager.RiskManager._get_daily_realized_loss", return_value=0.0),
    patch("src.trading.risk_manager.RiskManager._get_consecutive_losses", return_value=0),
    patch("src.trading.execution.save_order_run_summary"),
]


class TestRunDailyOrdersWithRepository:
    """prediction_repo パラメータ経由で _load_latest_predictions をパッチせずにテストする"""

    def test_buy_orders_via_repository(self):
        broker = InMemoryBrokerAdapter(initial_balance=10_000_000.0)
        repo = _make_repo_with_predictions(n=3)

        _ = [p.start() for p in _COMMON_PATCHES]
        try:
            stats = run_daily_orders(broker, market="jp", mode="paper", prediction_repo=repo)
            assert stats["buy_orders"] > 0
            assert stats["errors"] == 0
        finally:
            for p in _COMMON_PATCHES:
                p.stop()

    def test_empty_repo_no_orders(self):
        broker = InMemoryBrokerAdapter()
        repo = InMemoryPredictionRepository()  # 空

        _ = [p.start() for p in _COMMON_PATCHES]
        try:
            stats = run_daily_orders(broker, market="jp", mode="paper", prediction_repo=repo)
            assert stats["buy_orders"] == 0
        finally:
            for p in _COMMON_PATCHES:
                p.stop()

    def test_no_repo_falls_back_to_db(self):
        """prediction_repo=None のとき、従来の _load_latest_predictions を使う（後方互換）"""
        broker = InMemoryBrokerAdapter()

        with patch(
            "src.trading.execution._load_latest_predictions",
            return_value=pd.DataFrame(),
        ) as mock_load:
            _ = [p.start() for p in _COMMON_PATCHES]
            try:
                run_daily_orders(broker, market="jp", mode="paper", prediction_repo=None)
                mock_load.assert_called_once_with("jp")
            finally:
                for p in _COMMON_PATCHES:
                    p.stop()

    def test_with_repo_skips_db_load(self):
        """prediction_repo を渡したとき、_load_latest_predictions は呼ばれない"""
        broker = InMemoryBrokerAdapter()
        repo = _make_repo_with_predictions(n=1)

        with patch("src.trading.execution._load_latest_predictions") as mock_load:
            _ = [p.start() for p in _COMMON_PATCHES]
            try:
                run_daily_orders(broker, market="jp", mode="paper", prediction_repo=repo)
                mock_load.assert_not_called()
            finally:
                for p in _COMMON_PATCHES:
                    p.stop()
