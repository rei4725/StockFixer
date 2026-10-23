"""Unit Test: 現実的スリッページ（#494）

- Backtester._get_slippage の「フラット基準 + 動的インパクト加算」挙動
- simulate_trading に非ゼロコストが反映されること
- runner のデフォルト解決（市場別 / 明示 override / 動的 ON/OFF）
"""

from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.backtest.backtester import Backtester
from src.backtest.pipeline.runner import _resolve_slippage, default_slippage_for
from src.backtest.slippage import make_slippage_fn


def _make_bt(slippage=0.0, slippage_fn=None):
    return Backtester(
        model_manager=MagicMock(),
        signal_generator=MagicMock(),
        data_loader=MagicMock(),
        start_date=None,
        end_date=None,
        market="jp",
        symbol="7203",
        initial_cash=1_000_000,
        fee_rate=0.0,
        slippage=slippage,
        slippage_fn=slippage_fn,
    )


class TestGetSlippage:
    def test_base_only_without_fn(self):
        bt = _make_bt(slippage=0.001)
        assert bt._get_slippage(100, 50.0, 1000.0) == pytest.approx(0.001)

    def test_adds_dynamic_impact(self):
        # 動的項 = alpha(0.10) * sqrt(qty/vol) = 0.10 * sqrt(100/400) = 0.05
        bt = _make_bt(slippage=0.001, slippage_fn=make_slippage_fn(alpha=0.10))
        assert bt._get_slippage(100, 50.0, 400.0) == pytest.approx(0.001 + 0.05)

    def test_base_only_when_no_volume(self):
        bt = _make_bt(slippage=0.001, slippage_fn=make_slippage_fn(alpha=0.10))
        assert bt._get_slippage(100, 50.0, 0.0) == pytest.approx(0.001)

    def test_base_only_when_zero_qty(self):
        bt = _make_bt(slippage=0.001, slippage_fn=make_slippage_fn(alpha=0.10))
        assert bt._get_slippage(0, 50.0, 1000.0) == pytest.approx(0.001)

    def test_dynamic_increases_with_participation(self):
        bt = _make_bt(slippage=0.0, slippage_fn=make_slippage_fn(alpha=0.10))
        small = bt._get_slippage(10, 50.0, 10_000.0)
        large = bt._get_slippage(1000, 50.0, 10_000.0)
        assert large > small > 0


class TestSimulateTradingCost:
    def test_nonzero_slippage_reduces_return(self, sample_price_df):
        signal = pd.Series([1, 0, 0, -1, 0], index=sample_price_df.index)

        bt_free = _make_bt(slippage=0.0)
        _, m_free = bt_free.simulate_trading(sample_price_df, signal)

        bt_cost = _make_bt(slippage=0.01)
        _, m_cost = bt_cost.simulate_trading(sample_price_df, signal)

        # スリッページを乗せると同一シグナルでもリターンが低下する
        assert m_cost["total_return"] < m_free["total_return"]

    def test_dynamic_slippage_applies_without_error(self, sample_price_df):
        signal = pd.Series([1, 0, 0, -1, 0], index=sample_price_df.index)
        bt = _make_bt(slippage=0.0005, slippage_fn=make_slippage_fn(alpha=0.10))
        result_df, metrics = bt.simulate_trading(sample_price_df, signal)
        assert metrics["num_trades"] >= 1


class TestBuyCashCheckWithDynamicSlippage:
    """動的スリッページ込みの買付代金が現金を超えないこと（過剰買付 #483 の再発防止）。"""

    def test_full_sizing_low_volume_cash_never_negative(self):
        idx = pd.date_range("2024-01-01", periods=3, freq="D")
        df = pd.DataFrame(
            {"Close": [100.0, 100.0, 100.0], "Volume": [1000.0, 1000.0, 1000.0]},
            index=idx,
        )
        signal = pd.Series([1, 0, 0], index=idx)
        bt = _make_bt(slippage=0.001, slippage_fn=make_slippage_fn(alpha=0.10))
        result_df, _ = bt.simulate_trading(df, signal)

        buys = result_df[result_df["action"] == "buy"]
        assert len(buys) == 1
        qty = int(buys["qty"].iloc[0])
        assert qty > 0
        assert buys["cash"].iloc[0] >= 0
        # 1 単元増やすと現金を超える（必要以上に切り下げていない・#965）
        assert qty % bt.execution.lot_size == 0
        cost_plus_one = bt.execution.buy_cost(qty + bt.execution.lot_size, 100.0, 1000.0)
        assert cost_plus_one > 1_000_000


class TestExecutionVolumeNoLookahead:
    """翌バー始値約定のコストに、約定バー自身の出来高を使わないこと。"""

    @staticmethod
    def _run(exec_bar_volume, signal_values, exec_bar):
        idx = pd.date_range("2024-01-01", periods=5, freq="D")
        volume = [1000.0] * 5
        volume[exec_bar] = exec_bar_volume
        df = pd.DataFrame(
            {
                "Open": [100.0, 101.0, 102.0, 103.0, 104.0],
                "Close": [100.0, 101.0, 102.0, 103.0, 104.0],
                "Volume": volume,
            },
            index=idx,
        )
        signal = pd.Series(signal_values, index=idx)
        bt = _make_bt(slippage=0.001, slippage_fn=make_slippage_fn(alpha=0.10))
        assert bt.execution_lag > 0
        result_df, _ = bt.simulate_trading(df, signal)
        return result_df

    def test_buy_cost_ignores_execution_bar_volume(self):
        # bar0 の買いシグナルは bar1 始値で約定。bar1 の出来高だけを変える
        low = self._run(1000.0, [1, 0, 0, 0, 0], exec_bar=1)
        high = self._run(1_000_000_000.0, [1, 0, 0, 0, 0], exec_bar=1)
        buy_low = low[low["action"] == "buy"].iloc[0]
        buy_high = high[high["action"] == "buy"].iloc[0]
        assert buy_low["qty"] == buy_high["qty"]
        assert buy_low["cash"] == pytest.approx(buy_high["cash"])

    def test_sell_proceeds_ignores_execution_bar_volume(self):
        # bar0 買い→bar1 約定、bar2 売り→bar3 始値で約定。bar3 の出来高だけを変える
        low = self._run(1000.0, [1, 0, -1, 0, 0], exec_bar=3)
        high = self._run(1_000_000_000.0, [1, 0, -1, 0, 0], exec_bar=3)
        sell_low = low[low["action"] == "sell"].iloc[0]
        sell_high = high[high["action"] == "sell"].iloc[0]
        assert sell_low["cash"] == pytest.approx(sell_high["cash"])


class TestDefaultSlippageResolution:
    def test_default_by_market(self):
        assert default_slippage_for("us") == pytest.approx(0.0005)
        assert default_slippage_for("jp") == pytest.approx(0.0010)
        # 未知市場は US 既定にフォールバック
        assert default_slippage_for("xx") == pytest.approx(0.0005)

    def test_resolve_uses_default_when_none(self):
        eff, fn = _resolve_slippage("jp", None, dynamic_slippage=True)
        assert eff == pytest.approx(0.0010)
        assert callable(fn)

    def test_resolve_respects_explicit_override(self):
        eff, fn = _resolve_slippage("us", 0.002, dynamic_slippage=False)
        assert eff == pytest.approx(0.002)
        assert fn is None
