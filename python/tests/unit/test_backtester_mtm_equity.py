"""#492: バックテストの mark-to-market 日次 equity による drawdown 検証

simulate_trading が保有中の含み損益を資産曲線へ反映し、決済を待たずに
max_drawdown を計上することを保証する回帰テスト。
"""

from unittest.mock import MagicMock

import pandas as pd

from src.backtest.backtester import Backtester


def _make_backtester(**kwargs) -> Backtester:
    # #492 の mark-to-market 機構を同バー約定(execution_lag=0)で隔離検証する。
    # 翌バー約定(#493)の挙動は test_backtester_unit で別途検証する。
    kwargs.setdefault("execution_lag", 0)
    return Backtester(
        model_manager=MagicMock(),
        signal_generator=MagicMock(),
        data_loader=MagicMock(),
        start_date=None,
        end_date=None,
        market="jp",
        symbol="MTM",
        initial_cash=1_000_000,
        fee_rate=0.0,
        slippage=0.0,
        **kwargs,
    )


def test_unrealized_loss_is_reflected_in_drawdown():
    """保有中（未決済）の含み損が max_drawdown に反映される（#492 の核心）。

    シナリオ: バー0で全額買い、以降ホールドのまま最終バーで強制決済。
    途中バーで -20% まで下落し、最終的に +20% で終わる。
    - 確定損益ベース（旧挙動）: 決済は最終1回のみ → drawdown ≈ 0
    - mark-to-market（新挙動）: 保有中の谷 -20% を捉えて drawdown = -0.2
    """
    dates = pd.date_range("2024-01-01", periods=3, freq="B")
    df = pd.DataFrame({"Close": [100.0, 80.0, 120.0]}, index=dates)
    signal = pd.Series([1, 0, 0], index=dates)  # 初バーで買い、以降ホールド

    bt = _make_backtester()
    _, metrics = bt.simulate_trading(df, signal)

    # 谷（80）での含み損 -20% が drawdown に計上される
    assert metrics["max_drawdown"] == -0.2
    assert metrics["gross_max_drawdown"] == -0.2
    # total_return は確定ベースのまま（最終 +20%）— 据え置きを保証
    assert metrics["total_return"] == 0.2


def test_drawdown_zero_when_monotonic_increase():
    """単調増加で保有し続ければ含み損は無く drawdown は 0。"""
    dates = pd.date_range("2024-01-01", periods=3, freq="B")
    df = pd.DataFrame({"Close": [100.0, 110.0, 120.0]}, index=dates)
    signal = pd.Series([1, 0, 0], index=dates)

    bt = _make_backtester()
    _, metrics = bt.simulate_trading(df, signal)

    assert metrics["max_drawdown"] == 0.0
    assert metrics["total_return"] == 0.2
