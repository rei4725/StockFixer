"""#1418: 本番で指値になる流動性の低い銘柄は、翌バーが指値に届いた日だけ指値で約定する。

本番の発注（trading/execution/params.py の _choose_order_params）は、約定の前の日までの
平均出来高が少なければ引けの指値、値幅が広いだけなら寄付の指値を出す。買いは現在値＋
余裕の幅、売りは現在値−余裕の幅に置く。Backtester も同じ判定と定数で約定を決める。
"""

from __future__ import annotations

import pandas as pd
import pytest

from config.settings import LIMIT_ORDER_AVG_VOLUME_THRESHOLD
from src.backtest.backtester import Backtester

_LOW_VOLUME = LIMIT_ORDER_AVG_VOLUME_THRESHOLD / 50
_HIGH_VOLUME = LIMIT_ORDER_AVG_VOLUME_THRESHOLD * 2
# 1000 円の帯の呼値は 1 円。余裕の幅は max(1000 * 0.1%, 1) = 1 円で、買いの指値は 1001 円
_BUY_LIMIT = 1001.0
_SELL_LIMIT = 999.0


def _bt() -> Backtester:
    return Backtester(
        model_manager=None,
        signal_generator=None,
        data_loader=None,
        start_date=None,
        end_date=None,
        market="jp",
        symbol="7203",
        initial_cash=1_000_000,
        fee_rate=0.0,
        slippage=0.0,
        execution_lag=1,
    )


def _df(
    next_bars: list[tuple[float, float, float, float]], volume: float, wide: bool
) -> pd.DataFrame:
    """1000 円で動かない5本の後に next_bars（Open, High, Low, Close）を並べる。

    wide=True なら前の5本の値幅を3%にする（値幅が広い＝寄付の指値）。
    """
    half = 15.0 if wide else 0.0
    bars = [(1000.0, 1000.0 + half, 1000.0 - half, 1000.0)] * 5 + list(next_bars)
    idx = pd.date_range("2024-01-01", periods=len(bars), freq="B")
    return pd.DataFrame(
        {
            "Open": [b[0] for b in bars],
            "High": [b[1] for b in bars],
            "Low": [b[2] for b in bars],
            "Close": [b[3] for b in bars],
            "Volume": [volume] * len(bars),
        },
        index=idx,
    )


def _signal(df: pd.DataFrame, at: dict[int, int]) -> pd.Series:
    values = [at.get(i, 0) for i in range(len(df))]
    return pd.Series(values, index=df.index)


def _rows(result_df: pd.DataFrame, action: str) -> pd.DataFrame:
    if result_df.empty:
        return result_df
    return result_df[result_df["action"] == action]


class TestCloseLimitOnLowVolume:
    """出来高が少ない銘柄は引けの指値。翌バーの終値で判定する"""

    def test_gap_up_beyond_buffer_does_not_fill_buy(self):
        df = _df(
            [(1050.0, 1060.0, 1040.0, 1050.0), (1050.0, 1050.0, 1050.0, 1050.0)], _LOW_VOLUME, False
        )
        result_df, _ = _bt().simulate_trading(df, _signal(df, {4: 1}))
        assert _rows(result_df, "buy").empty

    def test_close_within_limit_fills_buy_at_limit_price(self):
        df = _df(
            [(1050.0, 1060.0, 990.0, 1000.0), (1000.0, 1000.0, 1000.0, 1000.0)], _LOW_VOLUME, False
        )
        result_df, _ = _bt().simulate_trading(df, _signal(df, {4: 1}))
        buys = _rows(result_df, "buy")
        assert len(buys) == 1
        assert buys["price"].iloc[0] == pytest.approx(_BUY_LIMIT)
        assert buys["date"].iloc[0] == df.index[5]

    def test_intraday_low_alone_does_not_fill_close_limit(self):
        """引けの指値は終値で判定する。安値が指値に届いても、終値が上なら約定しない"""
        df = _df(
            [(1000.0, 1060.0, 990.0, 1050.0), (1050.0, 1050.0, 1050.0, 1050.0)], _LOW_VOLUME, False
        )
        result_df, _ = _bt().simulate_trading(df, _signal(df, {4: 1}))
        assert _rows(result_df, "buy").empty

    def test_gap_down_does_not_fill_sell_and_keeps_position(self):
        df = _df(
            [
                (1000.0, 1000.0, 1000.0, 1000.0),
                (900.0, 910.0, 890.0, 900.0),
                (900.0, 900.0, 900.0, 900.0),
            ],
            _LOW_VOLUME,
            False,
        )
        result_df, _ = _bt().simulate_trading(df, _signal(df, {4: 1, 5: -1}))
        assert len(_rows(result_df, "buy")) == 1
        assert _rows(result_df, "sell").empty
        assert len(_rows(result_df, "final_sell")) == 1


class TestOpenLimitOnWideRange:
    """値幅が広いだけの銘柄は寄付の指値。翌バーの高値・安値で判定する"""

    def test_gap_up_beyond_buffer_does_not_fill_buy(self):
        df = _df(
            [(1050.0, 1060.0, 1040.0, 1050.0), (1050.0, 1050.0, 1050.0, 1050.0)], _HIGH_VOLUME, True
        )
        result_df, _ = _bt().simulate_trading(df, _signal(df, {4: 1}))
        assert _rows(result_df, "buy").empty

    def test_low_reaching_limit_fills_buy_at_limit_price(self):
        df = _df(
            [(1050.0, 1060.0, 995.0, 1050.0), (1050.0, 1050.0, 1050.0, 1050.0)], _HIGH_VOLUME, True
        )
        result_df, _ = _bt().simulate_trading(df, _signal(df, {4: 1}))
        buys = _rows(result_df, "buy")
        assert len(buys) == 1
        assert buys["price"].iloc[0] == pytest.approx(_BUY_LIMIT)

    def test_high_reaching_limit_fills_short_at_limit_price(self):
        bt = _bt()
        bt.enable_short = True
        df = _df([(950.0, 1005.0, 940.0, 950.0), (950.0, 950.0, 950.0, 950.0)], _HIGH_VOLUME, True)
        result_df, _ = bt.simulate_trading(df, _signal(df, {4: -1}))
        shorts = _rows(result_df, "short")
        assert len(shorts) == 1
        assert shorts["price"].iloc[0] == pytest.approx(_SELL_LIMIT)


class TestMarketOnLiquidSymbol:
    """流動性が十分な銘柄は、今までどおり翌バーの始値で約定する"""

    def test_gap_up_still_fills_at_next_open(self):
        df = _df(
            [(1050.0, 1060.0, 1040.0, 1050.0), (1050.0, 1050.0, 1050.0, 1050.0)],
            _HIGH_VOLUME,
            False,
        )
        result_df, _ = _bt().simulate_trading(df, _signal(df, {4: 1}))
        buys = _rows(result_df, "buy")
        assert len(buys) == 1
        assert buys["price"].iloc[0] == pytest.approx(1050.0)

    def test_metrics_exclude_fill_day(self):
        """約定の日の出来高が少なくても、前の日までが十分なら成行のまま（先読みしない）"""
        df = _df(
            [(1050.0, 1060.0, 1040.0, 1050.0), (1050.0, 1050.0, 1050.0, 1050.0)],
            _HIGH_VOLUME,
            False,
        )
        df.loc[df.index[5], "Volume"] = 1.0
        result_df, _ = _bt().simulate_trading(df, _signal(df, {4: 1}))
        buys = _rows(result_df, "buy")
        assert len(buys) == 1
        assert buys["price"].iloc[0] == pytest.approx(1050.0)
