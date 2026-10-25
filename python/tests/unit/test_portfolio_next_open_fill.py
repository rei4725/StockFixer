"""Issue #1099: 日 t のスコアで決めた売買を、翌営業日 t+1 の始値で約定させる。

行 t のスコアは t の引けの後にしか出せないため、t の終値で買うと
t の引けから t+1 の寄りまでの値動き（夜の窓）を取れたことになる。
"""

from unittest.mock import patch

import pandas as pd
import pytest

from src.backtest.execution import ExecutionModel, TradingCosts

_NAN = float("nan")


def _run(score_matrix, close_matrix, open_matrix, rebalance_dates, top_n=1):
    from src.backtest.portfolio import _simulate_portfolio

    with patch(
        "src.backtest.portfolio.simulation._get_portfolio_symbol_sector",
        return_value="Technology",
    ):
        equity_df, holdings = _simulate_portfolio(
            score_matrix,
            close_matrix,
            rebalance_dates,
            top_n=top_n,
            initial_cash=10_000,
            execution=ExecutionModel(TradingCosts(fee_rate=0.0)),
            max_sector_positions=5,
            open_matrix=open_matrix,
        )
    return equity_df.set_index("date"), holdings


def _day(dates, i):
    return str(dates[i])[:10]


class TestPortfolioNextOpenFill:
    def _matrices(self):
        dates = pd.date_range("2026-01-05", periods=4, freq="B")
        score_matrix = pd.DataFrame(
            {"A": [0.9, 0.1, 0.1, 0.1], "B": [0.1, 0.9, 0.9, 0.9]}, index=dates
        )
        close_matrix = pd.DataFrame(
            {"A": [100.0, 110.0, 120.0, 130.0], "B": [50.0, 50.0, 50.0, 50.0]}, index=dates
        )
        open_matrix = pd.DataFrame(
            {"A": [999.0, 125.0, 115.0, 125.0], "B": [50.0, 50.0, 40.0, 60.0]}, index=dates
        )
        return dates, score_matrix, close_matrix, open_matrix

    def test_buys_and_sells_at_next_open(self):
        """t のスコアで選んだ銘柄を t+1 の始値で買い、前の保有も t+1 の始値で売ること"""
        dates, score_matrix, close_matrix, open_matrix = self._matrices()

        equity, holdings = _run(score_matrix, close_matrix, open_matrix, [dates[0], dates[1]])

        # 決めた日 t には約定しない
        assert equity.loc[_day(dates, 0), "portfolio_value"] == pytest.approx(10_000)
        # t+1 の始値 125 で A を 80 株、t+1 の終値 110 で評価
        assert equity.loc[_day(dates, 1), "portfolio_value"] == pytest.approx(8_800)
        # t+2 の始値 115 で A を売り 9,200、始値 40 で B を 230 株、終値 50 で評価
        assert equity.loc[_day(dates, 2), "portfolio_value"] == pytest.approx(11_500)

        a_buy = [h for h in holdings if h["symbol"] == "A"][0]
        assert a_buy["price"] == pytest.approx(125.0)
        assert a_buy["qty"] == 80
        assert a_buy["rebalance_date"] == _day(dates, 0)
        assert a_buy["fill_date"] == _day(dates, 1)

    def test_equal_weight_fills_at_next_open(self):
        """等分の基準も t+1 の始値で売買すること"""
        dates, score_matrix, close_matrix, open_matrix = self._matrices()

        equity, _ = _run(score_matrix, close_matrix, open_matrix, [dates[0], dates[1]])

        assert equity.loc[_day(dates, 0), "equal_weight_value"] == pytest.approx(10_000)
        # A 40 株 @125・B 100 株 @50 を買い、終値で 40*110 + 100*50
        assert equity.loc[_day(dates, 1), "equal_weight_value"] == pytest.approx(9_400)
        # 始値で 40*115 + 100*40 = 8,600 に戻し、4,300 ずつ A 37 株 @115・B 107 株 @40
        # 残りの現金 65 と、終値で 37*120 + 107*50
        assert equity.loc[_day(dates, 2), "equal_weight_value"] == pytest.approx(9_855)

    def test_last_day_score_is_not_filled(self):
        """最後の日のスコアは約定させないこと"""
        dates, score_matrix, close_matrix, open_matrix = self._matrices()

        equity, holdings = _run(score_matrix, close_matrix, open_matrix, [dates[3]])

        assert holdings == []
        assert equity.loc[_day(dates, 3), "portfolio_value"] == pytest.approx(10_000)
        assert equity.loc[_day(dates, 3), "equal_weight_value"] == pytest.approx(10_000)

    def test_no_buy_without_next_open(self):
        """t+1 に始値の無い銘柄は買わないこと"""
        dates = pd.date_range("2026-01-05", periods=3, freq="B")
        score_matrix = pd.DataFrame({"A": [0.9, 0.9, 0.9]}, index=dates)
        close_matrix = pd.DataFrame({"A": [100.0, 100.0, 100.0]}, index=dates)
        open_matrix = pd.DataFrame({"A": [100.0, _NAN, 100.0]}, index=dates)

        equity, holdings = _run(score_matrix, close_matrix, open_matrix, [dates[0]])

        assert holdings == []
        assert equity.loc[_day(dates, 1), "portfolio_value"] == pytest.approx(10_000)
        assert equity.loc[_day(dates, 1), "equal_weight_value"] == pytest.approx(10_000)
