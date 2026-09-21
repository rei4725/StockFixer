"""エンジンの台帳・equity 整合（不変条件）。"""

import unittest
from unittest.mock import patch

import pandas as pd

from src.backtest.longterm import engine
from src.backtest.longterm.config import LongtermBacktestConfig
from src.screening.types import TrendCandidate

_START = "2024-01-02"
_END = "2024-06-28"


def _closes():
    dates = pd.bdate_range(_START, _END).strftime("%Y-%m-%d")
    rows = []
    for i, d in enumerate(dates):
        rows.append({"symbol": "UP", "ts": d, "close": 10.0 + i * 0.5})
        rows.append({"symbol": "FLAT", "ts": d, "close": 20.0})
    return pd.DataFrame(rows)


def _candidate(symbol, score):
    """TrendCandidate は 10 フィールドすべてが必須。"""
    return TrendCandidate(
        market="us",
        symbol=symbol,
        score=score,
        close=10.0,
        dist_from_52w_high=-0.05,
        above_200dma=True,
        sma200_rising=True,
        return_6m=0.3,
        return_12m=0.6,
        avg_volume=1_000_000.0,
    )


def _run(**kw):
    def _load(market, end_date=None):
        df = _closes()
        if end_date is not None:
            df = df[df["ts"] <= str(end_date)]
        return df.reset_index(drop=True)

    def _screen(market, top_n, as_of):
        return [_candidate("UP", 1.0), _candidate("FLAT", 0.5)]

    cfg = LongtermBacktestConfig.build(
        market="us", start=_START, end=_END, rescreen_freq="monthly", **kw
    )
    with patch.object(
        engine, "load_price_map", side_effect=lambda m, e: _price_map(_load(m, e))
    ), patch.object(engine, "screen_trend_candidates", side_effect=_screen), patch.object(
        engine,
        "fetch_benchmark_returns",
        return_value={"ticker": "^GSPC", "total_return": 0.1},
    ):
        return engine.run_longterm_backtest(cfg)


def _price_map(raw):
    out = {}
    norm = pd.DataFrame(
        {
            "symbol": raw["symbol"],
            "date": raw["ts"].astype(str),
            "Close": raw["close"].astype(float),
        }
    )
    for sym, g in norm.groupby("symbol", sort=False):
        out[str(sym)] = g[["date", "Close"]].sort_values("date").reset_index(drop=True)
    return out


class TestLedgerInvariants(unittest.TestCase):
    def test_equity_never_negative(self):
        equity, _, _ = _run(max_positions=2)
        self.assertTrue((equity["portfolio_value"] >= 0).all())

    def test_final_equity_matches_cash_plus_holdings(self):
        """最終行の equity は現金 + 保有評価に一致する（Portfolio.equity の定義）。"""
        equity, metrics, _ = _run(max_positions=2)
        self.assertAlmostEqual(
            float(equity["portfolio_value"].iloc[-1]), metrics["final_cash"], places=2
        )

    def test_trades_have_expected_columns(self):
        _, _, trades = _run(max_positions=2)
        for col in ("symbol", "entry_date", "exit_date", "multiple", "max_multiple"):
            self.assertIn(col, trades.columns)


if __name__ == "__main__":
    unittest.main()
