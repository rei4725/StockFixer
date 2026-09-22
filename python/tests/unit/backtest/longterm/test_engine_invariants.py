"""エンジンの台帳・equity 整合（不変条件）。"""

import unittest
from unittest.mock import patch

import pandas as pd

from src.backtest.longterm import engine
from src.backtest.longterm.config import LongtermBacktestConfig
from src.backtest.longterm.portfolio import Portfolio
from src.domain.types import TrendCandidate

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


def _run_with_portfolio(**kw):
    """`_run` と同じシナリオを実行し、内部で使われた `Portfolio` も返す。

    `run_longterm_backtest` の戻り値シグネチャ（3-tuple）は他のファサード・CLI・
    既存テストに波及するため変更しない。代わりに `engine.Portfolio` を薄い
    スパイ（コンストラクタで自分自身を捕捉するだけのサブクラス）で差し替える、
    最小侵襲なテスト側だけの観測手段を使う。`load_price_map` 等を
    `patch.object(engine, ...)` で差し替えている既存の `_run` と同じ作法。
    """
    captured: dict = {}

    class _SpyPortfolio(Portfolio):
        def __init__(self, cash):
            super().__init__(cash)
            captured["portfolio"] = self

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
    ), patch.object(
        engine, "Portfolio", side_effect=_SpyPortfolio
    ):
        equity, metrics, trades = engine.run_longterm_backtest(cfg)
    return equity, metrics, trades, captured["portfolio"], cfg


class TestLedgerInvariants(unittest.TestCase):
    def test_equity_never_negative(self):
        equity, _, _ = _run(max_positions=2)
        self.assertTrue((equity["portfolio_value"] >= 0).all())

    def test_final_equity_matches_cash_plus_holdings(self):
        """最終行の equity は現金 + 保有評価に一致する（Portfolio.equity の定義）。

        旧版は `metrics["final_cash"]` と比較していたが、`final_cash` は
        `equity_df` の最終行から `compute_longterm_metrics` が丸めて複製した
        だけの値であり、比較しても `round(x, 2) == x` に帰着するトートロジー
        だった（`src/backtest/longterm/metrics.py` 参照）。ここでは equity_df
        とは独立に追跡されている `Portfolio.cash` / 保有ポジションの時価評価
        （フィクスチャの終値から直接算出）と突き合わせる。
        """
        equity, _, _, portfolio, _ = _run_with_portfolio(max_positions=2)
        last_date = equity["date"].iloc[-1]
        closes = _closes()
        last_close = {
            sym: float(
                closes[(closes["symbol"] == sym) & (closes["ts"] == last_date)]["close"].iloc[0]
            )
            for sym in ("UP", "FLAT")
        }
        holdings_value = sum(
            pos.market_value(last_close[symbol]) for symbol, pos in portfolio.positions.items()
        )
        self.assertAlmostEqual(
            float(equity["portfolio_value"].iloc[-1]),
            portfolio.cash + holdings_value,
            places=2,
        )

    def test_ledger_net_cash_flow_matches_cash_drawdown(self):
        """台帳（`portfolio.ledger`）の現金インパクトが実際の現金減少と一致する。

        `TradeLedger.net_cash_flow()`（買付支払 − 全売却受取）は
        `initial_cash - 最終現金残高` に一致すべき不変条件。フィクスチャは
        UP を 10 → 約72.5 まで線形上昇させるため、40週トレーリング MA は
        期間中ずっと窓未充足（=判定なし）、価格は単調増加なので trailing
        stop も発火しない。結果として scale_out（2x/5x 部分利確）が実際に
        発生し、台帳が空のまま 0 == 0 で自明成立する事態にはならない
        （非空・scale_out 存在を明示的に確認する）。
        """
        _, _, _, portfolio, cfg = _run_with_portfolio(max_positions=2)
        ledger_df = portfolio.ledger.to_frame()
        self.assertFalse(ledger_df.empty)
        self.assertTrue((ledger_df["action"] == "scale_out").any())
        self.assertAlmostEqual(
            portfolio.ledger.net_cash_flow(),
            cfg.initial_cash - portfolio.cash,
            places=2,
        )

    def test_trades_have_expected_columns(self):
        _, _, trades = _run(max_positions=2)
        for col in ("symbol", "entry_date", "exit_date", "multiple", "max_multiple"):
            self.assertIn(col, trades.columns)


if __name__ == "__main__":
    unittest.main()
