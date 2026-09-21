# 長期バックテストのクラス分割 実装計画（Phase 2）

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `longterm_backtest.py` の生 dict と 11 引数を `LongtermBacktestConfig` / `Portfolio` / `OpenPosition` に置き換え、取引台帳を敷き、整数株とエントリー約定ラグを導入する。

**Architecture:** `src/backtest/longterm/` パッケージに config / portfolio / ledger / prices / engine / metrics / reporting を分割する。`src/backtest/longterm_backtest.py` は re-export ファサードとして残し、CLI の import 互換を保つ。メトリクスは `core.compute_metrics` を呼ばず、`metrics/stats.py` に抽出した純粋統計関数のみを共有する。

**Tech Stack:** Python 3.12 / pandas / dataclasses / pytest / mypy / flake8 / black / isort / import-linter

**Spec:** `docs/superpowers/specs/2026-09-21-longterm-portfolio-types-design.md`

## Global Constraints

- 作業ブランチは既存の `feature/execution-model`。ベースは `develop`。
- コミットメッセージは Conventional Commits（`feat:` / `fix:` / `refactor:` / `test:` / `docs:` / `chore:`）。
- 全コマンドは `python/` ディレクトリから実行する。Windows では `py` ではなく `python -m pytest` を使う。
- **ローカルのユニットテストは空 DB でないと 86 万行の dev DB を読んで timeout する。** 実行前に `STOCKFIXER_DB_PATH` 等でテスト用 DB に向けるか、DB に触れないテストのみを対象に絞ること。
- `pre-commit` の commit-msg フックは cp932 で落ちることがある。`PYTHONUTF8=1` を設定して回避する。
- **`src/backtest/metrics/core.py` の計算式は 1 文字も変更しない。** `compute_metrics` の `sharpe_ratio` は戦略ファクトリーの合格ゲートの入力である。
- **`src/screening/hold_engine.py` には手を入れない。** 本番のスクリーニング経路が共用している。
- `.env` / `src/env/**` / `*.pem` は読み書きしない。
- VERSION は develop の最新を基準に +1 する。PR-4 は patch、PR-5 と PR-6 は minor。
- PR ボディには `## version_impact` / `## version_rationale` / `## VERSION 更新` / `## VERSION 未更新理由` の 4 見出しが必須。
- PR のマージは squash ではなく通常 merge。`gh pr merge` は auto mode が拒否するため、ユーザーが手動実行する。

## ファイル構成

| ファイル | 責務 |
|---|---|
| `src/backtest/longterm/__init__.py` | 公開 API の再輸出 |
| `src/backtest/longterm/config.py` | `LongtermBacktestConfig`（frozen dataclass） |
| `src/backtest/longterm/ledger.py` | `TradeRecord` / `TradeLedger` |
| `src/backtest/longterm/portfolio.py` | `OpenPosition` / `ClosedTrade` / `Portfolio` |
| `src/backtest/longterm/prices.py` | 価格ロード・カレンダー・リスクリーン日・close 索引 |
| `src/backtest/longterm/engine.py` | `run_longterm_backtest` / `enter_candidates` |
| `src/backtest/longterm/metrics.py` | 倍率系メトリクス + stats 呼び出し |
| `src/backtest/longterm/reporting.py` | `save_results` / `build_conclusion` |
| `src/backtest/metrics/stats.py` | 純粋統計関数（core から抽出） |
| `src/backtest/longterm_backtest.py` | re-export ファサード |

## PR の区切り

| PR | タスク | 振る舞い | version_impact |
|---|---|---|---|
| PR-4 | Task 1〜6 | 不変 | patch |
| PR-5 | Task 7〜8 | 変わる | minor |
| PR-6 | Task 9〜11 | 指標追加・文言 | minor |

**PR-4 が振る舞い不変であることは、既存ユニットテストが数値を変えずに通ることで証明する。** ここで数値が動いたら型分割にバグがある。先に進んではならない。

---

# PR-4: 型分割（振る舞い不変）

## Task 1: `longterm/` パッケージを作り価格ヘルパーを移設する

**Files:**
- Create: `src/backtest/longterm/__init__.py`
- Create: `src/backtest/longterm/prices.py`
- Modify: `src/backtest/longterm_backtest.py`
- Test: `tests/unit/backtest/longterm/test_prices.py`

**Interfaces:**
- Consumes: `src.utils.db.market_data.load_raw_closes`
- Produces: `load_price_map(market: str, end: str) -> dict[str, pd.DataFrame]`, `build_calendar(price_map, start, end) -> list[str]`, `make_rescreen_dates(calendar, start, freq) -> list[str]`, `close_lookups(price_map, calendar) -> dict[str, dict[str, float]]`, `FREQ_OFFSETS: dict[str, pd.DateOffset]`

- [ ] **Step 1: 空パッケージを作る**

```bash
mkdir -p src/backtest/longterm
printf '"""長期コホート・バックテストのパッケージ。"""\n' > src/backtest/longterm/__init__.py
```

- [ ] **Step 2: 失敗するテストを書く**

`tests/unit/backtest/longterm/test_prices.py`:

```python
"""longterm/prices.py の価格ヘルパー。"""

import unittest

import pandas as pd

from src.backtest.longterm import prices


def _frame(dates, closes):
    return pd.DataFrame({"date": dates, "Close": closes})


class TestBuildCalendar(unittest.TestCase):
    def test_union_of_trading_days_within_window(self):
        price_map = {
            "A": _frame(["2024-01-02", "2024-01-03"], [10.0, 11.0]),
            "B": _frame(["2024-01-03", "2024-01-04"], [20.0, 21.0]),
        }
        cal = prices.build_calendar(price_map, "2024-01-01", "2024-01-03")
        self.assertEqual(cal, ["2024-01-02", "2024-01-03"])


class TestMakeRescreenDates(unittest.TestCase):
    def test_quarterly_rounds_forward_to_trading_day(self):
        cal = ["2024-01-02", "2024-04-02", "2024-07-02"]
        out = prices.make_rescreen_dates(cal, "2024-01-01", "quarterly")
        self.assertEqual(out, ["2024-01-02", "2024-04-02", "2024-07-02"])

    def test_unknown_freq_falls_back_to_quarterly(self):
        cal = ["2024-01-02", "2024-04-02"]
        self.assertEqual(
            prices.make_rescreen_dates(cal, "2024-01-01", "bogus"),
            prices.make_rescreen_dates(cal, "2024-01-01", "quarterly"),
        )

    def test_empty_calendar_returns_empty(self):
        self.assertEqual(prices.make_rescreen_dates([], "2024-01-01", "quarterly"), [])


class TestCloseLookups(unittest.TestCase):
    def test_forward_fills_missing_days(self):
        price_map = {"A": _frame(["2024-01-02", "2024-01-04"], [10.0, 12.0])}
        cal = ["2024-01-02", "2024-01-03", "2024-01-04"]
        out = prices.close_lookups(price_map, cal)
        self.assertEqual(out["A"]["2024-01-03"], 10.0)
        self.assertEqual(out["A"]["2024-01-04"], 12.0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3: テストが落ちることを確認する**

Run: `python -m pytest tests/unit/backtest/longterm/test_prices.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'src.backtest.longterm.prices'`）

- [ ] **Step 4: `prices.py` に 4 関数を移設する**

`src/backtest/longterm_backtest.py` の `_load_price_map`(53-76) / `_build_calendar`(78-85) / `_make_rescreen_dates`(87-112) / `_close_lookups`(114-124) と `_FREQ_OFFSETS`(45-52) を、先頭の `_` を外して `src/backtest/longterm/prices.py` へ**そのままの実装で**移す。ロジックは一切変えない。必要な import は `bisect_left` / `pandas` / `load_raw_closes` / `get_logger`。

- [ ] **Step 5: ファサードから再輸出する**

`src/backtest/longterm_backtest.py` の該当関数定義を削除し、冒頭の import に差し替える。既存の内部呼び出し名（`_load_price_map` 等）は当面そのまま使えるよう別名を張る。

```python
from src.backtest.longterm.prices import (
    FREQ_OFFSETS as _FREQ_OFFSETS,
    build_calendar as _build_calendar,
    close_lookups as _close_lookups,
    load_price_map as _load_price_map,
    make_rescreen_dates as _make_rescreen_dates,
)
```

- [ ] **Step 6: 新旧テストが通ることを確認する**

Run: `python -m pytest tests/unit/backtest/longterm/test_prices.py tests/unit/test_longterm_backtest.py -v`
Expected: 全 PASS。**既存テストの数値が 1 つも変わっていないこと。**

- [ ] **Step 7: コミット**

```bash
git add src/backtest/longterm/ src/backtest/longterm_backtest.py tests/unit/backtest/longterm/test_prices.py
git commit -m "refactor: 長期バックテストの価格ヘルパーを longterm/prices.py へ移設する"
```

---

## Task 2: `LongtermBacktestConfig` を導入する

**Files:**
- Create: `src/backtest/longterm/config.py`
- Test: `tests/unit/backtest/longterm/test_config.py`

**Interfaces:**
- Consumes: `src.backtest.execution.TradingCosts`, `src.backtest.execution.DEFAULT_FEE_RATE`, `src.screening.types.HoldRules`
- Produces: `LongtermBacktestConfig`（frozen dataclass、フィールドは下記）と `LongtermBacktestConfig.build(...)` クラスメソッド

- [ ] **Step 1: 失敗するテストを書く**

`tests/unit/backtest/longterm/test_config.py`:

```python
"""LongtermBacktestConfig の既定値とコスト解決。"""

import unittest

from src.backtest.longterm.config import LongtermBacktestConfig


class TestBuild(unittest.TestCase):
    def test_us_slippage_default(self):
        cfg = LongtermBacktestConfig.build(market="us")
        self.assertAlmostEqual(cfg.costs.slippage_rate, 0.0005)
        self.assertAlmostEqual(cfg.costs.fee_rate, 0.001)

    def test_jp_slippage_default(self):
        cfg = LongtermBacktestConfig.build(market="jp")
        self.assertAlmostEqual(cfg.costs.slippage_rate, 0.0010)

    def test_explicit_slippage_wins(self):
        cfg = LongtermBacktestConfig.build(market="us", slippage=0.01)
        self.assertAlmostEqual(cfg.costs.slippage_rate, 0.01)

    def test_defaults(self):
        cfg = LongtermBacktestConfig.build(market="us")
        self.assertEqual(cfg.rescreen_freq, "quarterly")
        self.assertEqual(cfg.top_n, 30)
        self.assertEqual(cfg.max_positions, 10)
        self.assertEqual(cfg.n_trials, 0)
        self.assertEqual(cfg.benchmark_ticker, "^GSPC")

    def test_execution_lag_defaults_to_zero_in_pr4(self):
        """PR-4 は振る舞い不変。lag の既定を 1 にするのは PR-5。"""
        cfg = LongtermBacktestConfig.build(market="us")
        self.assertEqual(cfg.execution_lag, 0)

    def test_is_frozen(self):
        cfg = LongtermBacktestConfig.build(market="us")
        with self.assertRaises(Exception):
            cfg.top_n = 99  # type: ignore[misc]

    def test_passthrough_kwargs(self):
        cfg = LongtermBacktestConfig.build(market="us", top_n=5, max_positions=2)
        self.assertEqual(cfg.top_n, 5)
        self.assertEqual(cfg.max_positions, 2)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: テストが落ちることを確認する**

Run: `python -m pytest tests/unit/backtest/longterm/test_config.py -v`
Expected: FAIL（`ModuleNotFoundError`）

- [ ] **Step 3: `config.py` を書く**

```python
"""長期コホート・バックテストの設定値。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from src.backtest.execution import DEFAULT_FEE_RATE, TradingCosts
from src.screening.types import HoldRules


@dataclass(frozen=True)
class LongtermBacktestConfig:
    """1 回の長期バックテストを定義する不変の設定。

    execution_lag: エントリー約定をリスクリーン日から何営業日ずらすか。
                   0 = 当日 Close 約定（従来）、1 = 翌営業日 Close 約定。
    n_trials:      DSR（過学習ガード）の試行回数。0 なら DSR を計算しない。
    """

    market: str = "us"
    start: str = "2021-01-01"
    end: str = "2026-01-01"
    rescreen_freq: str = "quarterly"
    top_n: int = 30
    initial_cash: float = 1_000_000.0
    max_positions: int = 10
    execution_lag: int = 0
    benchmark_ticker: str = "^GSPC"
    n_trials: int = 0
    costs: TradingCosts = field(default_factory=TradingCosts)
    rules: HoldRules = field(default_factory=HoldRules)

    @classmethod
    def build(
        cls,
        *,
        market: str = "us",
        fee_rate: float = DEFAULT_FEE_RATE,
        slippage: Optional[float] = None,
        rules: Optional[HoldRules] = None,
        **kwargs: Any,
    ) -> "LongtermBacktestConfig":
        """市場別スリッページ既定の解決を含む組み立て。CLI はこれを呼ぶ。"""
        return cls(
            market=market,
            costs=TradingCosts.for_market(market, fee_rate, slippage),
            rules=rules or HoldRules(),
            **kwargs,
        )
```

- [ ] **Step 4: テストが通ることを確認する**

Run: `python -m pytest tests/unit/backtest/longterm/test_config.py -v`
Expected: 7 passed

- [ ] **Step 5: コミット**

```bash
git add src/backtest/longterm/config.py tests/unit/backtest/longterm/test_config.py
git commit -m "feat: LongtermBacktestConfig を導入する"
```

---

## Task 3: 取引台帳を導入する

**Files:**
- Create: `src/backtest/longterm/ledger.py`
- Test: `tests/unit/backtest/longterm/test_ledger.py`

**Interfaces:**
- Consumes: なし（pandas のみ）
- Produces: `TradeRecord`（frozen dataclass: `date: str`, `action: str`, `symbol: str`, `price: float`, `qty: float`, `amount: float`, `cash: float`）、`TradeLedger` with `record_buy(date, symbol, price, qty, amount, cash) -> None` / `record_sell(date, symbol, price, qty, amount, cash, action="sell") -> None` / `to_frame() -> pd.DataFrame` / `net_cash_flow() -> float`

`amount` は符号なしの約定金額（買いは支払額、売りは受取額）。`cash` は約定**後**の現金残高。

- [ ] **Step 1: 失敗するテストを書く**

`tests/unit/backtest/longterm/test_ledger.py`:

```python
"""TradeLedger の記録と再構成。"""

import unittest

from src.backtest.longterm.ledger import TradeLedger


class TestLedger(unittest.TestCase):
    def test_records_in_order(self):
        led = TradeLedger()
        led.record_buy("2024-01-02", "A", 10.0, 100, 1001.0, 8999.0)
        led.record_sell("2024-02-02", "A", 12.0, 100, 1198.0, 10197.0)
        df = led.to_frame()
        self.assertEqual(list(df["action"]), ["buy", "sell"])
        self.assertEqual(list(df["symbol"]), ["A", "A"])
        self.assertEqual(
            list(df.columns),
            ["date", "action", "symbol", "price", "qty", "amount", "cash"],
        )

    def test_net_cash_flow_matches_cash_delta(self):
        """買い支払 − 売り受取 が現金の減少分と一致する。"""
        led = TradeLedger()
        led.record_buy("2024-01-02", "A", 10.0, 100, 1001.0, 8999.0)
        led.record_sell("2024-02-02", "A", 12.0, 100, 1198.0, 10197.0)
        self.assertAlmostEqual(led.net_cash_flow(), 1001.0 - 1198.0)

    def test_scale_out_action_label(self):
        led = TradeLedger()
        led.record_sell("2024-02-02", "A", 12.0, 20, 239.0, 239.0, action="scale_out")
        self.assertEqual(led.to_frame()["action"].iloc[0], "scale_out")

    def test_empty_frame_has_columns(self):
        df = TradeLedger().to_frame()
        self.assertTrue(df.empty)
        self.assertIn("cash", df.columns)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: テストが落ちることを確認する**

Run: `python -m pytest tests/unit/backtest/longterm/test_ledger.py -v`
Expected: FAIL（`ModuleNotFoundError`）

- [ ] **Step 3: `ledger.py` を書く**

```python
"""長期バックテストの取引台帳。

役割は 3 つ。
  1. 監査証跡
  2. 「現金 + 保有評価 == equity」の不変条件をテストする根拠
  3. 銘柄・数量を取り違えない損益列の出所

metrics/core.compute_metrics には渡さない。core の FIFO 突合は銘柄も数量も
見ないため、多銘柄同時保有では損益が誤る（spec 参照）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd

_COLUMNS = ["date", "action", "symbol", "price", "qty", "amount", "cash"]


@dataclass(frozen=True)
class TradeRecord:
    """1 件の約定。amount は符号なし金額、cash は約定後の現金残高。"""

    date: str
    action: str
    symbol: str
    price: float
    qty: float
    amount: float
    cash: float


class TradeLedger:
    """約定を発生順に追記する台帳。"""

    def __init__(self) -> None:
        self.records: List[TradeRecord] = []

    def record_buy(
        self, date: str, symbol: str, price: float, qty: float, amount: float, cash: float
    ) -> None:
        self.records.append(TradeRecord(date, "buy", symbol, price, qty, amount, cash))

    def record_sell(
        self,
        date: str,
        symbol: str,
        price: float,
        qty: float,
        amount: float,
        cash: float,
        action: str = "sell",
    ) -> None:
        self.records.append(TradeRecord(date, action, symbol, price, qty, amount, cash))

    def net_cash_flow(self) -> float:
        """買いの支払総額 − 売りの受取総額。現金の減少分と一致すべき値。"""
        paid = sum(r.amount for r in self.records if r.action == "buy")
        received = sum(r.amount for r in self.records if r.action != "buy")
        return float(paid - received)

    def to_frame(self) -> pd.DataFrame:
        if not self.records:
            return pd.DataFrame(columns=_COLUMNS)
        return pd.DataFrame([vars(r) for r in self.records], columns=_COLUMNS)
```

- [ ] **Step 4: テストが通ることを確認する**

Run: `python -m pytest tests/unit/backtest/longterm/test_ledger.py -v`
Expected: 4 passed

- [ ] **Step 5: コミット**

```bash
git add src/backtest/longterm/ledger.py tests/unit/backtest/longterm/test_ledger.py
git commit -m "feat: 長期バックテストの取引台帳を追加する"
```

---

## Task 4: `OpenPosition` / `ClosedTrade` / `Portfolio` を導入する

**Files:**
- Create: `src/backtest/longterm/portfolio.py`
- Test: `tests/unit/backtest/longterm/test_portfolio.py`

**Interfaces:**
- Consumes: `src.backtest.execution.ExecutionModel`, `src.backtest.longterm.ledger.TradeLedger`, `src.screening.types.PositionEvent`
- Produces:
  - `OpenPosition(symbol, entry_date, entry_price, shares, current_hf=1.0, cost_basis=0.0, realized=0.0, events_by_date=...)` と `held_shares() -> float` / `market_value(price: float) -> float`
  - `ClosedTrade`（frozen、フィールドは spec の通り）
  - `Portfolio(cash: float)` と `enter(pos) -> None` / `scale_out(symbol, ev, execution) -> None` / `close(symbol, ev, execution, max_multiple) -> ClosedTrade` / `equity(close_lookups, date) -> float`

**注記（spec からの意図的なずれ）**: spec は `shares: int` と書いているが、PR-4 は振る舞い不変のため `shares: float`（端株）で導入する。`int` への変更は Task 7 で行う。

- [ ] **Step 1: 失敗するテストを書く**

`tests/unit/backtest/longterm/test_portfolio.py`:

```python
"""OpenPosition / Portfolio の現金推移と保有計算。"""

import unittest

from src.backtest.execution import ExecutionModel, TradingCosts
from src.backtest.longterm.portfolio import OpenPosition, Portfolio
from src.screening.types import PositionEvent


def _exec(fee=0.0, slip=0.0):
    return ExecutionModel(TradingCosts(fee_rate=fee, slippage_rate=slip))


def _pos(shares=100.0, price=10.0):
    return OpenPosition(
        symbol="A",
        entry_date="2024-01-02",
        entry_price=price,
        shares=shares,
        cost_basis=shares * price,
    )


def _ev(action, price, held_fraction, reason="x", date="2024-02-02"):
    return PositionEvent(
        date=date,
        action=action,
        price=price,
        held_fraction=held_fraction,
        reason=reason,
        multiple=price / 10.0,
    )


class TestOpenPosition(unittest.TestCase):
    def test_held_shares_reflects_scale_out(self):
        pos = _pos()
        pos.current_hf = 0.8
        self.assertAlmostEqual(pos.held_shares(), 80.0)

    def test_market_value(self):
        pos = _pos()
        pos.current_hf = 0.5
        self.assertAlmostEqual(pos.market_value(12.0), 600.0)


class TestPortfolio(unittest.TestCase):
    def test_enter_deducts_cost_basis(self):
        pf = Portfolio(cash=10_000.0)
        pf.enter(_pos())
        self.assertAlmostEqual(pf.cash, 9_000.0)
        self.assertIn("A", pf.positions)
        self.assertEqual(pf.ledger.to_frame()["action"].iloc[0], "buy")

    def test_scale_out_adds_proceeds_and_lowers_hf(self):
        pf = Portfolio(cash=10_000.0)
        pf.enter(_pos())
        pf.scale_out("A", _ev("scale_out", 20.0, 0.8), _exec())
        # 20% = 20株 を 20.0 で売却 → +400
        self.assertAlmostEqual(pf.cash, 9_400.0)
        self.assertAlmostEqual(pf.positions["A"].current_hf, 0.8)
        self.assertAlmostEqual(pf.positions["A"].realized, 400.0)

    def test_close_returns_trade_with_realized_pnl(self):
        pf = Portfolio(cash=10_000.0)
        pf.enter(_pos())
        trade = pf.close("A", _ev("exit", 15.0, 0.0), _exec(), max_multiple=1.5)
        self.assertNotIn("A", pf.positions)
        self.assertAlmostEqual(pf.cash, 10_500.0)
        # 取得原価 1000 に対し受取 1500
        self.assertAlmostEqual(trade.realized_pnl, 500.0)
        self.assertAlmostEqual(trade.return_rate, 0.5)
        self.assertAlmostEqual(trade.multiple, 1.5)
        self.assertEqual(trade.exit_reason, "x")

    def test_realized_pnl_counts_scale_out(self):
        pf = Portfolio(cash=10_000.0)
        pf.enter(_pos())
        pf.scale_out("A", _ev("scale_out", 20.0, 0.8), _exec())
        trade = pf.close("A", _ev("exit", 15.0, 0.0), _exec(), max_multiple=2.0)
        # 受取 = 400（部分利確） + 80株 * 15 = 1200 → 計 1600、原価 1000
        self.assertAlmostEqual(trade.realized_pnl, 600.0)

    def test_cash_never_negative_on_costs(self):
        pf = Portfolio(cash=10_000.0)
        pf.enter(_pos())
        pf.close("A", _ev("exit", 1.0, 0.0), _exec(fee=0.001, slip=0.0005))
        self.assertGreaterEqual(pf.cash, 0.0)

    def test_equity_sums_cash_and_holdings(self):
        pf = Portfolio(cash=10_000.0)
        pf.enter(_pos())
        lookups = {"A": {"2024-03-01": 12.0}}
        self.assertAlmostEqual(pf.equity(lookups, "2024-03-01"), 9_000.0 + 1_200.0)

    def test_equity_missing_price_treated_as_zero(self):
        pf = Portfolio(cash=10_000.0)
        pf.enter(_pos())
        self.assertAlmostEqual(pf.equity({"A": {}}, "2024-03-01"), 9_000.0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: テストが落ちることを確認する**

Run: `python -m pytest tests/unit/backtest/longterm/test_portfolio.py -v`
Expected: FAIL（`ModuleNotFoundError`）

- [ ] **Step 3: `portfolio.py` を書く**

```python
"""長期バックテストの保有ポジションとポートフォリオ集約。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from src.backtest.execution import ExecutionModel
from src.backtest.longterm.ledger import TradeLedger
from src.screening.types import PositionEvent


@dataclass
class OpenPosition:
    """保有中の 1 銘柄。

    cost_basis は手数料・スリッページ込みの取得原価で、損益計算の正本である。
    realized は部分利確で既に受け取った金額の累計。
    """

    symbol: str
    entry_date: str
    entry_price: float
    shares: float
    current_hf: float = 1.0
    cost_basis: float = 0.0
    realized: float = 0.0
    events_by_date: Dict[str, List[PositionEvent]] = field(default_factory=dict)

    def held_shares(self) -> float:
        return self.shares * self.current_hf

    def market_value(self, price: float) -> float:
        return self.held_shares() * price


@dataclass(frozen=True)
class ClosedTrade:
    """決済済みポジション 1 件。trades_df の 1 行 + 損益。"""

    symbol: str
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    multiple: float
    max_multiple: float
    exit_reason: str
    held_days: int
    realized_pnl: float
    return_rate: float


class Portfolio:
    """現金と保有ポジションを持つ可変の集約。

    日次ループで 1200 回以上更新されるため frozen にはしない。
    """

    def __init__(self, cash: float) -> None:
        self.cash = cash
        self.positions: Dict[str, OpenPosition] = {}
        self.ledger = TradeLedger()
        self.closed: List[ClosedTrade] = []

    def enter(self, pos: OpenPosition) -> None:
        """取得原価 pos.cost_basis を現金から引いて保有に加える。"""
        self.cash -= pos.cost_basis
        self.positions[pos.symbol] = pos
        self.ledger.record_buy(
            pos.entry_date, pos.symbol, pos.entry_price, pos.shares, pos.cost_basis, self.cash
        )

    def scale_out(self, symbol: str, ev: PositionEvent, execution: ExecutionModel) -> None:
        """部分利確。保有比率の減少分だけ売却して現金を回収する。"""
        pos = self.positions[symbol]
        delta = pos.current_hf - ev.held_fraction
        if delta <= 0:
            return
        sold = pos.shares * delta
        proceeds = execution.sell_proceeds(sold, ev.price)
        self.cash += proceeds
        pos.realized += proceeds
        pos.current_hf = ev.held_fraction
        self.ledger.record_sell(
            ev.date, symbol, ev.price, sold, proceeds, self.cash, action="scale_out"
        )

    def close(
        self,
        symbol: str,
        ev: PositionEvent,
        execution: ExecutionModel,
        max_multiple: float,
    ) -> ClosedTrade:
        """残りを全売却してポジションを閉じ、確定した ClosedTrade を返す。"""
        import pandas as pd

        pos = self.positions.pop(symbol)
        sold = pos.held_shares()
        proceeds = execution.sell_proceeds(sold, ev.price)
        self.cash += proceeds
        pos.realized += proceeds
        pos.current_hf = 0.0
        self.ledger.record_sell(ev.date, symbol, ev.price, sold, proceeds, self.cash)

        pnl = pos.realized - pos.cost_basis
        rate = pnl / pos.cost_basis if pos.cost_basis > 0 else 0.0
        mult = ev.price / pos.entry_price if pos.entry_price > 0 else 0.0
        held_days = (pd.Timestamp(ev.date) - pd.Timestamp(pos.entry_date)).days

        trade = ClosedTrade(
            symbol=symbol,
            entry_date=pos.entry_date,
            exit_date=ev.date,
            entry_price=pos.entry_price,
            exit_price=ev.price,
            multiple=mult,
            max_multiple=max_multiple,
            exit_reason=ev.reason,
            held_days=held_days,
            realized_pnl=pnl,
            return_rate=rate,
        )
        self.closed.append(trade)
        return trade

    def equity(self, close_lookups: Dict[str, Dict[str, float]], date: str) -> float:
        """現金 + 保有の時価評価。"""
        held = sum(
            p.market_value(close_lookups.get(p.symbol, {}).get(date, 0.0))
            for p in self.positions.values()
        )
        return self.cash + held
```

`import pandas as pd` は関数内ではなくモジュール冒頭に置くこと（上のコードは可読性のため関数内に書いてあるが、実装では先頭へ移す）。

- [ ] **Step 4: テストが通ることを確認する**

Run: `python -m pytest tests/unit/backtest/longterm/test_portfolio.py -v`
Expected: 8 passed

- [ ] **Step 5: コミット**

```bash
git add src/backtest/longterm/portfolio.py tests/unit/backtest/longterm/test_portfolio.py
git commit -m "feat: OpenPosition / ClosedTrade / Portfolio を導入する"
```

---

## Task 5: エンジンを `Config` + `Portfolio` に載せ替える

**Files:**
- Create: `src/backtest/longterm/engine.py`
- Create: `src/backtest/longterm/metrics.py`
- Create: `src/backtest/longterm/reporting.py`
- Modify: `src/backtest/longterm_backtest.py`（ファサード化）
- Modify: `run_longterm_backtest.py`
- Test: `tests/unit/backtest/longterm/test_engine_invariants.py`

**Interfaces:**
- Consumes: Task 1〜4 の全て
- Produces:
  - `engine.run_longterm_backtest(config: LongtermBacktestConfig) -> tuple[pd.DataFrame, dict, pd.DataFrame]`
  - `engine.enter_candidates(config, portfolio, date, price_map, execution) -> None`
  - `metrics.compute_longterm_metrics(equity_df, trades_df, config, benchmark) -> dict`
  - `reporting.save_results(equity_df, trades_df, market) -> tuple[str, str]`
  - `reporting.build_conclusion(metrics, market, start, end, max_positions) -> str`（引数は PR-4 では現行のまま。変更は Task 11）

**重要**: `run_longterm_backtest` のキーワード引数版も残す。`engine.py` に `run_longterm_backtest(config)` を置き、ファサード側に旧シグネチャのラッパを置く。

- [ ] **Step 1: 不変条件の失敗するテストを書く**

`tests/unit/backtest/longterm/test_engine_invariants.py`:

```python
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
    with patch.object(engine, "load_price_map", side_effect=lambda m, e: _price_map(_load(m, e))), \
         patch.object(engine, "screen_trend_candidates", side_effect=_screen), \
         patch.object(
             engine,
             "fetch_benchmark_returns",
             return_value={"ticker": "^GSPC", "total_return": 0.1},
         ):
        return engine.run_longterm_backtest(cfg)


def _price_map(raw):
    out = {}
    norm = pd.DataFrame(
        {"symbol": raw["symbol"], "date": raw["ts"].astype(str), "Close": raw["close"].astype(float)}
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
```

`_price_map` は `engine.load_price_map` の戻り値と同じ形（`dict[str, DataFrame(date, Close)]`）を作るヘルパーである。

- [ ] **Step 2: テストが落ちることを確認する**

Run: `python -m pytest tests/unit/backtest/longterm/test_engine_invariants.py -v`
Expected: FAIL（`ModuleNotFoundError: src.backtest.longterm.engine`）

- [ ] **Step 3: `metrics.py` と `reporting.py` を移設する**

`longterm_backtest.py` の `_compute_metrics`(358-424) を `longterm/metrics.py` に `compute_longterm_metrics` として移す。**計算式は変えない。** 引数は `(equity_df, trades_df, config, benchmark)` にし、`initial_cash` / `start` / `end` は `config` から取る。

`save_results`(427-434) と `build_conclusion`(436-462) を `longterm/reporting.py` へそのまま移す。

- [ ] **Step 4: `engine.py` を書く**

`longterm_backtest.py` の `_make_trade`(136-164) / `_events_by_date`(126-134、`group_events_by_date` に改名) / `_try_enter`(166-224) / `run_longterm_backtest`(226-356) を `engine.py` へ移し、以下に組み替える。

- `_try_enter` → `enter_candidates(config, portfolio, date, price_map, execution)`。`cash` の引数・戻り値を廃し `portfolio.cash` を読む。`OpenPosition` を組んで `portfolio.enter(pos)` を呼ぶ。`cost_basis` には従来の `spent = per_position` をそのまま入れる（振る舞い不変）。
- `simulate_position` が空を返した場合の巻き戻しは、`portfolio.enter` を呼ぶ**前**にイベント生成を済ませる順序に変えて不要にする。
- 日次ループの `open_positions` 操作を `portfolio.scale_out` / `portfolio.close` に置換する。`close` に渡す `max_multiple` は従来 `_make_trade` が計算していた「entry〜exit 区間の最高終値 / entry_price」を先に求めて渡す。
- 評価額は `portfolio.equity(close_lookups, date)` を使う。
- `trades_df` は `portfolio.closed` の `ClosedTrade` から従来の 9 列に整形して作る（`realized_pnl` / `return_rate` は PR-4 では列に出さない。振る舞い不変のため）。丸めは従来通り `round(..., 4)` / `held_days` は整数。

- [ ] **Step 5: ファサードを書き換える**

`src/backtest/longterm_backtest.py` を再輸出だけの薄いモジュールにする。

```python
"""長期コホート・バックテストのファサード（互換維持用）。

実装は src/backtest/longterm/ に分割済み。本モジュールは CLI の import 互換の
ために残す。テストからのモジュール属性 patch は src.backtest.longterm.engine
を対象にすること（本モジュールへの patch は engine に届かない）。
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from src.backtest.execution import DEFAULT_FEE_RATE
from src.backtest.longterm.config import LongtermBacktestConfig
from src.backtest.longterm.engine import enter_candidates  # noqa: F401
from src.backtest.longterm.engine import run_longterm_backtest as _run_with_config
from src.backtest.longterm.reporting import build_conclusion, save_results  # noqa: F401
from src.screening.types import HoldRules

__all__ = [
    "LongtermBacktestConfig",
    "build_conclusion",
    "run_longterm_backtest",
    "save_results",
]


def run_longterm_backtest(
    market: str = "us",
    start: str = "2021-01-01",
    end: str = "2026-01-01",
    rescreen_freq: str = "quarterly",
    top_n: int = 30,
    initial_cash: float = 1_000_000.0,
    max_positions: int = 10,
    fee_rate: float = DEFAULT_FEE_RATE,
    slippage: Optional[float] = None,
    benchmark_ticker: str = "^GSPC",
    rules: Optional[HoldRules] = None,
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """旧シグネチャの互換ラッパ。新規の呼び出しは Config 版を使うこと。"""
    config = LongtermBacktestConfig.build(
        market=market,
        start=start,
        end=end,
        rescreen_freq=rescreen_freq,
        top_n=top_n,
        initial_cash=initial_cash,
        max_positions=max_positions,
        fee_rate=fee_rate,
        slippage=slippage,
        benchmark_ticker=benchmark_ticker,
        rules=rules,
    )
    return _run_with_config(config)
```

- [ ] **Step 6: CLI を Config 版に切り替える**

`run_longterm_backtest.py` の `main()` で `LongtermBacktestConfig.build(...)` を組み、`from src.backtest.longterm.engine import run_longterm_backtest` を呼ぶ形にする。引数パース以外のロジックは足さない（`run_*.py` は CLI ラッパのみという規約）。

- [ ] **Step 7: 新規テストが通ることを確認する**

Run: `python -m pytest tests/unit/backtest/longterm/ -v`
Expected: 全 PASS

- [ ] **Step 8: コミット**

```bash
git add src/backtest/longterm/ src/backtest/longterm_backtest.py run_longterm_backtest.py tests/unit/backtest/longterm/
git commit -m "refactor: 長期バックテストを Config と Portfolio に載せ替える"
```

---

## Task 6: 既存テストの patch 先を付け替え、PR-4 を仕上げる

**Files:**
- Modify: `tests/unit/test_longterm_backtest.py:130-136, 211-215`
- Modify: `python/VERSION`

- [ ] **Step 1: patch 先を `lb.engine` に付け替える**

`tests/unit/test_longterm_backtest.py` の冒頭 import を足す。

```python
from src.backtest.longterm import engine as lb_engine
```

`patch.object(lb, "load_raw_closes", ...)` は `load_price_map` の内部に移ったため、`patch.object(lb_engine, "load_price_map", ...)` に置き換える（戻り値は `dict[str, pd.DataFrame]`）。`screen_trend_candidates` と `fetch_benchmark_returns` も `lb_engine` を対象にする。**数値アサートは一切書き換えないこと。** 書き換えが必要になったら型分割にバグがある。

- [ ] **Step 2: 既存テストが数値を変えずに通ることを確認する**

Run: `python -m pytest tests/unit/test_longterm_backtest.py -v`
Expected: 全 PASS。数値アサートの期待値を変更していないこと。

- [ ] **Step 3: backtest 関連の全ユニットを回す**

Run: `python -m pytest tests/unit/backtest/ tests/unit/test_longterm_backtest.py tests/unit/test_backtest_metrics.py -v`
Expected: 全 PASS

- [ ] **Step 4: VERSION を patch 上げする**

`python/VERSION` を develop 最新の値 +patch にする。**PowerShell の `Set-Content -Encoding utf8` は BOM を付ける。** 使うなら書いた後に `xxd python/VERSION | head -1` で BOM が無いことを確認する。

- [ ] **Step 5: CI 相当の一括チェックを走らせる**

`ci-preflight` スキルを使う。手動なら:

```bash
cd python && black . && isort . && flake8 . && mypy src/
PYTHONUTF8=1 lint-imports
```

Expected: 全て exit 0

- [ ] **Step 6: コミットして push、PR を作る**

```bash
git add python/VERSION tests/unit/test_longterm_backtest.py
git commit -m "test: 長期バックテストのテスト patch 先を engine へ付け替える"
git push -u origin feature/execution-model
```

PR ボディに必須 4 見出しを書く。`version_impact` は `patch`、`version_rationale` は「振る舞い不変の内部構造変更のため」。

**PR 本文に「既存ユニットテストの数値アサートを 1 件も変更していない」ことを明記する。** これが振る舞い不変の証拠である。

---

# PR-5: 整数株とエントリー約定ラグ（振る舞いが変わる）

## Task 7: 整数株に切り替える

**Files:**
- Modify: `src/backtest/longterm/portfolio.py`（`shares: float` → `int`）
- Modify: `src/backtest/longterm/engine.py`（`enter_candidates` の株数算出）
- Test: `tests/unit/backtest/longterm/test_integer_shares.py`

**Interfaces:**
- Consumes: `ExecutionModel.max_affordable_qty(cash, price) -> int`, `ExecutionModel.buy_cost(qty, price) -> float`
- Produces: `OpenPosition.shares: int`

- [ ] **Step 1: 失敗するテストを書く**

`tests/unit/backtest/longterm/test_integer_shares.py`:

```python
"""エントリー株数が整数で、予算を超えないこと。"""

import unittest

from src.backtest.execution import ExecutionModel, TradingCosts


class TestIntegerShares(unittest.TestCase):
    def test_qty_is_integer_and_within_budget(self):
        ex = ExecutionModel(TradingCosts(fee_rate=0.001, slippage_rate=0.0005))
        budget, price = 100_000.0, 333.33
        qty = ex.max_affordable_qty(budget, price)
        self.assertIsInstance(qty, int)
        self.assertLessEqual(ex.buy_cost(qty, price), budget)
        self.assertGreater(ex.buy_cost(qty + 1, price), budget)

    def test_expensive_symbol_yields_zero(self):
        ex = ExecutionModel(TradingCosts())
        self.assertEqual(ex.max_affordable_qty(100.0, 1_000.0), 0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: テストを走らせる**

Run: `python -m pytest tests/unit/backtest/longterm/test_integer_shares.py -v`
Expected: PASS（`ExecutionModel` は Phase 1 で実装済みのため。これは前提の確認）

- [ ] **Step 3: `enter_candidates` を整数株に変える**

```python
shares = execution.max_affordable_qty(per_position, entry_price)
if shares <= 0:
    continue
cost = execution.buy_cost(shares, entry_price)
pos = OpenPosition(
    symbol=symbol,
    entry_date=entry_date,
    entry_price=entry_price,
    shares=shares,
    cost_basis=cost,
    events_by_date=group_events_by_date(events),
)
portfolio.enter(pos)
```

`OpenPosition.shares` の型注釈を `int` に変える。`held_shares()` は部分利確で端数が出るため戻り値は `float` のまま。

- [ ] **Step 4: 既存テストへの影響を確認する**

Run: `python -m pytest tests/unit/test_longterm_backtest.py tests/unit/backtest/longterm/ -v`
Expected: PASS。**落ちたテストがあれば期待値を新しい値に更新し、差分の理由（端株切り捨て）をコミットメッセージに書く。**

- [ ] **Step 5: コミット**

```bash
git add src/backtest/longterm/ tests/unit/
git commit -m "feat: 長期バックテストのエントリー株数を整数株に統一する"
```

---

## Task 8: エントリー約定ラグを導入する

**Files:**
- Modify: `src/backtest/longterm/config.py`（`execution_lag` 既定を 1 に）
- Modify: `src/backtest/longterm/engine.py`
- Modify: `run_longterm_backtest.py`（`--execution-lag` を追加）
- Modify: `tests/unit/backtest/longterm/test_config.py`
- Test: `tests/unit/backtest/longterm/test_entry_lag.py`

**Interfaces:**
- Consumes: `LongtermBacktestConfig.execution_lag`
- Produces: `enter_candidates(config, portfolio, screen_date, entry_date, price_map, execution)` — スクリーン日と約定日を分離した 6 引数版

- [ ] **Step 1: 失敗するテストを書く**

`tests/unit/backtest/longterm/test_entry_lag.py`:

```python
"""エントリー約定ラグ。"""

import unittest

from src.backtest.longterm.config import LongtermBacktestConfig
from src.backtest.longterm.engine import resolve_entry_date


class TestResolveEntryDate(unittest.TestCase):
    def setUp(self):
        self.cal = ["2024-01-02", "2024-01-03", "2024-01-04"]

    def test_lag_zero_is_same_day(self):
        self.assertEqual(resolve_entry_date(self.cal, "2024-01-02", 0), "2024-01-02")

    def test_lag_one_is_next_trading_day(self):
        self.assertEqual(resolve_entry_date(self.cal, "2024-01-02", 1), "2024-01-03")

    def test_last_day_with_lag_returns_none(self):
        self.assertIsNone(resolve_entry_date(self.cal, "2024-01-04", 1))

    def test_unknown_date_returns_none(self):
        self.assertIsNone(resolve_entry_date(self.cal, "2023-12-31", 1))


class TestConfigDefault(unittest.TestCase):
    def test_execution_lag_defaults_to_one(self):
        self.assertEqual(LongtermBacktestConfig.build(market="us").execution_lag, 1)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: テストが落ちることを確認する**

Run: `python -m pytest tests/unit/backtest/longterm/test_entry_lag.py -v`
Expected: FAIL（`ImportError: cannot import name 'resolve_entry_date'`）

- [ ] **Step 3: `resolve_entry_date` を実装する**

`engine.py` に追加する。

```python
def resolve_entry_date(calendar: list[str], screen_date: str, lag: int) -> Optional[str]:
    """スクリーン日から lag 営業日後の約定日を返す。

    カレンダー末尾を越える場合は None（そのリスクリーン日はエントリーを見送る）。
    """
    i = bisect_left(calendar, screen_date)
    if i >= len(calendar) or calendar[i] != screen_date:
        return None
    j = i + lag
    return calendar[j] if j < len(calendar) else None
```

- [ ] **Step 4: 日次ループで約定日を分離する**

`enter_candidates` の引数を `(config, portfolio, screen_date, entry_date, price_map, execution)` にする。`screen_trend_candidates(as_of=screen_date)` はスクリーン日で、価格の引き当てと `simulate_position(entry_date=entry_date)` は約定日で行う。日次ループ側は次のようにする。

```python
if date in rescreen_dates:
    entry_date = resolve_entry_date(calendar, date, config.execution_lag)
    if entry_date is not None:
        enter_candidates(config, portfolio, date, entry_date, price_map, execution)
```

- [ ] **Step 5: `config.py` の既定値を 1 にし、Task 2 のテストを更新する**

`execution_lag: int = 1` にする。`test_config.py` の `test_execution_lag_defaults_to_zero_in_pr4` を削除し、`test_entry_lag.py` の `TestConfigDefault` に責務を移す。

- [ ] **Step 6: CLI に `--execution-lag` を追加する**

```python
parser.add_argument(
    "--execution-lag",
    type=int,
    default=1,
    help="エントリー約定をリスクリーン日から何営業日ずらすか（0=当日Close）",
)
```

`main()` の `LongtermBacktestConfig.build(...)` に `execution_lag=args.execution_lag` を渡す。

- [ ] **Step 7: テストを走らせる**

Run: `python -m pytest tests/unit/backtest/longterm/ tests/unit/test_longterm_backtest.py -v`
Expected: 全 PASS。既存テストの期待値が動いた場合は新しい値に更新する。

- [ ] **Step 8: lag=0 と lag=1 の対照を取る**

使い捨てスクリプトで同条件・`execution_lag` のみを変えた 2 本を走らせ、`total_return` / `cagr` / `max_drawdown` / `n_trades` を並べた表を作る。**スクリプトはリポジトリに残さない。** 出力はスクラッチパッドに置き、PR 本文へ貼る。

本番 DB の読み取りが auto mode に拒否される場合は、合成データでの対照に切り替え、PR 本文にその旨を明記する。

- [ ] **Step 9: VERSION を minor 上げしてコミット、PR を作る**

```bash
git add src/backtest/longterm/ run_longterm_backtest.py tests/unit/ python/VERSION
git commit -m "feat: 長期バックテストのエントリーを翌営業日約定にする"
git push
```

PR 本文に F-1（端株→整数株）と L-1（lag）の対照表を貼り、悪化幅が理論上限内であることを示す。

---

# PR-6: メトリクスと結論文

## Task 9: `metrics/stats.py` を抽出する

**Files:**
- Create: `src/backtest/metrics/stats.py`
- Modify: `src/backtest/metrics/core.py`
- Modify: `src/backtest/metrics/__init__.py`
- Test: `tests/unit/backtest/test_metrics_stats.py`

**Interfaces:**
- Produces: `max_drawdown(equity: pd.Series) -> float`, `sharpe_per_trade(pnls: list[float], risk_free_per_trade: float = 0.0) -> float`, `annualize_sharpe(spt: float, trades_per_year: float) -> float`, `profit_factor(wins: list[float], losses: list[float]) -> float`, `cagr(initial: float, final: float, years: float) -> float`

- [ ] **Step 1: 失敗するテストを書く**

`tests/unit/backtest/test_metrics_stats.py`:

```python
"""metrics/stats.py の純粋統計関数。"""

import math
import unittest

import pandas as pd

from src.backtest.metrics import stats


class TestMaxDrawdown(unittest.TestCase):
    def test_monotonic_rise_has_no_drawdown(self):
        self.assertAlmostEqual(stats.max_drawdown(pd.Series([1.0, 2.0, 3.0])), 0.0)

    def test_halving_is_minus_half(self):
        self.assertAlmostEqual(stats.max_drawdown(pd.Series([100.0, 50.0])), -0.5)

    def test_empty_series_is_zero(self):
        self.assertAlmostEqual(stats.max_drawdown(pd.Series(dtype=float)), 0.0)


class TestSharpe(unittest.TestCase):
    def test_zero_variance_is_zero(self):
        self.assertAlmostEqual(stats.sharpe_per_trade([1.0, 1.0, 1.0]), 0.0)

    def test_annualize_scales_by_sqrt(self):
        self.assertAlmostEqual(stats.annualize_sharpe(0.5, 4.0), 0.5 * math.sqrt(4.0))

    def test_annualize_zero_frequency_is_zero(self):
        self.assertAlmostEqual(stats.annualize_sharpe(0.5, 0.0), 0.0)


class TestProfitFactor(unittest.TestCase):
    def test_ratio_of_gross_win_to_gross_loss(self):
        self.assertAlmostEqual(stats.profit_factor([10.0, 20.0], [-10.0]), 3.0)

    def test_no_losses_is_infinite(self):
        self.assertEqual(stats.profit_factor([10.0], []), math.inf)

    def test_no_trades_is_zero(self):
        self.assertAlmostEqual(stats.profit_factor([], []), 0.0)


class TestCagr(unittest.TestCase):
    def test_doubling_in_one_year(self):
        self.assertAlmostEqual(stats.cagr(100.0, 200.0, 1.0), 1.0)

    def test_total_loss_is_minus_one(self):
        self.assertAlmostEqual(stats.cagr(100.0, 0.0, 1.0), -1.0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: テストが落ちることを確認する**

Run: `python -m pytest tests/unit/backtest/test_metrics_stats.py -v`
Expected: FAIL（`ImportError`）

- [ ] **Step 3: `stats.py` を作る**

`core.py` の `_max_drawdown`(323-329) / `_sharpe_per_trade`(280-293) / `_annualize_sharpe`(316-320) を**計算式そのままで**移す。`profit_factor` は `core.py` 内の `profit_factor` 算出式を関数化したもの、`cagr` は `longterm_backtest.py` の

```python
cagr = (final_value / initial_cash) ** (1.0 / years) - 1.0 if final_value > 0 else -1.0
```

を関数化したものにする。`years` は呼び出し側で `max(days / 365.25, 1e-9)` を計算して渡す。

- [ ] **Step 4: `core.py` を差し替える**

`core.py` の該当 3 関数の定義を削除し、冒頭に

```python
from src.backtest.metrics.stats import annualize_sharpe, max_drawdown, sharpe_per_trade

_max_drawdown = max_drawdown
_sharpe_per_trade = sharpe_per_trade
_annualize_sharpe = annualize_sharpe
```

を置く。`profit_factor` のインライン式も `stats.profit_factor` 呼び出しに置き換えるが、**結果が一致することを既存テストで確認するまで置き換えない**（式の差異があれば置き換えを見送り、`stats.profit_factor` は longterm 専用として残す）。

- [ ] **Step 5: 既存メトリクステストが無改変で通ることを確認する**

Run: `python -m pytest tests/unit/test_backtest_metrics.py tests/unit/backtest/ -v`
Expected: 全 PASS。**1 件でも落ちたら抽出が式を変えている。差し戻すこと。**

- [ ] **Step 6: Backtester の数値ピン止めが通ることを確認する**

Run: `python -m pytest tests/unit/ -k "backtester" -v`
Expected: 全 PASS

- [ ] **Step 7: コミット**

```bash
git add src/backtest/metrics/ tests/unit/backtest/test_metrics_stats.py
git commit -m "refactor: メトリクスの純粋統計関数を metrics/stats.py に抽出する"
```

---

## Task 10: longterm のメトリクスを拡充する

**Files:**
- Modify: `src/backtest/longterm/metrics.py`
- Modify: `src/backtest/longterm/engine.py`（`ClosedTrade` から損益列を作る）
- Modify: `run_longterm_backtest.py`（出力キー）
- Modify: `tests/unit/test_longterm_backtest.py`（`n_trades` → `num_trades`）
- Test: `tests/unit/backtest/longterm/test_longterm_metrics.py`

**Interfaces:**
- Consumes: `src.backtest.metrics.stats`, `src.backtest.longterm.portfolio.ClosedTrade`
- Produces: `compute_longterm_metrics(equity_df, trades_df, config, benchmark, closed) -> dict`。追加キーは `sharpe_ratio` / `sharpe_per_trade` / `profit_factor` / `calmar_ratio`。`n_trades` は `num_trades` に改名。

- [ ] **Step 1: 失敗するテストを書く**

`tests/unit/backtest/longterm/test_longterm_metrics.py`:

```python
"""longterm メトリクスの損益対応と追加指標。"""

import unittest

import pandas as pd

from src.backtest.longterm.config import LongtermBacktestConfig
from src.backtest.longterm.metrics import compute_longterm_metrics
from src.backtest.longterm.portfolio import ClosedTrade


def _trade(symbol, pnl, rate, multiple):
    return ClosedTrade(
        symbol=symbol,
        entry_date="2024-01-02",
        exit_date="2024-06-28",
        entry_price=10.0,
        exit_price=10.0 * multiple,
        multiple=multiple,
        max_multiple=multiple,
        exit_reason="ma_break",
        held_days=178,
        realized_pnl=pnl,
        return_rate=rate,
    )


def _cfg():
    return LongtermBacktestConfig.build(
        market="us", start="2024-01-02", end="2024-12-31", initial_cash=1000.0
    )


class TestLongtermMetrics(unittest.TestCase):
    def setUp(self):
        self.equity = pd.DataFrame(
            {"date": ["2024-01-02", "2024-06-28"], "portfolio_value": [1000.0, 1400.0]}
        )
        self.closed = [_trade("A", 500.0, 0.5, 1.5), _trade("B", -100.0, -0.2, 0.8)]
        self.trades = pd.DataFrame(
            [
                {"multiple": t.multiple, "max_multiple": t.max_multiple, "held_days": t.held_days}
                for t in self.closed
            ]
        )

    def test_key_is_num_trades(self):
        m = compute_longterm_metrics(self.equity, self.trades, _cfg(), {}, self.closed)
        self.assertEqual(m["num_trades"], 2)
        self.assertNotIn("n_trades", m)

    def test_profit_factor_from_realized_pnl(self):
        """銘柄ごとに正しく対になった損益から算出される（core の FIFO は使わない）。"""
        m = compute_longterm_metrics(self.equity, self.trades, _cfg(), {}, self.closed)
        self.assertAlmostEqual(m["profit_factor"], 5.0)

    def test_sharpe_and_calmar_present(self):
        m = compute_longterm_metrics(self.equity, self.trades, _cfg(), {}, self.closed)
        self.assertIn("sharpe_ratio", m)
        self.assertIn("calmar_ratio", m)

    def test_dsr_absent_when_no_trials(self):
        m = compute_longterm_metrics(self.equity, self.trades, _cfg(), {}, self.closed)
        self.assertNotIn("dsr", m)

    def test_multiple_buckets_preserved(self):
        m = compute_longterm_metrics(self.equity, self.trades, _cfg(), {}, self.closed)
        self.assertEqual(m["n_2x"], 0)
        self.assertIn("pct_2x", m)

    def test_empty_trades_does_not_crash(self):
        m = compute_longterm_metrics(
            self.equity, pd.DataFrame(columns=["multiple", "max_multiple", "held_days"]), _cfg(), {}, []
        )
        self.assertEqual(m["num_trades"], 0)
        self.assertAlmostEqual(m["profit_factor"], 0.0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: テストが落ちることを確認する**

Run: `python -m pytest tests/unit/backtest/longterm/test_longterm_metrics.py -v`
Expected: FAIL

- [ ] **Step 3: `compute_longterm_metrics` を拡張する**

引数に `closed: list[ClosedTrade]` を足す。既存の倍率系・勝率・ベンチマーク比較はそのまま残し、以下を足す。

```python
pnls = [t.realized_pnl for t in closed]
wins = [p for p in pnls if p >= 0]
losses = [p for p in pnls if p < 0]
spt = stats.sharpe_per_trade(pnls)
years = max((pd.Timestamp(config.end) - pd.Timestamp(config.start)).days / 365.25, 1e-9)
trades_per_year = len(pnls) / years if pnls else 0.0
result["sharpe_per_trade"] = round(spt, 6)
result["sharpe_ratio"] = round(stats.annualize_sharpe(spt, trades_per_year), 4)
pf = stats.profit_factor(wins, losses)
result["profit_factor"] = round(pf, 4) if pf != math.inf else None
result["calmar_ratio"] = round(cagr / abs(max_dd), 4) if max_dd < 0 else None
if config.n_trials > 0:
    result["dsr"] = deflated_sharpe_ratio(spt, config.n_trials, len(pnls))
```

`cagr` と `max_dd` は `stats.cagr` / `stats.max_drawdown` に置き換える。`n_trades` キーを `num_trades` に改名する。

- [ ] **Step 4: 呼び出し側を更新する**

`engine.py` が `portfolio.closed` を `compute_longterm_metrics` に渡す。`run_longterm_backtest.py` の `_print_summary` のキー一覧から `n_trades` を除き、`num_trades` / `sharpe_ratio` / `profit_factor` / `calmar_ratio` を足す。`tests/unit/test_longterm_backtest.py` の `metrics["n_trades"]` を `metrics["num_trades"]` に置換する。

- [ ] **Step 5: テストを走らせる**

Run: `python -m pytest tests/unit/backtest/longterm/ tests/unit/test_longterm_backtest.py -v`
Expected: 全 PASS

- [ ] **Step 6: コミット**

```bash
git add src/backtest/longterm/ run_longterm_backtest.py tests/unit/
git commit -m "feat: 長期バックテストに sharpe / profit_factor / calmar を追加する"
```

---

## Task 11: 結論文を設定から組み立てる

**Files:**
- Modify: `src/backtest/longterm/reporting.py`
- Modify: `run_longterm_backtest.py`
- Test: `tests/unit/backtest/longterm/test_conclusion.py`

**Interfaces:**
- Produces: `build_conclusion(metrics: dict, config: LongtermBacktestConfig) -> str`（旧: `(metrics, market, start, end, max_positions)`）

- [ ] **Step 1: 失敗するテストを書く**

`tests/unit/backtest/longterm/test_conclusion.py`:

```python
"""結論文が設定を反映すること。"""

import unittest

from src.backtest.longterm.config import LongtermBacktestConfig
from src.backtest.longterm.reporting import build_conclusion
from src.screening.types import HoldRules

_METRICS = {
    "initial_cash": 1_000_000.0,
    "final_cash": 2_000_000.0,
    "total_return": 1.0,
    "cagr": 0.2,
    "max_drawdown": -0.3,
    "n_2x": 1,
    "n_3x": 0,
    "n_5x": 0,
    "n_10x": 0,
}


class TestBuildConclusion(unittest.TestCase):
    def test_weekly_freq_is_reflected(self):
        cfg = LongtermBacktestConfig.build(market="us", rescreen_freq="weekly")
        self.assertIn("週次スクリーン", build_conclusion(_METRICS, cfg))
        self.assertNotIn("四半期", build_conclusion(_METRICS, cfg))

    def test_quarterly_freq_is_reflected(self):
        cfg = LongtermBacktestConfig.build(market="us", rescreen_freq="quarterly")
        self.assertIn("四半期スクリーン", build_conclusion(_METRICS, cfg))

    def test_trail_ma_weeks_is_reflected(self):
        cfg = LongtermBacktestConfig.build(market="us", rules=HoldRules(trail_ma_weeks=30))
        self.assertIn("30週線割れ", build_conclusion(_METRICS, cfg))

    def test_missing_benchmark_is_stated(self):
        cfg = LongtermBacktestConfig.build(market="us")
        self.assertIn("ベンチマーク比較は", build_conclusion(_METRICS, cfg))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: テストが落ちることを確認する**

Run: `python -m pytest tests/unit/backtest/longterm/test_conclusion.py -v`
Expected: FAIL（引数の数が合わない）

- [ ] **Step 3: `build_conclusion` を書き換える**

シグネチャを `(metrics: dict, config: LongtermBacktestConfig) -> str` にし、文字列リテラルを設定から組み立てる。

```python
_FREQ_LABELS = {
    "weekly": "週次",
    "monthly": "月次",
    "quarterly": "四半期",
    "yearly": "年次",
    "annual": "年次",
}

freq_label = _FREQ_LABELS.get(config.rescreen_freq, config.rescreen_freq)
ma_weeks = config.rules.trail_ma_weeks
```

本文の該当箇所を

```python
f"{config.market} を {config.start}〜{config.end} に{freq_label}スクリーン・"
f"最大{config.max_positions}銘柄保有・{ma_weeks}週線割れまでホールドの規律で運用した場合、"
```

に置き換える。他の文言は変えない。

- [ ] **Step 4: CLI を更新する**

`run_longterm_backtest.py` の `build_conclusion(metrics, args.market, args.start, args.end, args.max_positions)` を `build_conclusion(metrics, config)` にする。

- [ ] **Step 5: テストを走らせる**

Run: `python -m pytest tests/unit/backtest/longterm/ tests/unit/test_longterm_backtest.py -v`
Expected: 全 PASS

- [ ] **Step 6: 全ユニットテストと CI 相当チェックを走らせる**

```bash
python -m pytest tests/unit/ -n 2 -v \
  --ignore=tests/unit/test_predict_unified_lightgbm_alignment.py \
  --cov=src --cov-branch --cov-report= --cov-fail-under=0
```

続けて `ci-preflight` スキルで lint / mypy / import-linter を通す。

- [ ] **Step 7: VERSION を minor 上げしてコミット、push、PR を作る**

```bash
git add src/backtest/longterm/ run_longterm_backtest.py tests/unit/ python/VERSION
git commit -m "fix: 結論文がスクリーン頻度とトレーリング期間を反映するようにする"
git push
```

---

# マージ

- [ ] **Step 1: CI を確認する**

```bash
gh pr checks <番号>
```

全 10 本が pass するまで待つ。

- [ ] **Step 2: ユーザーが手動でマージする**

`gh pr merge` は auto mode の classifier が "Merge Without Review" として拒否する。ユーザーに次を依頼する。

```
! gh pr merge <番号> --merge
```

**`--squash` ではなく `--merge`。** 個別コミットを保持する方針である。

- [ ] **Step 3: マージ後に memory を更新する**

`project_execution_model_phase1.md` に Phase 2 の完了を追記するか、`project_execution_model_phase2.md` を新設して `MEMORY.md` に 1 行足す。記録すべきは「`core.compute_metrics` の FIFO が銘柄非対応である」という発見と、Phase 3（ポート化）が次であること。
