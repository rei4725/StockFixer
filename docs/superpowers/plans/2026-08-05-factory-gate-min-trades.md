# 戦略ファクトリー 銘柄あたり最低取引数ゲート 実装計画

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 銘柄あたりの取引数が少なすぎる銘柄を集計から除外し、少取引アーティファクト（取引2回で Sharpe 25 のような発散値）が戦略ファクトリーのゲートを通過しないようにする。

**Architecture:** 銘柄別メトリクスの集計を `factory.py` から純関数モジュール `factory_aggregation.py` へ切り出し、そこで最低取引数フィルタを適用する。候補と対照群は同じ `evaluate_hypothesis` を通るため、`champion_sharpe` も自動的に浄化される。ゲートには有効銘柄数の下限を追加する。

**Tech Stack:** Python 3.12 / dataclasses / pytest / unittest（既存テストは `unittest.TestCase` スタイル）

**設計書:** `docs/superpowers/specs/2026-08-05-factory-gate-min-trades-design.md`
**Issue:** #625

## Global Constraints

- 作業ブランチ: `fix/factory-gate-min-trades-per-symbol`（作成済み）。PR のベースは `develop`。
- すべてのコマンドは `python/` ディレクトリから実行する。Windows では `py` を使う（`python` ではない）。
- 日本語を含むファイルを扱うため、pytest / スクリプト実行時は `PYTHONUTF8=1` を付ける。
- 新規定数は `python/config/settings.py` に `Field` 定義とモジュール直下の再エクスポートの**両方**を書く（既存の全 `FACTORY_GATE_*` がこの二重定義パターンになっている）。
- `factory.py` は現在 531 行。CI の File Size Guard は **1ファイル600行**が上限。この計画で `factory.py` の行数を増やさないこと。
- 型は dict ではなく dataclass を使う（CLAUDE.md「Types over dicts」）。
- ログは `get_logger(__name__)` を使い、`except: pass` は禁止。
- `check-ci.ps1` は壊れているため使わない。個別コマンドを実行する。
- ローカルの全ユニットテスト実行は開発用 DB のサイズによって遅くなることがある。TDD ループ中は対象ファイルだけを指定して実行し、全体実行は Task 6 で一度だけ行う。

---

### Task 1: 集計の純関数モジュールを作る

少取引銘柄を除外する集計ロジックを、DataFrame に依存しない純関数として新規作成する。これ単体でテスト可能であり、`factory.py` の行数も増えない。

**Files:**
- Create: `python/src/backtest/factory_aggregation.py`
- Test: `python/tests/unit/backtest/test_factory_aggregation.py`

**Interfaces:**
- Consumes: なし（このタスクが最初）
- Produces:
  - `SymbolMetrics(symbol: str, num_trades: int, sharpe_ratio: float, sharpe_per_trade: float, win_rate: float, total_return: float, max_drawdown: float)` — frozen dataclass
  - `AggregatedMetrics(sharpe_ratio: float, sharpe_per_trade: float, win_rate: float, total_return: float, max_drawdown: float, num_trades: int, n_symbols_with_signal: int, n_effective_symbols: int, avg_trades_per_symbol: float)` — 全フィールド既定値付き dataclass
  - `aggregate_symbol_metrics(rows: list[SymbolMetrics], min_trades_per_symbol: int) -> AggregatedMetrics`

- [ ] **Step 1: 失敗するテストを書く**

`python/tests/unit/backtest/test_factory_aggregation.py` を新規作成する。

```python
"""factory_aggregation の集計ロジックのテスト（#625）。"""

from __future__ import annotations

import unittest

from src.backtest.factory_aggregation import SymbolMetrics, aggregate_symbol_metrics


def _row(symbol: str, num_trades: int, sharpe: float, **kwargs) -> SymbolMetrics:
    defaults = dict(
        sharpe_per_trade=sharpe / 10,
        win_rate=0.6,
        total_return=0.1,
        max_drawdown=-0.05,
    )
    defaults.update(kwargs)
    return SymbolMetrics(
        symbol=symbol,
        num_trades=num_trades,
        sharpe_ratio=sharpe,
        **defaults,
    )


class TestAggregateSymbolMetrics(unittest.TestCase):
    def test_excludes_symbols_below_min_trades(self):
        # 2取引の発散 Sharpe(25.0) は除外され、5取引の 0.5 だけが残る
        rows = [_row("AAA", 2, 25.0), _row("BBB", 5, 0.5)]

        result = aggregate_symbol_metrics(rows, min_trades_per_symbol=3)

        self.assertEqual(result.n_effective_symbols, 1)
        self.assertAlmostEqual(result.sharpe_ratio, 0.5)

    def test_num_trades_counts_only_effective_symbols(self):
        rows = [_row("AAA", 1, 0.0), _row("BBB", 2, 25.0), _row("CCC", 8, 0.4)]

        result = aggregate_symbol_metrics(rows, min_trades_per_symbol=3)

        self.assertEqual(result.num_trades, 8)

    def test_reports_signal_and_effective_symbol_counts(self):
        rows = [_row("AAA", 1, 0.0), _row("BBB", 2, 25.0), _row("CCC", 8, 0.4)]

        result = aggregate_symbol_metrics(rows, min_trades_per_symbol=3)

        self.assertEqual(result.n_symbols_with_signal, 3)
        self.assertEqual(result.n_effective_symbols, 1)
        # 平均取引数はフィルタ前の母数で割る: (1 + 2 + 8) / 3
        self.assertAlmostEqual(result.avg_trades_per_symbol, 11 / 3)

    def test_healthy_distribution_is_unchanged_by_filter(self):
        # 全銘柄が閾値以上なら、フィルタの有無で結果が一致する（回帰テスト）
        rows = [_row("AAA", 5, 0.8), _row("BBB", 9, 0.4), _row("CCC", 12, 0.6)]

        filtered = aggregate_symbol_metrics(rows, min_trades_per_symbol=3)
        unfiltered = aggregate_symbol_metrics(rows, min_trades_per_symbol=1)

        self.assertEqual(filtered, unfiltered)
        self.assertAlmostEqual(filtered.sharpe_ratio, 0.6)
        self.assertEqual(filtered.n_effective_symbols, 3)

    def test_no_effective_symbols_returns_zeros_without_error(self):
        rows = [_row("AAA", 1, 0.0), _row("BBB", 2, 25.0)]

        result = aggregate_symbol_metrics(rows, min_trades_per_symbol=3)

        self.assertEqual(result.n_effective_symbols, 0)
        self.assertEqual(result.sharpe_ratio, 0.0)
        self.assertEqual(result.num_trades, 0)
        self.assertEqual(result.max_drawdown, 0.0)
        # 診断用の値はフィルタ前の実態を残す
        self.assertEqual(result.n_symbols_with_signal, 2)
        self.assertAlmostEqual(result.avg_trades_per_symbol, 1.5)

    def test_empty_rows_returns_zeros(self):
        result = aggregate_symbol_metrics([], min_trades_per_symbol=3)

        self.assertEqual(result.n_symbols_with_signal, 0)
        self.assertEqual(result.avg_trades_per_symbol, 0.0)
        self.assertEqual(result.sharpe_ratio, 0.0)

    def test_max_drawdown_is_worst_of_effective_symbols(self):
        # 除外される銘柄(-0.90)の DD は採用されない
        rows = [
            _row("AAA", 2, 25.0, max_drawdown=-0.90),
            _row("BBB", 5, 0.5, max_drawdown=-0.08),
            _row("CCC", 6, 0.5, max_drawdown=-0.20),
        ]

        result = aggregate_symbol_metrics(rows, min_trades_per_symbol=3)

        self.assertAlmostEqual(result.max_drawdown, -0.20)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: テストを実行して失敗を確認する**

```bash
cd python
PYTHONUTF8=1 py -m pytest tests/unit/backtest/test_factory_aggregation.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.backtest.factory_aggregation'`

- [ ] **Step 3: 実装を書く**

`python/src/backtest/factory_aggregation.py` を新規作成する。

```python
"""
戦略ファクトリー（#369）: 銘柄別評価結果の集計（#625）

少取引銘柄の Sharpe は分母がほぼ 0 になり発散する（取引が 2 回なら
std = |a-b|/√2 であり、2 回のリターンが近いほど Sharpe が大きくなる）。
そのため銘柄あたり最低取引数を満たす銘柄だけを集計に採用する。

factory.py から切り出した純関数であり DataFrame に依存しない。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SymbolMetrics:
    """1銘柄・全期間シミュレーションの評価結果。"""

    symbol: str
    num_trades: int
    sharpe_ratio: float
    sharpe_per_trade: float
    win_rate: float
    total_return: float
    max_drawdown: float


@dataclass
class AggregatedMetrics:
    """有効銘柄のみで再集計した仮説単位のメトリクス。

    n_symbols_with_signal / avg_trades_per_symbol はフィルタ「前」の母数で算出する。
    フィルタがどれだけ効いたかを診断するための値であるため。
    """

    sharpe_ratio: float = 0.0
    sharpe_per_trade: float = 0.0
    win_rate: float = 0.0
    total_return: float = 0.0
    max_drawdown: float = 0.0
    num_trades: int = 0
    n_symbols_with_signal: int = 0
    n_effective_symbols: int = 0
    avg_trades_per_symbol: float = 0.0


def aggregate_symbol_metrics(
    rows: list[SymbolMetrics], min_trades_per_symbol: int
) -> AggregatedMetrics:
    """銘柄別の評価結果を、最低取引数を満たす銘柄だけで集計する。

    Args:
        rows: 買いシグナルが出た銘柄の評価結果（シグナル 0 の銘柄は含めない）
        min_trades_per_symbol: 集計に採用する銘柄あたり最低取引数

    Returns:
        AggregatedMetrics。有効銘柄が 0 件でも例外を投げず、集計値は 0 のまま
        診断用の n_symbols_with_signal / avg_trades_per_symbol だけを埋めて返す。
    """
    n_with_signal = len(rows)
    avg_trades = sum(r.num_trades for r in rows) / n_with_signal if n_with_signal else 0.0

    effective = [r for r in rows if r.num_trades >= min_trades_per_symbol]
    if not effective:
        return AggregatedMetrics(
            n_symbols_with_signal=n_with_signal,
            n_effective_symbols=0,
            avg_trades_per_symbol=avg_trades,
        )

    n = len(effective)
    return AggregatedMetrics(
        sharpe_ratio=sum(r.sharpe_ratio for r in effective) / n,
        sharpe_per_trade=sum(r.sharpe_per_trade for r in effective) / n,
        win_rate=sum(r.win_rate for r in effective) / n,
        total_return=sum(r.total_return for r in effective) / n,
        max_drawdown=min(r.max_drawdown for r in effective),
        num_trades=sum(r.num_trades for r in effective),
        n_symbols_with_signal=n_with_signal,
        n_effective_symbols=n,
        avg_trades_per_symbol=avg_trades,
    )
```

- [ ] **Step 4: テストを実行して成功を確認する**

```bash
cd python
PYTHONUTF8=1 py -m pytest tests/unit/backtest/test_factory_aggregation.py -v
```

Expected: PASS（7件）

- [ ] **Step 5: フォーマットと型チェック**

```bash
cd python
py -m black src/backtest/factory_aggregation.py tests/unit/backtest/test_factory_aggregation.py
py -m isort src/backtest/factory_aggregation.py tests/unit/backtest/test_factory_aggregation.py
py -m flake8 src/backtest/factory_aggregation.py tests/unit/backtest/test_factory_aggregation.py
py -m mypy src/backtest/factory_aggregation.py
```

Expected: すべてエラーなし

- [ ] **Step 6: コミット**

```bash
git add python/src/backtest/factory_aggregation.py python/tests/unit/backtest/test_factory_aggregation.py
git commit -m "feat: 銘柄あたり最低取引数で集計する純関数を追加 (#625)"
```

---

### Task 2: evaluate_hypothesis をフィルタ経由に切り替える

`FactoryEvaluation` に診断フィールドを追加し、`evaluate_hypothesis` の集計を Task 1 の純関数に委譲する。

**Files:**
- Modify: `python/config/settings.py`（`FACTORY_GATE_MIN_TRADES` 定義の直後、およびモジュール直下の再エクスポート部）
- Modify: `python/src/backtest/types.py`（`FactoryEvaluation`、`n_symbols` の直後）
- Modify: `python/src/backtest/factory.py:292-365`（`evaluate_hypothesis` 全体）
- Test: `python/tests/unit/backtest/test_factory_evaluate_filter.py`

**Interfaces:**
- Consumes: `SymbolMetrics` / `aggregate_symbol_metrics`（Task 1）
- Produces:
  - `config.settings.FACTORY_GATE_MIN_TRADES_PER_SYMBOL: int`（既定 3）
  - `FactoryEvaluation.n_symbols_with_signal: int` / `.n_effective_symbols: int` / `.avg_trades_per_symbol: float`
  - `evaluate_hypothesis(..., min_trades_per_symbol: Optional[int] = None)` — `None` のとき設定値を使う

- [ ] **Step 1: 失敗するテストを書く**

`python/tests/unit/backtest/test_factory_evaluate_filter.py` を新規作成する。実 Backtester を使わずスタブで密閉する。

```python
"""evaluate_hypothesis が少取引銘柄を集計から除外することのテスト（#625）。"""

from __future__ import annotations

import unittest

import pandas as pd

from src.backtest import factory
from src.backtest.types import FactoryHypothesis

_SPEC = {"type": "atomic", "rule": "rsi_contrarian", "params": {}}


class _StubRule:
    """常に全日買いシグナルを返すルール。"""

    def __init__(self, signal: pd.Series) -> None:
        self._signal = signal

    def generate_signal(self, df: pd.DataFrame) -> pd.Series:
        return self._signal


class _StubBacktester:
    """呼び出し順に既定のメトリクスを返す Backtester。"""

    def __init__(self, metrics_list: list[dict]) -> None:
        self._it = iter(metrics_list)

    def simulate_trading(self, df, signal):
        return None, next(self._it)


def _metrics(num_trades: int, sharpe: float) -> dict:
    return {
        "num_trades": num_trades,
        "sharpe_ratio": sharpe,
        "sharpe_per_trade": sharpe / 10,
        "win_rate": 0.9,
        "total_return": 0.1,
        "max_drawdown": -0.05,
    }


class TestEvaluateHypothesisFilter(unittest.TestCase):
    def setUp(self):
        index = pd.date_range("2024-01-01", periods=10, freq="D")
        self.df = pd.DataFrame({"Close": range(10)}, index=index)
        self.signal = pd.Series([1] * 10, index=index)

    def _run(self, metrics_list: list[dict], min_trades_per_symbol: int):
        """スタブを差し込んだ状態で evaluate_hypothesis を1回実行する。"""
        data = {"AAA": self.df, "BBB": self.df}
        hypothesis = FactoryHypothesis(rule_spec=_SPEC, market="jp")
        with unittest.mock.patch.object(
            factory, "build_rule", return_value=_StubRule(self.signal)
        ), unittest.mock.patch.object(
            factory, "_make_backtester", return_value=_StubBacktester(metrics_list)
        ):
            return factory.evaluate_hypothesis(
                hypothesis,
                data,
                windows=[],
                min_trades_per_symbol=min_trades_per_symbol,
            )

    def test_low_trade_symbol_is_excluded_from_average(self):
        result = self._run([_metrics(2, 25.0), _metrics(8, 0.5)], min_trades_per_symbol=3)

        self.assertEqual(result.n_symbols, 2)
        self.assertEqual(result.n_symbols_with_signal, 2)
        self.assertEqual(result.n_effective_symbols, 1)
        self.assertAlmostEqual(result.sharpe_ratio, 0.5)
        self.assertEqual(result.num_trades, 8)
        self.assertAlmostEqual(result.avg_trades_per_symbol, 5.0)

    def test_threshold_one_keeps_every_symbol(self):
        result = self._run([_metrics(2, 25.0), _metrics(8, 0.5)], min_trades_per_symbol=1)

        self.assertEqual(result.n_effective_symbols, 2)
        self.assertAlmostEqual(result.sharpe_ratio, 12.75)


if __name__ == "__main__":
    unittest.main()
```

ファイル先頭の import に `import unittest.mock` を含めること（`unittest` だけでは `unittest.mock` は解決されない）。

- [ ] **Step 2: テストを実行して失敗を確認する**

```bash
cd python
PYTHONUTF8=1 py -m pytest tests/unit/backtest/test_factory_evaluate_filter.py -v
```

Expected: FAIL — `evaluate_hypothesis() got an unexpected keyword argument 'min_trades_per_symbol'`

- [ ] **Step 3: 設定定数を追加する**

`python/config/settings.py:164` の `FACTORY_GATE_MIN_TRADES` の直後に追加する。

```python
    FACTORY_GATE_MIN_TRADES: int = Field(default=30)
    FACTORY_GATE_MIN_TRADES_PER_SYMBOL: int = Field(
        default=3
    )  # 少取引銘柄の Sharpe は発散するため集計から除外する（#625）
```

同ファイル下部の再エクスポート部、`FACTORY_GATE_MIN_TRADES: int = settings.FACTORY_GATE_MIN_TRADES` の直後に追加する。

```python
FACTORY_GATE_MIN_TRADES_PER_SYMBOL: int = settings.FACTORY_GATE_MIN_TRADES_PER_SYMBOL
```

- [ ] **Step 4: FactoryEvaluation にフィールドを追加する**

`python/src/backtest/types.py` の `FactoryEvaluation`、`n_symbols: int = 0` の直後に追加する。

```python
    n_symbols: int = 0
    n_symbols_with_signal: int = 0
    n_effective_symbols: int = 0
    avg_trades_per_symbol: float = 0.0
    dsr: float = float("nan")
```

- [ ] **Step 5: evaluate_hypothesis を書き換える**

`python/src/backtest/factory.py` の import に追加する（`from src.backtest.factory_report import write_report` の直前）。

```python
from src.backtest.factory_aggregation import SymbolMetrics, aggregate_symbol_metrics
```

`from config.settings import (...)` のブロックに `FACTORY_GATE_MIN_TRADES_PER_SYMBOL,` を追加する（アルファベット順で `FACTORY_GATE_MIN_TRADES,` の直後）。

`evaluate_hypothesis`（292-365行）を次の内容で置き換える。

```python
def evaluate_hypothesis(
    hypothesis: FactoryHypothesis,
    data_by_symbol: dict[str, pd.DataFrame],
    windows: list[tuple[pd.Timestamp, pd.Timestamp]],
    initial_cash: float = 1_000_000,
    fee_rate: float = 0.001,
    slippage: float = 0.001,
    stop_loss_pct: Optional[float] = 0.07,
    min_trades_per_symbol: Optional[int] = None,
) -> FactoryEvaluation:
    """1仮説を全銘柄 × 全期間 + 窓別に評価する。

    - 全期間メトリクス: 銘柄あたり最低取引数を満たす銘柄のみで集計する（#625）。
      取引が 2 回だけの銘柄は Sharpe が発散するため、混ぜると平均が壊れる。
    - 窓別リターン: 各窓を独立にシミュレーションし全銘柄で平均する。
      こちらは意図的にフィルタしない。PBO はバッチ単位の診断指標であり
      apply_gate の判定に使われないため、母数を変えると PBO の意味が変わる。
    """
    if min_trades_per_symbol is None:
        min_trades_per_symbol = FACTORY_GATE_MIN_TRADES_PER_SYMBOL
    rule = build_rule(hypothesis.rule_spec)
    backtester = _make_backtester(initial_cash, fee_rate, slippage, stop_loss_pct)

    symbol_rows: list[SymbolMetrics] = []
    window_returns_by_symbol: list[list[float]] = []

    for symbol, df in data_by_symbol.items():
        try:
            signal = rule.generate_signal(df)
            if int((signal == 1).sum()) == 0:
                window_returns_by_symbol.append([0.0] * len(windows))
                continue

            _, metrics = backtester.simulate_trading(df, signal)
            symbol_rows.append(
                SymbolMetrics(
                    symbol=symbol,
                    num_trades=int(metrics.get("num_trades", 0) or 0),
                    sharpe_ratio=float(metrics.get("sharpe_ratio", 0.0) or 0.0),
                    sharpe_per_trade=float(metrics.get("sharpe_per_trade", 0.0) or 0.0),
                    win_rate=float(metrics.get("win_rate", 0.0) or 0.0),
                    total_return=float(metrics.get("total_return", 0.0) or 0.0),
                    max_drawdown=float(metrics.get("max_drawdown", 0.0) or 0.0),
                )
            )

            sym_window_returns = []
            for w_start, w_end in windows:
                mask = (df.index >= w_start) & (df.index < w_end)
                w_df, w_sig = df.loc[mask], signal.loc[mask]
                if len(w_df) < 5 or int((w_sig == 1).sum()) == 0:
                    sym_window_returns.append(0.0)
                    continue
                _, w_metrics = backtester.simulate_trading(w_df, w_sig)
                sym_window_returns.append(float(w_metrics.get("total_return", 0.0) or 0.0))
            window_returns_by_symbol.append(sym_window_returns)
        except Exception:
            logger.warning(
                "仮説評価失敗（銘柄スキップ）: %s [%s]",
                hypothesis.label,
                symbol,
                exc_info=True,
            )

    aggregated = aggregate_symbol_metrics(symbol_rows, min_trades_per_symbol)
    window_returns = (
        np.mean(np.asarray(window_returns_by_symbol, dtype=float), axis=0).tolist()
        if window_returns_by_symbol
        else [0.0] * len(windows)
    )
    return FactoryEvaluation(
        hypothesis=hypothesis,
        sharpe_ratio=aggregated.sharpe_ratio,
        sharpe_per_trade=aggregated.sharpe_per_trade,
        win_rate=aggregated.win_rate,
        num_trades=aggregated.num_trades,
        max_drawdown=aggregated.max_drawdown,
        total_return=aggregated.total_return,
        window_returns=window_returns,
        n_symbols=len(data_by_symbol),
        n_symbols_with_signal=aggregated.n_symbols_with_signal,
        n_effective_symbols=aggregated.n_effective_symbols,
        avg_trades_per_symbol=aggregated.avg_trades_per_symbol,
    )
```

- [ ] **Step 6: テストを実行して成功を確認する**

```bash
cd python
PYTHONUTF8=1 py -m pytest tests/unit/backtest/test_factory_evaluate_filter.py tests/unit/backtest/test_factory_aggregation.py -v
```

Expected: PASS（9件）

- [ ] **Step 7: 既存の factory テストが壊れていないか確認する**

```bash
cd python
PYTHONUTF8=1 py -m pytest tests/unit/test_strategy_factory.py -v
```

Expected: PASS。`TestApplyGate` はこの時点ではまだ新条件を持たないため全件通る。`TestEvaluateHypothesis` の 2 件も、アサーションが `len(window_returns)` と `num_trades == 0` の緩い検証であるため通る。もし失敗した場合は原因を特定してから次へ進む（この時点で失敗するのは想定外である）。

- [ ] **Step 8: ファイル行数を確認する**

```bash
cd python
py scripts/check_file_size.py
```

Expected: 違反なし（`factory.py` は集計処理を外に出したため 531 行より短くなっているはず）

- [ ] **Step 9: フォーマットと型チェック**

```bash
cd python
py -m black src/backtest/factory.py src/backtest/types.py config/settings.py tests/unit/backtest/test_factory_evaluate_filter.py
py -m isort src/backtest/factory.py src/backtest/types.py config/settings.py tests/unit/backtest/test_factory_evaluate_filter.py
py -m flake8 src/backtest/factory.py src/backtest/types.py config/settings.py tests/unit/backtest/test_factory_evaluate_filter.py
py -m mypy src/
```

Expected: すべてエラーなし

- [ ] **Step 10: コミット**

```bash
git add python/config/settings.py python/src/backtest/types.py python/src/backtest/factory.py python/tests/unit/backtest/test_factory_evaluate_filter.py
git commit -m "feat: evaluate_hypothesis を銘柄あたり最低取引数フィルタ経由にする (#625)"
```

---

### Task 3: 有効銘柄数の下限をゲートに追加する

`apply_gate` に有効銘柄数の条件を足す。既存テストの前提が変わるため合わせて修正する。

**Files:**
- Modify: `python/config/settings.py`
- Modify: `python/src/backtest/factory.py`（`apply_gate`）
- Modify: `python/tests/unit/test_strategy_factory.py:145-155`（`TestApplyGate._make_eval`）、`:328-421`（`TestRunFactoryBatch`）
- Test: `python/tests/unit/backtest/test_factory_gate.py`

**Interfaces:**
- Consumes: `FactoryEvaluation.n_effective_symbols`（Task 2）
- Produces: `config.settings.FACTORY_GATE_MIN_EFFECTIVE_SYMBOLS: int`（既定 20）、`apply_gate` の不合格理由文字列 `"effective_symbols {n} < {threshold}"`

- [ ] **Step 1: 失敗するテストを書く**

`python/tests/unit/backtest/test_factory_gate.py` を新規作成する。

```python
"""有効銘柄数ゲートのテスト（#625）。"""

from __future__ import annotations

import unittest

from src.backtest import factory
from src.backtest.factory import apply_gate
from src.backtest.types import FactoryEvaluation, FactoryHypothesis

_SPEC = {"type": "atomic", "rule": "rsi_contrarian", "params": {}}


def _make_eval(**kwargs) -> FactoryEvaluation:
    defaults = dict(
        hypothesis=FactoryHypothesis(rule_spec=_SPEC, market="jp"),
        sharpe_ratio=2.0,
        num_trades=50,
        max_drawdown=-0.10,
        dsr=0.97,
        pbo=0.30,
        n_effective_symbols=50,
    )
    defaults.update(kwargs)
    return FactoryEvaluation(**defaults)


class TestEffectiveSymbolsGate(unittest.TestCase):
    def test_fails_when_effective_symbols_below_minimum(self):
        ev = _make_eval(n_effective_symbols=5)

        apply_gate(ev, champion_sharpe=1.0)

        self.assertFalse(ev.gate_passed)
        self.assertEqual(ev.gate_reasons, ["effective_symbols 5 < 20"])

    def test_passes_when_effective_symbols_at_minimum(self):
        ev = _make_eval(n_effective_symbols=20)

        apply_gate(ev, champion_sharpe=1.0)

        self.assertTrue(ev.gate_passed)

    def test_threshold_is_configurable(self):
        ev = _make_eval(n_effective_symbols=5)

        with unittest.mock.patch.object(factory, "FACTORY_GATE_MIN_EFFECTIVE_SYMBOLS", 3):
            apply_gate(ev, champion_sharpe=1.0)

        self.assertTrue(ev.gate_passed)

    def test_artifact_hypothesis_fails_on_both_trades_and_symbols(self):
        # #598 相当: フィルタ後は取引数 0 / 有効銘柄 0
        ev = _make_eval(sharpe_ratio=0.0, num_trades=0, n_effective_symbols=0)

        apply_gate(ev, champion_sharpe=1.0)

        self.assertFalse(ev.gate_passed)
        self.assertTrue(any("num_trades" in r for r in ev.gate_reasons))
        self.assertTrue(any("effective_symbols" in r for r in ev.gate_reasons))


if __name__ == "__main__":
    unittest.main()
```

ファイル先頭の import に `import unittest.mock` を含めること。

- [ ] **Step 2: テストを実行して失敗を確認する**

```bash
cd python
PYTHONUTF8=1 py -m pytest tests/unit/backtest/test_factory_gate.py -v
```

Expected: FAIL — `test_fails_when_effective_symbols_below_minimum` が `gate_reasons == []` で失敗し、`test_threshold_is_configurable` は `AttributeError: <module 'src.backtest.factory'> does not have the attribute 'FACTORY_GATE_MIN_EFFECTIVE_SYMBOLS'` で失敗する

- [ ] **Step 3: 設定定数を追加する**

`python/config/settings.py`、Task 2 で足した `FACTORY_GATE_MIN_TRADES_PER_SYMBOL` の直後に追加する。

```python
    FACTORY_GATE_MIN_EFFECTIVE_SYMBOLS: int = Field(
        default=20
    )  # 有効銘柄がこの数未満なら極端な集中とみなし不合格（#625）
```

再エクスポート部にも追加する。

```python
FACTORY_GATE_MIN_EFFECTIVE_SYMBOLS: int = settings.FACTORY_GATE_MIN_EFFECTIVE_SYMBOLS
```

- [ ] **Step 4: apply_gate に条件を追加する**

`python/src/backtest/factory.py` の `from config.settings import (...)` ブロックに `FACTORY_GATE_MIN_EFFECTIVE_SYMBOLS,` を追加する（アルファベット順で `FACTORY_GATE_MAX_PBO,` の後）。

`apply_gate` の `num_trades` 判定の直後に追加する。

```python
    if evaluation.num_trades < FACTORY_GATE_MIN_TRADES:
        reasons.append(f"num_trades {evaluation.num_trades} < {FACTORY_GATE_MIN_TRADES}")
    if evaluation.n_effective_symbols < FACTORY_GATE_MIN_EFFECTIVE_SYMBOLS:
        reasons.append(
            f"effective_symbols {evaluation.n_effective_symbols}"
            f" < {FACTORY_GATE_MIN_EFFECTIVE_SYMBOLS}"
        )
```

docstring の末尾に一文を足す。

```
    有効銘柄数（銘柄あたり最低取引数を満たした銘柄の数）が下限未満の場合も不合格とする。
    合計取引数だけでは「2銘柄 × 20取引」のような極端な集中を弾けないため（#625）。
```

- [ ] **Step 5: テストを実行して成功を確認する**

```bash
cd python
PYTHONUTF8=1 py -m pytest tests/unit/backtest/test_factory_gate.py -v
```

Expected: PASS（4件）

- [ ] **Step 6: 既存テストを新しい前提に合わせる**

`python/tests/unit/test_strategy_factory.py` の `TestApplyGate._make_eval`（145-155行）の `defaults` に 1 行足す。これがないと合格を期待する 3 件（`test_passes_when_all_conditions_met` / `test_high_pbo_does_not_block_gate` / `test_champion_nan_skips_champion_condition`）が既定値 0 のせいで失敗する。

```python
        defaults = dict(
            hypothesis=FactoryHypothesis(rule_spec=_ATOMIC_SPEC, market="jp"),
            sharpe_ratio=2.0,
            num_trades=50,
            max_drawdown=-0.10,
            dsr=0.97,
            pbo=0.30,
            n_effective_symbols=50,
        )
```

`TestRunFactoryBatch` は銘柄を 2 件しか使わないため、有効銘柄数が必ず下限を下回り `result.passed` が空になる。合格経路を検証している 3 件（`test_batch_evaluates_and_records_candidates` / `test_batch_calls_review_only_for_passed_hypotheses` / `test_review_none_still_writes_report`）が空ループで素通りしてカバレッジが失われるため、下限を 1 に下げるデコレータを各テストメソッドに追加する。

```python
    @patch("src.backtest.factory.FACTORY_GATE_MIN_EFFECTIVE_SYMBOLS", 1)
    @patch("src.backtest.factory.save_factory_run")
    @patch("src.backtest.factory.count_factory_runs", return_value=0)
    @patch("src.backtest.factory.load_factory_hashes", return_value=set())
    @patch("src.backtest.factory.get_backtest_data_port")
    def test_batch_evaluates_and_records_candidates(
        self, mock_port, mock_hashes, mock_count, mock_save
    ):
```

`@patch` は下から順に引数へ渡されるため、値を差し替えるだけの `FACTORY_GATE_MIN_EFFECTIVE_SYMBOLS` パッチを**最上段**に置けば既存の引数の並び（`mock_port, mock_hashes, mock_count, mock_save`）は変えなくてよい。同じデコレータを `test_batch_calls_review_only_for_passed_hypotheses` と `test_review_none_still_writes_report` にも最上段へ追加する（`test_batch_aborts_without_symbol_data` は合格経路を見ていないため不要）。

- [ ] **Step 7: 既存テストを実行して成功を確認する**

```bash
cd python
PYTHONUTF8=1 py -m pytest tests/unit/test_strategy_factory.py -v
```

Expected: PASS（全件）

- [ ] **Step 8: フォーマットと型チェック**

```bash
cd python
py -m black src/backtest/factory.py config/settings.py tests/unit/backtest/test_factory_gate.py tests/unit/test_strategy_factory.py
py -m isort src/backtest/factory.py config/settings.py tests/unit/backtest/test_factory_gate.py tests/unit/test_strategy_factory.py
py -m flake8 src/backtest/factory.py config/settings.py tests/unit/backtest/test_factory_gate.py tests/unit/test_strategy_factory.py
py -m mypy src/
```

Expected: すべてエラーなし

- [ ] **Step 9: コミット**

```bash
git add python/config/settings.py python/src/backtest/factory.py python/tests/unit/backtest/test_factory_gate.py python/tests/unit/test_strategy_factory.py
git commit -m "feat: 有効銘柄数の下限をファクトリーのゲートに追加 (#625)"
```

---

### Task 4: レポートに母数を出力する

Issue 本文のメトリクス表と `gate` ブロックに新指標を出し、母数が曖昧だったラベルを改称する。

**Files:**
- Modify: `python/src/backtest/factory_report.py`（import、`_build_issue_body`、`write_report`）
- Test: `python/tests/unit/backtest/test_factory_report_generated.py`（追記）

**Interfaces:**
- Consumes: `FactoryEvaluation.n_symbols_with_signal` / `.n_effective_symbols` / `.avg_trades_per_symbol`（Task 2）、`FACTORY_GATE_MIN_TRADES_PER_SYMBOL` / `FACTORY_GATE_MIN_EFFECTIVE_SYMBOLS`（Task 2/3）
- Produces: レポート JSON の `gate` に `n_symbols_with_signal` / `n_effective_symbols` / `avg_trades_per_symbol` を追加

- [ ] **Step 1: 失敗するテストを書く**

`python/tests/unit/backtest/test_factory_report_generated.py` の末尾に追記する。

```python
def test_issue_body_reports_symbol_denominators(tmp_path, monkeypatch):
    """Sharpe の母数が読み取れることを保証する（#625）。

    従来の「銘柄数 194」はデータ取得できた銘柄数であり Sharpe の母数ではなく、
    レビュー時に誤読を招いていた。
    """
    monkeypatch.setattr("src.backtest.factory_report.get_results_dir", lambda: str(tmp_path))

    hypothesis = FactoryHypothesis(
        rule_spec={"type": "atomic", "rule": "rsi_contrarian", "params": {}},
        market="jp",
    )
    evaluation = FactoryEvaluation(
        hypothesis=hypothesis,
        sharpe_ratio=1.6,
        dsr=0.99,
        pbo=0.1,
        num_trades=85,
        max_drawdown=-0.19,
        win_rate=0.85,
        total_return=0.09,
        window_returns=[0.01, 0.02],
        n_symbols=194,
        n_symbols_with_signal=69,
        n_effective_symbols=16,
        avg_trades_per_symbol=1.23,
    )

    path = write_report(evaluation, champion_sharpe=1.083, period=("2024-07-25", "2026-07-25"))

    import json

    with open(path, encoding="utf-8") as f:
        report = json.load(f)

    body = report["issue_body"]
    assert "データ取得銘柄数 194" in body
    assert "シグナル発生銘柄" in body
    assert "| 69 |" in body
    assert "有効銘柄" in body
    assert "銘柄あたり平均取引数" in body
    assert "1.23" in body
    # 母数が曖昧だった旧ラベルは残っていない
    assert "Sharpe（銘柄平均）" not in body
    assert "Sharpe（有効銘柄平均）" in body

    assert report["gate"]["n_symbols_with_signal"] == 69
    assert report["gate"]["n_effective_symbols"] == 16
    assert report["gate"]["avg_trades_per_symbol"] == 1.23
```

- [ ] **Step 2: テストを実行して失敗を確認する**

```bash
cd python
PYTHONUTF8=1 py -m pytest tests/unit/backtest/test_factory_report_generated.py -v
```

Expected: FAIL — `assert "データ取得銘柄数 194" in body`

- [ ] **Step 3: factory_report.py を修正する**

import ブロックに 2 定数を追加する。

```python
from config.settings import (
    FACTORY_GATE_CHAMPION_MARGIN,
    FACTORY_GATE_MAX_DRAWDOWN,
    FACTORY_GATE_MAX_PBO,
    FACTORY_GATE_MIN_DSR,
    FACTORY_GATE_MIN_EFFECTIVE_SYMBOLS,
    FACTORY_GATE_MIN_TRADES,
    FACTORY_GATE_MIN_TRADES_PER_SYMBOL,
)
```

`_build_issue_body` の評価期間の行を差し替える。

```python
- 評価期間: {period[0]} 〜 {period[1]}（{h.lookback_years}年、データ取得銘柄数 {evaluation.n_symbols}）
```

メトリクス表を差し替える。

```python
| 指標 | 値 | ゲート |
|---|---|---|
| Sharpe（有効銘柄平均） | {evaluation.sharpe_ratio:.3f} | {champion_cell} |
| Deflated Sharpe | {evaluation.dsr:.3f} | >= {FACTORY_GATE_MIN_DSR} |
| PBO | {evaluation.pbo:.3f} | <= {FACTORY_GATE_MAX_PBO} |
| 取引数（有効銘柄合計） | {evaluation.num_trades} | >= {FACTORY_GATE_MIN_TRADES} |
| シグナル発生銘柄 | {evaluation.n_symbols_with_signal} | - |
| 有効銘柄（{FACTORY_GATE_MIN_TRADES_PER_SYMBOL}取引以上） | {evaluation.n_effective_symbols} | >= {FACTORY_GATE_MIN_EFFECTIVE_SYMBOLS} |
| 銘柄あたり平均取引数 | {evaluation.avg_trades_per_symbol:.2f} | - |
| 最大DD（最悪銘柄） | {evaluation.max_drawdown:.2%} | >= {FACTORY_GATE_MAX_DRAWDOWN:.0%} |
| 勝率（有効銘柄平均） | {evaluation.win_rate:.2%} | - |
| リターン（有効銘柄平均） | {evaluation.total_return:.2%} | - |
```

`write_report` の `gate` 辞書に 3 キーを追加する。

```python
        "gate": {
            "sharpe_ratio": evaluation.sharpe_ratio,
            "dsr": evaluation.dsr,
            "pbo": evaluation.pbo,
            "num_trades": evaluation.num_trades,
            "max_drawdown": evaluation.max_drawdown,
            "champion_sharpe": champion_sharpe,
            "n_symbols_with_signal": evaluation.n_symbols_with_signal,
            "n_effective_symbols": evaluation.n_effective_symbols,
            "avg_trades_per_symbol": evaluation.avg_trades_per_symbol,
        },
```

- [ ] **Step 4: テストを実行して成功を確認する**

```bash
cd python
PYTHONUTF8=1 py -m pytest tests/unit/backtest/test_factory_report_generated.py tests/unit/test_strategy_factory.py -v
```

Expected: PASS（全件）。`test_strategy_factory.py` の `TestWriteReport` がメトリクス表の旧文言に依存していた場合はここで失敗するため、その場合は新しい文言に合わせて修正する。

- [ ] **Step 5: フォーマットと型チェック**

```bash
cd python
py -m black src/backtest/factory_report.py tests/unit/backtest/test_factory_report_generated.py
py -m isort src/backtest/factory_report.py tests/unit/backtest/test_factory_report_generated.py
py -m flake8 src/backtest/factory_report.py tests/unit/backtest/test_factory_report_generated.py
py -m mypy src/
```

Expected: すべてエラーなし

- [ ] **Step 6: コミット**

```bash
git add python/src/backtest/factory_report.py python/tests/unit/backtest/test_factory_report_generated.py python/tests/unit/test_strategy_factory.py
git commit -m "feat: ファクトリーレポートに Sharpe の母数を出力する (#625)"
```

---

### Task 5: 批判的レビューの入力に母数を渡す

Claude に渡すプロンプトが「対象銘柄数 194 / 取引数 85」しか含まないため、レビュアーが母数を誤認する。#598 のレビューは「194銘柄で約0.44件」と書いたが正しくは「69銘柄で約1.23件」だった。

**Files:**
- Modify: `python/src/backtest/hypothesis_review.py:65-83`（`_build_review_context`）
- Test: `python/tests/unit/test_hypothesis_review.py`（追記）

**Interfaces:**
- Consumes: `FactoryEvaluation.n_symbols_with_signal` / `.n_effective_symbols` / `.avg_trades_per_symbol`（Task 2）
- Produces: なし（プロンプト文字列の変更のみ）

- [ ] **Step 1: 既存テストの構造を確認する**

```bash
cd python
PYTHONUTF8=1 py -m pytest tests/unit/test_hypothesis_review.py -v
```

既存テストが `_build_review_context` を直接呼んでいるか、`review_hypothesis` 経由かを確認し、次のステップのテストを同じスタイルで書く。

- [ ] **Step 2: 失敗するテストを書く**

`python/tests/unit/test_hypothesis_review.py` の末尾に追記する。`FactoryEvaluation` / `FactoryHypothesis` の import が未追加なら足すこと。

```python
def test_review_context_includes_symbol_denominators():
    """レビュアーが Sharpe の母数を誤認しないよう母数を渡す（#625）。"""
    from src.backtest.hypothesis_review import _build_review_context
    from src.backtest.types import FactoryEvaluation, FactoryHypothesis

    evaluation = FactoryEvaluation(
        hypothesis=FactoryHypothesis(
            rule_spec={"type": "atomic", "rule": "rsi_contrarian", "params": {}},
            market="jp",
        ),
        sharpe_ratio=1.6,
        dsr=0.99,
        pbo=0.1,
        num_trades=85,
        max_drawdown=-0.19,
        win_rate=0.85,
        total_return=0.09,
        window_returns=[0.01],
        n_symbols=194,
        n_symbols_with_signal=69,
        n_effective_symbols=16,
        avg_trades_per_symbol=1.23,
    )

    context = _build_review_context(evaluation, champion_sharpe=1.083)

    assert "データ取得銘柄数: 194" in context
    assert "シグナル発生銘柄数: 69" in context
    assert "有効銘柄数（集計母数）: 16" in context
    assert "銘柄あたり平均取引数: 1.23" in context
```

- [ ] **Step 3: テストを実行して失敗を確認する**

```bash
cd python
PYTHONUTF8=1 py -m pytest tests/unit/test_hypothesis_review.py::test_review_context_includes_symbol_denominators -v
```

Expected: FAIL — `assert "データ取得銘柄数: 194" in context`

- [ ] **Step 4: プロンプトを修正する**

`python/src/backtest/hypothesis_review.py` の `_build_review_context` の戻り値を差し替える（69行目と72-78行目）。

```python
    return f"""## 仮説スペック
```json
{json.dumps(h.rule_spec, ensure_ascii=False, indent=2)}
```
マーケット: {h.market} / 評価期間: {h.lookback_years}年 / データ取得銘柄数: {evaluation.n_symbols}

## メトリクス
- Sharpe（有効銘柄平均）: {evaluation.sharpe_ratio:.3f}
- Deflated Sharpe (DSR): {evaluation.dsr:.3f}
- PBO: {evaluation.pbo:.3f}
- 取引数（有効銘柄合計）: {evaluation.num_trades}
- シグナル発生銘柄数: {evaluation.n_symbols_with_signal}
- 有効銘柄数（集計母数）: {evaluation.n_effective_symbols}
- 銘柄あたり平均取引数: {evaluation.avg_trades_per_symbol:.2f}
- 最大DD（最悪銘柄）: {evaluation.max_drawdown:.2%}
- 勝率（有効銘柄平均）: {evaluation.win_rate:.2%}
- リターン（有効銘柄平均）: {evaluation.total_return:.2%}
- {champion_line}

## 窓別リターン（銘柄平均）
{window_lines}
"""
```

- [ ] **Step 5: テストを実行して成功を確認する**

```bash
cd python
PYTHONUTF8=1 py -m pytest tests/unit/test_hypothesis_review.py -v
```

Expected: PASS（全件）

- [ ] **Step 6: フォーマットと型チェック**

```bash
cd python
py -m black src/backtest/hypothesis_review.py tests/unit/test_hypothesis_review.py
py -m isort src/backtest/hypothesis_review.py tests/unit/test_hypothesis_review.py
py -m flake8 src/backtest/hypothesis_review.py tests/unit/test_hypothesis_review.py
py -m mypy src/
```

Expected: すべてエラーなし

- [ ] **Step 7: コミット**

```bash
git add python/src/backtest/hypothesis_review.py python/tests/unit/test_hypothesis_review.py
git commit -m "feat: 批判的レビューの入力に Sharpe の母数を渡す (#625)"
```

---

### Task 6: 実データでの受け入れ検証と PR 作成

ユニットテストとは別に、jp 194 銘柄の実データで #598 / #564 が不合格になること、健全な対照群が無傷であること、新しい `champion_sharpe` の実測値を確認する。

**Files:**
- Create: `C:\Users\fuchi\AppData\Local\Temp\claude\C--src-StockFixer\4a899873-5122-406c-b278-947d09d36bfa\scratchpad\acceptance_625.py`（スクラッチパッド。リポジトリにはコミットしない）
- Modify: `python/VERSION`
- Modify: `docs/superpowers/specs/2026-08-05-factory-gate-min-trades-design.md`（受け入れ条件のチェックと champion 実測値の記録）

**Interfaces:**
- Consumes: Task 1〜5 のすべて
- Produces: PR

- [ ] **Step 1: 受け入れ検証スクリプトを書く**

スクラッチパッドに `acceptance_625.py` を作成する。データ取得は既存のキャッシュ（`scratchpad/cache/jp_2024-07-25_2026-07-25_194.pkl`）を再利用する。キャッシュが無い場合は `_load_symbol_data` で取得する。

```python
"""#625 の受け入れ検証（使い捨て）。

新ゲートで #598 / #564 が不合格になること、健全な対照群が無傷であること、
新方式の champion_sharpe を実測する。
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(r"C:\src\StockFixer\python")))

from src.backtest.factory import (  # noqa: E402
    _load_symbol_data,
    _window_bounds,
    apply_gate,
    control_hypotheses,
    evaluate_hypothesis,
)
from src.backtest.types import FactoryHypothesis  # noqa: E402
from src.orchestration.port_wiring import wire_ports  # noqa: E402
from src.watchlist.batch_runner import load_target_symbols  # noqa: E402

START, END = "2024-07-25", "2026-07-25"
CACHE = Path(__file__).parent / "cache" / f"jp_{START}_{END}_194.pkl"

SPEC_598 = {
    "type": "and",
    "rules": [
        {"type": "atomic", "rule": "volatility_breakout", "params": {"buy_k": 0.8, "sell_k": 1.2}},
        {
            "type": "atomic",
            "rule": "rsi_contrarian",
            "params": {"oversold": 30.0, "overbought": 70.0},
        },
    ],
}
SPEC_564 = {
    "type": "and",
    "rules": [
        {"type": "atomic", "rule": "bollinger_band", "params": {"sell_at_upper": False}},
        {"type": "atomic", "rule": "volatility_breakout", "params": {"buy_k": 0.8, "sell_k": 1.2}},
    ],
}


def main() -> None:
    wire_ports()
    if CACHE.exists():
        with open(CACHE, "rb") as f:
            data = pickle.load(f)
    else:
        symbols = [t.symbol for t in load_target_symbols() if t.market == "jp"]
        data = _load_symbol_data("jp", symbols, START, END)
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        with open(CACHE, "wb") as f:
            pickle.dump(data, f)
    windows = _window_bounds(START, END, 8)

    # 対照群を先に評価して champion を決める
    champion_sharpe, champion_label = float("-inf"), ""
    print("=== 対照群（新方式） ===")
    for h in control_hypotheses("jp"):
        ev = evaluate_hypothesis(h, data, windows)
        print(
            f"  {h.label[:44]:<44} Sharpe={ev.sharpe_ratio:>8.3f} "
            f"有効銘柄={ev.n_effective_symbols:>3} 取引={ev.num_trades:>5}"
        )
        if ev.num_trades > 0 and ev.sharpe_ratio > champion_sharpe:
            champion_sharpe, champion_label = ev.sharpe_ratio, h.label
    print(f">>> champion_sharpe = {champion_sharpe:.3f} ({champion_label})")

    print("\n=== 仮説（不合格になるべき） ===")
    for name, spec in [("#598", SPEC_598), ("#564", SPEC_564)]:
        ev = evaluate_hypothesis(FactoryHypothesis(rule_spec=spec, market="jp"), data, windows)
        apply_gate(ev, champion_sharpe)
        print(
            f"  {name}: Sharpe={ev.sharpe_ratio:.3f} シグナル発生銘柄={ev.n_symbols_with_signal} "
            f"有効銘柄={ev.n_effective_symbols} 取引={ev.num_trades} "
            f"平均取引数={ev.avg_trades_per_symbol:.2f}"
        )
        print(f"    gate_passed={ev.gate_passed} reasons={ev.gate_reasons}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 受け入れ検証を実行する**

```bash
cd python
PYTHONUTF8=1 py "C:\Users\fuchi\AppData\Local\Temp\claude\C--src-StockFixer\4a899873-5122-406c-b278-947d09d36bfa\scratchpad\acceptance_625.py"
```

Expected:
- #598 / #564 とも `gate_passed=False` で、`reasons` に `effective_symbols 0 < 20` を含む
- 健全な対照群 4 本（bollinger_band / ema_momentum / macd_rsi / volatility_breakout）の有効銘柄数が 193〜194
- `champion_sharpe` が bollinger_band ≈ 0.782 になる（設計書の想定値）

想定と異なる値が出た場合は先に進まず原因を調べること。特に champion が bollinger_band 以外になった場合は、フィルタが設計と違う挙動をしている。

- [ ] **Step 3: 実測値を設計書に記録する**

`docs/superpowers/specs/2026-08-05-factory-gate-min-trades-design.md` の受け入れ条件のチェックボックスを埋め、実測した `champion_sharpe` を「champion_sharpe は下がる（想定内）」節の末尾に追記する。

```markdown
**実測（実装後、jp 194銘柄 2024-07-25〜2026-07-25）:** champion_sharpe = <実測値> (<ルール名>)
```

- [ ] **Step 4: 全ユニットテストとカバレッジゲートを実行する**

```bash
cd python
PYTHONUTF8=1 py -m pytest tests/unit/ -q --cov=src --cov-branch --cov-report=term-missing --cov-fail-under=80
```

Expected: PASS、カバレッジ 80% 以上。失敗した場合は原因を特定して修正する。

- [ ] **Step 5: 残りの CI チェックを実行する**

```bash
cd python
py -m pylint src/backtest/factory_aggregation.py src/backtest/factory.py src/backtest/factory_report.py src/backtest/hypothesis_review.py --errors-only
PYTHONUTF8=1 py -m lint_imports --config .importlinter
py scripts/check_file_size.py
```

Expected: すべてエラーなし。`lint_imports` の設定ファイル名が異なる場合は `pyproject.toml` / `setup.cfg` の `[importlinter]` 設定を確認して合わせる。

- [ ] **Step 6: VERSION を更新する**

```bash
cd C:/src/StockFixer
git fetch
git show origin/develop:python/VERSION
```

`origin/develop` の最新値を基準に **minor** を +1 する（振る舞いを変える機能追加であり破壊的変更ではない）。`python/VERSION` を書き換える。

この計画を書いた時点の `develop` は `2.3.0` であり、その場合は `2.4.0` になる。ただし develop が進んでいる可能性があるため、必ず `git show origin/develop:python/VERSION` の実際の値を基準にすること。

- [ ] **Step 7: コミットして push する**

```bash
git add python/VERSION docs/superpowers/specs/2026-08-05-factory-gate-min-trades-design.md
git commit -m "chore: VERSION を更新し受け入れ実測値を設計書に記録 (#625)"
git push -u origin fix/factory-gate-min-trades-per-symbol
```

- [ ] **Step 8: PR を作成する**

PR 本文には CI の `validate-pr-body` が要求する 4 セクションを必ず含める。`version_before` / `version_after` は Step 6 の実際の値に置き換えること。

```bash
gh pr create --repo rei4725/StockFixer --base develop \
  --title "fix: 戦略ファクトリーのゲートに銘柄あたり最低取引数を追加 (#625)" \
  --body "$(cat <<'EOF'
## 概要

戦略ファクトリーのゲートが「1銘柄あたり1〜2取引 × 数十銘柄」という統計的アーティファクトを弾けない問題を修正する。Closes #625

`_sharpe_per_trade` は取引 PnL の mean/std であり、取引が 2 回だけの銘柄では `std = |a-b|/√2` となって Sharpe が発散する。`evaluate_hypothesis` はこれを銘柄平均していたため、#598 の報告 Sharpe 1.596 は「2取引の16銘柄」だけが作った値だった。

## 変更内容

- 集計を純関数 `src/backtest/factory_aggregation.py` に切り出し、銘柄あたり最低取引数（既定3）を満たす銘柄だけで集計する
- `apply_gate` に有効銘柄数の下限（既定20）を追加
- レポート / Issue 本文 / 批判的レビュー入力に「シグナル発生銘柄数」「有効銘柄数」「銘柄あたり平均取引数」を出力し、母数が曖昧だったラベルを改称
- 対照群も同じ経路を通るため `champion_sharpe` も浄化される

## champion_sharpe が下がることについて

最も汚染されていた対照群がチャンピオンだったため、jp の champion は rsi_contrarian(1.083) から bollinger_band(≈0.782) に交代し基準が約28%下がる。1.083 が偽の数値だったのでこの低下は正しい。デプロイ後の初回バッチで合格候補が増える可能性があるが、候補側も新フィルタを通る必要があるため基準は緩んでいない。

## 検証

- ユニットテスト追加（集計の純関数 / ゲート / レポート / レビュー入力）
- 実データ（jp 194銘柄、2024-07-25〜2026-07-25）で #598 / #564 が `effective_symbols 0 < 20` により不合格になることを確認
- 健全な対照群4本が194銘柄のまま無傷であることを確認

設計書: `docs/superpowers/specs/2026-08-05-factory-gate-min-trades-design.md`

## version_impact
minor

## version_rationale
ファクトリーの評価・ゲート判定の振る舞いを変える機能追加であり、外部インターフェースの破壊的変更は含まない。

## VERSION 更新
- version_update_required: yes
- version_before: X.Y.Z
- version_after: X.Y.Z

## VERSION 未更新理由
該当なし
EOF
)"
```

- [ ] **Step 9: CI の結果を確認する**

```bash
gh pr checks --repo rei4725/StockFixer --watch
```

Expected: すべて green。失敗したジョブがあればログを確認して修正する。

---

## 自己レビュー結果

**仕様カバレッジ:** 設計書の各節を Task に対応付けた。

| 設計書の節 | 対応 Task |
|---|---|
| 閾値（2定数） | Task 2（MIN_TRADES_PER_SYMBOL）/ Task 3（MIN_EFFECTIVE_SYMBOLS） |
| モジュール構成（factory_aggregation.py） | Task 1 |
| データフロー・各指標の母数 | Task 1（純関数）/ Task 2（配線） |
| `apply_gate` の追加条件 | Task 3 |
| レポート出力 | Task 4 |
| 批判的レビューへの入力 | Task 5 |
| 窓別リターンは据え置く | Task 2 Step 5（docstring にコメントとして明記） |
| 後方互換・champion 低下 | Task 6 Step 2/3（実測して記録） |
| テスト（9項目） | Task 1（1〜6）/ Task 3（7、8）/ Task 4（9） |
| 受け入れ検証（実データ） | Task 6 Step 1/2 |
| 受け入れ条件（champion 実測記録） | Task 6 Step 3 |

**未カバーだった項目と対処:** 設計書のテスト項目8「閾値が env で上書きできる」は、モジュール直下の定数を pydantic Settings 経由で読む構造上、env の再読込には `importlib.reload` が必要で密閉性を損なう。Task 3 Step 1 の `test_threshold_is_configurable` で `factory` モジュール属性のパッチによる上書き可能性を検証する形に置き換えた（`config/settings.py` の二重定義パターン自体は既存の全 `FACTORY_GATE_*` と同一であり、env 上書きの仕組みは既存テストで担保されている）。

**型の一貫性:** `SymbolMetrics` / `AggregatedMetrics` / `aggregate_symbol_metrics` の名前とフィールド名は Task 1 の定義と Task 2 の使用箇所で一致している。`FactoryEvaluation` の新フィールド名（`n_symbols_with_signal` / `n_effective_symbols` / `avg_trades_per_symbol`）は Task 2〜5 を通じて同一である。
