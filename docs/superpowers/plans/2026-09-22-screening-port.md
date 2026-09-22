# backtest → screening のポート化 実装計画（Phase 3）

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `backtest → screening` の直接依存を解消し、`.importlinter` の `ignore_imports` を両契約とも空にする。

**Architecture:** BC を跨ぐ型 3 つを共有カーネル `src/domain/types.py` へ移し、関数 2 本は既存の `backtest/data_port.py` と同じ骨格のポートで依存を逆転する。アダプタは提供側の screening BC に置き、`orchestration/port_wiring.py` で結線する。

**Tech Stack:** Python 3.12 / dataclasses / typing.Protocol / pytest / mypy / import-linter

**Spec:** `docs/superpowers/specs/2026-09-22-screening-port-design.md`

## Global Constraints

- 作業ブランチは既存の `feature/screening-port`（develop `8f4e7a2` から分岐、spec コミット `92b92f0` を含む）。ベースは `develop`。
- **振る舞いは 1 つも変えない。** 既存テストが期待値を変えずに通ることが正しさの証明。数値アサートの変更が必要になったら移設か委譲にバグがある。
- 全コマンドは `python/` ディレクトリから実行する。テストは `python -m pytest`。
- **ローカルのユニットテストは全件実行しない。** dev DB が 86 万行あり DB に触れるテストが timeout する。各タスクが指定するファイルのみ実行する。
- `pre-commit` の commit-msg フックが cp932 で落ちたら `PYTHONUTF8=1` を設定する。
- コミットメッセージは Conventional Commits。
- **`src/screening/hold_engine.py` と `src/screening/trend_screener.py` の実装（関数本体）は変更しない。** import 行の書き換えのみ許可。
- `run_*.py` は CLI ラッパのみ（引数パース + サービス呼び出し）。`wire_ports()` の呼び出しは合成であり、これは許可される（他の `run_backtest*.py` も同じ形）。
- `.env` / `src/env/**` / `*.pem` は読み書きしない。
- VERSION は develop 最新を基準に patch 上げ。PR-7 と PR-8 でそれぞれ 1 回。
- PR ボディには `## version_impact` / `## version_rationale` / `## VERSION 更新` / `## VERSION 未更新理由` の 4 見出しが必須。
- PR のマージは squash ではなく通常 merge。`gh pr merge` は auto mode が拒否するためユーザーが手動実行する。

## ファイル構成

| ファイル | 責務 |
|---|---|
| `src/domain/types.py` | 共有カーネル。`HoldRules` / `PositionEvent` / `TrendCandidate` を追加 |
| `src/screening/types.py` | screening 固有型のみ（`Position` / `MultibaggerCandidate` / `ValueCandidate`）を残す |
| `src/backtest/screening_port.py` | 新規。`BacktestScreeningPort`(Protocol) + `set_` / `get_` |
| `src/screening/backtest_adapter.py` | 新規。`BacktestScreeningAdapter`（具象） |
| `src/orchestration/port_wiring.py` | `wire_ports()` に 1 行追加 |
| `run_longterm_backtest.py` | `wire_ports()` の呼び出しを追加 |
| `python/.importlinter` | 両契約の `ignore_imports` を空にし、古いコメントを整理 |

## PR の区切り

| PR | タスク | 消える依存 | version_impact |
|---|---|---|---|
| PR-7 | Task 1〜2 | 4 / 6 | patch |
| PR-8 | Task 3〜6 | 2 / 6（残り全部） | patch |

---

# PR-7: 型の移設

## Task 1: 3 型を `domain/types.py` へ移す

**Files:**
- Modify: `src/domain/types.py`
- Modify: `src/screening/types.py:1-45`
- Test: `tests/unit/domain/test_shared_screening_types.py`

**Interfaces:**
- Produces: `src.domain.types.HoldRules`, `src.domain.types.PositionEvent`, `src.domain.types.TrendCandidate`（フィールド・既定値は移設前と同一）

- [ ] **Step 1: 失敗するテストを書く**

`tests/unit/domain/test_shared_screening_types.py`:

```python
"""domain へ移設した screening 共有型の定義が変わっていないこと。"""

import unittest
from dataclasses import fields

from src.domain.types import HoldRules, PositionEvent, TrendCandidate


class TestHoldRules(unittest.TestCase):
    def test_defaults(self):
        r = HoldRules()
        self.assertEqual(r.trail_ma_weeks, 40)
        self.assertAlmostEqual(r.trail_stop_pct, 0.35)
        self.assertEqual(r.scale_out_multiples, [2.0, 5.0])
        self.assertAlmostEqual(r.scale_out_fraction, 0.20)

    def test_scale_out_multiples_not_shared_between_instances(self):
        """default_factory なので、インスタンス間で list を共有しない。"""
        a, b = HoldRules(), HoldRules()
        a.scale_out_multiples.append(10.0)
        self.assertEqual(b.scale_out_multiples, [2.0, 5.0])


class TestFieldNames(unittest.TestCase):
    def test_position_event_fields(self):
        self.assertEqual(
            [f.name for f in fields(PositionEvent)],
            ["date", "action", "price", "held_fraction", "reason", "multiple"],
        )

    def test_trend_candidate_fields(self):
        self.assertEqual(
            [f.name for f in fields(TrendCandidate)],
            [
                "market",
                "symbol",
                "score",
                "close",
                "dist_from_52w_high",
                "above_200dma",
                "sma200_rising",
                "return_6m",
                "return_12m",
                "avg_volume",
            ],
        )

    def test_trend_candidate_requires_all_fields(self):
        with self.assertRaises(TypeError):
            TrendCandidate(market="us", symbol="AAPL")  # type: ignore[call-arg]


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: テストが落ちることを確認する**

Run: `python -m pytest tests/unit/domain/test_shared_screening_types.py -v`
Expected: FAIL（`ImportError: cannot import name 'HoldRules' from 'src.domain.types'`）

- [ ] **Step 3: 3 型を `domain/types.py` へ移す**

`src/screening/types.py` の 7-44 行（`HoldRules` / `PositionEvent` / `TrendCandidate` の 3 つの dataclass）を、**定義をそのまま** `src/domain/types.py` の末尾へ移す。フィールド名・型・既定値・コメント・docstring を 1 文字も変えない。

`src/domain/types.py` は既に `from dataclasses import dataclass, field` と `from typing import Optional` を持つため、import の追加は不要。

- [ ] **Step 4: `screening/types.py` の未使用 import を外す**

3 型を抜くと `field` が未使用になる（残る 3 型は `Optional` のみ使用）。

```python
from dataclasses import dataclass
from typing import Optional
```

`flake8` が F401 を出すので、これを外さないと lint が落ちる。

- [ ] **Step 5: テストが通ることを確認する**

Run: `python -m pytest tests/unit/domain/test_shared_screening_types.py -v`
Expected: 5 passed

- [ ] **Step 6: コミット**

```bash
git add python/src/domain/types.py python/src/screening/types.py python/tests/unit/domain/test_shared_screening_types.py
git commit -m "refactor: BC を跨ぐ screening 型 3 つを domain/types.py へ移す"
```

---

## Task 2: 全利用者の import を書き換える

**Files:**
- Modify: 下表の 18 ファイル
- Test: 既存テスト全般（新規なし）

**Interfaces:**
- Consumes: Task 1 の `src.domain.types.{HoldRules, PositionEvent, TrendCandidate}`

**注意**: `src/screening/positions.py:23`（`Position`）と `src/screening/value_screener.py:17`（`ValueCandidate`）、`tests/unit/test_value_screener.py:10` は**変更しない**。移設対象外の型のみを import しているため。

- [ ] **Step 1: 移設対象の型のみを import している 14 ファイルを置換する**

そのまま `from src.screening.types import` を `from src.domain.types import` に変える（import する名前は変えない）。

| ファイル:行 | 現在 |
|---|---|
| `src/backtest/longterm/config.py:9` | `from src.screening.types import HoldRules` |
| `src/backtest/longterm/engine.py:45` | `from src.screening.types import TrendCandidate` |
| `src/backtest/longterm/portfolio.py:12` | `from src.screening.types import PositionEvent` |
| `src/backtest/longterm_backtest.py:23` | `from src.screening.types import HoldRules` |
| `src/screening/hold_engine.py:15` | `from src.screening.types import HoldRules, PositionEvent` |
| `src/screening/trend_screener.py:16` | `from src.screening.types import TrendCandidate` |
| `tests/unit/backtest/longterm/test_conclusion.py:7` | `from src.screening.types import HoldRules` |
| `tests/unit/backtest/longterm/test_engine_invariants.py:11` | `from src.screening.types import TrendCandidate` |
| `tests/unit/backtest/longterm/test_entry_lag.py:13` | `from src.screening.types import PositionEvent, TrendCandidate` |
| `tests/unit/backtest/longterm/test_integer_shares.py:12` | `from src.screening.types import PositionEvent, TrendCandidate` |
| `tests/unit/backtest/longterm/test_portfolio.py:7` | `from src.screening.types import PositionEvent` |
| `tests/unit/test_hold_engine.py:12` | `from src.screening.types import HoldRules, PositionEvent` |
| `tests/unit/test_longterm_backtest.py:16` | `from src.screening.types import TrendCandidate` |
| `tests/unit/test_quality_gate.py:5` | `from src.screening.types import TrendCandidate` |

- [ ] **Step 2: 混在して import している 4 ファイルを 2 行に割る**

`src/orchestration/multibagger_job.py:25`:

```python
from src.domain.types import PositionEvent
from src.screening.types import MultibaggerCandidate, Position
```

`src/screening/quality_gate.py:18`:

```python
from src.domain.types import TrendCandidate
from src.screening.types import MultibaggerCandidate
```

`src/screening/__init__.py:17`:

```python
from src.domain.types import TrendCandidate
from src.screening.types import Position
```

`tests/unit/test_multibagger_job.py:9`:

```python
from src.domain.types import PositionEvent
from src.screening.types import MultibaggerCandidate
```

`src/screening/__init__.py` の `__all__` は変更しない（再輸出する名前は同じ）。

- [ ] **Step 3: 取りこぼしが無いことを確認する**

Run:

```bash
grep -rn "from src.screening.types import" --include=*.py src/ tests/ | grep -E "HoldRules|PositionEvent|TrendCandidate"
```

Expected: 出力なし（1 件でも残っていたら書き換え漏れ）

- [ ] **Step 4: 該当テストを実行する**

Run:

```bash
python -m pytest tests/unit/domain/ tests/unit/backtest/ tests/unit/test_longterm_backtest.py \
  tests/unit/test_hold_engine.py tests/unit/test_quality_gate.py \
  tests/unit/test_multibagger_job.py tests/unit/test_value_screener.py -v
```

Expected: 全 PASS。**期待値の変更は 1 件も不要**。

- [ ] **Step 5: lint / 型 / 契約を確認する**

```bash
black --check . && isort --check-only . && flake8 . && mypy src/
PYTHONUTF8=1 lint-imports
```

Expected: すべて exit 0。`lint-imports` は依然として 2 contracts kept（この時点では型絡み 4 依存が `ignore_imports` に残ったまま「unmatched」警告になる。次のステップで消す）。

- [ ] **Step 6: `.importlinter` から型絡みの 4 依存を消す**

両契約の `ignore_imports` から次の 4 行を削除する（Contract 1 と Contract 2 の両方、計 8 行）。

```
    src.backtest.longterm_backtest -> src.screening.types
    src.backtest.longterm.engine -> src.screening.types
    src.backtest.longterm.config -> src.screening.types
    src.backtest.longterm.portfolio -> src.screening.types
```

関数絡みの 2 行（`-> src.screening.hold_engine` と `-> src.screening.trend_screener`）は PR-8 まで残す。

Contract 1 のコメントから、型に言及した次の 2 文を削除する（依存が消えたため）。

```
# longterm.config は LongtermBacktestConfig の rules フィールドが HoldRules を保持。
# longterm.engine -> screening.types はエントリー約定ラグ導入（Task 8）で
# screen_trend_candidates の戻り値（TrendCandidate のリスト）をキューへ
# 積むための型ヒントとして追加。
```

- [ ] **Step 7: 契約が通ることを確認する**

Run: `PYTHONUTF8=1 lint-imports`
Expected: `Contracts: 2 kept, 0 broken.`、unmatched 警告も出ないこと

- [ ] **Step 8: VERSION を patch 上げしてコミットする**

`python/VERSION` を develop 最新の patch +1 にする。**`printf '<version>\n' > python/VERSION` で書き、`xxd python/VERSION | head -1` で BOM が無く末尾が `0a` であることを確認する。** PowerShell の `Set-Content -Encoding utf8` は BOM を付けるので使わない。

```bash
git add python/src python/tests python/.importlinter python/VERSION
git commit -m "refactor: screening 共有型の参照を domain/types.py へ切り替える"
```

---

# PR-8: ポートの導入

## Task 3: `BacktestScreeningPort` を定義する

**Files:**
- Create: `src/backtest/screening_port.py`
- Test: `tests/unit/backtest/test_screening_port.py`

**Interfaces:**
- Consumes: `src.domain.types.{HoldRules, PositionEvent, TrendCandidate}`
- Produces: `BacktestScreeningPort`(Protocol, `runtime_checkable`)、`set_backtest_screening_port(port) -> None`、`get_backtest_screening_port() -> BacktestScreeningPort`

- [ ] **Step 1: 失敗するテストを書く**

`tests/unit/backtest/test_screening_port.py`:

```python
"""BacktestScreeningPort の注入と未注入エラー。"""

import unittest

import pandas as pd

from src.backtest import screening_port as sp


class _FakePort:
    def screen_trend_candidates(self, market, top_n, as_of):
        return []

    def simulate_position(self, prices, entry_date, rules):
        return []


class TestPortInjection(unittest.TestCase):
    def setUp(self):
        self._saved = sp._port

    def tearDown(self):
        sp._port = self._saved

    def test_raises_when_not_injected(self):
        sp._port = None
        with self.assertRaises(RuntimeError) as ctx:
            sp.get_backtest_screening_port()
        self.assertIn("wire_ports", str(ctx.exception))

    def test_returns_injected_instance(self):
        fake = _FakePort()
        sp.set_backtest_screening_port(fake)
        self.assertIs(sp.get_backtest_screening_port(), fake)

    def test_fake_satisfies_protocol(self):
        self.assertIsInstance(_FakePort(), sp.BacktestScreeningPort)

    def test_object_without_methods_does_not_satisfy_protocol(self):
        self.assertNotIsInstance(object(), sp.BacktestScreeningPort)

    def test_injected_port_is_callable_through_getter(self):
        sp.set_backtest_screening_port(_FakePort())
        port = sp.get_backtest_screening_port()
        self.assertEqual(port.screen_trend_candidates("us", 5, "2024-01-02"), [])
        self.assertEqual(
            port.simulate_position(pd.DataFrame({"date": [], "Close": []}), "2024-01-02", None), []
        )


if __name__ == "__main__":
    unittest.main()
```

`setUp` / `tearDown` でモジュール変数 `_port` を退避・復元しているのは、このテスト自身が他テストへ注入を漏らさないため。

- [ ] **Step 2: テストが落ちることを確認する**

Run: `python -m pytest tests/unit/backtest/test_screening_port.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'src.backtest.screening_port'`）

- [ ] **Step 3: `screening_port.py` を書く**

```python
"""
BacktestScreeningPort - screening BC への依存を逆転させるポート定義。

backtest BC はこのポートを通じて screening BC の純粋関数を利用する。
"""

from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable

import pandas as pd

from src.domain.types import HoldRules, PositionEvent, TrendCandidate


@runtime_checkable
class BacktestScreeningPort(Protocol):
    """バックテスト用 screening アクセスポートのインターフェース。"""

    def screen_trend_candidates(
        self, market: str, top_n: int, as_of: str
    ) -> list[TrendCandidate]: ...

    def simulate_position(
        self, prices: pd.DataFrame, entry_date: str, rules: HoldRules
    ) -> list[PositionEvent]: ...


_port: Optional[BacktestScreeningPort] = None


def set_backtest_screening_port(port: BacktestScreeningPort) -> None:
    """テストや orchestration から実装を注入するためのセッター。"""
    global _port
    _port = port


def get_backtest_screening_port() -> BacktestScreeningPort:
    """注入済みの BacktestScreeningPort を返す。未注入なら RuntimeError。

    注入は orchestration の合成ルートで行う:
    `src.orchestration.port_wiring.wire_ports()`（エントリポイント起動時に呼ぶ）。
    テストや個別注入は `set_backtest_screening_port()` を使う。
    """
    if _port is None:
        raise RuntimeError(
            "BacktestScreeningPort が未注入です。エントリポイントで "
            "src.orchestration.port_wiring.wire_ports() を呼ぶか、"
            "set_backtest_screening_port() で実装を注入してください。"
        )
    return _port
```

- [ ] **Step 4: テストが通ることを確認する**

Run: `python -m pytest tests/unit/backtest/test_screening_port.py -v`
Expected: 5 passed

- [ ] **Step 5: コミット**

```bash
git add python/src/backtest/screening_port.py python/tests/unit/backtest/test_screening_port.py
git commit -m "feat: backtest に screening アクセスポートを定義する"
```

---

## Task 4: アダプタを screening BC に置く

**Files:**
- Create: `src/screening/backtest_adapter.py`
- Test: `tests/unit/test_screening_backtest_adapter.py`

**Interfaces:**
- Consumes: Task 3 の `BacktestScreeningPort`
- Produces: `BacktestScreeningAdapter`（引数なしで構築できるクラス）

- [ ] **Step 1: 失敗するテストを書く**

`tests/unit/test_screening_backtest_adapter.py`:

```python
"""BacktestScreeningAdapter が Protocol を満たし実関数へ委譲すること。"""

import unittest
from unittest.mock import patch

import pandas as pd

from src.backtest.screening_port import BacktestScreeningPort
from src.domain.types import HoldRules
from src.screening.backtest_adapter import BacktestScreeningAdapter


class TestAdapter(unittest.TestCase):
    def test_satisfies_protocol(self):
        self.assertIsInstance(BacktestScreeningAdapter(), BacktestScreeningPort)

    def test_screen_delegates_with_same_arguments(self):
        with patch("src.screening.trend_screener.screen_trend_candidates") as m:
            m.return_value = ["sentinel"]
            out = BacktestScreeningAdapter().screen_trend_candidates("us", 30, "2024-01-02")
        self.assertEqual(out, ["sentinel"])
        m.assert_called_once_with(market="us", top_n=30, as_of="2024-01-02")

    def test_simulate_delegates_with_same_arguments(self):
        prices = pd.DataFrame({"date": ["2024-01-02"], "Close": [10.0]})
        rules = HoldRules()
        with patch("src.screening.hold_engine.simulate_position") as m:
            m.return_value = ["event"]
            out = BacktestScreeningAdapter().simulate_position(prices, "2024-01-02", rules)
        self.assertEqual(out, ["event"])
        args, kwargs = m.call_args
        self.assertIs(args[0], prices)
        self.assertEqual(kwargs["entry_date"], "2024-01-02")
        self.assertIs(kwargs["rules"], rules)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: テストが落ちることを確認する**

Run: `python -m pytest tests/unit/test_screening_backtest_adapter.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'src.screening.backtest_adapter'`）

- [ ] **Step 3: アダプタを書く**

```python
"""
BacktestScreeningAdapter - BacktestScreeningPort の具象実装。

screening BC の純粋関数を BacktestScreeningPort インターフェースに適合させる。
screening BC 内に配置することで backtest -> screening 直接依存を解消する。
Protocol は構造的なので、本モジュールは backtest を import しない。
"""

from __future__ import annotations

import pandas as pd

from src.domain.types import HoldRules, PositionEvent, TrendCandidate


class BacktestScreeningAdapter:
    """screening BC の関数を BacktestScreeningPort インターフェースに適合させるアダプター。"""

    def screen_trend_candidates(
        self, market: str, top_n: int, as_of: str
    ) -> list[TrendCandidate]:
        from src.screening.trend_screener import screen_trend_candidates

        return screen_trend_candidates(market=market, top_n=top_n, as_of=as_of)

    def simulate_position(
        self, prices: pd.DataFrame, entry_date: str, rules: HoldRules
    ) -> list[PositionEvent]:
        from src.screening.hold_engine import simulate_position

        return simulate_position(prices, entry_date=entry_date, rules=rules)
```

関数内 import は既存の `src/market_data/backtest_adapter.py` と同じ書式。

- [ ] **Step 4: テストが通ることを確認する**

Run: `python -m pytest tests/unit/test_screening_backtest_adapter.py -v`
Expected: 3 passed

- [ ] **Step 5: コミット**

```bash
git add python/src/screening/backtest_adapter.py python/tests/unit/test_screening_backtest_adapter.py
git commit -m "feat: screening に backtest 向けポートアダプタを追加する"
```

---

## Task 5: 結線してエンジンをポート経由にする

**Files:**
- Modify: `src/orchestration/port_wiring.py`
- Modify: `src/backtest/longterm/engine.py:43-44, 164, 291`
- Modify: `run_longterm_backtest.py`
- Modify: `tests/conftest.py:83-89`
- Modify: `tests/unit/backtest/longterm/test_engine_invariants.py`, `test_entry_lag.py`, `test_integer_shares.py`, `tests/unit/test_longterm_backtest.py`
- Test: `tests/unit/test_port_wiring.py`

**Interfaces:**
- Consumes: Task 3 の `set_backtest_screening_port` / `get_backtest_screening_port`、Task 4 の `BacktestScreeningAdapter`

- [ ] **Step 1: 失敗するテストを書く**

`tests/unit/test_port_wiring.py`:

```python
"""wire_ports() が screening ポートを注入すること。"""

import unittest

from src.backtest import screening_port as sp
from src.orchestration import port_wiring


class TestWirePorts(unittest.TestCase):
    def setUp(self):
        self._saved_port = sp._port
        self._saved_wired = port_wiring._wired

    def tearDown(self):
        sp._port = self._saved_port
        port_wiring._wired = self._saved_wired

    def test_injects_screening_port(self):
        sp._port = None
        port_wiring._wired = False
        port_wiring.wire_ports()
        self.assertIsNotNone(sp.get_backtest_screening_port())

    def test_is_idempotent(self):
        port_wiring._wired = False
        port_wiring.wire_ports()
        first = sp.get_backtest_screening_port()
        port_wiring.wire_ports()
        self.assertIs(sp.get_backtest_screening_port(), first)

    def test_force_reinjects(self):
        port_wiring._wired = False
        port_wiring.wire_ports()
        first = sp.get_backtest_screening_port()
        port_wiring.wire_ports(force=True)
        self.assertIsNot(sp.get_backtest_screening_port(), first)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: テストが落ちることを確認する**

Run: `python -m pytest tests/unit/test_port_wiring.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'src.backtest.screening_port'` にはならず、`wire_ports` が screening ポートを注入しないため `test_injects_screening_port` が `RuntimeError`）

- [ ] **Step 3: `wire_ports()` に注入を足す**

`src/orchestration/port_wiring.py` の `wire_ports()` 内に追加する。

```python
    from src.backtest.data_port import set_backtest_data_port
    from src.backtest.screening_port import set_backtest_screening_port
    from src.market_data.backtest_adapter import BacktestMarketDataAdapter
    from src.market_data.prediction_adapter import PredictionMarketDataAdapter
    from src.prediction.ports import set_market_data_port
    from src.screening.backtest_adapter import BacktestScreeningAdapter

    set_backtest_data_port(BacktestMarketDataAdapter())
    set_market_data_port(PredictionMarketDataAdapter())
    set_backtest_screening_port(BacktestScreeningAdapter())
    _wired = True
    logger.debug("ポート注入完了: BacktestDataPort / MarketDataPort / BacktestScreeningPort")
```

モジュール冒頭の docstring にある「BC（backtest / prediction）が定義する market_data アクセスポート」という説明も、screening ポートを含む形に直す。

- [ ] **Step 4: テストが通ることを確認する**

Run: `python -m pytest tests/unit/test_port_wiring.py -v`
Expected: 3 passed

- [ ] **Step 5: `conftest.py` の autouse fixture に 1 行足す**

`tests/conftest.py` の `_wire_default_ports` に追加する。

```python
    from src.backtest.data_port import set_backtest_data_port
    from src.backtest.screening_port import set_backtest_screening_port
    from src.market_data.backtest_adapter import BacktestMarketDataAdapter
    from src.market_data.prediction_adapter import PredictionMarketDataAdapter
    from src.prediction.ports import set_market_data_port
    from src.screening.backtest_adapter import BacktestScreeningAdapter

    set_backtest_data_port(BacktestMarketDataAdapter())
    set_market_data_port(PredictionMarketDataAdapter())
    set_backtest_screening_port(BacktestScreeningAdapter())
    yield
```

これが毎テスト前に既定アダプタを注入し直すため、個別に偽ポートを注入したテストの影響が次のテストへ漏れない。

- [ ] **Step 6: `engine.py` をポート経由にする**

`src/backtest/longterm/engine.py` の 43-44 行の 2 つの import を削除し、代わりに次を足す。

```python
from src.backtest.screening_port import get_backtest_screening_port
```

164 行を置き換える。

```python
        events = get_backtest_screening_port().simulate_position(
            series, entry_date=entry_date, rules=config.rules
        )
```

291 行を置き換える。

```python
                pending_entries[entry_date] = get_backtest_screening_port().screen_trend_candidates(
```

モジュール冒頭 docstring の「import について」節（19-22 行付近）を、ポート経由になった旨に書き換える。`.importlinter` への登録が不要になったことを明記する。

- [ ] **Step 7: 既存テストの patch を偽ポート注入に置き換える**

`patch.object(engine, "screen_trend_candidates", ...)` と `patch.object(engine, "simulate_position", ...)` は engine の属性ではなくなるため機能しない。次の 4 ファイルで置き換える。

- `tests/unit/backtest/longterm/test_engine_invariants.py`
- `tests/unit/backtest/longterm/test_entry_lag.py`
- `tests/unit/backtest/longterm/test_integer_shares.py`
- `tests/unit/test_longterm_backtest.py`

置き換えの形（`test_engine_invariants.py` の `_run` を例に）:

```python
class _StubScreeningPort:
    """screen は固定候補を返し、simulate は実装へ委譲する。"""

    def __init__(self, screen_fn):
        self._screen_fn = screen_fn

    def screen_trend_candidates(self, market, top_n, as_of):
        return self._screen_fn(market=market, top_n=top_n, as_of=as_of)

    def simulate_position(self, prices, entry_date, rules):
        from src.screening.hold_engine import simulate_position

        return simulate_position(prices, entry_date=entry_date, rules=rules)
```

`_run` の中で `set_backtest_screening_port(_StubScreeningPort(_screen))` を呼び、`patch.object(engine, "screen_trend_candidates", ...)` を削除する。`load_price_map` と `fetch_benchmark_returns` の patch はそのまま残す（engine のモジュール属性のままのため）。

**数値アサートは 1 件も変更しない。** 変更が必要になったら委譲にバグがある。

- [ ] **Step 8: CLI に `wire_ports()` を足す**

`run_longterm_backtest.py` の `main()` 冒頭で呼ぶ。他の `run_backtest*.py` と同じ形にする。

```python
def main():
    from src.orchestration.port_wiring import wire_ports

    wire_ports()
    args = parse_args()
    ...
```

- [ ] **Step 9: 該当テストを実行する**

Run:

```bash
python -m pytest tests/unit/backtest/ tests/unit/test_longterm_backtest.py \
  tests/unit/test_port_wiring.py tests/unit/test_screening_backtest_adapter.py \
  tests/unit/test_run_longterm_backtest_cli.py -v
```

Expected: 全 PASS。**既存の数値アサートは無改変**。

- [ ] **Step 10: コミット**

```bash
git add python/src python/tests python/run_longterm_backtest.py
git commit -m "refactor: 長期バックテストの screening 参照をポート経由にする"
```

---

## Task 6: `.importlinter` を空にして PR-8 を仕上げる

**Files:**
- Modify: `python/.importlinter`
- Modify: `python/VERSION`

- [ ] **Step 1: 残る 2 依存を消す**

両契約の `ignore_imports` から次を削除する（Contract 1 と Contract 2 の両方、計 4 行）。

```
    src.backtest.longterm.engine -> src.screening.hold_engine
    src.backtest.longterm.engine -> src.screening.trend_screener
```

これで両契約の `ignore_imports` が空になる。空になったキーは行ごと削除する（`ignore_imports =` だけ残さない）。

- [ ] **Step 2: 古いコメントを整理する**

Contract 1 の backtest → screening に関する説明ブロックを削除する。Contract 2 の次の記述を更新する。

- ロードマップの `[D] backtest → screening` の項を削除する（解消済みのため）
- 「削減目標」ブロックの「現状: 47件」は実態と合っていない。実測値に基づき、`ignore_imports` が 0 件になったことを書く。

- [ ] **Step 3: 契約が通ることを確認する**

Run: `PYTHONUTF8=1 lint-imports`
Expected: `Contracts: 2 kept, 0 broken.`

- [ ] **Step 4: 逆向きの依存が無いことを確認する**

Run:

```bash
grep -rn "from src.backtest" --include=*.py src/screening/
```

Expected: 出力なし（アダプタが backtest を import していたら Protocol の構造的性質を活かせていない）

- [ ] **Step 5: 全体のチェックを走らせる**

```bash
python -m pytest tests/unit/backtest/ tests/unit/domain/ tests/unit/test_longterm_backtest.py \
  tests/unit/test_hold_engine.py tests/unit/test_quality_gate.py tests/unit/test_multibagger_job.py \
  tests/unit/test_port_wiring.py tests/unit/test_screening_backtest_adapter.py \
  tests/unit/test_run_longterm_backtest_cli.py -v
black --check . && isort --check-only . && flake8 . && mypy src/
PYTHONUTF8=1 lint-imports
```

Expected: 全 PASS / 全 exit 0

- [ ] **Step 6: VERSION を patch 上げしてコミットする**

`python/VERSION` を PR-7 マージ後の develop 最新 +patch にする。**`printf '<version>\n' > python/VERSION` で書き、`xxd python/VERSION | head -1` で BOM が無く末尾が `0a` であることを確認する。**

```bash
git add python/.importlinter python/VERSION
git commit -m "chore: backtest -> screening の ignore_imports を撤去する"
```

---

# マージ

- [ ] **Step 1: CI を確認する**

```bash
gh pr checks <番号>
```

全チェックが pass するまで待つ。

- [ ] **Step 2: ユーザーが手動でマージする**

`gh pr merge` は auto mode の classifier が "Merge Without Review" として拒否する。ユーザーに次を依頼する。

```
! gh pr merge <番号> --merge
```

**`--squash` ではなく `--merge`。**

- [ ] **Step 3: マージ後に memory を更新する**

`project_execution_model_phase2.md` に Phase 3 完了を追記するか、Phase 3 用のファイルを新設して `MEMORY.md` に 1 行足す。記録すべきは「`ignore_imports` が 0 件になった」ことと、ポート注入によりモジュール属性 patch の脆さが解消されたこと。
