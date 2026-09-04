# レジームレバレッジ戦略ペーパートレードボット Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** STRATEGY.md 7章（強気相場・レバレッジ買い持ち戦略）をSPY単独・レバレッジ2.0倍・円建てでStockFixerのペーパートレードボットとして実装する。

**Architecture:** `src/trading/allocation_strategy/`（TQQQ/SHY配分戦略ボット）と同じ自己完結モジュールパターン。新規BC `src/trading/regime_leverage_strategy/` を新設し、既存の`PaperBroker`・`allocation_strategy`には一切手を入れない。状態は追記専用ログテーブル`regime_leverage_log`。週次ジョブ（金曜、レジーム転換＋新規エントリー判定）と日次ジョブ（毎営業日、初期損切り＋マージンコール判定）の2つのスケジュールジョブで構成する。

**Tech Stack:** Python, PostgreSQL（psycopg）, pandas, yfinance（`MarketDataPort`経由）, APScheduler

**Spec:** `docs/superpowers/specs/2026-09-03-regime-leverage-strategy-design.md`

## Global Constraints

- レバレッジ倍率: **2.0倍固定**
- 対象銘柄: **SPYのみ**（円建てで信用建て、為替リスクを含む）
- 初期資金: **1,000,000円**（`config/settings.py`に設定値として追加）
- 維持証拠金率（追証ライン）: **0.20**（`MARGIN_MAINTENANCE["JPY"]`と同じ値）
- 初期損切り幅: **entry - 3.0 × ATR(14)**
- 年利: **3.0%**（円建て信用金利、`INTEREST_ANNUAL["JPY"]`と同じ値）
- スリッページ: **片道0.1%**
- レジーム判定はドル建てSPYの200日移動平均線を使う（円建て判定はSTRATEGY.md 7.7節で検証済み・不採用）
- 週次ジョブは金曜の週足終値ベース、日次ジョブは当日安値ベースで判定する
- 金曜は日次ジョブ→週次ジョブの順で実行する

---

## ファイル構成

```
python/src/trading/regime_leverage_strategy/
  __init__.py          # 空
  types.py             # RegimeLeverageSnapshot, RegimeLeverageDecision dataclass
  indicators.py        # wilder_atr, build_weekly_frame（価格データからの指標計算、純粋関数）
  repository.py        # get_latest_snapshot / insert_snapshot（DB I/O）
  service.py           # decide_weekly_*, decide_daily_check, compute_equity_now（純粋ロジック）
                        # run_regime_leverage_weekly_check / run_regime_leverage_daily_margin_check（結合）

python/src/utils/db/migrations/
  0006_add_regime_leverage_log_postgres.sql

python/src/orchestration/jobs/periodic.py     # ジョブ関数を追加（既存ファイルに追記）
python/src/orchestration/scheduler.py          # re-export追加
python/run_scheduler.py                        # job_* 関数 + SCHEDULE_CONFIG登録
python/config/settings.py                      # 設定値追加

python/tests/unit/test_regime_leverage_indicators.py
python/tests/unit/test_regime_leverage_repository.py
python/tests/unit/test_regime_leverage_service.py
python/tests/unit/test_periodic_jobs.py         # 既存ファイルにジョブ関数のテストを追記
```

---

### Task 1: 設定値 + DBマイグレーション + types.py

**Files:**
- Modify: `python/config/settings.py`
- Create: `python/src/utils/db/migrations/0006_add_regime_leverage_log_postgres.sql`
- Create: `python/src/trading/regime_leverage_strategy/__init__.py`
- Create: `python/src/trading/regime_leverage_strategy/types.py`

**Interfaces:**
- Produces: `RegimeLeverageSnapshot`（DBの1行を表すfrozen dataclass）、`RegimeLeverageDecision`（判定結果を表すfrozen dataclass）。以降の全タスクがこの2つの型を使う。

- [ ] **Step 1: `config/settings.py` に設定値を追加**

`ALLOCATION_STRATEGY_*` の定義ブロック（`config/settings.py:61-65`付近）のすぐ後に追加する。まず`Settings`クラス内（`ALLOCATION_STRATEGY_REBALANCE_YEARS: int = Field(default=2)`の直後）に以下を追記:

```python
    REGIME_LEVERAGE_SYMBOL: str = Field(default="SPY")
    REGIME_LEVERAGE_RATIO: float = Field(default=2.0)
    REGIME_LEVERAGE_INITIAL_CAPITAL_JPY: float = Field(default=1_000_000.0)
    REGIME_LEVERAGE_MARGIN_MAINTENANCE: float = Field(default=0.20)
    REGIME_LEVERAGE_INITIAL_STOP_ATR_MULT: float = Field(default=3.0)
    REGIME_LEVERAGE_INTEREST_ANNUAL: float = Field(default=0.030)
    REGIME_LEVERAGE_SLIPPAGE_PCT: float = Field(default=0.001)
```

次に、モジュールレベル定数のエクスポートブロック（`ALLOCATION_STRATEGY_REBALANCE_YEARS: int = settings.ALLOCATION_STRATEGY_REBALANCE_YEARS`の直後、`config/settings.py:226`付近）に以下を追記:

```python
REGIME_LEVERAGE_SYMBOL: str = settings.REGIME_LEVERAGE_SYMBOL
REGIME_LEVERAGE_RATIO: float = settings.REGIME_LEVERAGE_RATIO
REGIME_LEVERAGE_INITIAL_CAPITAL_JPY: float = settings.REGIME_LEVERAGE_INITIAL_CAPITAL_JPY
REGIME_LEVERAGE_MARGIN_MAINTENANCE: float = settings.REGIME_LEVERAGE_MARGIN_MAINTENANCE
REGIME_LEVERAGE_INITIAL_STOP_ATR_MULT: float = settings.REGIME_LEVERAGE_INITIAL_STOP_ATR_MULT
REGIME_LEVERAGE_INTEREST_ANNUAL: float = settings.REGIME_LEVERAGE_INTEREST_ANNUAL
REGIME_LEVERAGE_SLIPPAGE_PCT: float = settings.REGIME_LEVERAGE_SLIPPAGE_PCT
```

- [ ] **Step 2: マイグレーションファイルを作成**

`python/src/utils/db/migrations/0006_add_regime_leverage_log_postgres.sql`:

```sql
-- 0006_add_regime_leverage_log_postgres: STRATEGY.md 7章(強気相場・レバレッジ買い持ち)
-- ペーパートレードの状態を追記専用ログとして記録するテーブル。id最大の行が現在の
-- 建玉・評価額の状態を表す。allocation_rebalance_logと異なり、週次(レジーム判定)と
-- 日次(マージンコール判定)の2種類のジョブが同じ状態を読み書きするため、action/reason
-- で発生源を区別する。
CREATE TABLE IF NOT EXISTS regime_leverage_log (
    id                    BIGSERIAL PRIMARY KEY,
    executed_at           TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    action                VARCHAR NOT NULL,
    reason                VARCHAR NOT NULL,
    spy_price_usd         DOUBLE PRECISION NOT NULL,
    usdjpy_rate           DOUBLE PRECISION NOT NULL,
    shares                DOUBLE PRECISION NOT NULL,
    entry_date            TIMESTAMP,
    entry_price_jpy       DOUBLE PRECISION,
    entry_commission_jpy  DOUBLE PRECISION,
    equity_at_entry_jpy   DOUBLE PRECISION,
    stop_price_jpy        DOUBLE PRECISION,
    equity_now_jpy        DOUBLE PRECISION NOT NULL,
    maintenance_ratio     DOUBLE PRECISION
);
```

- [ ] **Step 3: `types.py` を作成**

`python/src/trading/regime_leverage_strategy/__init__.py` は空ファイルとして作成する。

`python/src/trading/regime_leverage_strategy/types.py`:

```python
"""レジームレバレッジ戦略(TQQQ/短期債と同様の自己完結モジュール)の型定義。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass(frozen=True)
class RegimeLeverageSnapshot:
    """ある時点でのレジームレバレッジ戦略の建玉・評価額状態(regime_leverage_logの1行)。"""

    id: int
    executed_at: datetime
    action: str
    reason: str
    spy_price_usd: float
    usdjpy_rate: float
    shares: float
    entry_date: Optional[datetime]
    entry_price_jpy: Optional[float]
    entry_commission_jpy: Optional[float]
    equity_at_entry_jpy: Optional[float]
    stop_price_jpy: Optional[float]
    equity_now_jpy: float
    maintenance_ratio: Optional[float]


@dataclass(frozen=True)
class RegimeLeverageDecision:
    """週次/日次の判定結果(insert_snapshotへそのまま渡せる形)。"""

    action: str  # 'entry' | 'exit' | 'noop'
    reason: str  # 'regime_entry' | 'regime_flip' | 'initial_stop' | 'margin_call' | 'weekly_noop' | 'daily_noop'
    spy_price_usd: float
    usdjpy_rate: float
    shares: float
    entry_date: Optional[datetime]
    entry_price_jpy: Optional[float]
    entry_commission_jpy: Optional[float]
    equity_at_entry_jpy: Optional[float]
    stop_price_jpy: Optional[float]
    equity_now_jpy: float
    maintenance_ratio: Optional[float]
```

- [ ] **Step 4: 型チェックとlintを確認**

```powershell
cd python
py -m mypy src/trading/regime_leverage_strategy/types.py --ignore-missing-imports --implicit-optional
py -m black src/trading/regime_leverage_strategy/ config/settings.py --check
py -m flake8 src/trading/regime_leverage_strategy/ config/settings.py
```

Expected: すべて成功（差分があれば `py -m black` を `--check` なしで実行して整形する）

- [ ] **Step 5: Commit**

```bash
git add python/config/settings.py python/src/utils/db/migrations/0006_add_regime_leverage_log_postgres.sql python/src/trading/regime_leverage_strategy/
git commit -m "feat: レジームレバレッジ戦略の設定値・テーブル・型定義を追加"
```

---

### Task 2: repository.py

**Files:**
- Create: `python/src/trading/regime_leverage_strategy/repository.py`
- Test: `python/tests/unit/test_regime_leverage_repository.py`

**Interfaces:**
- Consumes: `RegimeLeverageSnapshot`, `RegimeLeverageDecision`（Task 1で定義）
- Produces: `get_latest_snapshot() -> Optional[RegimeLeverageSnapshot]`、`insert_snapshot(decision: RegimeLeverageDecision) -> None`。Task 5がこれを呼ぶ。

`src/trading/allocation_strategy/repository.py` と全く同じパターンを踏襲する（`_db_connection`コンテキストマネージャ、`fetchone`/`execute`）。

- [ ] **Step 1: 失敗するテストを書く**

`python/tests/unit/test_regime_leverage_repository.py`:

```python
"""ユニットテスト: src.trading.regime_leverage_strategy.repository"""

import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch


class TestGetLatestSnapshot(unittest.TestCase):
    def _mock_db(self, row):
        mock_con = MagicMock()
        mock_con.execute.return_value.fetchone.return_value = row
        return mock_con

    def test_returns_none_when_no_rows(self):
        from src.trading.regime_leverage_strategy.repository import get_latest_snapshot

        mock_con = self._mock_db(None)
        with patch("src.trading.regime_leverage_strategy.repository._db_connection") as mock_ctx:
            mock_ctx.return_value.__enter__ = MagicMock(return_value=mock_con)
            mock_ctx.return_value.__exit__ = MagicMock(return_value=False)
            result = get_latest_snapshot()
        self.assertIsNone(result)

    def test_parses_row_into_snapshot(self):
        from src.trading.regime_leverage_strategy.repository import get_latest_snapshot

        row = (
            3, datetime(2026, 9, 1), "entry", "regime_entry",
            560.0, 148.5, 3500.0,
            datetime(2026, 9, 1), 83160.0, 500.0, 1000000.0, 78960.0,
            1005000.0, None,
        )
        mock_con = self._mock_db(row)
        with patch("src.trading.regime_leverage_strategy.repository._db_connection") as mock_ctx:
            mock_ctx.return_value.__enter__ = MagicMock(return_value=mock_con)
            mock_ctx.return_value.__exit__ = MagicMock(return_value=False)
            result = get_latest_snapshot()

        self.assertEqual(result.id, 3)
        self.assertEqual(result.action, "entry")
        self.assertEqual(result.shares, 3500.0)
        self.assertEqual(result.entry_price_jpy, 83160.0)
        self.assertEqual(result.equity_now_jpy, 1005000.0)
        self.assertIsNone(result.maintenance_ratio)


class TestInsertSnapshot(unittest.TestCase):
    def test_executes_insert_with_expected_params(self):
        from src.trading.regime_leverage_strategy.repository import insert_snapshot
        from src.trading.regime_leverage_strategy.types import RegimeLeverageDecision

        decision = RegimeLeverageDecision(
            action="entry", reason="regime_entry",
            spy_price_usd=560.0, usdjpy_rate=148.5, shares=3500.0,
            entry_date=datetime(2026, 9, 1), entry_price_jpy=83160.0,
            entry_commission_jpy=500.0, equity_at_entry_jpy=1000000.0,
            stop_price_jpy=78960.0, equity_now_jpy=1005000.0, maintenance_ratio=None,
        )
        mock_con = MagicMock()
        with patch("src.trading.regime_leverage_strategy.repository._db_connection") as mock_ctx:
            mock_ctx.return_value.__enter__ = MagicMock(return_value=mock_con)
            mock_ctx.return_value.__exit__ = MagicMock(return_value=False)
            insert_snapshot(decision)

        mock_con.execute.assert_called_once()
        sql_call = mock_con.execute.call_args
        self.assertIn("INSERT INTO regime_leverage_log", sql_call[0][0])
        self.assertEqual(sql_call[0][1][0], "entry")
        self.assertEqual(sql_call[0][1][4], 3500.0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: テストが失敗することを確認**

```powershell
cd python
py -m pytest tests/unit/test_regime_leverage_repository.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.trading.regime_leverage_strategy.repository'`

- [ ] **Step 3: `repository.py` を実装**

```python
"""regime_leverage_log テーブルの読み書き。"""

from typing import Optional

from src.trading.regime_leverage_strategy.types import RegimeLeverageDecision, RegimeLeverageSnapshot
from src.utils.db._connection import _db_connection
from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_latest_snapshot() -> Optional[RegimeLeverageSnapshot]:
    """最新の状態行を返す。まだ1行も無ければ None。"""
    with _db_connection() as con:
        row = con.execute("""
            SELECT id, executed_at, action, reason, spy_price_usd, usdjpy_rate, shares,
                   entry_date, entry_price_jpy, entry_commission_jpy, equity_at_entry_jpy,
                   stop_price_jpy, equity_now_jpy, maintenance_ratio
            FROM regime_leverage_log
            ORDER BY id DESC
            LIMIT 1
            """).fetchone()
    if row is None:
        return None
    return RegimeLeverageSnapshot(
        id=row[0],
        executed_at=row[1],
        action=row[2],
        reason=row[3],
        spy_price_usd=row[4],
        usdjpy_rate=row[5],
        shares=row[6],
        entry_date=row[7],
        entry_price_jpy=row[8],
        entry_commission_jpy=row[9],
        equity_at_entry_jpy=row[10],
        stop_price_jpy=row[11],
        equity_now_jpy=row[12],
        maintenance_ratio=row[13],
    )


def insert_snapshot(decision: RegimeLeverageDecision) -> None:
    """新しい状態行を追記する。"""
    with _db_connection() as con:
        con.execute(
            """
            INSERT INTO regime_leverage_log (
                action, reason, spy_price_usd, usdjpy_rate, shares,
                entry_date, entry_price_jpy, entry_commission_jpy, equity_at_entry_jpy,
                stop_price_jpy, equity_now_jpy, maintenance_ratio
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            [
                decision.action,
                decision.reason,
                decision.spy_price_usd,
                decision.usdjpy_rate,
                decision.shares,
                decision.entry_date,
                decision.entry_price_jpy,
                decision.entry_commission_jpy,
                decision.equity_at_entry_jpy,
                decision.stop_price_jpy,
                decision.equity_now_jpy,
                decision.maintenance_ratio,
            ],
        )
    logger.info(
        "regime_leverage_log 追記: action=%s reason=%s shares=%.4f equity_now_jpy=%.2f",
        decision.action,
        decision.reason,
        decision.shares,
        decision.equity_now_jpy,
    )
```

- [ ] **Step 4: テストが通ることを確認**

```powershell
py -m pytest tests/unit/test_regime_leverage_repository.py -v
```

Expected: 3 passed

- [ ] **Step 5: lint確認とCommit**

```powershell
py -m black src/trading/regime_leverage_strategy/repository.py tests/unit/test_regime_leverage_repository.py
py -m isort src/trading/regime_leverage_strategy/repository.py tests/unit/test_regime_leverage_repository.py
py -m flake8 src/trading/regime_leverage_strategy/repository.py tests/unit/test_regime_leverage_repository.py
py -m mypy src/trading/regime_leverage_strategy/repository.py --ignore-missing-imports --implicit-optional
```

```bash
git add python/src/trading/regime_leverage_strategy/repository.py python/tests/unit/test_regime_leverage_repository.py
git commit -m "feat: regime_leverage_log のrepositoryを追加"
```

---

### Task 3: indicators.py（ATR・週足フレーム計算）

**Files:**
- Create: `python/src/trading/regime_leverage_strategy/indicators.py`
- Test: `python/tests/unit/test_regime_leverage_indicators.py`

**Interfaces:**
- Consumes: なし（pandas DataFrameのみ）
- Produces: `wilder_atr(df: pd.DataFrame, period: int = 14) -> pd.Series`、`build_weekly_frame(daily_df: pd.DataFrame) -> pd.DataFrame`（列: Close, Low, High, MA200, ATR14 を持つ日次df・週足dfそれぞれに対応。Task 4がこれを使う）

`trading-strategy/backtest/backtest.py` の `wilder_atr` と `backtest_regime.py` の `compute_regime_indicators` のロジックを移植する（バックテストとの計算式の一致を保つため、既存ロジックをそのまま再現する）。

- [ ] **Step 1: 失敗するテストを書く**

```python
"""ユニットテスト: src.trading.regime_leverage_strategy.indicators"""

import unittest

import numpy as np
import pandas as pd

from src.trading.regime_leverage_strategy.indicators import build_weekly_frame, wilder_atr


class TestWilderAtr(unittest.TestCase):
    def test_atr_is_nan_before_period(self):
        idx = pd.bdate_range("2026-01-01", periods=20)
        df = pd.DataFrame(
            {
                "High": np.full(20, 105.0),
                "Low": np.full(20, 95.0),
                "Close": np.full(20, 100.0),
            },
            index=idx,
        )
        atr = wilder_atr(df, period=14)
        self.assertTrue(atr.iloc[:13].isna().all())
        self.assertFalse(pd.isna(atr.iloc[13]))

    def test_atr_reflects_true_range(self):
        idx = pd.bdate_range("2026-01-01", periods=20)
        df = pd.DataFrame(
            {
                "High": np.full(20, 110.0),
                "Low": np.full(20, 90.0),
                "Close": np.full(20, 100.0),
            },
            index=idx,
        )
        atr = wilder_atr(df, period=14)
        # High-Low=20 が唯一の候補(前日終値との差より大きい)なのでATRは20に収束する
        self.assertAlmostEqual(atr.iloc[-1], 20.0, places=6)


class TestBuildWeeklyFrame(unittest.TestCase):
    def test_adds_ma200_and_week_close(self):
        idx = pd.bdate_range("2024-01-01", periods=260)
        rng = np.random.default_rng(0)
        prices = 100 + np.cumsum(rng.normal(0, 1, 260))
        df = pd.DataFrame(
            {
                "Open": prices,
                "High": prices + 1,
                "Low": prices - 1,
                "Close": prices,
                "Volume": np.full(260, 1_000_000),
            },
            index=idx,
        )
        result = build_weekly_frame(df)
        self.assertIn("MA200", result.columns)
        self.assertIn("ATR14", result.columns)
        # 200本目以降はMA200が非NaN
        self.assertFalse(pd.isna(result["MA200"].iloc[-1]))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: テストが失敗することを確認**

```powershell
py -m pytest tests/unit/test_regime_leverage_indicators.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.trading.regime_leverage_strategy.indicators'`

- [ ] **Step 3: `indicators.py` を実装**

```python
"""レジームレバレッジ戦略で使う指標計算(ATR・200日線)。

trading-strategy/backtest/backtest.py の wilder_atr、
backtest_regime.py の compute_regime_indicators と同じ計算式を使う
(バックテスト結果との整合性を保つため)。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_REGIME_MA = 200
_ATR_PERIOD = 14


def wilder_atr(df: pd.DataFrame, period: int = _ATR_PERIOD) -> pd.Series:
    """Wilderのスムージング法によるATRを計算する。"""
    prev_close = df["Close"].shift(1)
    tr = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.copy()
    atr.iloc[:period] = np.nan
    atr.iloc[period - 1] = tr.iloc[0:period].mean()
    for i in range(period, len(tr)):
        atr.iloc[i] = (atr.iloc[i - 1] * (period - 1) + tr.iloc[i]) / period
    return atr


def build_weekly_frame(daily_df: pd.DataFrame) -> pd.DataFrame:
    """日足dfにATR14・200日線を追加する(週足化はしない。呼び出し側が最終行=直近営業日
    を使う。7章の判定は「その週の金曜終値」だが、日次ジョブ実行時点では当該週がまだ
    確定していないため、呼び出し側で週の最終営業日かどうかを判定する)。"""
    df = daily_df.copy()
    df["ATR14"] = wilder_atr(df, _ATR_PERIOD)
    df["MA200"] = df["Close"].rolling(_REGIME_MA).mean()
    return df
```

- [ ] **Step 4: テストが通ることを確認**

```powershell
py -m pytest tests/unit/test_regime_leverage_indicators.py -v
```

Expected: 3 passed

- [ ] **Step 5: lint確認とCommit**

```powershell
py -m black src/trading/regime_leverage_strategy/indicators.py tests/unit/test_regime_leverage_indicators.py
py -m isort src/trading/regime_leverage_strategy/indicators.py tests/unit/test_regime_leverage_indicators.py
py -m flake8 src/trading/regime_leverage_strategy/indicators.py tests/unit/test_regime_leverage_indicators.py
py -m mypy src/trading/regime_leverage_strategy/indicators.py --ignore-missing-imports --implicit-optional
```

```bash
git add python/src/trading/regime_leverage_strategy/indicators.py python/tests/unit/test_regime_leverage_indicators.py
git commit -m "feat: レジームレバレッジ戦略のATR/200日線計算を追加"
```

---

### Task 4: service.py — 週次判定ロジック（純粋関数）

**Files:**
- Create: `python/src/trading/regime_leverage_strategy/service.py`
- Test: `python/tests/unit/test_regime_leverage_service.py`

**Interfaces:**
- Consumes: `RegimeLeverageSnapshot`, `RegimeLeverageDecision`（Task 1）
- Produces: `decide_weekly_entry(cash_jpy, week_close_usd, ma200_usd, atr14_usd, usdjpy_rate, now) -> RegimeLeverageDecision`、`decide_weekly_exit(snapshot, week_close_usd, ma200_usd, usdjpy_rate, now) -> RegimeLeverageDecision`、`compute_equity_now(snapshot, current_price_jpy, now) -> float`。Task 5・Task 6がこれらを使う。

- [ ] **Step 1: 失敗するテストを書く**

```python
"""ユニットテスト: src.trading.regime_leverage_strategy.service"""

import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from src.trading.regime_leverage_strategy.service import (
    compute_equity_now,
    decide_weekly_entry,
    decide_weekly_exit,
)
from src.trading.regime_leverage_strategy.types import RegimeLeverageSnapshot

# Task 5 (decide_daily_check) と Task 6 (run_regime_leverage_*) のテストも
# このファイルに追記される。MagicMock/patch/numpy/pandas はそれらのテストが使う。


def _holding_snapshot(**overrides):
    base = dict(
        id=1,
        executed_at=datetime(2026, 1, 2),
        action="entry",
        reason="regime_entry",
        spy_price_usd=500.0,
        usdjpy_rate=145.0,
        # 自己資金1,000,000円・レバレッジ2.0倍・entry_price_usd=500.5(スリッページ込み)
        # とした場合の株数と整合させる: floor((1,000,000/145.0)*2.0 / 500.5) = 27
        shares=27.0,
        entry_date=datetime(2026, 1, 2),
        entry_price_jpy=72500.0,  # 500.0 * 145.0
        entry_commission_jpy=0.0,
        equity_at_entry_jpy=1_000_000.0,
        stop_price_jpy=65000.0,
        equity_now_jpy=1_000_000.0,
        maintenance_ratio=None,
    )
    base.update(overrides)
    return RegimeLeverageSnapshot(**base)


class TestDecideWeeklyEntry(unittest.TestCase):
    def test_no_entry_when_regime_down(self):
        decision = decide_weekly_entry(
            cash_jpy=1_000_000.0,
            week_close_usd=480.0,
            ma200_usd=500.0,  # close < ma200 → 下降レジーム
            atr14_usd=5.0,
            usdjpy_rate=145.0,
            now=datetime(2026, 1, 9),
        )
        self.assertEqual(decision.action, "noop")
        self.assertEqual(decision.reason, "weekly_noop")
        self.assertEqual(decision.shares, 0.0)
        self.assertEqual(decision.equity_now_jpy, 1_000_000.0)

    def test_enters_with_leverage_when_regime_up(self):
        decision = decide_weekly_entry(
            cash_jpy=1_000_000.0,
            week_close_usd=500.0,
            ma200_usd=480.0,  # close > ma200 → 上昇レジーム
            atr14_usd=5.0,
            usdjpy_rate=145.0,
            now=datetime(2026, 1, 9),
        )
        self.assertEqual(decision.action, "entry")
        self.assertEqual(decision.reason, "regime_entry")
        # 建玉USD時価 = 1,000,000 / 145.0 * 2.0 = 13,793.10ドル相当 → shares = floor(13793.10 / entry_price)
        # entry_price_usd = 500.0 * 1.001 = 500.5、entry_price_jpy = 500.5 * 145.0
        self.assertGreater(decision.shares, 0)
        self.assertIsNotNone(decision.entry_price_jpy)
        self.assertIsNotNone(decision.stop_price_jpy)
        self.assertLess(decision.stop_price_jpy, decision.entry_price_jpy)


class TestDecideWeeklyExit(unittest.TestCase):
    def test_holds_when_regime_still_up(self):
        snap = _holding_snapshot()
        decision = decide_weekly_exit(
            snap, week_close_usd=520.0, ma200_usd=480.0, usdjpy_rate=146.0,
            now=datetime(2026, 1, 16),
        )
        self.assertEqual(decision.action, "noop")
        self.assertEqual(decision.shares, snap.shares)

    def test_exits_on_regime_flip(self):
        snap = _holding_snapshot()
        decision = decide_weekly_exit(
            snap, week_close_usd=470.0, ma200_usd=480.0, usdjpy_rate=146.0,
            now=datetime(2026, 1, 16),
        )
        self.assertEqual(decision.action, "exit")
        self.assertEqual(decision.reason, "regime_flip")
        self.assertEqual(decision.shares, 0.0)


class TestComputeEquityNow(unittest.TestCase):
    def test_flat_price_returns_equity_at_entry_minus_interest(self):
        snap = _holding_snapshot(entry_commission_jpy=0.0)
        equity = compute_equity_now(snap, current_price_jpy=snap.entry_price_jpy, now=datetime(2026, 1, 3))
        # 1日分の金利のみ差し引かれる: entry_price_jpy * shares * 0.03 / 365 * 1日
        expected_interest = snap.entry_price_jpy * snap.shares * 0.030 / 365 * 1
        self.assertAlmostEqual(equity, snap.equity_at_entry_jpy - expected_interest, places=2)

    def test_price_rise_increases_equity(self):
        snap = _holding_snapshot(entry_commission_jpy=0.0)
        higher_price = snap.entry_price_jpy * 1.05
        equity = compute_equity_now(snap, current_price_jpy=higher_price, now=snap.entry_date)
        expected_unrealized = (higher_price - snap.entry_price_jpy) * snap.shares
        self.assertAlmostEqual(equity, snap.equity_at_entry_jpy + expected_unrealized, places=2)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: テストが失敗することを確認**

```powershell
py -m pytest tests/unit/test_regime_leverage_service.py -v
```

Expected: `ModuleNotFoundError: No module named 'src.trading.regime_leverage_strategy.service'`

- [ ] **Step 3: `service.py` に週次ロジック部分を実装**

```python
"""レジームレバレッジ戦略(TQQQ/短期債と同様の自己完結モジュール)の判定ロジック。

STRATEGY.md 7章の週次(レジーム転換・新規エントリー)・日次(初期損切り・マージンコール)
判定を実装する。バックテスト(trading-strategy/backtest/backtest_regime_leverage.py)と
同じ計算式を使う。
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from config.settings import (
    REGIME_LEVERAGE_INITIAL_STOP_ATR_MULT,
    REGIME_LEVERAGE_INTEREST_ANNUAL,
    REGIME_LEVERAGE_MARGIN_MAINTENANCE,
    REGIME_LEVERAGE_RATIO,
    REGIME_LEVERAGE_SLIPPAGE_PCT,
)
from src.trading.regime_leverage_strategy.types import RegimeLeverageDecision, RegimeLeverageSnapshot

# 米国株信用の手数料(trading-strategy/backtest.pyのCOMMISSION_PCT["USD"]と同じ値)
_COMMISSION_PCT_USD = 0.0033
_COMMISSION_CAP_USD = 16.5


def _calc_commission_usd(notional_usd: float) -> float:
    return min(notional_usd * _COMMISSION_PCT_USD, _COMMISSION_CAP_USD)


def compute_equity_now(snapshot: RegimeLeverageSnapshot, current_price_jpy: float, now: datetime) -> float:
    """保有中ポジションの現在評価額(円)を再計算する。

    バックテストのように保有期間中の含み損益・金利を都度累積して持ち回るのではなく、
    エントリー時点の情報(entry_date/entry_price_jpy/equity_at_entry_jpy/
    entry_commission_jpy)から毎回再計算する(累積値の更新漏れによるバグを避けるため)。
    """
    if snapshot.entry_price_jpy is None or snapshot.entry_date is None or snapshot.equity_at_entry_jpy is None:
        raise ValueError("保有中でないsnapshotに対してcompute_equity_nowは呼べない")
    unrealized_pnl = (current_price_jpy - snapshot.entry_price_jpy) * snapshot.shares
    days_held = (now.date() - snapshot.entry_date.date()).days
    interest_accrued = (
        snapshot.entry_price_jpy * snapshot.shares * REGIME_LEVERAGE_INTEREST_ANNUAL / 365 * days_held
    )
    commission = snapshot.entry_commission_jpy or 0.0
    return snapshot.equity_at_entry_jpy + unrealized_pnl - interest_accrued - commission


def decide_weekly_entry(
    cash_jpy: float,
    week_close_usd: float,
    ma200_usd: float,
    atr14_usd: float,
    usdjpy_rate: float,
    now: datetime,
) -> RegimeLeverageDecision:
    """未保有時の週次判定: レジームが上昇なら新規エントリーする。"""
    regime_up = week_close_usd > ma200_usd
    if not regime_up:
        return RegimeLeverageDecision(
            action="noop", reason="weekly_noop",
            spy_price_usd=week_close_usd, usdjpy_rate=usdjpy_rate, shares=0.0,
            entry_date=None, entry_price_jpy=None, entry_commission_jpy=None,
            equity_at_entry_jpy=None, stop_price_jpy=None,
            equity_now_jpy=cash_jpy, maintenance_ratio=None,
        )

    entry_price_usd = week_close_usd * (1 + REGIME_LEVERAGE_SLIPPAGE_PCT)
    entry_price_jpy = entry_price_usd * usdjpy_rate
    notional_target_usd = (cash_jpy / usdjpy_rate) * REGIME_LEVERAGE_RATIO
    shares = float(int(notional_target_usd // entry_price_usd))
    if shares <= 0:
        return RegimeLeverageDecision(
            action="noop", reason="weekly_noop",
            spy_price_usd=week_close_usd, usdjpy_rate=usdjpy_rate, shares=0.0,
            entry_date=None, entry_price_jpy=None, entry_commission_jpy=None,
            equity_at_entry_jpy=None, stop_price_jpy=None,
            equity_now_jpy=cash_jpy, maintenance_ratio=None,
        )

    commission_jpy = _calc_commission_usd(entry_price_usd * shares) * usdjpy_rate
    stop_price_jpy = (week_close_usd - REGIME_LEVERAGE_INITIAL_STOP_ATR_MULT * atr14_usd) * usdjpy_rate

    return RegimeLeverageDecision(
        action="entry", reason="regime_entry",
        spy_price_usd=week_close_usd, usdjpy_rate=usdjpy_rate, shares=shares,
        entry_date=now, entry_price_jpy=entry_price_jpy, entry_commission_jpy=commission_jpy,
        equity_at_entry_jpy=cash_jpy, stop_price_jpy=stop_price_jpy,
        equity_now_jpy=cash_jpy - commission_jpy, maintenance_ratio=None,
    )


def decide_weekly_exit(
    snapshot: RegimeLeverageSnapshot,
    week_close_usd: float,
    ma200_usd: float,
    usdjpy_rate: float,
    now: datetime,
) -> RegimeLeverageDecision:
    """保有中の週次判定: レジーム転換のみ判定する(初期損切り・マージンコールは日次ジョブが担当)。"""
    current_price_jpy = week_close_usd * usdjpy_rate
    equity_now = compute_equity_now(snapshot, current_price_jpy, now)
    regime_up = week_close_usd > ma200_usd

    if regime_up:
        return RegimeLeverageDecision(
            action="noop", reason="weekly_noop",
            spy_price_usd=week_close_usd, usdjpy_rate=usdjpy_rate, shares=snapshot.shares,
            entry_date=snapshot.entry_date, entry_price_jpy=snapshot.entry_price_jpy,
            entry_commission_jpy=snapshot.entry_commission_jpy,
            equity_at_entry_jpy=snapshot.equity_at_entry_jpy, stop_price_jpy=snapshot.stop_price_jpy,
            equity_now_jpy=equity_now, maintenance_ratio=None,
        )

    exit_price_jpy = current_price_jpy * (1 - REGIME_LEVERAGE_SLIPPAGE_PCT)
    exit_equity = compute_equity_now(snapshot, exit_price_jpy, now)
    return RegimeLeverageDecision(
        action="exit", reason="regime_flip",
        spy_price_usd=week_close_usd, usdjpy_rate=usdjpy_rate, shares=0.0,
        entry_date=None, entry_price_jpy=None, entry_commission_jpy=None,
        equity_at_entry_jpy=None, stop_price_jpy=None,
        equity_now_jpy=exit_equity, maintenance_ratio=None,
    )
```

- [ ] **Step 4: テストが通ることを確認**

```powershell
py -m pytest tests/unit/test_regime_leverage_service.py -v
```

Expected: 6 passed

- [ ] **Step 5: lint確認とCommit**

```powershell
py -m black src/trading/regime_leverage_strategy/service.py tests/unit/test_regime_leverage_service.py
py -m isort src/trading/regime_leverage_strategy/service.py tests/unit/test_regime_leverage_service.py
py -m flake8 src/trading/regime_leverage_strategy/service.py tests/unit/test_regime_leverage_service.py
py -m mypy src/trading/regime_leverage_strategy/service.py --ignore-missing-imports --implicit-optional
```

```bash
git add python/src/trading/regime_leverage_strategy/service.py python/tests/unit/test_regime_leverage_service.py
git commit -m "feat: レジームレバレッジ戦略の週次判定ロジックを追加"
```

---

### Task 5: service.py — 日次判定ロジック（純粋関数）を追加

**Files:**
- Modify: `python/src/trading/regime_leverage_strategy/service.py`（Task 4で作成したファイルに追記）
- Modify: `python/tests/unit/test_regime_leverage_service.py`（追記）

**Interfaces:**
- Consumes: `compute_equity_now`（Task 4で定義、同ファイル内なのでそのまま呼べる）
- Produces: `decide_daily_check(snapshot, day_low_usd, usdjpy_rate, now) -> RegimeLeverageDecision`。Task 6がこれを使う。

- [ ] **Step 1: 失敗するテストを追記**

`test_regime_leverage_service.py` の末尾（`if __name__ == "__main__":` の直前）に追記:

```python
class TestDecideDailyCheck(unittest.TestCase):
    def test_noop_when_above_stop_and_maintenance(self):
        from src.trading.regime_leverage_strategy.service import decide_daily_check

        snap = _holding_snapshot()
        decision = decide_daily_check(
            snap, day_low_usd=510.0, usdjpy_rate=146.0, now=datetime(2026, 1, 5)
        )
        self.assertEqual(decision.action, "noop")
        self.assertEqual(decision.reason, "daily_noop")
        self.assertEqual(decision.shares, snap.shares)
        self.assertIsNotNone(decision.maintenance_ratio)

    def test_initial_stop_triggers_exit(self):
        from src.trading.regime_leverage_strategy.service import decide_daily_check

        snap = _holding_snapshot(stop_price_jpy=70000.0)
        # day_low_usdを円換算するとstop_price_jpy(70000)を下回る値にする
        decision = decide_daily_check(
            snap, day_low_usd=470.0, usdjpy_rate=145.0, now=datetime(2026, 1, 5)
        )
        self.assertEqual(decision.action, "exit")
        self.assertEqual(decision.reason, "initial_stop")
        self.assertEqual(decision.shares, 0.0)

    def test_margin_call_triggers_exit_before_stop_check(self):
        from src.trading.regime_leverage_strategy.service import decide_daily_check

        # レバレッジ2倍で建てた直後に急落し、維持率が0.20を割るケース
        snap = _holding_snapshot(
            shares=4000.0, entry_price_jpy=72500.0, equity_at_entry_jpy=1_000_000.0,
            entry_commission_jpy=0.0, stop_price_jpy=1000.0,  # stopには触れない値
        )
        decision = decide_daily_check(
            snap, day_low_usd=200.0, usdjpy_rate=145.0, now=datetime(2026, 1, 5)
        )
        self.assertEqual(decision.action, "exit")
        self.assertEqual(decision.reason, "margin_call")
```

- [ ] **Step 2: テストが失敗することを確認**

```powershell
py -m pytest tests/unit/test_regime_leverage_service.py::TestDecideDailyCheck -v
```

Expected: `ImportError: cannot import name 'decide_daily_check'`

- [ ] **Step 3: `service.py` に日次ロジックを追記**

ファイル末尾に追加:

```python
def decide_daily_check(
    snapshot: RegimeLeverageSnapshot,
    day_low_usd: float,
    usdjpy_rate: float,
    now: datetime,
) -> RegimeLeverageDecision:
    """保有中の日次判定: マージンコール→初期損切りの優先順位で当日安値ベースに判定する
    (バックテストのrun_levered_regimeと同じ優先順位)。"""
    day_low_jpy = day_low_usd * usdjpy_rate
    equity_at_low = compute_equity_now(snapshot, day_low_jpy, now)
    value_at_low = day_low_jpy * snapshot.shares
    maintenance_ratio = equity_at_low / value_at_low if value_at_low > 0 else 0.0

    if value_at_low > 0 and maintenance_ratio < REGIME_LEVERAGE_MARGIN_MAINTENANCE:
        exit_price_jpy = day_low_jpy * (1 - REGIME_LEVERAGE_SLIPPAGE_PCT)
        exit_equity = compute_equity_now(snapshot, exit_price_jpy, now)
        return RegimeLeverageDecision(
            action="exit", reason="margin_call",
            spy_price_usd=day_low_usd, usdjpy_rate=usdjpy_rate, shares=0.0,
            entry_date=None, entry_price_jpy=None, entry_commission_jpy=None,
            equity_at_entry_jpy=None, stop_price_jpy=None,
            equity_now_jpy=exit_equity, maintenance_ratio=maintenance_ratio,
        )

    if snapshot.stop_price_jpy is not None and day_low_jpy <= snapshot.stop_price_jpy:
        exit_price_jpy = snapshot.stop_price_jpy * (1 - REGIME_LEVERAGE_SLIPPAGE_PCT)
        exit_equity = compute_equity_now(snapshot, exit_price_jpy, now)
        return RegimeLeverageDecision(
            action="exit", reason="initial_stop",
            spy_price_usd=day_low_usd, usdjpy_rate=usdjpy_rate, shares=0.0,
            entry_date=None, entry_price_jpy=None, entry_commission_jpy=None,
            equity_at_entry_jpy=None, stop_price_jpy=None,
            equity_now_jpy=exit_equity, maintenance_ratio=maintenance_ratio,
        )

    return RegimeLeverageDecision(
        action="noop", reason="daily_noop",
        spy_price_usd=day_low_usd, usdjpy_rate=usdjpy_rate, shares=snapshot.shares,
        entry_date=snapshot.entry_date, entry_price_jpy=snapshot.entry_price_jpy,
        entry_commission_jpy=snapshot.entry_commission_jpy,
        equity_at_entry_jpy=snapshot.equity_at_entry_jpy, stop_price_jpy=snapshot.stop_price_jpy,
        equity_now_jpy=equity_at_low, maintenance_ratio=maintenance_ratio,
    )
```

- [ ] **Step 4: テストが通ることを確認**

```powershell
py -m pytest tests/unit/test_regime_leverage_service.py -v
```

Expected: 9 passed

- [ ] **Step 5: lint確認とCommit**

```powershell
py -m black src/trading/regime_leverage_strategy/service.py tests/unit/test_regime_leverage_service.py
py -m isort src/trading/regime_leverage_strategy/service.py tests/unit/test_regime_leverage_service.py
py -m flake8 src/trading/regime_leverage_strategy/service.py tests/unit/test_regime_leverage_service.py
py -m mypy src/trading/regime_leverage_strategy/service.py --ignore-missing-imports --implicit-optional
```

```bash
git add python/src/trading/regime_leverage_strategy/service.py python/tests/unit/test_regime_leverage_service.py
git commit -m "feat: レジームレバレッジ戦略の日次判定ロジック(マージンコール・初期損切り)を追加"
```

---

### Task 6: service.py — 結合関数（MarketDataPort + repository）

**Files:**
- Modify: `python/src/trading/regime_leverage_strategy/service.py`（追記）
- Modify: `python/tests/unit/test_regime_leverage_service.py`（追記）

**Interfaces:**
- Consumes: `MarketDataPort`（`src/domain/ports.py`）、`get_latest_snapshot`/`insert_snapshot`（Task 2）、`build_weekly_frame`/`wilder_atr`（Task 3）、`decide_weekly_entry`/`decide_weekly_exit`/`decide_daily_check`（Task 4・5）
- Produces: `run_regime_leverage_weekly_check(market_data_port) -> RegimeLeverageDecision`、`run_regime_leverage_daily_margin_check(market_data_port) -> Optional[RegimeLeverageDecision]`（未保有なら None）。Task 7（orchestration層）がこれらを呼ぶ。

- [ ] **Step 1: 失敗するテストを追記**

`test_regime_leverage_service.py` に追記:

```python
class TestRunRegimeLeverageWeeklyCheck(unittest.TestCase):
    @patch("src.trading.regime_leverage_strategy.service.insert_snapshot")
    @patch("src.trading.regime_leverage_strategy.service.get_latest_snapshot")
    def test_first_run_uses_initial_capital_and_enters_on_uptrend(self, mock_latest, mock_insert):
        from src.trading.regime_leverage_strategy.service import run_regime_leverage_weekly_check

        mock_latest.return_value = None
        mock_port = MagicMock()
        idx = pd.bdate_range("2025-01-01", periods=260)
        prices = pd.Series(np.linspace(400.0, 500.0, 260), index=idx)
        df = pd.DataFrame(
            {"Open": prices, "High": prices + 2, "Low": prices - 2, "Close": prices, "Volume": 1_000_000},
            index=idx,
        )
        mock_port.get_stock_data.return_value = df
        fx_df = pd.DataFrame({"Close": [145.0]}, index=[idx[-1]])
        mock_port.get_forex_data.return_value = fx_df

        decision = run_regime_leverage_weekly_check(mock_port)

        self.assertEqual(decision.action, "entry")
        mock_insert.assert_called_once()


class TestRunRegimeLeverageDailyMarginCheck(unittest.TestCase):
    @patch("src.trading.regime_leverage_strategy.service.get_latest_snapshot")
    def test_returns_none_when_not_holding(self, mock_latest):
        from src.trading.regime_leverage_strategy.service import run_regime_leverage_daily_margin_check

        mock_latest.return_value = None
        mock_port = MagicMock()
        result = run_regime_leverage_daily_margin_check(mock_port)
        self.assertIsNone(result)
        mock_port.get_stock_data.assert_not_called()
```

このテストで使う `pd`・`np`・`MagicMock` は既にファイル冒頭でimport済み(Task 4のテストで追加済み)。もし未importなら `import numpy as np` と `import pandas as pd` をファイル冒頭に追加する。

- [ ] **Step 2: テストが失敗することを確認**

```powershell
py -m pytest tests/unit/test_regime_leverage_service.py::TestRunRegimeLeverageWeeklyCheck -v
```

Expected: `ImportError: cannot import name 'run_regime_leverage_weekly_check'`

- [ ] **Step 3: `service.py` に結合関数を追記**

ファイル末尾に追加（`from __future__ import annotations` の下、importブロックに以下を追加する必要がある: `from datetime import timedelta`、`import pandas as pd`、`from src.domain.ports import MarketDataPort`、`from src.trading.regime_leverage_strategy.indicators import build_weekly_frame`、`from src.trading.regime_leverage_strategy.repository import get_latest_snapshot, insert_snapshot`）:

```python
def _load_spy_daily(market_data_port: MarketDataPort, symbol: str) -> pd.DataFrame:
    end = datetime.now()
    start = end - timedelta(days=400)  # 200日線を計算するため十分な余裕を持って取得
    df = market_data_port.get_stock_data(symbol, "us", start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
    return build_weekly_frame(df)


def _load_latest_usdjpy(market_data_port: MarketDataPort) -> float:
    end = datetime.now()
    start = end - timedelta(days=10)
    fx_df = market_data_port.get_forex_data("JPY=X", start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
    return float(fx_df["Close"].iloc[-1])


def run_regime_leverage_weekly_check(market_data_port: MarketDataPort) -> RegimeLeverageDecision:
    """週次ジョブ本体: レジーム転換の判定、または未保有時の新規エントリー判定を行い、
    結果をDBに記録して返す。"""
    from config.settings import REGIME_LEVERAGE_SYMBOL

    df = _load_spy_daily(market_data_port, REGIME_LEVERAGE_SYMBOL)
    usdjpy_rate = _load_latest_usdjpy(market_data_port)
    latest_row = df.iloc[-1]
    week_close_usd = float(latest_row["Close"])
    ma200_usd = float(latest_row["MA200"])
    atr14_usd = float(latest_row["ATR14"])
    now = datetime.now()

    snapshot = get_latest_snapshot()
    holding = snapshot is not None and snapshot.shares > 0

    if not holding:
        from config.settings import REGIME_LEVERAGE_INITIAL_CAPITAL_JPY

        cash_jpy = snapshot.equity_now_jpy if snapshot is not None else REGIME_LEVERAGE_INITIAL_CAPITAL_JPY
        decision = decide_weekly_entry(cash_jpy, week_close_usd, ma200_usd, atr14_usd, usdjpy_rate, now)
    else:
        decision = decide_weekly_exit(snapshot, week_close_usd, ma200_usd, usdjpy_rate, now)

    insert_snapshot(decision)
    return decision


def run_regime_leverage_daily_margin_check(market_data_port: MarketDataPort) -> Optional[RegimeLeverageDecision]:
    """日次ジョブ本体: 保有中の場合のみ、初期損切り・マージンコールを判定する。"""
    from config.settings import REGIME_LEVERAGE_SYMBOL

    snapshot = get_latest_snapshot()
    if snapshot is None or snapshot.shares <= 0:
        return None

    df = _load_spy_daily(market_data_port, REGIME_LEVERAGE_SYMBOL)
    usdjpy_rate = _load_latest_usdjpy(market_data_port)
    day_low_usd = float(df.iloc[-1]["Low"])
    now = datetime.now()

    decision = decide_daily_check(snapshot, day_low_usd, usdjpy_rate, now)
    insert_snapshot(decision)
    return decision
```

ファイル冒頭のimportブロックを以下のように更新する（Task 4で書いたimportに追記):

```python
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from config.settings import (
    REGIME_LEVERAGE_INITIAL_STOP_ATR_MULT,
    REGIME_LEVERAGE_INTEREST_ANNUAL,
    REGIME_LEVERAGE_MARGIN_MAINTENANCE,
    REGIME_LEVERAGE_RATIO,
    REGIME_LEVERAGE_SLIPPAGE_PCT,
)
from src.domain.ports import MarketDataPort
from src.trading.regime_leverage_strategy.indicators import build_weekly_frame
from src.trading.regime_leverage_strategy.repository import get_latest_snapshot, insert_snapshot
from src.trading.regime_leverage_strategy.types import RegimeLeverageDecision, RegimeLeverageSnapshot
```

- [ ] **Step 4: テストが通ることを確認**

```powershell
py -m pytest tests/unit/test_regime_leverage_service.py -v
```

Expected: 11 passed

- [ ] **Step 5: lint確認とCommit**

```powershell
py -m black src/trading/regime_leverage_strategy/service.py tests/unit/test_regime_leverage_service.py
py -m isort src/trading/regime_leverage_strategy/service.py tests/unit/test_regime_leverage_service.py
py -m flake8 src/trading/regime_leverage_strategy/service.py tests/unit/test_regime_leverage_service.py
py -m mypy src/trading/regime_leverage_strategy/service.py --ignore-missing-imports --implicit-optional
$env:PYTHONUTF8=1; lint-imports
```

Expected: import-linterも含めすべて成功（`src.trading` から `src.domain` への依存はshared kernelとして許可されている）

```bash
git add python/src/trading/regime_leverage_strategy/service.py python/tests/unit/test_regime_leverage_service.py
git commit -m "feat: レジームレバレッジ戦略の週次/日次ジョブ結合関数を追加"
```

---

### Task 7: orchestration層のジョブ登録とスケジューラ統合

**Files:**
- Modify: `python/src/orchestration/jobs/periodic.py`
- Modify: `python/src/orchestration/scheduler.py`
- Modify: `python/run_scheduler.py`
- Test: `python/tests/unit/test_periodic_jobs.py`（追記）

**Interfaces:**
- Consumes: `run_regime_leverage_weekly_check`/`run_regime_leverage_daily_margin_check`（Task 6）
- Produces: `run_regime_leverage_weekly_job() -> None`、`run_regime_leverage_daily_margin_job() -> None`（`periodic.py`）。`run_scheduler.py`のSCHEDULE_CONFIGから呼ばれる。

- [ ] **Step 1: 失敗するテストを書く**

`test_periodic_jobs.py` に追記（ファイル冒頭のimportに `from unittest.mock import MagicMock, patch` が無ければ追加）:

```python
class TestRunRegimeLeverageWeeklyJob(unittest.TestCase):
    @patch("src.trading.regime_leverage_strategy.service.run_regime_leverage_weekly_check")
    @patch("src.infrastructure.yfinance_market_data_adapter.YFinanceMarketDataAdapter")
    def test_calls_service_with_adapter(self, mock_adapter_cls, mock_run):
        from src.orchestration.jobs.periodic import run_regime_leverage_weekly_job

        mock_run.return_value = MagicMock(action="noop")
        run_regime_leverage_weekly_job()
        mock_run.assert_called_once()

    @patch("src.trading.regime_leverage_strategy.service.run_regime_leverage_weekly_check")
    def test_does_not_raise_on_failure(self, mock_run):
        from src.orchestration.jobs.periodic import run_regime_leverage_weekly_job

        mock_run.side_effect = Exception("boom")
        run_regime_leverage_weekly_job()  # 例外を吸収してログのみ出すこと


class TestRunRegimeLeverageDailyMarginJob(unittest.TestCase):
    @patch("src.trading.regime_leverage_strategy.service.run_regime_leverage_daily_margin_check")
    def test_does_not_raise_on_failure(self, mock_run):
        from src.orchestration.jobs.periodic import run_regime_leverage_daily_margin_job

        mock_run.side_effect = Exception("boom")
        run_regime_leverage_daily_margin_job()
```

- [ ] **Step 2: テストが失敗することを確認**

```powershell
py -m pytest tests/unit/test_periodic_jobs.py::TestRunRegimeLeverageWeeklyJob -v
```

Expected: `ImportError: cannot import name 'run_regime_leverage_weekly_job'`

- [ ] **Step 3: `periodic.py` にジョブ関数を追加**

`periodic.py` の `run_allocation_rebalance_job` 関数の直後に追加:

```python
def run_regime_leverage_weekly_job() -> None:
    """
    レジームレバレッジ戦略(STRATEGY.md 7章、SPY・レバレッジ2.0倍・円建て)の
    週次判定(レジーム転換・新規エントリー)を実行する。
    """
    logger.info("=== レジームレバレッジ戦略(週次) 実行開始 ===")
    try:
        from src.infrastructure.yfinance_market_data_adapter import YFinanceMarketDataAdapter
        from src.trading.regime_leverage_strategy.service import run_regime_leverage_weekly_check

        decision = run_regime_leverage_weekly_check(YFinanceMarketDataAdapter())
        logger.info(
            "レジームレバレッジ戦略(週次): action=%s reason=%s", decision.action, decision.reason
        )
    except Exception as e:
        logger.error("レジームレバレッジ戦略(週次)の実行に失敗しました: %s", e, exc_info=True)
        return
    logger.info("=== レジームレバレッジ戦略(週次) 実行完了 ===")


def run_regime_leverage_daily_margin_job() -> None:
    """
    レジームレバレッジ戦略の日次判定(初期損切り・マージンコール)を実行する。
    未保有の場合は何もしない。
    """
    logger.info("=== レジームレバレッジ戦略(日次) 実行開始 ===")
    try:
        from src.infrastructure.yfinance_market_data_adapter import YFinanceMarketDataAdapter
        from src.trading.regime_leverage_strategy.service import run_regime_leverage_daily_margin_check

        decision = run_regime_leverage_daily_margin_check(YFinanceMarketDataAdapter())
        if decision is not None:
            logger.info(
                "レジームレバレッジ戦略(日次): action=%s reason=%s maintenance_ratio=%s",
                decision.action,
                decision.reason,
                decision.maintenance_ratio,
            )
    except Exception as e:
        logger.error("レジームレバレッジ戦略(日次)の実行に失敗しました: %s", e, exc_info=True)
        return
    logger.info("=== レジームレバレッジ戦略(日次) 実行完了 ===")
```

- [ ] **Step 4: `scheduler.py` にre-exportを追加**

`src/orchestration/scheduler.py:26`付近の import ブロック（`run_allocation_rebalance_job,`の行）に2行追加:

```python
    run_allocation_rebalance_job,
    run_regime_leverage_daily_margin_job,
    run_regime_leverage_weekly_job,
```

`__all__` リスト（`"run_allocation_rebalance_job",`がある行、`scheduler.py:82`付近）にも追加:

```python
    "run_allocation_rebalance_job",
    "run_regime_leverage_daily_margin_job",
    "run_regime_leverage_weekly_job",
```

- [ ] **Step 5: テストが通ることを確認**

```powershell
py -m pytest tests/unit/test_periodic_jobs.py -v
```

Expected: 全件 passed（既存テスト含む）

- [ ] **Step 6: `run_scheduler.py` にジョブ登録**

`run_scheduler.py:170-175`（`job_allocation_rebalance`関数）の直後に追加:

```python
def job_regime_leverage_weekly():
    """レジームレバレッジ戦略(SPY・レバレッジ2.0倍)の週次判定（毎週金曜06:30）"""
    from src.orchestration.scheduler import run_regime_leverage_weekly_job

    run_regime_leverage_weekly_job()


def job_regime_leverage_daily_margin():
    """レジームレバレッジ戦略の日次マージンコールチェック（毎日06:15）"""
    from src.orchestration.scheduler import run_regime_leverage_daily_margin_job

    run_regime_leverage_daily_margin_job()
```

`SCHEDULE_CONFIG` 辞書の `"allocation_rebalance": {...}` エントリの直後（`run_scheduler.py:392`付近、次のキーの直前）に追加:

```python
    "regime_leverage_daily_margin": {
        "func": job_regime_leverage_daily_margin,
        "trigger": "cron",
        "period": "daily",
        "day_of_week": "mon-fri",
        "hour": 6,
        "minute": 15,
        "recovery_delay_minutes": 30,
        "description": "毎日06:15 - レジームレバレッジ戦略(SPY)の日次マージンコールチェック",
    },
    "regime_leverage_weekly": {
        "func": job_regime_leverage_weekly,
        "trigger": "cron",
        "period": "weekly",
        "day_of_week": "fri",
        "hour": 6,
        "minute": 30,
        "recovery_delay_minutes": 30,
        "description": "毎週金曜06:30 - レジームレバレッジ戦略(SPY)のレジーム転換・新規エントリー判定",
    },
```

日次(06:15)を週次(06:30)より前に置くことで、Global Constraintsに記載した
「金曜は日次→週次の順」を実行時刻の順序として保証する。

- [ ] **Step 7: `--run-now` の手動実行パスを確認・追加**

`run_scheduler.py:581`付近の `elif pipeline == "allocation_rebalance":` の直後に追加:

```python
    elif pipeline == "regime_leverage_daily_margin":
        queue_manager.run_job("regime_leverage_daily_margin", reason="manual", force=True)
    elif pipeline == "regime_leverage_weekly":
        queue_manager.run_job("regime_leverage_weekly", reason="manual", force=True)
```

- [ ] **Step 8: lint確認とCommit**

```powershell
py -m black src/orchestration/jobs/periodic.py src/orchestration/scheduler.py run_scheduler.py tests/unit/test_periodic_jobs.py
py -m isort src/orchestration/jobs/periodic.py src/orchestration/scheduler.py run_scheduler.py tests/unit/test_periodic_jobs.py
py -m flake8 src/orchestration/jobs/periodic.py src/orchestration/scheduler.py run_scheduler.py tests/unit/test_periodic_jobs.py
py -m mypy src/orchestration/jobs/periodic.py src/orchestration/scheduler.py run_scheduler.py --ignore-missing-imports --implicit-optional
py -m pytest tests/unit/test_periodic_jobs.py -v
```

```bash
git add python/src/orchestration/jobs/periodic.py python/src/orchestration/scheduler.py python/run_scheduler.py python/tests/unit/test_periodic_jobs.py
git commit -m "feat: レジームレバレッジ戦略のジョブをスケジューラに登録"
```

---

### Task 8: 実データでのエンドツーエンド検証

**Files:**
- なし（scratchpadでの検証のみ、本番コードの変更なし）

**Interfaces:**
- Consumes: Task 6・7で実装した全関数

このタスクはコード変更を行わない。advisorから過去に指摘された教訓（「モックだけで満足せず実データで一度動かす」）を踏まえ、実際にAPIを叩いて動作を確認する。

- [ ] **Step 1: scratchpadに検証スクリプトを書く**

（スクラッチパッドのパスは実行時のセッションディレクトリを使う。以下は内容の例）

```python
import sys
sys.path.insert(0, r"C:\src\StockFixer\python")

from src.infrastructure.yfinance_market_data_adapter import YFinanceMarketDataAdapter
from src.trading.regime_leverage_strategy.indicators import build_weekly_frame
from src.trading.regime_leverage_strategy.service import _load_spy_daily, _load_latest_usdjpy

port = YFinanceMarketDataAdapter()
df = _load_spy_daily(port, "SPY")
print("rows:", len(df), "latest date:", df.index[-1])
print(df[["Close", "MA200", "ATR14"]].tail(3))

rate = _load_latest_usdjpy(port)
print("USDJPY:", rate)
```

- [ ] **Step 2: 実行して確認する**

```powershell
cd C:\src\StockFixer\python
py <scratchpadのパス>
```

確認項目:
- `MA200`・`ATR14` が最終行でNaNでないこと（400日分取得しているので200日を超えているはず）
- `USDJPY` が140〜160程度の妥当な範囲であること
- 例外が発生しないこと

- [ ] **Step 3: `get_latest_snapshot`/`insert_snapshot` を実DBに対して1回動かす**

（重要: これは実際にDBへ行を書き込む。テスト用DBではなく本番DBに接続する設定になっていないか、`DATABASE_URL`環境変数を確認してから実行すること）

```powershell
$env:DATABASE_URL = "<開発用DBのURL、本番を指していないことを確認>"
py -c "
from src.trading.regime_leverage_strategy.service import run_regime_leverage_weekly_check
from src.infrastructure.yfinance_market_data_adapter import YFinanceMarketDataAdapter
decision = run_regime_leverage_weekly_check(YFinanceMarketDataAdapter())
print(decision)
"
```

このステップは実データへの書き込みを伴うため、実行前に必ずユーザーに確認を取ること。

- [ ] **Step 4: 発見した問題があればここで修正し、該当タスクに戻って再テストする**

（問題がなければこのステップは完了とマークするだけでよい）

---

### Task 9: 全体確認・最終コミット

**Files:**
- なし（確認のみ）

- [ ] **Step 1: 全体のunit testとカバレッジを確認**

```powershell
cd C:\src\StockFixer\python
py -m pytest tests/unit/ -n 2 -v `
  --ignore=tests/unit/test_predict_unified_lightgbm_alignment.py `
  --cov=src --cov-branch --cov-report= --cov-fail-under=0
py -m pytest tests/unit/test_predict_unified_lightgbm_alignment.py -v `
  --cov=src --cov-branch --cov-append --cov-report=term-missing --cov-fail-under=80
```

Expected: 全件 passed、カバレッジ80%以上

- [ ] **Step 2: import-linterを確認**

```powershell
$env:PYTHONUTF8=1
lint-imports
```

Expected: `Contracts: 2 kept, 0 broken.`

- [ ] **Step 3: pip-auditを確認（新規依存を追加していないため、通常は変更なしのはず）**

```powershell
$env:PYTHONUTF8=1
pip-audit -r requirements.txt
pip-audit -r requirements-dev.txt
```

- [ ] **Step 4: PRを作成する**

CLAUDE.mdの規約に従い、`version_impact: minor`（新機能追加）でPR本文を作成する。
バージョンは develop の最新から+1する。

---

## Self-Review チェック結果

- **Spec coverage**: spec docの全セクション（テーブルスキーマ・週次/日次ジョブ・モジュール構成・スケジューラ登録・テスト方針）に対応するタスクが存在する
- **Placeholder scan**: 各タスクに具体的なコードを記載済み。「TBD」等は無い
- **Type consistency**: `RegimeLeverageSnapshot`・`RegimeLeverageDecision` のフィールド名はTask 1で定義したものをTask 2〜7まで一貫して使用している
