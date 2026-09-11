# TQQQ/短期債 配分戦略ペーパートレードボット Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** TQQQ 80% / 短期債(SHY) 20%を目安に約2年ごとにリバランスする配分戦略を、StockFixerの既存アーキテクチャ（DB・Discord通知・APScheduler）に統合したペーパートレードボットとして実装する。

**Architecture:** 既存の `PaperBroker` は `.T` サフィックス・`market='jp'`・円建てがハードコードされており米国ティッカーに使えないため、`src/trading/allocation_strategy/` に新規の自己完結モジュールを作る。状態は「追記専用ログの最新1行が現在の状態を表す」という単一テーブル設計にし、ポジションテーブルとログテーブルを分けない。これにより「途中でクラッシュしたら半端な状態が残る」というリスクを構造的に排除する（状態遷移は常に1回のINSERTで完結する）。既存の `run_strategy_promotion_check` と同じ「try/except + 個別のDiscord通知try/except」パターンに揃えるが、ON/OFF機能フラグは設けない（ユーザー要望）。代わりに `SCHEDULE_CONFIG` へ `auto_schedule: False` で登録するため、このPRをマージしてもスケジューラは新ジョブを一切自動実行せず、`--run-now allocation_rebalance` による人間の明示的な実行のみが唯一のトリガーとなる。

**Tech Stack:** Python, PostgreSQL(psycopg3), pydantic-settings, APScheduler, pytest/unittest

**Spec:** なし（本チャットでのユーザー要望「TQQQ/短期債配分戦略のペーパートレードbotをStockFixerに統合」を、既存アーキテクチャ調査に基づき本プランに直接落とし込んでいる）

## Global Constraints

- この戦略には既存の類似ジョブ（`STRATEGY_PROMOTION_CHECK_ENABLED` 等）のようなON/OFF機能フラグを設けない。安全装置は `SCHEDULE_CONFIG` の `auto_schedule: False`（自動起動なし）のみとし、実行するかどうかは人間が `--run-now allocation_rebalance` を叩くことで明示的に判断する。
- `SCHEDULE_CONFIG` への新規エントリは `auto_schedule: False` で登録する（このPRでは自動実行を一切有効化しない。有効化はユーザー自身の別判断）。
- フラグが無いため `--run-now allocation_rebalance` は実行するたびに実際に本番DBへ状態を書き込み、実際にDiscordへ通知する。したがって本プランの実装中（ユニットテスト以外の場面）でこのコマンドを実際に実行してはならない。実際に動かすタイミングはユーザー自身が決める。
- 通貨は米ドル（USD）建て。既存の円建て関数（`send_paper_trade_position_report` 等）は一切再利用・変更しない。
- `PaperBroker` / `src/trading/brokers/` 配下のファイルは一切変更しない（JP本番ペーパートレードが依存する共有コードのため）。
- TQQQ比率・初期資金・リバランス間隔は `config/settings.py` の設定値として持ち、コード中にハードコードしない。
- 状態の永続化は新規テーブル `allocation_rebalance_log`（追記専用ログ）のみを使う。`system_config` テーブルは使わない（既にR-410ドリフト閾値用途で使われている汎用KVストアであり、ポジション・現金という金額を持つ状態を汎用KVに突っ込むと監査履歴が失われるため）。
- DB接続は `src.utils.db._connection._db_connection()` を使う。本番接続は `autocommit=True` のプール接続のため、複数文にまたがるトランザクション制御は不要（本プランは全ての状態遷移を単一INSERT文に設計しているため、そもそも複数文にまたがるトランザクションは発生しない）。
- 価格取得は `src.infrastructure.yfinance_market_data_adapter.YFinanceMarketDataAdapter.get_latest_price(symbol)` を使う。この関数は失敗時に `0.0` を返す契約なので、価格が `0.0` 以下の場合は状態を書き込まずに中断する。

---

### Task 1: DBマイグレーション — allocation_rebalance_log テーブル追加

**Files:**
- Create: `python/src/utils/db/migrations/0005_add_allocation_rebalance_log_postgres.sql`

**Interfaces:**
- Consumes: なし
- Produces: テーブル `allocation_rebalance_log(id, executed_at, action, tqqq_price, shy_price, tqqq_qty_before, shy_qty_before, cash_before, tqqq_qty_after, shy_qty_after, cash_after)`。Task 3 の `repository.py` がこのテーブルを読み書きする。

- [ ] **Step 1: マイグレーションファイルを作成する**

`python/src/utils/db/migrations/0005_add_allocation_rebalance_log_postgres.sql`:

```sql
-- 0005_add_allocation_rebalance_log_postgres: 配分戦略(TQQQ/短期債)ペーパートレードの
-- 状態を追記専用ログとして記録するテーブル。id最大の行が現在の建玉・現金の状態を表す。
CREATE TABLE IF NOT EXISTS allocation_rebalance_log (
    id              BIGSERIAL PRIMARY KEY,
    executed_at     TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    action          VARCHAR NOT NULL,
    tqqq_price      DOUBLE PRECISION NOT NULL,
    shy_price       DOUBLE PRECISION NOT NULL,
    tqqq_qty_before DOUBLE PRECISION NOT NULL,
    shy_qty_before  DOUBLE PRECISION NOT NULL,
    cash_before     DOUBLE PRECISION NOT NULL,
    tqqq_qty_after  DOUBLE PRECISION NOT NULL,
    shy_qty_after   DOUBLE PRECISION NOT NULL,
    cash_after      DOUBLE PRECISION NOT NULL
);
```

- [ ] **Step 2: マイグレーションランナーのユニットテストが壊れていないことを確認する**

`src/utils/db/migration_runner.py` のディスカバリは `tests/unit/utils/db/test_migration_runner.py` 内の一時ディレクトリのみを見るため、実ディレクトリにファイルを追加してもそのテストは影響を受けない。念のため実行して確認する。

Run: `python -m pytest tests/unit/utils/db/test_migration_runner.py -v`
Expected: PASS（既存の全テストが green のまま）

- [ ] **Step 3: Commit**

```bash
git add src/utils/db/migrations/0005_add_allocation_rebalance_log_postgres.sql
git commit -m "feat: 配分戦略ペーパートレード用のallocation_rebalance_logテーブルを追加"
```

---

### Task 2: 設定値追加（settings.py）

**Files:**
- Modify: `python/config/settings.py`

**Interfaces:**
- Consumes: なし
- Produces: `config.settings.ALLOCATION_STRATEGY_TQQQ_SYMBOL: str`、`config.settings.ALLOCATION_STRATEGY_BOND_SYMBOL: str`、`config.settings.ALLOCATION_STRATEGY_TQQQ_RATIO: float`、`config.settings.ALLOCATION_STRATEGY_INITIAL_CAPITAL: float`、`config.settings.ALLOCATION_STRATEGY_REBALANCE_YEARS: int`。Task 4（service.py）がこれらをimportする。ON/OFF機能フラグは設けない（Global Constraints参照）。

- [ ] **Step 1: `config/settings.py` の `Settings` クラス内、`PAPER_INITIAL_BALANCE` の直後に追記する**

`PAPER_INITIAL_BALANCE: float = Field(default=1_000_000.0)` の次の行から:

```python

    # ---------- 配分戦略ペーパートレード（trading/allocation_strategy） ----------
    ALLOCATION_STRATEGY_TQQQ_SYMBOL: str = Field(default="TQQQ")
    ALLOCATION_STRATEGY_BOND_SYMBOL: str = Field(default="SHY")
    ALLOCATION_STRATEGY_TQQQ_RATIO: float = Field(default=0.8)
    ALLOCATION_STRATEGY_INITIAL_CAPITAL: float = Field(default=100_000.0)
    ALLOCATION_STRATEGY_REBALANCE_YEARS: int = Field(default=2)
```

- [ ] **Step 2: `config/settings.py` のモジュールレベル re-export に追記する**

`PAPER_INITIAL_BALANCE: float = settings.PAPER_INITIAL_BALANCE` の直後に:

```python

ALLOCATION_STRATEGY_TQQQ_SYMBOL: str = settings.ALLOCATION_STRATEGY_TQQQ_SYMBOL
ALLOCATION_STRATEGY_BOND_SYMBOL: str = settings.ALLOCATION_STRATEGY_BOND_SYMBOL
ALLOCATION_STRATEGY_TQQQ_RATIO: float = settings.ALLOCATION_STRATEGY_TQQQ_RATIO
ALLOCATION_STRATEGY_INITIAL_CAPITAL: float = settings.ALLOCATION_STRATEGY_INITIAL_CAPITAL
ALLOCATION_STRATEGY_REBALANCE_YEARS: int = settings.ALLOCATION_STRATEGY_REBALANCE_YEARS
```

- [ ] **Step 3: importできることを確認する**

Run: `python -c "from config.settings import ALLOCATION_STRATEGY_TQQQ_RATIO, ALLOCATION_STRATEGY_INITIAL_CAPITAL; print(ALLOCATION_STRATEGY_TQQQ_RATIO, ALLOCATION_STRATEGY_INITIAL_CAPITAL)"`
Expected: `0.8 100000.0`

- [ ] **Step 4: Commit**

```bash
git add config/settings.py
git commit -m "feat: 配分戦略ペーパートレード用の設定値を追加"
```

---

### Task 3: 型定義 + リポジトリ層

**Files:**
- Create: `python/src/trading/allocation_strategy/__init__.py`
- Create: `python/src/trading/allocation_strategy/types.py`
- Create: `python/src/trading/allocation_strategy/repository.py`
- Test: `python/tests/unit/test_allocation_repository.py`

**Interfaces:**
- Consumes: `src.utils.db._connection._db_connection`
- Produces: `AllocationSnapshot`（dataclass）, `RebalanceOutcome`（dataclass）, `get_latest_snapshot() -> Optional[AllocationSnapshot]`, `insert_snapshot(action: str, tqqq_price: float, shy_price: float, tqqq_qty_before: float, shy_qty_before: float, cash_before: float, tqqq_qty_after: float, shy_qty_after: float, cash_after: float) -> None`。Task 4（service.py）がこれらを使う。

- [ ] **Step 1: パッケージ初期化ファイルを作成する**

`python/src/trading/allocation_strategy/__init__.py`: 空ファイル（既存の他BCサブパッケージと同様、内容なし）。

- [ ] **Step 2: 型定義を書く**

`python/src/trading/allocation_strategy/types.py`:

```python
"""配分戦略(TQQQ/短期債)ペーパートレードの型定義。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class AllocationSnapshot:
    """ある時点での配分戦略の建玉・現金状態(allocation_rebalance_logの1行)。"""

    id: int
    executed_at: datetime
    action: str
    tqqq_price: float
    shy_price: float
    tqqq_qty_before: float
    shy_qty_before: float
    cash_before: float
    tqqq_qty_after: float
    shy_qty_after: float
    cash_after: float


@dataclass(frozen=True)
class RebalanceOutcome:
    """run_allocation_rebalance() が実行した結果(Discord通知に使う)。"""

    action: str
    tqqq_price: float
    shy_price: float
    tqqq_qty_before: float
    shy_qty_before: float
    cash_before: float
    tqqq_qty_after: float
    shy_qty_after: float
    cash_after: float
```

- [ ] **Step 3: 失敗するテストを書く**

`python/tests/unit/test_allocation_repository.py`:

```python
"""ユニットテスト: src.trading.allocation_strategy.repository"""

import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch


class TestGetLatestSnapshot(unittest.TestCase):
    def _mock_db(self, row):
        mock_con = MagicMock()
        mock_con.execute.return_value.fetchone.return_value = row
        return mock_con

    def test_returns_none_when_no_rows(self):
        from src.trading.allocation_strategy.repository import get_latest_snapshot

        mock_con = self._mock_db(None)
        with patch("src.trading.allocation_strategy.repository._db_connection") as mock_ctx:
            mock_ctx.return_value.__enter__ = MagicMock(return_value=mock_con)
            mock_ctx.return_value.__exit__ = MagicMock(return_value=False)
            result = get_latest_snapshot()
        self.assertIsNone(result)

    def test_parses_row_into_snapshot(self):
        from src.trading.allocation_strategy.repository import get_latest_snapshot

        row = (
            7,
            datetime(2026, 1, 1),
            "rebalance",
            80.0,
            85.0,
            100.0,
            50.0,
            10.0,
            120.0,
            30.0,
            5.0,
        )
        mock_con = self._mock_db(row)
        with patch("src.trading.allocation_strategy.repository._db_connection") as mock_ctx:
            mock_ctx.return_value.__enter__ = MagicMock(return_value=mock_con)
            mock_ctx.return_value.__exit__ = MagicMock(return_value=False)
            result = get_latest_snapshot()

        self.assertEqual(result.id, 7)
        self.assertEqual(result.executed_at, datetime(2026, 1, 1))
        self.assertEqual(result.action, "rebalance")
        self.assertEqual(result.tqqq_price, 80.0)
        self.assertEqual(result.shy_price, 85.0)
        self.assertEqual(result.tqqq_qty_before, 100.0)
        self.assertEqual(result.shy_qty_before, 50.0)
        self.assertEqual(result.cash_before, 10.0)
        self.assertEqual(result.tqqq_qty_after, 120.0)
        self.assertEqual(result.shy_qty_after, 30.0)
        self.assertEqual(result.cash_after, 5.0)


class TestInsertSnapshot(unittest.TestCase):
    def test_executes_insert_with_expected_params(self):
        from src.trading.allocation_strategy.repository import insert_snapshot

        mock_con = MagicMock()
        with patch("src.trading.allocation_strategy.repository._db_connection") as mock_ctx:
            mock_ctx.return_value.__enter__ = MagicMock(return_value=mock_con)
            mock_ctx.return_value.__exit__ = MagicMock(return_value=False)
            insert_snapshot(
                action="initial",
                tqqq_price=80.0,
                shy_price=85.0,
                tqqq_qty_before=0.0,
                shy_qty_before=0.0,
                cash_before=100_000.0,
                tqqq_qty_after=1000.0,
                shy_qty_after=235.29,
                cash_after=0.0,
            )

        mock_con.execute.assert_called_once()
        sql_call = mock_con.execute.call_args
        self.assertIn("INSERT INTO allocation_rebalance_log", sql_call[0][0])
        self.assertEqual(
            sql_call[0][1],
            ["initial", 80.0, 85.0, 0.0, 0.0, 100_000.0, 1000.0, 235.29, 0.0],
        )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 4: テストが失敗することを確認する（モジュール未実装のため）**

Run: `python -m pytest tests/unit/test_allocation_repository.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'src.trading.allocation_strategy.repository'`）

- [ ] **Step 5: リポジトリを実装する**

`python/src/trading/allocation_strategy/repository.py`:

```python
"""allocation_rebalance_log テーブルの読み書き。"""

from typing import Optional

from src.trading.allocation_strategy.types import AllocationSnapshot
from src.utils.db._connection import _db_connection
from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_latest_snapshot() -> Optional[AllocationSnapshot]:
    """最新の状態行を返す。まだ1行も無ければ None。"""
    with _db_connection() as con:
        row = con.execute(
            """
            SELECT id, executed_at, action, tqqq_price, shy_price,
                   tqqq_qty_before, shy_qty_before, cash_before,
                   tqqq_qty_after, shy_qty_after, cash_after
            FROM allocation_rebalance_log
            ORDER BY id DESC
            LIMIT 1
            """
        ).fetchone()
    if row is None:
        return None
    return AllocationSnapshot(
        id=row[0],
        executed_at=row[1],
        action=row[2],
        tqqq_price=row[3],
        shy_price=row[4],
        tqqq_qty_before=row[5],
        shy_qty_before=row[6],
        cash_before=row[7],
        tqqq_qty_after=row[8],
        shy_qty_after=row[9],
        cash_after=row[10],
    )


def insert_snapshot(
    action: str,
    tqqq_price: float,
    shy_price: float,
    tqqq_qty_before: float,
    shy_qty_before: float,
    cash_before: float,
    tqqq_qty_after: float,
    shy_qty_after: float,
    cash_after: float,
) -> None:
    """新しい状態行を追記する。"""
    with _db_connection() as con:
        con.execute(
            """
            INSERT INTO allocation_rebalance_log (
                action, tqqq_price, shy_price,
                tqqq_qty_before, shy_qty_before, cash_before,
                tqqq_qty_after, shy_qty_after, cash_after
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            [
                action,
                tqqq_price,
                shy_price,
                tqqq_qty_before,
                shy_qty_before,
                cash_before,
                tqqq_qty_after,
                shy_qty_after,
                cash_after,
            ],
        )
    logger.info(
        "allocation_rebalance_log 追記: action=%s tqqq_qty=%.4f shy_qty=%.4f cash=%.2f",
        action,
        tqqq_qty_after,
        shy_qty_after,
        cash_after,
    )
```

- [ ] **Step 6: テストが通ることを確認する**

Run: `python -m pytest tests/unit/test_allocation_repository.py -v`
Expected: PASS（3テスト）

- [ ] **Step 7: Commit**

```bash
git add src/trading/allocation_strategy/__init__.py src/trading/allocation_strategy/types.py src/trading/allocation_strategy/repository.py tests/unit/test_allocation_repository.py
git commit -m "feat: 配分戦略ペーパートレードの型定義とリポジトリ層を追加"
```

---

### Task 4: サービス層（リバランス判定 + 発注計算ロジック）

**Files:**
- Create: `python/src/trading/allocation_strategy/service.py`
- Test: `python/tests/unit/test_allocation_service.py`

**Interfaces:**
- Consumes: `src.trading.allocation_strategy.repository.get_latest_snapshot`, `src.trading.allocation_strategy.repository.insert_snapshot`, `src.trading.allocation_strategy.types.RebalanceOutcome`, `src.trading.allocation_strategy.types.AllocationSnapshot`, `src.infrastructure.yfinance_market_data_adapter.YFinanceMarketDataAdapter`, `config.settings.ALLOCATION_STRATEGY_TQQQ_SYMBOL/BOND_SYMBOL/TQQQ_RATIO/INITIAL_CAPITAL/REBALANCE_YEARS`
- Produces: `run_allocation_rebalance() -> Optional[RebalanceOutcome]`。Task 6（periodic.py）がこれを呼ぶ。

- [ ] **Step 1: 失敗するテストを書く**

`python/tests/unit/test_allocation_service.py`:

```python
"""ユニットテスト: src.trading.allocation_strategy.service"""

import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch

from src.trading.allocation_strategy.types import AllocationSnapshot


class TestRunAllocationRebalance(unittest.TestCase):
    def _patch_settings(self):
        return (
            patch("config.settings.ALLOCATION_STRATEGY_TQQQ_SYMBOL", "TQQQ"),
            patch("config.settings.ALLOCATION_STRATEGY_BOND_SYMBOL", "SHY"),
            patch("config.settings.ALLOCATION_STRATEGY_TQQQ_RATIO", 0.8),
            patch("config.settings.ALLOCATION_STRATEGY_INITIAL_CAPITAL", 100_000.0),
            patch("config.settings.ALLOCATION_STRATEGY_REBALANCE_YEARS", 2),
        )

    def _start_settings_patchers(self):
        patchers = self._patch_settings()
        for p in patchers:
            p.start()
        self.addCleanup(lambda: [p.stop() for p in patchers])

    @patch("src.infrastructure.yfinance_market_data_adapter.YFinanceMarketDataAdapter")
    @patch("src.trading.allocation_strategy.service.insert_snapshot")
    @patch("src.trading.allocation_strategy.service.get_latest_snapshot")
    @patch("src.trading.allocation_strategy.service.datetime")
    def test_creates_initial_position_when_no_prior_state(
        self, mock_datetime, mock_get_latest, mock_insert, mock_adapter_cls
    ):
        mock_datetime.now.return_value = datetime(2026, 1, 1)
        mock_get_latest.return_value = None
        mock_adapter = MagicMock()
        mock_adapter.get_latest_price.side_effect = lambda symbol: {
            "TQQQ": 100.0,
            "SHY": 50.0,
        }[symbol]
        mock_adapter_cls.return_value = mock_adapter
        self._start_settings_patchers()

        from src.trading.allocation_strategy.service import run_allocation_rebalance

        outcome = run_allocation_rebalance()

        self.assertIsNotNone(outcome)
        self.assertEqual(outcome.action, "initial")
        self.assertAlmostEqual(outcome.tqqq_qty_after, 800.0)
        self.assertAlmostEqual(outcome.shy_qty_after, 400.0)
        self.assertAlmostEqual(outcome.cash_after, 0.0)
        mock_insert.assert_called_once()

    @patch("src.infrastructure.yfinance_market_data_adapter.YFinanceMarketDataAdapter")
    @patch("src.trading.allocation_strategy.service.insert_snapshot")
    @patch("src.trading.allocation_strategy.service.get_latest_snapshot")
    @patch("src.trading.allocation_strategy.service.datetime")
    def test_skips_when_already_run_today(
        self, mock_datetime, mock_get_latest, mock_insert, mock_adapter_cls
    ):
        mock_datetime.now.return_value = datetime(2026, 1, 1, 15, 0, 0)
        mock_get_latest.return_value = AllocationSnapshot(
            id=1,
            executed_at=datetime(2026, 1, 1, 10, 0, 0),
            action="initial",
            tqqq_price=100.0,
            shy_price=50.0,
            tqqq_qty_before=0.0,
            shy_qty_before=0.0,
            cash_before=100_000.0,
            tqqq_qty_after=800.0,
            shy_qty_after=400.0,
            cash_after=0.0,
        )

        from src.trading.allocation_strategy.service import run_allocation_rebalance

        outcome = run_allocation_rebalance()

        self.assertIsNone(outcome)
        mock_insert.assert_not_called()
        mock_adapter_cls.assert_not_called()

    @patch("src.infrastructure.yfinance_market_data_adapter.YFinanceMarketDataAdapter")
    @patch("src.trading.allocation_strategy.service.insert_snapshot")
    @patch("src.trading.allocation_strategy.service.get_latest_snapshot")
    @patch("src.trading.allocation_strategy.service.datetime")
    def test_skips_when_rebalance_not_yet_due(
        self, mock_datetime, mock_get_latest, mock_insert, mock_adapter_cls
    ):
        mock_datetime.now.return_value = datetime(2026, 1, 1)
        mock_get_latest.return_value = AllocationSnapshot(
            id=1,
            executed_at=datetime(2025, 6, 1),
            action="initial",
            tqqq_price=100.0,
            shy_price=50.0,
            tqqq_qty_before=0.0,
            shy_qty_before=0.0,
            cash_before=100_000.0,
            tqqq_qty_after=800.0,
            shy_qty_after=400.0,
            cash_after=0.0,
        )
        self._start_settings_patchers()

        from src.trading.allocation_strategy.service import run_allocation_rebalance

        outcome = run_allocation_rebalance()

        self.assertIsNone(outcome)
        mock_insert.assert_not_called()
        mock_adapter_cls.assert_not_called()

    @patch("src.infrastructure.yfinance_market_data_adapter.YFinanceMarketDataAdapter")
    @patch("src.trading.allocation_strategy.service.insert_snapshot")
    @patch("src.trading.allocation_strategy.service.get_latest_snapshot")
    @patch("src.trading.allocation_strategy.service.datetime")
    def test_rebalances_when_due(
        self, mock_datetime, mock_get_latest, mock_insert, mock_adapter_cls
    ):
        mock_datetime.now.return_value = datetime(2026, 9, 1)
        mock_get_latest.return_value = AllocationSnapshot(
            id=1,
            executed_at=datetime(2024, 1, 1),
            action="initial",
            tqqq_price=100.0,
            shy_price=50.0,
            tqqq_qty_before=0.0,
            shy_qty_before=0.0,
            cash_before=100_000.0,
            tqqq_qty_after=800.0,
            shy_qty_after=400.0,
            cash_after=0.0,
        )
        mock_adapter = MagicMock()
        mock_adapter.get_latest_price.side_effect = lambda symbol: {
            "TQQQ": 120.0,
            "SHY": 50.0,
        }[symbol]
        mock_adapter_cls.return_value = mock_adapter
        self._start_settings_patchers()

        from src.trading.allocation_strategy.service import run_allocation_rebalance

        outcome = run_allocation_rebalance()

        self.assertIsNotNone(outcome)
        self.assertEqual(outcome.action, "rebalance")
        self.assertAlmostEqual(outcome.tqqq_qty_before, 800.0)
        self.assertAlmostEqual(outcome.shy_qty_before, 400.0)
        self.assertAlmostEqual(outcome.tqqq_qty_after, 773.333333, places=4)
        self.assertAlmostEqual(outcome.shy_qty_after, 464.0, places=4)
        self.assertAlmostEqual(outcome.cash_after, 0.0, places=4)
        mock_insert.assert_called_once()

    @patch("src.infrastructure.yfinance_market_data_adapter.YFinanceMarketDataAdapter")
    @patch("src.trading.allocation_strategy.service.insert_snapshot")
    @patch("src.trading.allocation_strategy.service.get_latest_snapshot")
    @patch("src.trading.allocation_strategy.service.datetime")
    def test_aborts_when_price_fetch_fails(
        self, mock_datetime, mock_get_latest, mock_insert, mock_adapter_cls
    ):
        mock_datetime.now.return_value = datetime(2026, 1, 1)
        mock_get_latest.return_value = None
        mock_adapter = MagicMock()
        mock_adapter.get_latest_price.side_effect = lambda symbol: {
            "TQQQ": 100.0,
            "SHY": 0.0,
        }[symbol]
        mock_adapter_cls.return_value = mock_adapter
        self._start_settings_patchers()

        from src.trading.allocation_strategy.service import run_allocation_rebalance

        outcome = run_allocation_rebalance()

        self.assertIsNone(outcome)
        mock_insert.assert_not_called()


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: テストが失敗することを確認する**

Run: `python -m pytest tests/unit/test_allocation_service.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'src.trading.allocation_strategy.service'`）

- [ ] **Step 3: サービスを実装する**

`python/src/trading/allocation_strategy/service.py`:

```python
"""配分戦略(TQQQ 80% / 短期債 20%目安、約2年ごとリバランス)のペーパートレード実行ロジック。"""

from datetime import datetime, timedelta
from typing import Optional

from src.trading.allocation_strategy.repository import get_latest_snapshot, insert_snapshot
from src.trading.allocation_strategy.types import RebalanceOutcome
from src.utils.logger import get_logger

logger = get_logger(__name__)


def run_allocation_rebalance() -> Optional[RebalanceOutcome]:
    """配分戦略の初期建玉作成、または期日到来時のリバランスを実行する。

    まだ状態が無ければ初期建玉を作成する(action="initial")。
    既に状態があり、前回実行から ALLOCATION_STRATEGY_REBALANCE_YEARS 年以上
    (365日換算の近似)経過していればリバランスする(action="rebalance")。
    それ以外(未到来、または本日既に実行済み)は None を返し何もしない。
    価格取得に失敗した場合(0.0以下)も None を返し、状態は書き込まない。
    """
    from config.settings import (
        ALLOCATION_STRATEGY_BOND_SYMBOL,
        ALLOCATION_STRATEGY_INITIAL_CAPITAL,
        ALLOCATION_STRATEGY_REBALANCE_YEARS,
        ALLOCATION_STRATEGY_TQQQ_RATIO,
        ALLOCATION_STRATEGY_TQQQ_SYMBOL,
    )
    from src.infrastructure.yfinance_market_data_adapter import YFinanceMarketDataAdapter

    latest = get_latest_snapshot()
    now = datetime.now()

    if latest is not None:
        if latest.executed_at.date() == now.date():
            logger.info("配分戦略: 本日は既に実行済みのためスキップ")
            return None
        due_at = latest.executed_at + timedelta(days=365 * ALLOCATION_STRATEGY_REBALANCE_YEARS)
        if now < due_at:
            logger.info(
                "配分戦略: リバランス期日未到来のためスキップ（次回予定: %s）",
                due_at.date(),
            )
            return None

    market_data = YFinanceMarketDataAdapter()
    tqqq_price = market_data.get_latest_price(ALLOCATION_STRATEGY_TQQQ_SYMBOL)
    shy_price = market_data.get_latest_price(ALLOCATION_STRATEGY_BOND_SYMBOL)
    if tqqq_price <= 0 or shy_price <= 0:
        logger.error(
            "配分戦略: 価格取得失敗のため中止（tqqq_price=%s, shy_price=%s）",
            tqqq_price,
            shy_price,
        )
        return None

    if latest is None:
        action = "initial"
        tqqq_qty_before = 0.0
        shy_qty_before = 0.0
        cash_before = ALLOCATION_STRATEGY_INITIAL_CAPITAL
        total_value = ALLOCATION_STRATEGY_INITIAL_CAPITAL
    else:
        action = "rebalance"
        tqqq_qty_before = latest.tqqq_qty_after
        shy_qty_before = latest.shy_qty_after
        cash_before = latest.cash_after
        total_value = tqqq_qty_before * tqqq_price + shy_qty_before * shy_price + cash_before

    tqqq_qty_after = (total_value * ALLOCATION_STRATEGY_TQQQ_RATIO) / tqqq_price
    shy_target_value = total_value * (1 - ALLOCATION_STRATEGY_TQQQ_RATIO)
    shy_qty_after = shy_target_value / shy_price
    cash_after = total_value - (tqqq_qty_after * tqqq_price + shy_qty_after * shy_price)

    insert_snapshot(
        action=action,
        tqqq_price=tqqq_price,
        shy_price=shy_price,
        tqqq_qty_before=tqqq_qty_before,
        shy_qty_before=shy_qty_before,
        cash_before=cash_before,
        tqqq_qty_after=tqqq_qty_after,
        shy_qty_after=shy_qty_after,
        cash_after=cash_after,
    )

    return RebalanceOutcome(
        action=action,
        tqqq_price=tqqq_price,
        shy_price=shy_price,
        tqqq_qty_before=tqqq_qty_before,
        shy_qty_before=shy_qty_before,
        cash_before=cash_before,
        tqqq_qty_after=tqqq_qty_after,
        shy_qty_after=shy_qty_after,
        cash_after=cash_after,
    )
```

- [ ] **Step 4: テストが通ることを確認する**

Run: `python -m pytest tests/unit/test_allocation_service.py -v`
Expected: PASS（5テスト）

- [ ] **Step 5: Commit**

```bash
git add src/trading/allocation_strategy/service.py tests/unit/test_allocation_service.py
git commit -m "feat: 配分戦略ペーパートレードのリバランス判定・計算ロジックを追加"
```

---

### Task 5: Discord通知関数

**Files:**
- Modify: `python/src/reporting/discord/discord_notification_specs.py`
- Modify: `python/src/reporting/discord/notifications_model.py`
- Modify: `python/src/reporting/discord/discord_utils.py`
- Test: `python/tests/unit/test_allocation_discord_notification.py`

**Interfaces:**
- Consumes: `src.reporting.discord.webhook_sender.send_status_fields`, `src.reporting.discord.discord_notification_specs.NotificationSpec`, `src.utils.japan_time.format_jst`
- Produces: `send_allocation_rebalance_report(action: str, tqqq_price: float, shy_price: float, tqqq_qty_before: float, shy_qty_before: float, cash_before: float, tqqq_qty_after: float, shy_qty_after: float, cash_after: float) -> bool`（`discord_utils` からimport可能）。Task 6（periodic.py）がこれを呼ぶ。

- [ ] **Step 1: `discord_notification_specs.py` に通知スペックを追記する**

`COLOR_CAUTION = 0xFFAA00` の直後、`DAILY_PIPELINE_COMPLETION = ...` の直前に:

```python

ALLOCATION_REBALANCE_COMPLETION = NotificationSpec("💰 配分戦略 建玉更新", COLOR_SUCCESS)
```

- [ ] **Step 2: 失敗するテストを書く**

`python/tests/unit/test_allocation_discord_notification.py`:

```python
"""ユニットテスト: send_allocation_rebalance_report"""

import unittest
from unittest.mock import patch


class TestSendAllocationRebalanceReport(unittest.TestCase):
    @patch("src.reporting.discord.notifications_model.send_status_fields", return_value=True)
    def test_sends_status_fields_with_expected_spec(self, mock_send):
        from src.reporting.discord.discord_notification_specs import (
            ALLOCATION_REBALANCE_COMPLETION,
        )
        from src.reporting.discord.notifications_model import send_allocation_rebalance_report

        result = send_allocation_rebalance_report(
            action="initial",
            tqqq_price=100.0,
            shy_price=50.0,
            tqqq_qty_before=0.0,
            shy_qty_before=0.0,
            cash_before=100_000.0,
            tqqq_qty_after=800.0,
            shy_qty_after=400.0,
            cash_after=0.0,
        )

        self.assertTrue(result)
        mock_send.assert_called_once()
        spec_arg, fields_arg = mock_send.call_args[0][0], mock_send.call_args[0][1]
        self.assertEqual(spec_arg, ALLOCATION_REBALANCE_COMPLETION)
        field_names = [f["name"] for f in fields_arg]
        self.assertIn("TQQQ", field_names)
        self.assertIn("SHY", field_names)

    @patch("src.reporting.discord.notifications_model.send_status_fields", return_value=True)
    def test_rebalance_action_included_in_fields(self, mock_send):
        from src.reporting.discord.notifications_model import send_allocation_rebalance_report

        send_allocation_rebalance_report(
            action="rebalance",
            tqqq_price=120.0,
            shy_price=50.0,
            tqqq_qty_before=800.0,
            shy_qty_before=400.0,
            cash_before=0.0,
            tqqq_qty_after=773.33,
            shy_qty_after=464.0,
            cash_after=0.0,
        )

        fields_arg = mock_send.call_args[0][1]
        type_field = next(f for f in fields_arg if f["name"] == "📌 種別")
        self.assertEqual(type_field["value"], "リバランス")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3: テストが失敗することを確認する**

Run: `python -m pytest tests/unit/test_allocation_discord_notification.py -v`
Expected: FAIL（`ImportError: cannot import name 'ALLOCATION_REBALANCE_COMPLETION'` または `send_allocation_rebalance_report`）

- [ ] **Step 4: `notifications_model.py` に通知関数を追記する（ファイル末尾）**

```python


def send_allocation_rebalance_report(
    action: str,
    tqqq_price: float,
    shy_price: float,
    tqqq_qty_before: float,
    shy_qty_before: float,
    cash_before: float,
    tqqq_qty_after: float,
    shy_qty_after: float,
    cash_after: float,
) -> bool:
    """配分戦略(TQQQ/短期債)の初期建玉作成・リバランス結果を Discord に通知する。

    Args:
        action: "initial"（初期建玉作成）または "rebalance"（リバランス）
    """
    from src.reporting.discord.discord_notification_specs import ALLOCATION_REBALANCE_COMPLETION

    action_label = "初期建玉作成" if action == "initial" else "リバランス"
    fields = [
        {"name": "🕐 時刻", "value": format_jst(fmt=DISCORD_DATETIME_FORMAT), "inline": True},
        {"name": "📌 種別", "value": action_label, "inline": True},
        {
            "name": "TQQQ",
            "value": f"{tqqq_qty_before:.4f}株 → {tqqq_qty_after:.4f}株 @ ${tqqq_price:,.2f}",
            "inline": False,
        },
        {
            "name": "SHY",
            "value": f"{shy_qty_before:.4f}株 → {shy_qty_after:.4f}株 @ ${shy_price:,.2f}",
            "inline": False,
        },
        {
            "name": "現金",
            "value": f"${cash_before:,.2f} → ${cash_after:,.2f}",
            "inline": False,
        },
    ]
    return send_status_fields(ALLOCATION_REBALANCE_COMPLETION, fields)
```

- [ ] **Step 5: `discord_utils.py` の re-export リストに追記する**

`from src.reporting.discord.notifications_model import (  # noqa: F401  # re-export（#497 第4弾）` ブロックは既存項目がアルファベット順に並んでいる。`send_allocation_rebalance_report` はアルファベット順で先頭に来るため、ブロック全体を次の内容に置き換える:

```python
from src.reporting.discord.notifications_model import (  # noqa: F401  # re-export（#497 第4弾）
    send_allocation_rebalance_report,
    send_factory_completion,
    send_feature_suggestion_notification,
    send_optimization_completion,
    send_promotion_result,
    send_shadow_evaluation_notification,
    send_shap_batch_summary,
    send_shap_notification,
    send_strategy_promotion_detected,
)
```

- [ ] **Step 6: テストが通ることを確認する**

Run: `python -m pytest tests/unit/test_allocation_discord_notification.py -v`
Expected: PASS（2テスト）

- [ ] **Step 7: Commit**

```bash
git add src/reporting/discord/discord_notification_specs.py src/reporting/discord/notifications_model.py src/reporting/discord/discord_utils.py tests/unit/test_allocation_discord_notification.py
git commit -m "feat: 配分戦略ペーパートレードのDiscord通知関数を追加"
```

---

### Task 6: 定期実行ジョブ + scheduler.py ファサード再エクスポート

**Files:**
- Modify: `python/src/orchestration/jobs/periodic.py`
- Modify: `python/src/orchestration/scheduler.py`
- Test: `python/tests/unit/test_allocation_rebalance_job.py`

**Interfaces:**
- Consumes: `src.trading.allocation_strategy.service.run_allocation_rebalance`, `src.reporting.discord.discord_utils.send_allocation_rebalance_report`
- Produces: `run_allocation_rebalance_job() -> None`（`src.orchestration.scheduler` からimport可能）。ON/OFFフラグは設けない（Global Constraints参照）。Task 7（run_scheduler.py）がこれを呼ぶ。

- [ ] **Step 1: 失敗するテストを書く**

`python/tests/unit/test_allocation_rebalance_job.py`:

```python
from unittest.mock import patch

from src.orchestration.jobs.periodic import run_allocation_rebalance_job


class TestRunAllocationRebalanceJob:
    @patch("src.reporting.discord.discord_utils.send_allocation_rebalance_report")
    @patch("src.trading.allocation_strategy.service.run_allocation_rebalance")
    def test_runs_and_notifies_when_outcome_present(self, mock_run, mock_notify):
        from src.trading.allocation_strategy.types import RebalanceOutcome

        mock_run.return_value = RebalanceOutcome(
            action="initial",
            tqqq_price=100.0,
            shy_price=50.0,
            tqqq_qty_before=0.0,
            shy_qty_before=0.0,
            cash_before=100_000.0,
            tqqq_qty_after=800.0,
            shy_qty_after=400.0,
            cash_after=0.0,
        )

        run_allocation_rebalance_job()

        mock_run.assert_called_once()
        mock_notify.assert_called_once_with(
            action="initial",
            tqqq_price=100.0,
            shy_price=50.0,
            tqqq_qty_before=0.0,
            shy_qty_before=0.0,
            cash_before=100_000.0,
            tqqq_qty_after=800.0,
            shy_qty_after=400.0,
            cash_after=0.0,
        )

    @patch("src.reporting.discord.discord_utils.send_allocation_rebalance_report")
    @patch("src.trading.allocation_strategy.service.run_allocation_rebalance", return_value=None)
    def test_does_not_notify_when_service_returns_none(self, mock_run, mock_notify):
        run_allocation_rebalance_job()

        mock_run.assert_called_once()
        mock_notify.assert_not_called()

    @patch(
        "src.trading.allocation_strategy.service.run_allocation_rebalance",
        side_effect=Exception("network error"),
    )
    def test_does_not_raise_on_service_failure(self, mock_run):
        run_allocation_rebalance_job()  # 例外を送出しないことを確認

    @patch(
        "src.reporting.discord.discord_utils.send_allocation_rebalance_report",
        side_effect=Exception("discord down"),
    )
    @patch("src.trading.allocation_strategy.service.run_allocation_rebalance")
    def test_does_not_raise_on_notification_failure(self, mock_run, mock_notify):
        from src.trading.allocation_strategy.types import RebalanceOutcome

        mock_run.return_value = RebalanceOutcome(
            action="initial",
            tqqq_price=100.0,
            shy_price=50.0,
            tqqq_qty_before=0.0,
            shy_qty_before=0.0,
            cash_before=100_000.0,
            tqqq_qty_after=800.0,
            shy_qty_after=400.0,
            cash_after=0.0,
        )

        run_allocation_rebalance_job()  # 例外を送出しないことを確認
```

- [ ] **Step 2: テストが失敗することを確認する**

Run: `python -m pytest tests/unit/test_allocation_rebalance_job.py -v`
Expected: FAIL（`ImportError: cannot import name 'run_allocation_rebalance_job'`）

- [ ] **Step 3: `periodic.py` にジョブ関数を追記する（ファイル末尾）**

```python


def run_allocation_rebalance_job() -> None:
    """
    配分戦略(TQQQ 80% / 短期債 20%目安、約2年ごとリバランス)のペーパートレードを実行する。

    ON/OFFフラグは設けない。自動実行させない安全装置は run_scheduler.py の
    SCHEDULE_CONFIG 側（auto_schedule: False）にあり、実行するかどうかは
    --run-now allocation_rebalance を叩く人間の判断に委ねる。
    """
    logger.info("=== 配分戦略 実行開始 ===")
    try:
        from src.trading.allocation_strategy.service import run_allocation_rebalance

        outcome = run_allocation_rebalance()
    except Exception as e:
        logger.error("配分戦略の実行に失敗しました: %s", e, exc_info=True)
        return
    logger.info("=== 配分戦略 実行完了 ===")

    if outcome is None:
        return

    try:
        from src.reporting.discord.discord_utils import send_allocation_rebalance_report

        send_allocation_rebalance_report(
            action=outcome.action,
            tqqq_price=outcome.tqqq_price,
            shy_price=outcome.shy_price,
            tqqq_qty_before=outcome.tqqq_qty_before,
            shy_qty_before=outcome.shy_qty_before,
            cash_before=outcome.cash_before,
            tqqq_qty_after=outcome.tqqq_qty_after,
            shy_qty_after=outcome.shy_qty_after,
            cash_after=outcome.cash_after,
        )
    except Exception as e:
        logger.error("配分戦略通知失敗: %s", e, exc_info=True)
```

- [ ] **Step 4: テストが通ることを確認する**

Run: `python -m pytest tests/unit/test_allocation_rebalance_job.py -v`
Expected: PASS（4テスト）

- [ ] **Step 5: `scheduler.py` ファサードに再エクスポートを追加する**

`from src.orchestration.jobs.periodic import (` ブロックを次のように変更する:

```python
from src.orchestration.jobs.periodic import (
    run_allocation_rebalance_job,
    run_monthly_report_job,
    run_nightly_strategy_factory,
    run_strategy_promotion_check,
)
```

`__all__` リストの `# periodic` セクションを次のように変更する:

```python
    # periodic
    "run_monthly_report_job",
    "run_nightly_strategy_factory",
    "run_strategy_promotion_check",
    "run_allocation_rebalance_job",
```

- [ ] **Step 6: importできることを確認する**

Run: `python -c "from src.orchestration.scheduler import run_allocation_rebalance_job; print(run_allocation_rebalance_job)"`
Expected: 関数オブジェクトが表示される（エラーなし）

- [ ] **Step 7: Commit**

```bash
git add src/orchestration/jobs/periodic.py src/orchestration/scheduler.py tests/unit/test_allocation_rebalance_job.py
git commit -m "feat: 配分戦略ペーパートレードの定期実行ジョブを追加"
```

---

### Task 7: run_scheduler.py への配線（手動実行のみ・自動起動なし）

**Files:**
- Modify: `python/run_scheduler.py`

**Interfaces:**
- Consumes: `src.orchestration.scheduler.run_allocation_rebalance_job`
- Produces: `py run_scheduler.py --run-now allocation_rebalance` で手動実行可能。`SCHEDULE_CONFIG["allocation_rebalance"]`（`auto_schedule: False` のため自動起動はしない）。

- [ ] **Step 1: ジョブラッパー関数を追加する**

`job_strategy_promotion_check()` 関数の直後（171行目付近、`# ── イベントリスナー` セクションの直前）に追記する:

```python


def job_allocation_rebalance():
    """配分戦略(TQQQ/短期債)ペーパートレード実行（SCHEDULE_CONFIGでauto_schedule: False、既定は手動実行のみ）"""
    from src.orchestration.scheduler import run_allocation_rebalance_job

    run_allocation_rebalance_job()
```

- [ ] **Step 2: `SCHEDULE_CONFIG` にエントリを追加する**

`"strategy_promotion_check": { ... },` エントリの直後、`}` （`SCHEDULE_CONFIG` の閉じ括弧）の直前に追記する:

```python
    "allocation_rebalance": {
        "func": job_allocation_rebalance,
        "trigger": "cron",
        "period": "daily",
        "day_of_week": "mon-sun",
        "hour": 6,
        "minute": 0,
        "recovery_delay_minutes": 30,
        "max_executions_per_period": 1,
        "description": "毎日 06:00 - 配分戦略(TQQQ/短期債)リバランス判定",
        # 既定では自動起動しない。--run-now allocation_rebalance による手動実行のみ許可。
        # 自動スケジュールを有効化する（auto_schedule: True に変更する）かどうかは
        # ユーザー自身の判断で後日行う。
        "auto_schedule": False,
    },
```

- [ ] **Step 3: `--run-now` の `choices` リストに追加する**

```python
            "promotion_check",
            "allocation_rebalance",
        ],
```

（`"promotion_check",` の直後に `"allocation_rebalance",` を追加する）

- [ ] **Step 4: `run_now()` 関数に分岐を追加する**

`elif pipeline == "promotion_check":` ブロックの直後に追記する:

```python
    elif pipeline == "allocation_rebalance":
        queue_manager.run_job("allocation_rebalance", reason="manual", force=True)
```

- [ ] **Step 5: モジュールとしてimportできることを確認する**

Run: `python -c "import run_scheduler; print('allocation_rebalance' in run_scheduler.SCHEDULE_CONFIG)"`
Expected: `True`

- [ ] **Step 6: `--run-now allocation_rebalance` を実際には実行しない**

このジョブにはON/OFFフラグが無いため、`python run_scheduler.py --run-now allocation_rebalance` を実行すると実際にyfinanceから価格取得し、本番DBの `allocation_rebalance_log` に初期建玉（`ALLOCATION_STRATEGY_INITIAL_CAPITAL` 分、既定 $100,000）を書き込み、実際にDiscordへ通知が送信される。これは実データを動かす実運用開始の判断そのものであり、実装検証の一部として自動的に行ってはならない。配線が正しいことは Step 5（import確認）と Task 6 のユニットテスト（モック済み、実DB・実API・実Discordに一切触れない）で十分に検証済みである。実際に `--run-now allocation_rebalance` を叩いて戦略を開始するタイミングは、この実装が完了した後にユーザー自身が判断する。

- [ ] **Step 7: ユニットテスト全体を実行し、既存テストを壊していないことを確認する**

Run: `python -m pytest tests/unit/ -v --cov=src --cov-branch --cov-report=term-missing --cov-fail-under=80`
Expected: PASS（新規追加分含め全green、カバレッジ80%以上）

- [ ] **Step 8: Commit**

```bash
git add run_scheduler.py
git commit -m "feat: 配分戦略ペーパートレードをrun_scheduler.pyに配線（既定は手動実行のみ）"
```

---

## 自己レビュー結果（このプラン作成時に実施済み）

- **仕様網羅性**: 「TQQQ 80%/短期債20%、2年ごとリバランス」の戦略ロジック(Task 4)、StockFixerへの統合(Task 6-7)、既存JP資産への非干渉(Global Constraints)、Discord通知(Task 5)、DB永続化(Task 1, 3)を全てタスク化済み。
- **プレースホルダ**: なし。全タスクに完全なコードを記載。
- **型の一貫性**: `RebalanceOutcome`/`AllocationSnapshot` のフィールド名は types.py → repository.py → service.py → periodic.py → notifications_model.py まで一貫して `tqqq_qty_before/after` 等の命名で統一。
- **advisorレビューで指摘された3点への対応**:
  1. 状態を `system_config` に置かない → `allocation_rebalance_log` 追記専用ログ単独テーブルに変更済み（Task 1, 3）。
  2. 同日二重実行・クラッシュ時の半端な状態 → 状態遷移を単一INSERTに設計し、かつ `auto_schedule: False` によりrecovery pollerがそもそもこのジョブに触れないことを確認済み（Task 7 Step 2のコメント、Global Constraints）。加えて同日ガードをservice.py自体にも実装（Task 4）。
  3. 価格取得失敗(0.0)時のガード → service.py内で明示的にチェックし、状態を書き込まず中断（Task 4）。
- **ユーザー修正の反映**: 当初案にあった `ALLOCATION_STRATEGY_ENABLED` ON/OFFフラグはユーザー指示により撤去。安全装置は `auto_schedule: False` のみとし、`--run-now allocation_rebalance` が実データを動かす実運用開始の判断そのものになる点をGlobal ConstraintsとTask 7 Step 6に明記した。
