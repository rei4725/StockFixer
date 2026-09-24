# DB 所有権の疎結合化 Phase 4b（`paper_real_diff`）実装計画

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `paper_real_diff` テーブルの所有権を `src/prediction/db/` から剥がし、書き側を `TradeDiffSink` ポート、読み側を `AnalyticsQuery` ポートへ移して、`src/utils/db` プロキシ経由の trading→prediction / reporting→prediction の隠れた辺を 2 本消す。

**Architecture:** Phase 4a で確立した 5 ステップ（値オブジェクト → ポート → アダプタ → 合成ルート結線 → 旧経路撤去）を `paper_real_diff` テーブルに対してなぞる。書き側は `TradeDiffRecord` 値オブジェクト + `TradeDiffSink` ポートを `src/domain/` に立て、SQL は `src/infrastructure/persistence/trade_diff_repository.py` へ一文字も変えずに移設する。読み側は reporting が自分で DB を取りに行くのをやめ、最外周の入口（`run_dashboard.py` / `run_monthly_report.py` / `orchestration`）が `AnalyticsQuery` を構築して渡す形に押し上げる。

**Tech Stack:** Python 3.12 / psycopg (Postgres) / pandas / unittest + pytest / import-linter / mypy / black / isort / flake8 / pylint

**Spec:** `docs/superpowers/specs/2026-09-22-db-ownership-decoupling-design.md`（§6.1 書き側 / §6.2 読み側 / §6.3 除去する保険 / §6.4 呼び出し元ゼロの関数）

**先行計画:** `docs/superpowers/plans/2026-09-22-db-ownership-decoupling-phase4a.md`（PR#736 / v2.15.0 でマージ済み）

---

## Global Constraints

- **SQL は一文字も変更しない。** アダプタへの移設は「場所の変更」に限定する。差分レビューで SQL 行が変わって見えてはならない。
- **`| None = None` の任意注入をしてはならない。** 本計画で導入するポートはすべて必須引数とし、合成ルートが必ず渡す。None フォールバックが残る限り旧経路が生き続け、「ポートを入れたのに何も変わらない」状態になる。
- **ポートを生成するのは最外周の入口のみ。** `run_*.py` / `src/orchestration/` / Discord Bot / `src/api/`。BC の内側でアダプタを `import` してはならない。
- **`src/domain/` は標準ライブラリと `pandas` 以外を import しない。** `.importlinter` の `domain-purity` 契約（`forbidden`）が強制する。
- **Windows でのローカル import-linter 実行には `PYTHONUTF8=1` が必須。** `.importlinter` の日本語コメントで `'cp932' codec can't decode byte` になる。また `py -m importlinter.cli lint-imports` は無出力 exit 0 を返すため使ってはならない。正しい呼び出しは `py -c "from importlinter.cli import lint_imports_command; lint_imports_command()"`。
- **`git commit -m A -m B` は trailer を壊す。** git は最終段落のみを trailer と見なすため、`Co-Authored-By` と `Claude-Session` は 1 つの `-m` にまとめる。
- **`Set-Content -Encoding utf8` は PowerShell 5.1 では BOM を付ける。** `VERSION` を書き換えたら `xxd python/VERSION | head -1` で BOM 無しを確認する。
- **unit テストは `DATABASE_URL` 未設定なら 5433 のテスト専用 Postgres に向く**（#548 対策）。worktree には `python/.env` が無いため本番 DB への誤到達は構造的に防がれている。この状態を壊さない。
- **コミットメッセージは Conventional Commits**（`feat:` / `fix:` / `refactor:` / `test:` / `docs:` / `chore:`）。ベースブランチは `develop`。
- 作業ディレクトリは断りがない限り `python/`。Windows では `python` ではなく `py` を使う。

## 本計画のスコープ

設計書 §5 が挙げた Phase 4b の対象のうち、**束① の `paper_real_diff`（書き＋読み＋reporting 押し上げ）のみ**を扱う。

| 対象 | 本計画 | 備考 |
|---|---|---|
| `upsert_paper_real_diff`（書き） | ✅ Task 1〜7 | trading BC の module-level import を外す |
| `load_paper_real_diff_summary`（読み） | ✅ Task 8〜9 | reporting 3 ファイルの押し上げ |
| `load_open_close_advantage_summary` | ✅ Task 10（削除） | 本番呼び出し元ゼロ（設計書 §6.4） |
| 束② accuracy / drift / weekly | ❌ Phase 4c | `AnalyticsQuery` に 3 メソッド追加する形で続く |
| 束③ `prediction_results` | ❌ Phase 4c | `PredictionResultRepository` の DB アダプタ新設 |
| プロキシ全撤去・`allow_indirect_imports` 削除 | ❌ Phase 4c | 本計画は `_PREDICTION_DB` を 17 → 15 に減らすのみ |

**設計書からの意図的な逸脱が 1 点ある。** 設計書 §6.2 は `AnalyticsQuery` を 4 メソッド（`paper_real_diff_summary` / `drift_summary` / `prediction_accuracy` / `weekly_accuracy_snapshots`）で定義しているが、本計画では **`paper_real_diff_summary` の 1 メソッドのみ**を定義する。残る 3 つは束②（Phase 4c）の対象であり、実装の当てのない `@abstractmethod` を先に置くと、アダプタ 2 本（Postgres / InMemory）が中身の無いメソッドを抱えることになるためである。メソッドの追加は Phase 4c で行い、その時点でも上限 10 メソッドの設計規則は変わらない。

---

## File Structure

### 新規作成

| ファイル | 責務 |
|---|---|
| `python/src/infrastructure/persistence/trade_diff_repository.py` | `paper_real_diff` テーブルへの書き込み（`PostgresTradeDiffSink`）。`upsert_paper_real_diff` の SQL とマージロジックをそのまま持つ |
| `python/src/infrastructure/persistence/analytics_query.py` | reporting 向け読み取りクエリ（`PostgresAnalyticsQuery`）。`load_paper_real_diff_summary` の SQL をそのまま持つ |
| `python/tests/unit/test_trade_diff_sink.py` | `TradeDiffRecord` / `TradeDiffSink` / `InMemoryTradeDiffSink` のユニットテスト |
| `python/tests/unit/test_persistence_trade_diff_repository.py` | `PostgresTradeDiffSink` の SQL 回帰テスト |
| `python/tests/unit/test_analytics_query.py` | `AnalyticsQuery` ポートと 2 実装のテスト |
| `python/tests/unit/test_composition_roots.py` | **合成ルートが本物のアダプタを渡していることの検証**（Phase 4a で生まれた新しい失敗様式への対処） |

### 変更

| ファイル | 変更内容 |
|---|---|
| `python/src/domain/types.py` | `TradeDiffRecord` dataclass を追加 |
| `python/src/domain/ports.py` | `TradeDiffSink` / `AnalyticsQuery` を追加 |
| `python/src/infrastructure/in_memory.py` | `InMemoryTradeDiffSink` / `InMemoryAnalyticsQuery` を追加 |
| `python/src/utils/db/_connection.py` | `_db_connection` の公開別名 `db_connection` を追加 |
| `python/src/utils/db/__init__.py` | `db_connection` を再輸出。`_PREDICTION_DB` から 2 名を削除（17 → 15） |
| `python/src/trading/execution/recording.py` | module-level `from src.utils.db import upsert_paper_real_diff` を撤去し、`trade_diff_sink` を必須引数で受ける |
| `python/src/trading/execution/runner.py` | `run_daily_orders` に `trade_diff_sink` 必須キーワードを追加し、4 箇所の `_record_order` と `_sync_live_execution_diffs` へ貫通させる |
| `python/src/trading/execution/sl_tp.py` | `_check_sl_tp_exits` に `trade_diff_sink` を貫通させる |
| `python/src/trading/claude_agent.py` | `run_claude_trader` / `_handle_place_order` に `trade_diff_sink` を貫通させる |
| `python/src/trading/brokers/paper/paper_broker.py` | `record_diff: Callable[..., None] \| None = None` を `trade_diff_sink: TradeDiffSink`（必須）へ置換 |
| `python/src/orchestration/jobs/daily.py` | 2 箇所の合成ルートで `PostgresTradeDiffSink()` を注入 |
| `python/run_auto_trade.py` | 合成を `main(argv)` 関数へ切り出し（テスト可能にする）、`PostgresTradeDiffSink()` を注入 |
| `python/run_claude_trader.py` | 同上 |
| `python/src/reporting/dashboard.py` | `run_dashboard` が `AnalyticsQuery` を受け取り、`_section_paper_real_diff` はデータを受け取る形に |
| `python/src/reporting/kpi.py` | `get_monthly_kpis` が `diff_summary: dict` を受け取る（自分で読まない） |
| `python/src/reporting/monthly.py` | `run_monthly_report` が `AnalyticsQuery` を受け取り `kpi` へデータを渡す |
| `python/src/reporting/query_service.py` | `get_monthly_report_summary` が `AnalyticsQuery` を受け取り `run_monthly_report` へ引き渡す |
| `python/src/api/external_v1.py` / `python/src/reporting/discord/discord_bot.py` | 入口として `PostgresAnalyticsQuery()` を構築して渡す |
| `python/src/reporting/discord/notifications_report.py` | `diff_summary` の None フォールバック（自分で DB を読む保険）を撤去 |
| `python/run_dashboard.py` / `python/run_monthly_report.py` | `PostgresAnalyticsQuery()` を構築して渡す |
| `python/src/orchestration/jobs/periodic.py` / `weekly.py` | 同上 |
| `python/src/prediction/db/paper_real_diff.py` | 削除 |
| `python/src/prediction/db/__init__.py` | `paper_real_diff` の 3 名の再輸出を削除 |
| `python/tests/unit/test_db_prediction.py` | `paper_real_diff` 関連テストを新テストへ移設し、旧テストを削除 |
| `python/VERSION` | 2.15.0 → 2.16.0 |

---

## Task 1: `TradeDiffRecord` 値オブジェクトを domain に定義する

**Files:**
- Modify: `python/src/domain/types.py`（末尾に追加。`OrderRunSummary` は 249-266 行）
- Test: `python/tests/unit/test_trade_diff_sink.py`（新規）

**Interfaces:**
- Consumes: なし（最初のタスク）
- Produces: `src.domain.types.TradeDiffRecord` — フィールドは `market: str` / `symbol: str` / `predicted_at: str` / `side: int` / `signal_price: float` / `mode: str` / `order_id: str` / `actual_price: Optional[float] = None` / `checked_at: Optional[datetime] = None` / `order_session: str = "open"` / `split_ratio: Optional[float] = None`

このフィールド順・既定値は現行 `upsert_paper_real_diff()` の引数順と既定値をそのまま写したものである。写し間違いは静かな誤記録になるため、必ず元関数（`python/src/prediction/db/paper_real_diff.py:11-22`）と突き合わせること。

- [ ] **Step 1: 失敗するテストを書く**

`python/tests/unit/test_trade_diff_sink.py` を新規作成する。

```python
"""ユニットテスト: TradeDiffRecord / TradeDiffSink / InMemoryTradeDiffSink。"""

import unittest
from datetime import datetime

from src.domain.types import TradeDiffRecord


class TestTradeDiffRecord(unittest.TestCase):
    def test_required_fields_and_defaults(self):
        rec = TradeDiffRecord(
            market="jp",
            symbol="7203",
            predicted_at="2026-09-23T00:00:00",
            side=1,
            signal_price=1000.0,
            mode="paper",
            order_id="ord-1",
        )
        self.assertEqual(rec.market, "jp")
        self.assertEqual(rec.symbol, "7203")
        self.assertEqual(rec.predicted_at, "2026-09-23T00:00:00")
        self.assertEqual(rec.side, 1)
        self.assertEqual(rec.signal_price, 1000.0)
        self.assertEqual(rec.mode, "paper")
        self.assertEqual(rec.order_id, "ord-1")
        self.assertIsNone(rec.actual_price)
        self.assertIsNone(rec.checked_at)
        self.assertEqual(rec.order_session, "open")
        self.assertIsNone(rec.split_ratio)

    def test_optional_fields_are_settable(self):
        now = datetime(2026, 9, 23, 9, 0, 0)
        rec = TradeDiffRecord(
            market="jp",
            symbol="7203",
            predicted_at="2026-09-23T00:00:00",
            side=1,
            signal_price=1000.0,
            mode="live",
            order_id="ord-2",
            actual_price=1005.0,
            checked_at=now,
            order_session="close",
            split_ratio=0.5,
        )
        self.assertEqual(rec.actual_price, 1005.0)
        self.assertEqual(rec.checked_at, now)
        self.assertEqual(rec.order_session, "close")
        self.assertEqual(rec.split_ratio, 0.5)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: テストが失敗することを確認する**

```bash
cd python
py -m pytest tests/unit/test_trade_diff_sink.py -v
```

Expected: FAIL — `ImportError: cannot import name 'TradeDiffRecord' from 'src.domain.types'`

- [ ] **Step 3: 最小実装を書く**

`python/src/domain/types.py` の import 節に `datetime` を追加する。ファイル先頭は現状こうなっている:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
```

`dataclasses` の行の**上**に次の 1 行を挿入する（isort の標準ライブラリ順序に従う）:

```python
from datetime import datetime
```

そのうえで、`OrderRunSummary` の定義（`min_change_ratio: float` の行）の直後、ファイル末尾に追加する:

```python


@dataclass
class TradeDiffRecord:
    """paper / real 約定価格の乖離追跡 1 件（paper_real_diff テーブルの 1 行に対応）。

    フィールドは旧 upsert_paper_real_diff() の引数をそのまま写したもの。
    side: OrderSide の整数値。mode: "paper" / "live"。
    order_session: "open"（寄付）または "close"（引け）。
    """

    market: str
    symbol: str
    predicted_at: str
    side: int
    signal_price: float
    mode: str
    order_id: str
    actual_price: Optional[float] = None
    checked_at: Optional[datetime] = None
    order_session: str = "open"
    split_ratio: Optional[float] = None
```

- [ ] **Step 4: テストが通ることを確認する**

```bash
cd python
py -m pytest tests/unit/test_trade_diff_sink.py -v
```

Expected: PASS（2 passed）

- [ ] **Step 5: domain の純粋性が保たれていることを確認する**

```bash
cd python
PYTHONUTF8=1 py -c "from importlinter.cli import lint_imports_command; lint_imports_command()"
```

Expected: `Contracts: 3 kept, 0 broken.`（`domain-purity` が緑であること。`datetime` は標準ライブラリなので違反にならない）

- [ ] **Step 6: コミット**

```bash
git add python/src/domain/types.py python/tests/unit/test_trade_diff_sink.py
git commit -m "$(cat <<'EOF'
feat: TradeDiffRecord 値オブジェクトを domain に追加する

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01N4FK4xZ3jgfaWpZcMDv964
EOF
)"
```

---

## Task 2: `TradeDiffSink` ポートとインメモリ実装

**Files:**
- Modify: `python/src/domain/ports.py`（`OrderRunSink` の定義の直後に追加）
- Modify: `python/src/infrastructure/in_memory.py`（`InMemoryOrderRunSink`（195-203 行）の直後に追加、および先頭の import 節）
- Test: `python/tests/unit/test_trade_diff_sink.py`（Task 1 で作成済み・追記）

**Interfaces:**
- Consumes: `src.domain.types.TradeDiffRecord`（Task 1）
- Produces:
  - `src.domain.ports.TradeDiffSink` — ABC。`record(self, record: TradeDiffRecord) -> None` の 1 メソッド
  - `src.infrastructure.in_memory.InMemoryTradeDiffSink` — `.recorded: list[TradeDiffRecord]` を公開する偽実装

`InMemoryOrderRunSink` が属性名 `saved` を使っているのに対し、こちらは `recorded` を使う。メソッド名が `save` ではなく `record` であることに対応させるためで、2 つの Sink を同じテストで取り違えないようにする意図もある。

- [ ] **Step 1: 失敗するテストを書く**

`python/tests/unit/test_trade_diff_sink.py` の末尾（`if __name__ == "__main__":` の**前**）に追記する。import 節にも追加する。

ファイル先頭の import を次に差し替える:

```python
import unittest
from datetime import datetime

from src.domain.ports import TradeDiffSink
from src.domain.types import TradeDiffRecord
from src.infrastructure.in_memory import InMemoryTradeDiffSink
```

末尾に追記するテストクラス:

```python
class TestInMemoryTradeDiffSink(unittest.TestCase):
    def _record(self, symbol: str = "7203") -> TradeDiffRecord:
        return TradeDiffRecord(
            market="jp",
            symbol=symbol,
            predicted_at="2026-09-23T00:00:00",
            side=1,
            signal_price=1000.0,
            mode="paper",
            order_id=f"ord-{symbol}",
        )

    def test_implements_port(self):
        self.assertIsInstance(InMemoryTradeDiffSink(), TradeDiffSink)

    def test_starts_empty(self):
        self.assertEqual(InMemoryTradeDiffSink().recorded, [])

    def test_record_appends_in_order(self):
        sink = InMemoryTradeDiffSink()
        sink.record(self._record("7203"))
        sink.record(self._record("6758"))
        self.assertEqual([r.symbol for r in sink.recorded], ["7203", "6758"])

    def test_port_cannot_be_instantiated(self):
        with self.assertRaises(TypeError):
            TradeDiffSink()  # type: ignore[abstract]
```

- [ ] **Step 2: テストが失敗することを確認する**

```bash
cd python
py -m pytest tests/unit/test_trade_diff_sink.py -v
```

Expected: FAIL — `ImportError: cannot import name 'TradeDiffSink' from 'src.domain.ports'`

- [ ] **Step 3: ポートを実装する**

`python/src/domain/ports.py` の import 行を差し替える:

```python
from src.domain.types import OrderRunSummary, TradeDiffRecord
```

`OrderRunSink` クラスの定義の直後に追加する:

```python


class TradeDiffSink(ABC):
    """paper / real 約定価格の乖離記録の書き込みポート。

    trading BC は自らの約定結果を記録するが、記録先（テーブル・DB）を知らない。
    実装は src/infrastructure/persistence/ に置き、合成ルートが注入する。
    """

    @abstractmethod
    def record(self, record: TradeDiffRecord) -> None:
        """約定乖離レコードを 1 件記録する（同一キーは上書き）"""
```

- [ ] **Step 4: インメモリ実装を書く**

`python/src/infrastructure/in_memory.py` の `from src.domain.ports import (...)` に `TradeDiffSink` を、`from src.domain.types import OrderRunSummary` を `from src.domain.types import OrderRunSummary, TradeDiffRecord` に差し替える。アルファベット順を保つため、ports の import は次になる:

```python
from src.domain.ports import (
    AlertLevel,
    BrokerPort,
    MarketDataPort,
    NotificationPort,
    OrderRunSink,
    PredictionResultRepository,
    StockFeatureRepository,
    TradeDiffSink,
)
from src.domain.types import OrderRunSummary, TradeDiffRecord
```

`InMemoryOrderRunSink` クラスの直後（ファイル末尾）に追加する:

```python


class InMemoryTradeDiffSink(TradeDiffSink):
    """インメモリ約定乖離 Sink（テスト用）"""

    def __init__(self) -> None:
        self.recorded: list[TradeDiffRecord] = []

    def record(self, record: TradeDiffRecord) -> None:
        self.recorded.append(record)
```

- [ ] **Step 5: テストが通ることを確認する**

```bash
cd python
py -m pytest tests/unit/test_trade_diff_sink.py -v
```

Expected: PASS（6 passed）

- [ ] **Step 6: 契約を確認する**

```bash
cd python
PYTHONUTF8=1 py -c "from importlinter.cli import lint_imports_command; lint_imports_command()"
```

Expected: `Contracts: 3 kept, 0 broken.`

- [ ] **Step 7: コミット**

```bash
git add python/src/domain/ports.py python/src/infrastructure/in_memory.py python/tests/unit/test_trade_diff_sink.py
git commit -m "$(cat <<'EOF'
feat: TradeDiffSink ポートとインメモリ実装を追加する

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01N4FK4xZ3jgfaWpZcMDv964
EOF
)"
```

---

## Task 3: `_db_connection` の公開別名を用意する

**Files:**
- Modify: `python/src/utils/db/_connection.py`（末尾に別名を追加）
- Modify: `python/src/utils/db/__init__.py`（30-37 行の `from src.utils.db._connection import (...)` に追加）
- Modify: `python/src/infrastructure/persistence/order_run_repository.py`（4 行目の import を差し替え）
- Test: `python/tests/unit/test_db_connection_alias.py`（新規）

**Interfaces:**
- Consumes: なし
- Produces: `src.utils.db.db_connection` — `_db_connection` と同一オブジェクトを指す公開別名（コンテキストマネージャを返す callable）

このタスクを Task 4 より前に置く理由は、本計画でアダプタを 2 本（`trade_diff_repository.py` / `analytics_query.py`）新設するためである。Phase 4c でさらに 2 本増える見込みであり、`_db_connection` という私的シンボルを 4 ファイルで参照する形を先に潰しておく。

- [ ] **Step 1: 失敗するテストを書く**

`python/tests/unit/test_db_connection_alias.py` を新規作成する。

```python
"""ユニットテスト: src.utils.db.db_connection 公開別名。

infrastructure/persistence/ のアダプタが私的シンボル _db_connection を
直接参照せずに済むよう、公開別名が同一実体を指していることを保証する。
"""

import unittest


class TestDbConnectionAlias(unittest.TestCase):
    def test_alias_is_the_same_object(self):
        from src.utils.db import db_connection
        from src.utils.db._connection import _db_connection

        self.assertIs(db_connection, _db_connection)

    def test_alias_is_importable_from_package_root(self):
        import src.utils.db as db_module

        self.assertTrue(hasattr(db_module, "db_connection"))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: テストが失敗することを確認する**

```bash
cd python
py -m pytest tests/unit/test_db_connection_alias.py -v
```

Expected: FAIL — `ImportError: cannot import name 'db_connection' from 'src.utils.db'`

- [ ] **Step 3: 別名を定義する**

`python/src/utils/db/_connection.py` の末尾に追加する:

```python


# 公開別名。infrastructure/persistence/ のアダプタはこちらを使う。
# （モジュール名が `_connection` である以上、`_db_connection` は二重に私的であり、
#   アダプタごとに私的シンボルを参照して回る形を避ける）
db_connection = _db_connection
```

`python/src/utils/db/__init__.py` の接続管理 import を次に差し替える:

```python
# --- 接続管理 ---
from src.utils.db._connection import (
    _db_connection,
    close_connection,
    db_connection,
    get_readonly_connection,
    init_tables,
    set_test_connection,
)
```

- [ ] **Step 4: テストが通ることを確認する**

```bash
cd python
py -m pytest tests/unit/test_db_connection_alias.py -v
```

Expected: PASS（2 passed）

- [ ] **Step 5: 既存アダプタを公開別名に寄せる**

`python/src/infrastructure/persistence/order_run_repository.py` の 4 行目を差し替える。

変更前:
```python
from src.utils.db._connection import _db_connection
```

変更後:
```python
from src.utils.db import db_connection
```

同ファイル内の `with _db_connection() as con:` を `with db_connection() as con:` に差し替える（1 箇所）。

- [ ] **Step 6: 既存の order_run_sink テストが緑のままであることを確認する**

```bash
cd python
py -m pytest tests/unit/test_in_memory_order_run_sink.py tests/unit/test_db_connection_alias.py -v
```

Expected: PASS

- [ ] **Step 7: わざと壊して別名が効いていることを確かめる**

`python/src/utils/db/_connection.py` の末尾に足した `db_connection = _db_connection` を一時的にコメントアウトし、Step 4 のテストを再実行する。

Expected: FAIL（`ImportError`）。確認できたらコメントアウトを戻し、もう一度 Step 4 を実行して PASS に戻ることを確認する。

- [ ] **Step 8: コミット**

```bash
git add python/src/utils/db/_connection.py python/src/utils/db/__init__.py python/src/infrastructure/persistence/order_run_repository.py python/tests/unit/test_db_connection_alias.py
git commit -m "$(cat <<'EOF'
refactor: _db_connection の公開別名 db_connection を用意する

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01N4FK4xZ3jgfaWpZcMDv964
EOF
)"
```

---

## Task 4: `PostgresTradeDiffSink` アダプタ（SQL の移設）

**Files:**
- Create: `python/src/infrastructure/persistence/trade_diff_repository.py`
- Test: `python/tests/unit/test_persistence_trade_diff_repository.py`（新規）
- Reference（読むだけ・この時点では変更しない）: `python/src/prediction/db/paper_real_diff.py:11-131`

**Interfaces:**
- Consumes: `TradeDiffRecord`（Task 1）/ `TradeDiffSink`（Task 2）/ `db_connection`（Task 3）
- Produces: `src.infrastructure.persistence.trade_diff_repository.PostgresTradeDiffSink` — 引数なしで構築できる `TradeDiffSink` 実装

**このタスクの肝は「移設であって書き換えではない」こと。** 元関数の SELECT / DELETE / INSERT の 3 文、`merged` 辞書のマージ規則、`price_diff` の算出、`split_ratio` の二段構えの扱いをすべてそのまま持ってくる。引数 12 個を `record.` 属性参照に置き換えるだけが差分である。

- [ ] **Step 1: 失敗するテストを書く**

`python/tests/unit/test_persistence_trade_diff_repository.py` を新規作成する。実 DB には触らず、偽の接続オブジェクトへ渡された SQL と引数を検証する。

```python
"""ユニットテスト: PostgresTradeDiffSink（SQL 移設の回帰検証）。

実 DB には接続せず、db_connection を偽物に差し替えて
発行される SQL と引数を検証する。SQL 本文が変わると落ちる。
"""

import unittest
from contextlib import contextmanager
from datetime import datetime
from unittest.mock import patch

from src.domain.ports import TradeDiffSink
from src.domain.types import TradeDiffRecord
from src.infrastructure.persistence.trade_diff_repository import PostgresTradeDiffSink


class _FakeCursor:
    def __init__(self, row):
        self._row = row

    def fetchone(self):
        return self._row


class _FakeConnection:
    """execute() の呼び出しを (sql, params) で記録する偽接続。"""

    def __init__(self, existing_row=None):
        self.calls: list[tuple[str, list]] = []
        self._existing_row = existing_row

    def execute(self, sql, params=None):
        self.calls.append((sql, params))
        return _FakeCursor(self._existing_row)


def _patched_connection(con):
    @contextmanager
    def _cm():
        yield con

    return _cm


def _record(**overrides) -> TradeDiffRecord:
    base = dict(
        market="jp",
        symbol="7203",
        predicted_at="2026-09-23T00:00:00",
        side=1,
        signal_price=1000.0,
        mode="paper",
        order_id="ord-1",
    )
    base.update(overrides)
    return TradeDiffRecord(**base)  # type: ignore[arg-type]


class TestPostgresTradeDiffSink(unittest.TestCase):
    def test_implements_port(self):
        self.assertIsInstance(PostgresTradeDiffSink(), TradeDiffSink)

    def test_issues_select_delete_insert_in_order(self):
        con = _FakeConnection(existing_row=None)
        with patch(
            "src.infrastructure.persistence.trade_diff_repository.db_connection",
            _patched_connection(con),
        ):
            PostgresTradeDiffSink().record(_record())

        self.assertEqual(len(con.calls), 3)
        self.assertIn("SELECT", con.calls[0][0])
        self.assertIn("FROM paper_real_diff", con.calls[0][0])
        self.assertTrue(con.calls[1][0].startswith("DELETE FROM paper_real_diff"))
        self.assertIn("INSERT INTO paper_real_diff", con.calls[2][0])

    def test_paper_mode_fills_paper_columns_and_slippage(self):
        con = _FakeConnection(existing_row=None)
        checked = datetime(2026, 9, 23, 9, 0, 0)
        with patch(
            "src.infrastructure.persistence.trade_diff_repository.db_connection",
            _patched_connection(con),
        ):
            PostgresTradeDiffSink().record(
                _record(mode="paper", actual_price=1010.0, checked_at=checked)
            )

        insert_params = con.calls[2][1]
        # 並び: market, symbol, predicted_at, side, signal_price,
        #       paper_order_id, real_order_id, paper_price, real_price,
        #       paper_slippage, real_slippage, price_diff,
        #       paper_filled_at, real_checked_at, created_at, order_session, split_ratio
        self.assertEqual(insert_params[5], "ord-1")  # paper_order_id
        self.assertIsNone(insert_params[6])  # real_order_id
        self.assertEqual(insert_params[7], 1010.0)  # paper_price
        self.assertAlmostEqual(insert_params[9], 0.01)  # paper_slippage
        self.assertEqual(insert_params[12], checked)  # paper_filled_at
        self.assertEqual(insert_params[15], "open")  # order_session

    def test_live_mode_fills_real_columns(self):
        con = _FakeConnection(existing_row=None)
        with patch(
            "src.infrastructure.persistence.trade_diff_repository.db_connection",
            _patched_connection(con),
        ):
            PostgresTradeDiffSink().record(_record(mode="live", actual_price=990.0))

        insert_params = con.calls[2][1]
        self.assertIsNone(insert_params[5])  # paper_order_id
        self.assertEqual(insert_params[6], "ord-1")  # real_order_id
        self.assertEqual(insert_params[8], 990.0)  # real_price
        self.assertAlmostEqual(insert_params[10], -0.01)  # real_slippage

    def test_price_diff_computed_when_both_sides_present(self):
        # 既存行に paper_price=1000 があり、今回 live で 1005 を記録する
        existing = (
            1000.0,  # signal_price
            "paper-1",  # paper_order_id
            None,  # real_order_id
            1000.0,  # paper_price
            None,  # real_price
            0.0,  # paper_slippage
            None,  # real_slippage
            datetime(2026, 9, 22, 9, 0, 0),  # paper_filled_at
            None,  # real_checked_at
            datetime(2026, 9, 22, 8, 0, 0),  # created_at
            "open",  # order_session
            1.0,  # split_ratio
        )
        con = _FakeConnection(existing_row=existing)
        with patch(
            "src.infrastructure.persistence.trade_diff_repository.db_connection",
            _patched_connection(con),
        ):
            PostgresTradeDiffSink().record(_record(mode="live", actual_price=1005.0))

        insert_params = con.calls[2][1]
        self.assertAlmostEqual(insert_params[11], 5.0)  # price_diff = real - paper

    def test_zero_signal_price_yields_none_slippage(self):
        con = _FakeConnection(existing_row=None)
        with patch(
            "src.infrastructure.persistence.trade_diff_repository.db_connection",
            _patched_connection(con),
        ):
            PostgresTradeDiffSink().record(
                _record(signal_price=0.0, mode="paper", actual_price=1010.0)
            )

        insert_params = con.calls[2][1]
        self.assertIsNone(insert_params[9])  # paper_slippage


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: テストが失敗することを確認する**

```bash
cd python
py -m pytest tests/unit/test_persistence_trade_diff_repository.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.infrastructure.persistence.trade_diff_repository'`

- [ ] **Step 3: アダプタを書く**

`python/src/infrastructure/persistence/trade_diff_repository.py` を新規作成する。SQL とマージ規則は `src/prediction/db/paper_real_diff.py` から一字一句そのまま移す。

```python
"""paper_real_diff テーブル: paper / real 約定価格の乖離追跡のアダプタ。"""

from datetime import datetime
from typing import Any

from src.domain.ports import TradeDiffSink
from src.domain.types import TradeDiffRecord
from src.utils.db import db_connection
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PostgresTradeDiffSink(TradeDiffSink):
    """paper_real_diff テーブルへ書き込む TradeDiffSink 実装。

    同一キー (market, symbol, predicted_at, side) の既存行があれば
    その値を引き継いだうえで DELETE + INSERT で置き換える。
    """

    def record(self, record: TradeDiffRecord) -> None:
        checked_at = record.checked_at or datetime.now()
        with db_connection() as con:
            row = con.execute(
                """
            SELECT signal_price, paper_order_id, real_order_id, paper_price, real_price,
                   paper_slippage, real_slippage, paper_filled_at, real_checked_at, created_at,
                   order_session, split_ratio
            FROM paper_real_diff
            WHERE market = %s AND symbol = %s AND predicted_at = %s AND side = %s
            """,
                [record.market, record.symbol, record.predicted_at, record.side],
            ).fetchone()

            merged: dict[str, Any] = {
                "signal_price": record.signal_price,
                "paper_order_id": None,
                "real_order_id": None,
                "paper_price": None,
                "real_price": None,
                "paper_slippage": None,
                "real_slippage": None,
                "paper_filled_at": None,
                "real_checked_at": None,
                "created_at": checked_at,
                "order_session": record.order_session,
                "split_ratio": record.split_ratio,
            }
            if row:
                merged.update(
                    {
                        "signal_price": (
                            float(row[0]) if row[0] is not None else record.signal_price
                        ),
                        "paper_order_id": row[1],
                        "real_order_id": row[2],
                        "paper_price": row[3],
                        "real_price": row[4],
                        "paper_slippage": row[5],
                        "real_slippage": row[6],
                        "paper_filled_at": row[7],
                        "real_checked_at": row[8],
                        "created_at": row[9] or checked_at,
                        "order_session": row[10] or record.order_session,
                        "split_ratio": row[11] if len(row) > 11 else record.split_ratio,
                    }
                )
            if record.split_ratio is not None:
                merged["split_ratio"] = record.split_ratio

            merged["signal_price"] = record.signal_price
            if record.mode == "paper":
                merged["paper_order_id"] = record.order_id
                if record.actual_price is not None:
                    merged["paper_price"] = record.actual_price
                    merged["paper_slippage"] = (
                        (record.actual_price - record.signal_price) / record.signal_price
                        if record.signal_price
                        else None
                    )
                    merged["paper_filled_at"] = checked_at
            else:
                merged["real_order_id"] = record.order_id
                if record.actual_price is not None:
                    merged["real_price"] = record.actual_price
                    merged["real_slippage"] = (
                        (record.actual_price - record.signal_price) / record.signal_price
                        if record.signal_price
                        else None
                    )
                    merged["real_checked_at"] = checked_at

            price_diff = None
            if merged["paper_price"] is not None and merged["real_price"] is not None:
                price_diff = float(merged["real_price"]) - float(merged["paper_price"])

            con.execute(
                "DELETE FROM paper_real_diff WHERE market = %s AND symbol = %s "
                "AND predicted_at = %s AND side = %s",
                [record.market, record.symbol, record.predicted_at, record.side],
            )
            con.execute(
                """
            INSERT INTO paper_real_diff (
                market, symbol, predicted_at, side, signal_price,
                paper_order_id, real_order_id, paper_price, real_price,
                paper_slippage, real_slippage, price_diff,
                paper_filled_at, real_checked_at, created_at, updated_at, order_session,
                split_ratio
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    CURRENT_TIMESTAMP, %s, %s)
            """,
                [
                    record.market,
                    record.symbol,
                    record.predicted_at,
                    record.side,
                    merged["signal_price"],
                    merged["paper_order_id"],
                    merged["real_order_id"],
                    merged["paper_price"],
                    merged["real_price"],
                    merged["paper_slippage"],
                    merged["real_slippage"],
                    price_diff,
                    merged["paper_filled_at"],
                    merged["real_checked_at"],
                    merged["created_at"],
                    merged["order_session"],
                    merged.get("split_ratio"),
                ],
            )
```

- [ ] **Step 4: テストが通ることを確認する**

```bash
cd python
py -m pytest tests/unit/test_persistence_trade_diff_repository.py -v
```

Expected: PASS（6 passed）

- [ ] **Step 5: 移設が忠実であることを機械的に確かめる**

SQL 本文が旧実装と同一であることを差分で確認する。

```bash
cd python
py - <<'EOF'
import re
old = open("src/prediction/db/paper_real_diff.py", encoding="utf-8").read()
new = open("src/infrastructure/persistence/trade_diff_repository.py", encoding="utf-8").read()
pat = re.compile(r'"""\s*\n\s*(SELECT|INSERT INTO|DELETE FROM).*?"""', re.S)
norm = lambda s: [" ".join(m.group(0).split()) for m in pat.finditer(s)]
o, n = norm(old), norm(new)
print("old SQL blocks:", len(o), "new SQL blocks:", len(n))
for a, b in zip(o, n):
    print("MATCH" if a == b else "DIFF", a[:60])
EOF
```

Expected: SELECT と INSERT の 2 ブロックがいずれも `MATCH`。`DIFF` が出たら移設に手が入っている — 直すこと。
（DELETE 文は三重引用符ではなく連結文字列のため上の正規表現には掛からない。目視で 2 行が一致していることを確認する）

- [ ] **Step 6: 旧実装がまだ生きていることを確認する（この時点では両方存在する）**

```bash
cd python
py -m pytest tests/unit/test_db_prediction.py -v -k paper_real_diff
```

Expected: PASS（旧テストはまだ旧実装を叩いている。Task 10 で移設・削除する）

- [ ] **Step 7: コミット**

```bash
git add python/src/infrastructure/persistence/trade_diff_repository.py python/tests/unit/test_persistence_trade_diff_repository.py
git commit -m "$(cat <<'EOF'
feat: paper_real_diff の Postgres アダプタを infrastructure に追加する

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01N4FK4xZ3jgfaWpZcMDv964
EOF
)"
```

---

## Task 5: `recording.py` の module-level import を外し Sink 注入にする

**Files:**
- Modify: `python/src/trading/execution/recording.py`（7 行目の import 撤去、`_sync_live_execution_diffs` と `_record_order` の署名変更）
- Test: `python/tests/unit/test_recording.py`（既存・呼び出し側を新署名に合わせる）

**Interfaces:**
- Consumes: `TradeDiffSink` / `TradeDiffRecord` / `InMemoryTradeDiffSink`
- Produces:
  - `_record_order(market, predicted_at, symbol, side, qty, signal_price, order_price, order_type, order_result, broker, mode, *, trade_diff_sink: TradeDiffSink, order_session: str = "open", split_ratio: float = 1.0, horizon: int | None = None) -> None`
  - `_sync_live_execution_diffs(broker: BrokerBase, trade_diff_sink: TradeDiffSink) -> None`

**このタスクが Phase 4b の主目的である。** `recording.py:7` の module-level `from src.utils.db import upsert_paper_real_diff` が、trading→prediction の辺を import 時に生かしている当のものである。

`trade_diff_sink` を**キーワード専用の必須引数**にする理由は、既存の位置引数（`order_session` 等）の順序を崩さずに必須性を保てるためである。`| None = None` にしてはならない。

- [ ] **Step 1: 既存テストを新署名に書き換える（この時点では失敗する）**

`python/tests/unit/test_recording.py` の import 節に追加する:

```python
from src.infrastructure.in_memory import InMemoryTradeDiffSink
```

同ファイル内の `_record_order(` 呼び出し 3 箇所（35 / 67 / 94 行付近）それぞれの引数末尾に次を追加する:

```python
            trade_diff_sink=InMemoryTradeDiffSink(),
```

さらに、Sink に実際に届くことを検証するテストをファイル末尾に追加する:

```python
class TestRecordOrderUsesInjectedSink(unittest.TestCase):
    """_record_order が注入された Sink へ記録すること（module-level import 撤去の回帰）。"""

    @patch("src.trading.execution.recording._link_paper_order_metadata")
    def test_records_into_injected_sink(self, _mock_link):
        from src.infrastructure.in_memory import InMemoryTradeDiffSink
        from src.trading.brokers.base import OrderSide, OrderType

        sink = InMemoryTradeDiffSink()
        _record_order(
            market="jp",
            predicted_at="2026-09-23T00:00:00",
            symbol="7203",
            side=OrderSide.BUY,
            qty=100,
            signal_price=1000.0,
            order_price=1000.0,
            order_type=OrderType.MARKET,
            order_result={"order_id": "ord-1", "fill_price": 1002.0},
            broker=MagicMock(),
            mode="paper",
            trade_diff_sink=sink,
        )

        self.assertEqual(len(sink.recorded), 1)
        rec = sink.recorded[0]
        self.assertEqual(rec.symbol, "7203")
        self.assertEqual(rec.mode, "paper")
        self.assertEqual(rec.order_id, "ord-1")
        self.assertEqual(rec.actual_price, 1002.0)

    def test_module_has_no_prediction_db_import(self):
        """src.utils.db からの関数 import が残っていないこと。"""
        import src.trading.execution.recording as mod

        self.assertFalse(hasattr(mod, "upsert_paper_real_diff"))
```

`unittest` / `patch` / `MagicMock` が未 import ならファイル先頭に追加する:

```python
import unittest
from unittest.mock import MagicMock, patch
```

- [ ] **Step 2: テストが失敗することを確認する**

```bash
cd python
py -m pytest tests/unit/test_recording.py -v
```

Expected: FAIL — `TypeError: _record_order() got an unexpected keyword argument 'trade_diff_sink'`

- [ ] **Step 3: `recording.py` を書き換える**

ファイル冒頭の import を差し替える。**7 行目の `from src.utils.db import upsert_paper_real_diff` を削除する**のが要点である。

変更前:
```python
from src.trading.brokers.base import BrokerBase, OrderSide, OrderType
from src.utils.db import upsert_paper_real_diff
from src.utils.db._connection import _db_connection
```

変更後:
```python
from src.domain.ports import TradeDiffSink
from src.domain.types import TradeDiffRecord
from src.trading.brokers.base import BrokerBase, OrderSide, OrderType
from src.utils.db import db_connection
```

`_link_paper_order_metadata` 内の `with _db_connection() as con:` を `with db_connection() as con:` に差し替える。

`_sync_live_execution_diffs` を次に差し替える:

```python
def _sync_live_execution_diffs(broker: BrokerBase, trade_diff_sink: TradeDiffSink) -> None:
    for order in broker.get_orders():
        order_id = str(order.get("order_id") or "")
        price = order.get("price")
        if not order_id or price in (None, ""):
            continue
        try:
            actual_price = float(price)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if actual_price <= 0:
            continue

        with db_connection() as con:
            row = con.execute(
                """
                SELECT market, symbol, predicted_at, side, signal_price
                FROM paper_real_diff
                WHERE real_order_id = %s
                """,
                [order_id],
            ).fetchone()
        if row is None:
            continue

        trade_diff_sink.record(
            TradeDiffRecord(
                market=str(row[0]),
                symbol=str(row[1]),
                predicted_at=str(row[2]),
                side=int(row[3]),
                signal_price=float(row[4] or 0.0),
                mode="live",
                order_id=order_id,
                actual_price=actual_price,
            )
        )
```

`_record_order` の署名を次に差し替える（`trade_diff_sink` をキーワード専用の必須引数として `*` の後ろに置く）:

```python
def _record_order(
    market: str,
    predicted_at: str,
    symbol: str,
    side: OrderSide,
    qty: int,
    signal_price: float,
    order_price: float,
    order_type: OrderType,
    order_result: dict[str, Any],
    broker: BrokerBase,
    mode: str,
    *,
    trade_diff_sink: TradeDiffSink,
    order_session: str = "open",
    split_ratio: float = 1.0,
    horizon: int | None = None,
) -> None:
```

同関数末尾の `upsert_paper_real_diff(...)` の呼び出しを差し替える:

```python
    trade_diff_sink.record(
        TradeDiffRecord(
            market=market,
            symbol=symbol,
            predicted_at=predicted_at,
            side=int(side),
            signal_price=signal_price,
            mode=mode,
            order_id=order_id,
            actual_price=fill_price,
            order_session=order_session,
            split_ratio=split_ratio,
        )
    )
```

- [ ] **Step 4: テストが通ることを確認する**

```bash
cd python
py -m pytest tests/unit/test_recording.py -v
```

Expected: PASS

- [ ] **Step 5: 呼び出し側がまだ壊れていることを確認する（想定内）**

```bash
cd python
py -m pytest tests/unit/test_order_execution_pipeline.py tests/unit/test_sl_tp_exits.py tests/unit/test_claude_agent.py -v
```

Expected: FAIL（`_record_order` の呼び出し側が `trade_diff_sink` を渡していないため）。Task 6 で解消する。**このタスクは単独ではコミットしない** — Task 6 とまとめて 1 コミットにする。

---

## Task 6: `runner.py` / `sl_tp.py` / `claude_agent.py` へ必須注入を貫通させる

**Files:**
- Modify: `python/src/trading/execution/runner.py`（`run_daily_orders` 署名 65-73 行、`_record_order` 呼び出し 248 / 371 / 436 / 540 行、`_check_sl_tp_exits` 呼び出し 195 行、`_sync_live_execution_diffs` 呼び出し 572 行）
- Modify: `python/src/trading/execution/sl_tp.py`（`_check_sl_tp_exits` 署名 15-21 行、`_record_order` 呼び出し 58 行）
- Modify: `python/src/trading/claude_agent.py`（`_handle_place_order` 署名 231-241 行、`_record_order` 呼び出し 322 行、`run_claude_trader` 署名 371-375 行、`_handle_place_order` 呼び出し 524 行）
- Test: `python/tests/unit/test_order_execution_pipeline.py` / `test_sl_tp_exits.py` / `test_claude_agent.py` / `test_notification_adapters.py` / `test_repository_pattern.py`

**Interfaces:**
- Consumes: Task 5 の `_record_order` / `_sync_live_execution_diffs` 新署名
- Produces:
  - `run_daily_orders(broker, *, order_run_sink: OrderRunSink, trade_diff_sink: TradeDiffSink, market="jp", mode="paper", market_data=None, notifier=None, prediction_repo=None) -> OrderExecutionStats`
  - `_check_sl_tp_exits(broker, market, mode, market_data, stats, trade_diff_sink: TradeDiffSink) -> set[str]`（返り値は SL/TP が発動した銘柄の集合。`runner.py:210` が `predictions["symbol"].isin(...)` で使う。**返り値の型は変更しない**）
  - `run_claude_trader(broker, market="jp", mode="paper", *, trade_diff_sink: TradeDiffSink) -> dict[str, Any]`
  - `_handle_place_order(symbol, side_str, reasoning, market, broker, risk, predictions_cache, mode, stats, *, trade_diff_sink: TradeDiffSink) -> dict[str, Any]`

`order_run_sink` と同様、`trade_diff_sink` も `run_daily_orders` の `*` の後ろに既定値なしで置く。`prediction_repo: PredictionResultRepository | None = None` は本計画では触らない（束③＝Phase 4c の対象）。

- [ ] **Step 1: `sl_tp.py` を書き換える**

import を差し替える:

```python
from src.domain.ports import MarketDataPort, TradeDiffSink
```

`_check_sl_tp_exits` の署名に末尾引数を追加する:

```python
def _check_sl_tp_exits(
    broker: BrokerBase,
    market: str,
    mode: str,
    market_data: MarketDataPort | None,
    stats: OrderExecutionStats,
    trade_diff_sink: TradeDiffSink,
) -> set[str]:
```

**返り値の型注記は現行のまま `set[str]` を維持すること。** 追加するのは `trade_diff_sink` 引数のみである。

58 行付近の `_record_order(` 呼び出しの引数末尾（`order_session=order_session,` の次の行）に追加:

```python
                trade_diff_sink=trade_diff_sink,
```

- [ ] **Step 2: `runner.py` を書き換える**

`src.domain.ports` からの import 行に `TradeDiffSink` を追加する。

`run_daily_orders` の署名を次に差し替える:

```python
def run_daily_orders(
    broker: BrokerBase,
    *,
    order_run_sink: OrderRunSink,
    trade_diff_sink: TradeDiffSink,
    market: str = "jp",
    mode: str = "paper",
    market_data: MarketDataPort | None = None,
    notifier: NotificationPort | None = None,
    prediction_repo: PredictionResultRepository | None = None,
) -> OrderExecutionStats:
```

docstring の Args に 1 行足す:

```
        trade_diff_sink: 約定乖離の記録先（TradeDiffSink 実装。合成ルートが必ず渡す）
```

195 行の呼び出しを差し替える:

```python
    sl_tp_triggered = _check_sl_tp_exits(broker, market, mode, market_data, stats, trade_diff_sink)
```

572 行の呼び出しを差し替える:

```python
        _sync_live_execution_diffs(broker, trade_diff_sink)
```

248 / 371 / 436 / 540 行の `_record_order(` 呼び出し 4 箇所それぞれの引数末尾に追加する（各呼び出しのインデントに合わせること）:

```python
                trade_diff_sink=trade_diff_sink,
```

4 箇所すべてに入ったことを確認する:

```bash
cd python
grep -c "trade_diff_sink=trade_diff_sink" src/trading/execution/runner.py
```

Expected: `4`

- [ ] **Step 3: `claude_agent.py` を書き換える**

`src.domain.ports` からの import に `TradeDiffSink` を追加する（該当行が無ければ新規に追加する）。

`_handle_place_order` の署名末尾にキーワード専用引数を追加:

```python
def _handle_place_order(
    symbol: str,
    side_str: str,
    reasoning: str,
    market: str,
    broker: BrokerBase,
    risk: RiskManager,
    predictions_cache: pd.DataFrame,
    mode: str,
    stats: dict[str, Any],
    *,
    trade_diff_sink: TradeDiffSink,
) -> dict[str, Any]:
```

322 行の `_record_order(` 呼び出しの引数末尾（`order_session=order_session,` の次）に追加:

```python
            trade_diff_sink=trade_diff_sink,
```

`run_claude_trader` の署名を差し替える:

```python
def run_claude_trader(
    broker: BrokerBase,
    market: str = "jp",
    mode: str = "paper",
    *,
    trade_diff_sink: TradeDiffSink,
) -> dict[str, Any]:
```

docstring の Args に 1 行足す:

```
        trade_diff_sink: 約定乖離の記録先（TradeDiffSink 実装。合成ルートが必ず渡す）
```

524 行の `_handle_place_order(` 呼び出しの引数末尾に追加:

```python
                    trade_diff_sink=trade_diff_sink,
```

- [ ] **Step 4: 既存テストの呼び出し側を新署名に合わせる**

```bash
cd python
grep -rn "run_daily_orders(" tests/ | grep -v "def "
grep -rn "_check_sl_tp_exits(\|run_claude_trader(\|_handle_place_order(" tests/ | grep -v "def "
```

ヒットした各呼び出しに `trade_diff_sink=InMemoryTradeDiffSink(),` を追加し、そのファイルの import に次を足す（既に `InMemoryOrderRunSink` を import しているファイルは同じ import 文にまとめる）:

```python
from src.infrastructure.in_memory import InMemoryTradeDiffSink
```

- [ ] **Step 5: trading 系テストが全部通ることを確認する**

```bash
cd python
py -m pytest tests/unit/test_recording.py tests/unit/test_order_execution_pipeline.py tests/unit/test_sl_tp_exits.py tests/unit/test_claude_agent.py tests/unit/test_notification_adapters.py tests/unit/test_repository_pattern.py -v
```

Expected: PASS（全件）

- [ ] **Step 6: `| None = None` を持ち込んでいないことを確認する**

```bash
cd python
grep -n "trade_diff_sink" src/trading/execution/runner.py src/trading/execution/sl_tp.py src/trading/execution/recording.py src/trading/claude_agent.py | grep "None"
```

Expected: 出力なし。1 行でも出たら任意注入になっているので直すこと。

- [ ] **Step 7: コミット（Task 5 の変更とまとめて）**

```bash
git add python/src/trading/ python/tests/unit/
git commit -m "$(cat <<'EOF'
refactor: 約定乖離の記録を TradeDiffSink ポート経由にする

recording.py の module-level `from src.utils.db import upsert_paper_real_diff`
を撤去し、trading BC が import 時に prediction BC を引き込む辺を断つ。
trade_diff_sink は必須キーワード引数とし、None フォールバックを設けない。

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01N4FK4xZ3jgfaWpZcMDv964
EOF
)"
```

---

## Task 7: `PaperBroker` の Callable 注入をポート注入に替え、合成ルートを結線する

**Files:**
- Modify: `python/src/trading/brokers/paper/paper_broker.py`（`__init__` 38-44 行、約定処理 205-220 行）
- Modify: `python/src/orchestration/jobs/daily.py`（192 / 206 行、334 / 341 行）
- Modify: `python/run_auto_trade.py`（全面：`main(argv)` への切り出しを含む）
- Modify: `python/run_claude_trader.py`（同様）
- Test: `python/tests/unit/test_paper_broker.py`（既存・偽 Callable を `InMemoryTradeDiffSink` に置換）
- Test: `python/tests/unit/test_composition_roots.py`（新規）

**Interfaces:**
- Consumes: `TradeDiffSink` / `TradeDiffRecord` / `PostgresTradeDiffSink` / Task 6 の `run_daily_orders` 新署名
- Produces:
  - `PaperBroker(market_data_port: MarketDataPort, trade_diff_sink: TradeDiffSink)` — 第 2 引数は必須
  - `run_auto_trade.build_broker(mode: str, trade_diff_sink: TradeDiffSink) -> BrokerBase`
  - `run_auto_trade.main(argv: list[str] | None = None) -> int`

**このタスクに Phase 4a で判明した新しい失敗様式への対処が入る。** DI 後は「呼び出しがあれば書かれた」が成り立たない。合成ルートが `InMemoryTradeDiffSink()` を本番側に貼り間違えても**全テストが緑のまま本番の書き込みが止まる**。`run_auto_trade.py` は現状テストが 1 本も無く、合成が `if __name__ == "__main__":` の中にあるため検証できない。これを `main(argv)` へ切り出して検証可能にする。

- [ ] **Step 1: `PaperBroker` のテストを新署名に書き換える（失敗する）**

`python/tests/unit/test_paper_broker.py` の 80-83 行付近を差し替える。

変更前:
```python
        self._mock_record_diff = MagicMock()
        ...
            record_diff=self._mock_record_diff,
```

変更後:
```python
        self._trade_diff_sink = InMemoryTradeDiffSink()
        ...
            trade_diff_sink=self._trade_diff_sink,
```

import に追加:
```python
from src.infrastructure.in_memory import InMemoryTradeDiffSink
```

173-176 行のアサーションを差し替える:

```python
        self.assertEqual(len(self._trade_diff_sink.recorded), 1)
        rec = self._trade_diff_sink.recorded[0]
        self.assertEqual(rec.market, "jp")
        self.assertEqual(rec.symbol, "7203")
        self.assertAlmostEqual(rec.actual_price, 1000.0)
        self.assertEqual(rec.mode, "paper")
```

- [ ] **Step 2: テストが失敗することを確認する**

```bash
cd python
py -m pytest tests/unit/test_paper_broker.py -v
```

Expected: FAIL — `TypeError: PaperBroker.__init__() got an unexpected keyword argument 'trade_diff_sink'`

- [ ] **Step 3: `PaperBroker` を書き換える**

`Callable` が他で使われていなければ `typing` からの import を削除する。`src.domain.ports` からの import に `TradeDiffSink` を、`src.domain.types` から `TradeDiffRecord` を追加する。

`__init__` を差し替える:

```python
    def __init__(
        self,
        market_data_port: MarketDataPort,
        trade_diff_sink: TradeDiffSink,
    ) -> None:
        self._market_data = market_data_port
        self._trade_diff_sink = trade_diff_sink
```

205-220 行付近の約定時の記録を差し替える。**`self._record_diff is not None` の条件を落とす**のが要点である（必須注入になったため常に存在する）:

```python
                if market and predicted_at and signal_price is not None:
                    self._trade_diff_sink.record(
                        TradeDiffRecord(
                            market=str(market),
                            symbol=str(symbol),
                            predicted_at=str(predicted_at),
                            side=int(side),
                            signal_price=float(signal_price),
                            mode="paper",
                            order_id=str(order_id),
                            actual_price=fill_price,
                        )
                    )
```

- [ ] **Step 4: テストが通ることを確認する**

```bash
cd python
py -m pytest tests/unit/test_paper_broker.py -v
```

Expected: PASS

- [ ] **Step 5: `run_auto_trade.py` を検証可能な形に切り出す**

先頭の import 節を次に差し替える:

```python
import argparse
import sys

from src.domain.ports import TradeDiffSink
from src.infrastructure.persistence.order_run_repository import PostgresOrderRunSink
from src.infrastructure.persistence.trade_diff_repository import PostgresTradeDiffSink
from src.orchestration.port_wiring import wire_ports
from src.utils.logger import get_logger
```

32 行目以降（`_build_broker` 定義から末尾まで）を次で全面的に置き換える:

```python
def build_broker(mode: str, trade_diff_sink: TradeDiffSink):
    """mode に応じた Broker インスタンスを返す"""
    if mode == "live":
        import os

        from src.trading.brokers.kabu.kabu_client import KabuBroker

        api_password = os.environ.get("KABU_API_PASSWORD")
        if not api_password:
            logger.critical(
                "live モードには環境変数 KABU_API_PASSWORD が必要です。"
                "kabu STATION® アプリを起動した上で設定してください。"
            )
            sys.exit(1)
        return KabuBroker(api_password=api_password)
    else:
        from src.infrastructure.yfinance_market_data_adapter import YFinanceMarketDataAdapter
        from src.trading.brokers.paper.paper_broker import PaperBroker

        return PaperBroker(
            market_data_port=YFinanceMarketDataAdapter(),
            trade_diff_sink=trade_diff_sink,
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="StockFixer 自動発注スクリプト")
    parser.add_argument(
        "--mode",
        choices=["paper", "live"],
        default="paper",
        help="paper: ペーパートレード（デフォルト）, live: 本番（kabu STATION® API）",
    )
    parser.add_argument(
        "--market",
        default="jp",
        help="対象マーケット（デフォルト: jp）",
    )
    parser.add_argument(
        "--settle",
        action="store_true",
        help="ペーパートレードの pending 注文を約定処理する",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """合成ルート。本番アダプタを構築して発注パイプラインへ注入する。"""
    args = parse_args(argv)

    try:
        trade_diff_sink = PostgresTradeDiffSink()
        broker = build_broker(args.mode, trade_diff_sink)

        if args.settle:
            if args.mode != "paper":
                logger.warning("--settle は paper モードでのみ有効です")
                return 1

            settled = broker.settle_pending_orders()
            print(f"約定処理完了: {len(settled)} 件")
            for s in settled:
                print(f"  {s['symbol']} {s['qty']}株 @ {s['fill_price']:.1f}円")
        else:
            from src.trading.execution import run_daily_orders

            stats = run_daily_orders(
                broker=broker,
                order_run_sink=PostgresOrderRunSink(),
                trade_diff_sink=trade_diff_sink,
                market=args.market,
                mode=args.mode,
            )
            print(
                f"発注完了 — 買い: {stats['buy_orders']} 売り: {stats['sell_orders']} "
                f"スキップ: {stats['skipped']} エラー: {stats['errors']}"
            )
        return 0

    except Exception as e:
        logger.critical(f"自動発注スクリプト 致命的エラー: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
```

`wire_ports()` の module-level 呼び出し（29 行目）はそのまま残す。冪等かつ DB に触れないため、テストからの import で副作用は起きない。

- [ ] **Step 6: `run_claude_trader.py` を同様に切り出す**

`_build_broker` を `build_broker(mode: str, trade_diff_sink: TradeDiffSink)` に改名し、関数内の `from src.prediction.db import upsert_paper_real_diff` を削除して `record_diff=upsert_paper_real_diff` を `trade_diff_sink=trade_diff_sink` に差し替える。`if __name__ == "__main__":` の本体を `main(argv: list[str] | None = None) -> int` へ切り出し、冒頭で `trade_diff_sink = PostgresTradeDiffSink()` を構築して `build_broker` と `run_claude_trader(...)` の両方へ渡す。ファイル先頭に次を追加する:

```python
from src.domain.ports import TradeDiffSink
from src.infrastructure.persistence.trade_diff_repository import PostgresTradeDiffSink
```

- [ ] **Step 7: `orchestration/jobs/daily.py` の合成ルート 2 箇所を結線する**

`run_daily_auto_order`（189-215 行付近）の関数内 import から `from src.utils.db import upsert_paper_real_diff` を削除し、代わりに次を追加する:

```python
    from src.infrastructure.persistence.trade_diff_repository import PostgresTradeDiffSink
```

同関数を次のように書き換える（`trade_diff_sink` を 1 度だけ作り、broker と `run_daily_orders` の両方へ渡す）:

```python
    trade_diff_sink = PostgresTradeDiffSink()

    if mode == "live":
        from src.trading.brokers.kabu.kabu_client import KabuBroker

        broker = KabuBroker()
        market_data = None
    else:
        market_data = YFinanceMarketDataAdapter()
        broker = PaperBroker(
            market_data_port=market_data,
            trade_diff_sink=trade_diff_sink,
        )

    try:
        stats = run_daily_orders(
            broker=broker,
            order_run_sink=PostgresOrderRunSink(),
            trade_diff_sink=trade_diff_sink,
            market="jp",
            mode=mode,
            market_data=market_data,
```

（以降の引数はそのまま残す）

約定処理側（325-345 行付近）の関数内 import から `from src.prediction.db import upsert_paper_real_diff` を削除し、次に差し替える:

```python
    from src.infrastructure.persistence.trade_diff_repository import PostgresTradeDiffSink
    from src.infrastructure.yfinance_market_data_adapter import YFinanceMarketDataAdapter
    from src.trading.brokers.paper.paper_broker import PaperBroker

    logger.info("=== ペーパートレード約定処理開始 ===")
    try:
        broker = PaperBroker(
            market_data_port=YFinanceMarketDataAdapter(),
            trade_diff_sink=PostgresTradeDiffSink(),
        )
```

- [ ] **Step 8: 合成ルート検証テストを書く**

`python/tests/unit/test_composition_roots.py` を新規作成する。

```python
"""合成ルートが本物のアダプタを注入していることの検証。

Phase 4a で生まれた失敗様式への対処。DI 後は「呼び出しがあれば書かれた」が
成り立たず、合成ルートが InMemory 実装を本番側へ貼り間違えても
全テストが緑のまま本番の書き込みだけが止まる。合成ルートそのものを検証する。
"""

import unittest
from unittest.mock import MagicMock, patch

from src.infrastructure.persistence.order_run_repository import PostgresOrderRunSink
from src.infrastructure.persistence.trade_diff_repository import PostgresTradeDiffSink


class TestRunAutoTradeCompositionRoot(unittest.TestCase):
    def test_paper_broker_gets_the_injected_sink(self):
        import run_auto_trade

        sink = PostgresTradeDiffSink()
        broker = run_auto_trade.build_broker("paper", sink)

        self.assertIs(broker._trade_diff_sink, sink)

    def test_main_injects_postgres_adapters(self):
        import run_auto_trade

        captured: dict = {}

        def _capture(**kwargs):
            captured.update(kwargs)
            return {"buy_orders": 0, "sell_orders": 0, "skipped": 0, "errors": 0}

        with patch("src.trading.execution.run_daily_orders", _capture), patch.object(
            run_auto_trade, "build_broker", return_value=MagicMock()
        ):
            rc = run_auto_trade.main(["--mode", "paper", "--market", "jp"])

        self.assertEqual(rc, 0)
        self.assertIsInstance(captured["order_run_sink"], PostgresOrderRunSink)
        self.assertIsInstance(captured["trade_diff_sink"], PostgresTradeDiffSink)

    def test_broker_and_pipeline_share_one_sink(self):
        """broker に渡した Sink と run_daily_orders に渡す Sink が同一実体であること。"""
        import run_auto_trade

        captured: dict = {}
        seen_sinks: list = []

        def _capture(**kwargs):
            captured.update(kwargs)
            return {"buy_orders": 0, "sell_orders": 0, "skipped": 0, "errors": 0}

        def _build(mode, trade_diff_sink):
            seen_sinks.append(trade_diff_sink)
            return MagicMock()

        with patch("src.trading.execution.run_daily_orders", _capture), patch.object(
            run_auto_trade, "build_broker", _build
        ):
            run_auto_trade.main(["--mode", "paper"])

        self.assertIs(seen_sinks[0], captured["trade_diff_sink"])


class TestDailyJobCompositionRoot(unittest.TestCase):
    def test_run_daily_auto_order_injects_postgres_adapters(self):
        from src.orchestration.jobs import daily

        captured: dict = {}

        def _capture(**kwargs):
            captured.update(kwargs)
            return {
                "buy_orders": 0,
                "sell_orders": 0,
                "short_orders": 0,
                "skipped": 0,
                "skipped_min_change": 0,
                "errors": 0,
                "trading_stopped": False,
                "total_turnover": 0.0,
            }

        with patch("src.trading.execution.run_daily_orders", _capture), patch(
            "src.infrastructure.yfinance_market_data_adapter.YFinanceMarketDataAdapter"
        ):
            try:
                daily.run_daily_auto_order()
            except Exception:
                pass  # Discord 通知など後続処理の失敗はここでは問わない

        self.assertIsInstance(captured.get("order_run_sink"), PostgresOrderRunSink)
        self.assertIsInstance(captured.get("trade_diff_sink"), PostgresTradeDiffSink)


if __name__ == "__main__":
    unittest.main()
```

`run_daily_orders` を `src.trading.execution` の名前空間で patch しているのは、`run_auto_trade.main()` と `daily.run_daily_auto_order()` がどちらも関数内で `from src.trading.execution import run_daily_orders` しているためである。呼び出し側のモジュール属性ではなく定義元を差し替える必要がある。

- [ ] **Step 9: テストが通ることを確認する**

```bash
cd python
py -m pytest tests/unit/test_composition_roots.py -v
```

Expected: PASS（4 passed）

`ModuleNotFoundError: No module named 'run_auto_trade'` が出る場合は、`python/` が `sys.path` に入っていない。`python/tests/unit/conftest.py`（無ければ新規作成）に次を足す:

```python
import sys
from pathlib import Path

_PYTHON_ROOT = Path(__file__).resolve().parents[2]
if str(_PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(_PYTHON_ROOT))
```

- [ ] **Step 10: わざと壊して、このテストが本当に貼り間違いを捕まえることを確かめる**

`python/run_auto_trade.py` の `main()` 内、`trade_diff_sink = PostgresTradeDiffSink()` を一時的に次へ差し替える:

```python
        from src.infrastructure.in_memory import InMemoryTradeDiffSink

        trade_diff_sink = InMemoryTradeDiffSink()
```

```bash
cd python
py -m pytest tests/unit/test_composition_roots.py -v
```

Expected: FAIL（`test_main_injects_postgres_adapters` が `AssertionError`）。**ここで FAIL しなければテストが無意味なので、テストのほうを直すこと。** 確認できたら元に戻し、再実行して PASS になることを確認する。

- [ ] **Step 11: trading 系と orchestration 系のテストを通す**

```bash
cd python
py -m pytest tests/unit/test_paper_broker.py tests/unit/test_scheduler_pipeline_unit.py tests/unit/test_composition_roots.py tests/unit/test_order_execution_pipeline.py -v
```

Expected: PASS（全件）

- [ ] **Step 12: `record_diff` が 1 箇所も残っていないことを確認する**

```bash
cd python
grep -rn "record_diff" --include=*.py src/ tests/ run_*.py
```

Expected: 出力なし

- [ ] **Step 13: コミット**

```bash
git add python/src/trading/brokers/paper/paper_broker.py python/src/orchestration/jobs/daily.py python/run_auto_trade.py python/run_claude_trader.py python/tests/unit/
git commit -m "$(cat <<'EOF'
refactor: PaperBroker の Callable 注入を TradeDiffSink に替え合成ルートを結線する

run_auto_trade.py の合成を main(argv) へ切り出し、本物のアダプタが
注入されていることを test_composition_roots.py で検証する。

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01N4FK4xZ3jgfaWpZcMDv964
EOF
)"
```

---

## Task 8: 読み取りポート `AnalyticsQuery` とアダプタ 2 本

**Files:**
- Modify: `python/src/domain/ports.py`（`TradeDiffSink` の直後に追加）
- Create: `python/src/infrastructure/persistence/analytics_query.py`
- Modify: `python/src/infrastructure/in_memory.py`（`InMemoryTradeDiffSink` の直後に追加）
- Test: `python/tests/unit/test_analytics_query.py`（新規）
- Reference: `python/src/prediction/db/paper_real_diff.py:133-170`（`load_paper_real_diff_summary`）

**Interfaces:**
- Consumes: `db_connection`（Task 3）
- Produces:
  - `src.domain.ports.AnalyticsQuery` — ABC。`paper_real_diff_summary(self, recent_days: int = 7) -> dict`
  - `src.infrastructure.persistence.analytics_query.PostgresAnalyticsQuery` — 引数なしで構築可
  - `src.infrastructure.in_memory.InMemoryAnalyticsQuery` — `__init__(self, paper_real_diff: dict | None = None)`。テストが返り値を仕込める

既定値 `recent_days: int = 7` は旧関数の既定値をそのまま写したものである（`dashboard.py` は 30、`kpi.py` は 30 を明示的に渡すため、既定値の差で挙動が変わる箇所は無い）。

- [ ] **Step 1: 失敗するテストを書く**

`python/tests/unit/test_analytics_query.py` を新規作成する。

```python
"""ユニットテスト: AnalyticsQuery ポートと 2 実装。"""

import unittest
from contextlib import contextmanager
from unittest.mock import patch

from src.domain.ports import AnalyticsQuery
from src.infrastructure.in_memory import InMemoryAnalyticsQuery
from src.infrastructure.persistence.analytics_query import PostgresAnalyticsQuery

_EMPTY = {
    "tracked_count": 0,
    "comparable_count": 0,
    "avg_paper_slippage": 0.0,
    "avg_real_slippage": 0.0,
    "avg_abs_price_diff": 0.0,
    "avg_abs_diff_ratio": 0.0,
    "max_abs_price_diff": 0.0,
}


class _FakeCursor:
    def __init__(self, row):
        self._row = row

    def fetchone(self):
        return self._row


class _FakeConnection:
    def __init__(self, row):
        self.calls: list[tuple[str, list]] = []
        self._row = row

    def execute(self, sql, params=None):
        self.calls.append((sql, params))
        return _FakeCursor(self._row)


def _patched_connection(con):
    @contextmanager
    def _cm():
        yield con

    return _cm


class TestPortContract(unittest.TestCase):
    def test_port_cannot_be_instantiated(self):
        with self.assertRaises(TypeError):
            AnalyticsQuery()  # type: ignore[abstract]

    def test_both_implementations_satisfy_port(self):
        self.assertIsInstance(InMemoryAnalyticsQuery(), AnalyticsQuery)
        self.assertIsInstance(PostgresAnalyticsQuery(), AnalyticsQuery)


class TestInMemoryAnalyticsQuery(unittest.TestCase):
    def test_defaults_to_zeros(self):
        self.assertEqual(InMemoryAnalyticsQuery().paper_real_diff_summary(), _EMPTY)

    def test_returns_seeded_value(self):
        seeded = dict(_EMPTY, tracked_count=5, avg_paper_slippage=0.01)
        q = InMemoryAnalyticsQuery(paper_real_diff=seeded)
        self.assertEqual(q.paper_real_diff_summary(recent_days=30), seeded)


class TestPostgresAnalyticsQuery(unittest.TestCase):
    def test_maps_row_to_summary_dict(self):
        row = (10, 7, 0.001, 0.002, 3.5, 0.0035, 9.0)
        con = _FakeConnection(row)
        with patch(
            "src.infrastructure.persistence.analytics_query.db_connection",
            _patched_connection(con),
        ):
            result = PostgresAnalyticsQuery().paper_real_diff_summary(recent_days=7)

        self.assertEqual(result["tracked_count"], 10)
        self.assertEqual(result["comparable_count"], 7)
        self.assertAlmostEqual(result["avg_paper_slippage"], 0.001)
        self.assertAlmostEqual(result["avg_real_slippage"], 0.002)
        self.assertAlmostEqual(result["avg_abs_price_diff"], 3.5)
        self.assertAlmostEqual(result["avg_abs_diff_ratio"], 0.0035)
        self.assertAlmostEqual(result["max_abs_price_diff"], 9.0)

    def test_null_row_becomes_zeros(self):
        con = _FakeConnection((None, None, None, None, None, None, None))
        with patch(
            "src.infrastructure.persistence.analytics_query.db_connection",
            _patched_connection(con),
        ):
            result = PostgresAnalyticsQuery().paper_real_diff_summary()

        self.assertEqual(result, _EMPTY)

    def test_queries_paper_real_diff_table(self):
        con = _FakeConnection((0, 0, None, None, None, None, None))
        with patch(
            "src.infrastructure.persistence.analytics_query.db_connection",
            _patched_connection(con),
        ):
            PostgresAnalyticsQuery().paper_real_diff_summary(recent_days=14)

        sql, params = con.calls[0]
        self.assertIn("FROM paper_real_diff", sql)
        self.assertIn("FILTER", sql)
        self.assertEqual(len(params), 1)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: テストが失敗することを確認する**

```bash
cd python
py -m pytest tests/unit/test_analytics_query.py -v
```

Expected: FAIL — `ImportError: cannot import name 'AnalyticsQuery' from 'src.domain.ports'`

- [ ] **Step 3: ポートを定義する**

`python/src/domain/ports.py` の `TradeDiffSink` の直後に追加する:

```python


class AnalyticsQuery(ABC):
    """reporting BC 向けの読み取り専用問い合わせポート。

    reporting は整形屋であり、データを自分で取りに行かない。最外周の入口
    （run_*.py / orchestration / Discord Bot / api）がこのポートを構築して渡す。

    メソッド数が 10 を超えた場合はファサード化の兆候とみなし、設計を見直すこと。
    Phase 4c で drift_summary / prediction_accuracy / weekly_accuracy_snapshots が加わる。
    """

    @abstractmethod
    def paper_real_diff_summary(self, recent_days: int = 7) -> dict:
        """直近期間の paper / real 乖離サマリーを返す"""
```

- [ ] **Step 4: Postgres アダプタを書く**

`python/src/infrastructure/persistence/analytics_query.py` を新規作成する。SQL は `load_paper_real_diff_summary` からそのまま移す。

```python
"""reporting 向けの読み取り専用クエリのアダプタ。"""

from datetime import datetime, timedelta

from src.domain.ports import AnalyticsQuery
from src.utils.db import db_connection
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PostgresAnalyticsQuery(AnalyticsQuery):
    """Postgres から reporting 向けの集計を読む AnalyticsQuery 実装。"""

    def paper_real_diff_summary(self, recent_days: int = 7) -> dict:
        since = datetime.now() - timedelta(days=recent_days)
        with db_connection() as con:
            row = con.execute(
                """
            SELECT
                COUNT(*) AS tracked_count,
                COUNT(*) FILTER (
                    WHERE paper_price IS NOT NULL AND real_price IS NOT NULL
                ) AS comparable_count,
                AVG(paper_slippage) AS avg_paper_slippage,
                AVG(real_slippage) AS avg_real_slippage,
                AVG(ABS(price_diff)) AS avg_abs_price_diff,
                AVG(ABS(price_diff / NULLIF(signal_price, 0))) AS avg_abs_diff_ratio,
                MAX(ABS(price_diff)) AS max_abs_price_diff
            FROM paper_real_diff
            WHERE COALESCE(paper_filled_at, real_checked_at, created_at) >= %s
            """,
                [since],
            ).fetchone()

        return {
            "tracked_count": int(row[0] or 0),
            "comparable_count": int(row[1] or 0),
            "avg_paper_slippage": float(row[2] or 0.0),
            "avg_real_slippage": float(row[3] or 0.0),
            "avg_abs_price_diff": float(row[4] or 0.0),
            "avg_abs_diff_ratio": float(row[5] or 0.0),
            "max_abs_price_diff": float(row[6] or 0.0),
        }
```

- [ ] **Step 5: インメモリ実装を書く**

`python/src/infrastructure/in_memory.py` の ports import に `AnalyticsQuery` を追加し（アルファベット順で `AlertLevel` の次）、ファイル末尾に追加する:

```python


class InMemoryAnalyticsQuery(AnalyticsQuery):
    """インメモリ読み取りクエリ（テスト用）。

    コンストラクタで返り値を仕込む。仕込まない場合はゼロ値を返す。
    """

    _EMPTY_PAPER_REAL_DIFF: dict = {
        "tracked_count": 0,
        "comparable_count": 0,
        "avg_paper_slippage": 0.0,
        "avg_real_slippage": 0.0,
        "avg_abs_price_diff": 0.0,
        "avg_abs_diff_ratio": 0.0,
        "max_abs_price_diff": 0.0,
    }

    def __init__(self, paper_real_diff: Optional[dict] = None) -> None:
        self._paper_real_diff = (
            paper_real_diff
            if paper_real_diff is not None
            else dict(self._EMPTY_PAPER_REAL_DIFF)
        )

    def paper_real_diff_summary(self, recent_days: int = 7) -> dict:
        return self._paper_real_diff
```

- [ ] **Step 6: テストが通ることを確認する**

```bash
cd python
py -m pytest tests/unit/test_analytics_query.py -v
```

Expected: PASS（7 passed）

- [ ] **Step 7: SQL の移設が忠実であることを確かめる**

`src/prediction/db/paper_real_diff.py` の `load_paper_real_diff_summary` の SELECT 文と、新アダプタの SELECT 文を並べて目視で突き合わせる。空白の正規化後に一致することを次で機械的に確認する:

```bash
cd python
py - <<'EOF'
import re
old = open("src/prediction/db/paper_real_diff.py", encoding="utf-8").read()
new = open("src/infrastructure/persistence/analytics_query.py", encoding="utf-8").read()
pat = re.compile(r'"""\s*\n\s*SELECT.*?"""', re.S)
o = [" ".join(m.group(0).split()) for m in pat.finditer(old)]
n = [" ".join(m.group(0).split()) for m in pat.finditer(new)]
target = [s for s in o if "tracked_count" in s]
print("MATCH" if target and target[0] == n[0] else "DIFF")
EOF
```

Expected: `MATCH`

- [ ] **Step 8: コミット**

```bash
git add python/src/domain/ports.py python/src/infrastructure/ python/tests/unit/test_analytics_query.py
git commit -m "$(cat <<'EOF'
feat: AnalyticsQuery 読み取りポートと Postgres/インメモリ実装を追加する

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01N4FK4xZ3jgfaWpZcMDv964
EOF
)"
```

---

## Task 9: reporting の読み取りを入口へ押し上げる

**Files:**
- Modify: `python/src/reporting/kpi.py`（15 行 import、43-52 行 `get_monthly_kpis`、66-82 行の 2 関数）
- Modify: `python/src/reporting/monthly.py`（20 行 import、62 行 `run_monthly_report`、91 / 141 行の `get_monthly_kpis()` 呼び出し）
- Modify: `python/src/reporting/dashboard.py`（21-22 行 import、46 行、104-115 行 `_section_paper_real_diff`、197 行 `run_dashboard`、255 行）
- Modify: `python/src/reporting/query_service.py`（`get_monthly_report_summary` 184-188 行）
- Modify: `python/src/api/external_v1.py`（188-190 行）/ `python/src/reporting/discord/discord_bot.py`（17 / 293 行）
- Modify: `python/src/reporting/discord/notifications_report.py`（27-31 行の署名、51-55 行の関数内 import、136-137 行の None フォールバック）
- Modify: `python/run_dashboard.py` / `python/run_monthly_report.py`
- Modify: `python/src/orchestration/jobs/periodic.py`（61-68 行）/ `python/src/orchestration/jobs/weekly.py`（153-175 行）
- Test: `python/tests/unit/test_dashboard.py` / `test_monthly_report_pipeline.py` / `test_scheduler_pipeline_unit.py` / `test_discord_utils.py`（565 / 876 / 886 行の `send_weekly_report` 呼び出し）

**Interfaces:**
- Consumes: `AnalyticsQuery` / `PostgresAnalyticsQuery` / `InMemoryAnalyticsQuery`（Task 8）
- Produces:
  - `get_monthly_kpis(days: int = _REPORT_DAYS, *, diff_summary: dict) -> MonthlyKPI`
  - `run_monthly_report(target_month: Optional[str] = None, *, analytics: AnalyticsQuery) -> MonthlyReportSummary`
  - `save_monthly_report_to_file(..., *, analytics: AnalyticsQuery)`（既存引数はそのまま）
  - `run_dashboard(recent_days: int = 30, drift_n: int = 20, *, analytics: AnalyticsQuery) -> None`
  - `_section_paper_real_diff(diff: dict) -> list[list]`
  - `send_weekly_report(accuracy_df, horizon: int = 1, *, diff_summary: dict, llm_review: Optional[str] = None) -> bool`（`diff_summary` がキーワード専用の必須引数になる）
  - `get_monthly_report_summary(target_month: str | None = None, *, analytics: AnalyticsQuery) -> MonthlyReportSummary`（`query_service.py`）

設計書 §6.2 の線引きに従う。`dashboard` / `query_service` / `monthly` は入口ゆえポートを持ち、`kpi` と `discord/notifications_report` は内側ゆえデータを受け取る。`drift_summary` / `prediction_accuracy` は束②（Phase 4c）のため、`kpi.py` の `_compute_hit_rate` / `_compute_drift_count` は**本計画では変更しない**。`kpi.py` の `from src.utils.db import ...` の行は `load_drift_summary, load_prediction_accuracy` のみが残る。

- [ ] **Step 1: `kpi.py` を書き換える（データを受け取る形へ）**

15 行の import を差し替える:

```python
from src.utils.db import load_drift_summary, load_prediction_accuracy
```

`_compute_avg_slippage`（66-74 行）と `_load_diff_summary`（76-82 行）の 2 関数を**削除**し、`get_monthly_kpis` を差し替える:

```python
def get_monthly_kpis(days: int = _REPORT_DAYS, *, diff_summary: dict) -> MonthlyKPI:
    """hit_rate / avg_slippage / drift_count を集約して返す。

    diff_summary は入口が AnalyticsQuery から取得して渡す（自分では読みに行かない）。
    """
    avg_slippage = diff_summary.get("avg_paper_slippage")
    return MonthlyKPI(
        hit_rate=_compute_hit_rate(days),
        avg_slippage=float(avg_slippage) if avg_slippage is not None else None,
        drift_count=_compute_drift_count(days),
        diff_summary=diff_summary,
    )
```

`_EMPTY_DIFF` は入口側（`monthly.py`）の例外時フォールバックで使うため残す。

- [ ] **Step 2: `monthly.py` を書き換える（入口としてポートを持つ）**

import に追加:

```python
from src.domain.ports import AnalyticsQuery
from src.reporting.kpi import _EMPTY_DIFF, get_monthly_kpis
```

`run_monthly_report` の署名を差し替える:

```python
def run_monthly_report(
    target_month: Optional[str] = None,
    *,
    analytics: AnalyticsQuery,
) -> MonthlyReportSummary:
```

91 行の `kpi = get_monthly_kpis()` を差し替える:

```python
    try:
        diff_summary = analytics.paper_real_diff_summary(recent_days=30)
    except Exception as e:
        logger.error(f"diff_summary 取得失敗: {e}", exc_info=True)
        diff_summary = dict(_EMPTY_DIFF)
    kpi = get_monthly_kpis(diff_summary=diff_summary)
```

`save_monthly_report_to_file`（114 行）にも `*, analytics: AnalyticsQuery` を追加し、141 行の `_kpi = get_monthly_kpis()` を同じ形に差し替える。

- [ ] **Step 3: `dashboard.py` を書き換える**

21-23 行の import を差し替える:

```python
from src.domain.ports import AnalyticsQuery
from src.reporting.monthly import run_monthly_report
from src.utils.db import db_connection, load_drift_summary, load_experiment_runs
```

（`from src.utils.db._connection import _db_connection` の行は削除し、ファイル内の `_db_connection()` 呼び出しを `db_connection()` に差し替える）

`_section_monthly_kpi`（44 行）に `analytics` を貫通させる:

```python
def _section_monthly_kpi(analytics: AnalyticsQuery) -> Optional[list[list]]:
```

その中（46 行）の `summary = run_monthly_report()` を差し替える:

```python
    summary = run_monthly_report(analytics=analytics)
```

`_section_paper_real_diff` をデータ受け取りに変える:

```python
def _section_paper_real_diff(d: dict) -> list[list]:
    """paper/real 乖離サマリーを表形式の行リストで返す。

    Args:
        d: AnalyticsQuery.paper_real_diff_summary() の戻り値
    """
    return [
        ["追跡件数", str(d["tracked_count"])],
        ["比較可能件数", str(d["comparable_count"])],
        ["Paper Slippage 平均", f"{d['avg_paper_slippage'] * 100:.4f}%"],
        ["Real Slippage 平均", f"{d['avg_real_slippage'] * 100:.4f}%"],
        ["価格乖離率 平均 (abs)", f"{d['avg_abs_diff_ratio'] * 100:.4f}%"],
        ["価格乖離 最大 (abs)", f"¥{d['max_abs_price_diff']:.2f}"],
    ]
```

`run_dashboard` の署名を差し替える:

```python
def run_dashboard(
    recent_days: int = 30,
    drift_n: int = 20,
    *,
    analytics: AnalyticsQuery,
) -> None:
```

255 行の呼び出しを差し替える:

```python
        rows = _section_paper_real_diff(analytics.paper_real_diff_summary(recent_days=recent_days))
```

**218 行**の `rows = _section_monthly_kpi()` を `rows = _section_monthly_kpi(analytics)` に差し替える。

- [ ] **Step 4: `query_service.py` を書き換える**

対象の関数は `get_monthly_report_summary`（184-188 行）である。ファイル先頭に `from src.domain.ports import AnalyticsQuery` を足し、次に差し替える:

```python
def get_monthly_report_summary(
    target_month: str | None = None,
    *,
    analytics: AnalyticsQuery,
) -> MonthlyReportSummary:
    """月次KPIサマリーを取得する。Discord /monthlyreport コマンド向け。"""
    from src.reporting.monthly import run_monthly_report

    return run_monthly_report(target_month=target_month, analytics=analytics)
```

呼び出し元は 2 箇所である。いずれも `analytics=PostgresAnalyticsQuery()` を渡す形に直す。

`python/src/api/external_v1.py:188-190`:

```python
            from src.infrastructure.persistence.analytics_query import PostgresAnalyticsQuery
            from src.reporting.query_service import get_monthly_report_summary

            summary = get_monthly_report_summary(
                target_month=target_month, analytics=PostgresAnalyticsQuery()
            )
```

`python/src/reporting/discord/discord_bot.py:293`:

```python
        summary = get_monthly_report_summary(target_month, analytics=PostgresAnalyticsQuery())
```

同ファイル先頭に `from src.infrastructure.persistence.analytics_query import PostgresAnalyticsQuery` を追加する。Discord Bot は最外周の入口であるため、ここでアダプタを構築してよい。

漏れが無いことを確認する:

```bash
cd python
grep -rn "get_monthly_report_summary(" --include=*.py src/ run_*.py tests/ | grep -v "def "
```

Expected: ヒットしたすべての呼び出しに `analytics=` が付いていること。

- [ ] **Step 5: `notifications_report.py` の保険を外す**

署名を差し替える（`diff_summary` を必須にする）:

```python
def send_weekly_report(
    accuracy_df=None,
    horizon: int = 1,
    diff_summary: Optional[dict] = None,
    llm_review: Optional[str] = None,
) -> bool:
```

↓

```python
def send_weekly_report(
    accuracy_df,
    horizon: int = 1,
    *,
    diff_summary: dict,
    llm_review: Optional[str] = None,
) -> bool:
```

**`horizon` の既定値 `1` は残す。** 既存テスト `tests/unit/test_discord_utils.py:876` が `horizon` を省略して呼んでいるためで、必須化の対象は「自分で DB を読む保険」を持っていた `diff_summary` だけである。`diff_summary` はキーワード専用の必須引数にする。

これにより `tests/unit/test_discord_utils.py:886` の `send_weekly_report(accuracy_df=None)` が `TypeError` になる。同テストは「精度データが無ければ False を返す」ことを確かめているので、`diff_summary={"tracked_count": 0}` を渡す形に直す:

```python
        result = send_weekly_report(accuracy_df=None, diff_summary={"tracked_count": 0})
```

同ファイル 876 行も `diff_summary` を既に渡しているためそのままでよい。

51-55 行の関数内 import から `load_paper_real_diff_summary` を削除する:

```python
    from src.utils.db import load_drift_summary, load_weekly_accuracy_snapshots
```

136-137 行の **None フォールバックを削除する**:

```python
    if diff_summary is None:
        diff_summary = load_paper_real_diff_summary(recent_days=7)
```

↓ この 2 行を丸ごと削除する。直後の `if diff_summary.get("tracked_count", 0) > 0:` はそのまま残る。

docstring の Args も直す:

```
        accuracy_df: load_drift_summary() の戻り値 DataFrame
        diff_summary: paper/real 乖離サマリー（入口が AnalyticsQuery から取得して渡す）
```

- [ ] **Step 6: `weekly.py` の呼び出しを結線する**

153-157 行の関数内 import から `load_paper_real_diff_summary` を削除し、`PostgresAnalyticsQuery` を追加する:

```python
        from src.infrastructure.persistence.analytics_query import PostgresAnalyticsQuery
        from src.prediction.db import load_drift_summary, save_weekly_accuracy_snapshot
```

169 行を差し替える:

```python
        diff_summary = PostgresAnalyticsQuery().paper_real_diff_summary(recent_days=7)
```

- [ ] **Step 7: 最外周の入口 3 本を結線する**

`python/run_dashboard.py`:

```python
from src.infrastructure.persistence.analytics_query import PostgresAnalyticsQuery
from src.reporting.dashboard import run_dashboard
```

`run_dashboard(...)` の呼び出しに `analytics=PostgresAnalyticsQuery()` を追加する。

`python/run_monthly_report.py`:

```python
from src.infrastructure.persistence.analytics_query import PostgresAnalyticsQuery
from src.reporting.monthly import run_monthly_report
```

33 行の `run_monthly_report(target_month=args.month)` を差し替える:

```python
    summary = run_monthly_report(target_month=args.month, analytics=PostgresAnalyticsQuery())
```

`python/src/orchestration/jobs/periodic.py` の 61-68 行:

```python
    from src.infrastructure.persistence.analytics_query import PostgresAnalyticsQuery
    from src.reporting.monthly import run_monthly_report, save_monthly_report_to_file

    ...
        analytics = PostgresAnalyticsQuery()
        summary = run_monthly_report(analytics=analytics)
```

`save_monthly_report_to_file(...)` の呼び出しにも `analytics=analytics` を追加する。

- [ ] **Step 8: テストの patch を正しい流儀に直す**

`tests/unit/test_dashboard.py:125` の `@patch("src.reporting.dashboard.load_paper_real_diff_summary")` は、dashboard が自分で読まなくなったため不要になる。該当テストを `InMemoryAnalyticsQuery` を渡す形に書き換える:

```python
    def test_section_paper_real_diff_formats_percentages(self):
        rows = _section_paper_real_diff(
            {
                "tracked_count": 10,
                "comparable_count": 7,
                "avg_paper_slippage": 0.001,
                "avg_real_slippage": 0.002,
                "avg_abs_price_diff": 3.5,
                "avg_abs_diff_ratio": 0.0035,
                "max_abs_price_diff": 9.0,
            }
        )
        self.assertEqual(rows[0], ["追跡件数", "10"])
        self.assertEqual(rows[2], ["Paper Slippage 平均", "0.1000%"])
```

`run_dashboard(` を呼んでいるテストには `analytics=InMemoryAnalyticsQuery()` を渡す。

`tests/unit/test_monthly_report_pipeline.py:133-146` の `@patch("src.reporting.kpi.load_paper_real_diff_summary")` 3 箇所も不要になる。`get_monthly_kpis(diff_summary=...)` を直接呼ぶ形に書き換える:

```python
    @patch("src.reporting.kpi._compute_drift_count", return_value=0)
    @patch("src.reporting.kpi._compute_hit_rate", return_value=0.6)
    def test_avg_slippage_comes_from_injected_summary(self, _hit, _drift):
        kpi = get_monthly_kpis(diff_summary={"avg_paper_slippage": 0.0123})
        self.assertAlmostEqual(kpi.avg_slippage, 0.0123)
```

`tests/unit/test_scheduler_pipeline_unit.py:393 / 423` の `@patch("src.prediction.db.load_paper_real_diff_summary")` は、`weekly.py` が `PostgresAnalyticsQuery` を使うようになるため patch 先を差し替える:

```python
    @patch("src.infrastructure.persistence.analytics_query.PostgresAnalyticsQuery")
```

- [ ] **Step 9: reporting 系テストが全部通ることを確認する**

```bash
cd python
py -m pytest tests/unit/test_dashboard.py tests/unit/test_monthly_report_pipeline.py tests/unit/test_scheduler_pipeline_unit.py tests/unit/test_analytics_query.py -v
```

Expected: PASS（全件）

- [ ] **Step 10: 保険が本当に外れたことを確かめる**

```bash
cd python
grep -n "load_paper_real_diff_summary" src/reporting/ -r
```

Expected: 出力なし（reporting の 4 ファイルから 1 つ残らず消えていること）

```bash
cd python
grep -n "if diff_summary is None" src/reporting/discord/notifications_report.py
```

Expected: 出力なし

- [ ] **Step 11: コミット**

```bash
git add python/src/reporting/ python/src/orchestration/jobs/ python/run_dashboard.py python/run_monthly_report.py python/tests/unit/
git commit -m "$(cat <<'EOF'
refactor: reporting の paper/real 乖離読み取りを AnalyticsQuery へ押し上げる

notifications_report.py の「引数が None なら自分で DB を読む」保険を撤去し、
入口（run_dashboard / run_monthly_report / orchestration）がポートを構築して渡す形にする。

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01N4FK4xZ3jgfaWpZcMDv964
EOF
)"
```

---

## Task 10: 旧経路の撤去と VERSION 更新

**Files:**
- Delete: `python/src/prediction/db/paper_real_diff.py`
- Modify: `python/src/prediction/db/__init__.py`（26-29 行の import、49-51 行の `__all__`）
- Modify: `python/src/utils/db/__init__.py`（`_PREDICTION_DB` から 2 名を削除）
- Modify: `python/tests/unit/test_db_prediction.py`（18-31 行の import、612-716 行の paper_real_diff 系テストを削除）
- Modify: `python/VERSION`
- Test: `python/tests/unit/test_architecture_dynamic_import_guard.py`（許容リストは**まだ空にしない** — プロキシ本体は Phase 4c まで残る）

**Interfaces:**
- Consumes: Task 4 / Task 8 のアダプタ（旧関数の置き換え先が揃っていること）
- Produces: `_DbPackageProxy._PREDICTION_DB` が 15 要素（`load_paper_real_diff_summary` と `upsert_paper_real_diff` が抜けた状態）

`load_open_close_advantage_summary` は本番呼び出し元がゼロである（設計書 §6.4）。移設せずに削除する。削除前に改めて確認すること。

- [ ] **Step 1: 呼び出し元ゼロを改めて確認する**

```bash
cd python
grep -rn "load_open_close_advantage_summary" --include=*.py src/ run_*.py | grep -v "src/prediction/db/"
```

Expected: 出力なし。1 件でも出たら削除してはならない — その時点で設計書 §6.4 の前提が崩れているので、計画を止めてユーザーに報告する。

- [ ] **Step 2: 旧経路がまだ使われていないことを確認する**

```bash
cd python
grep -rn "upsert_paper_real_diff\|load_paper_real_diff_summary" --include=*.py src/ run_*.py | grep -v "src/prediction/db/\|src/utils/db/__init__.py"
```

Expected: 出力なし。残っていれば Task 5〜9 のいずれかが未完了である。

- [ ] **Step 3: テストを新実装側へ移す**

`python/tests/unit/test_db_prediction.py` から次を削除する:
- 18-19 行の `load_open_close_advantage_summary` / `load_paper_real_diff_summary` の import
- 31 行の `upsert_paper_real_diff` の import
- 612-716 行付近の `paper_real_diff` 系テストメソッド全部（`test_load_paper_real_diff_summary_returns_aggregates` / `test_load_open_close_advantage_summary_groups_by_session` / `test_load_open_close_advantage_summary_empty_when_no_fills` と、それらの前提を作る `upsert_paper_real_diff(...)` 呼び出しを含むテスト）

削除対象の範囲を先に確認する:

```bash
cd python
grep -n "def test_" tests/unit/test_db_prediction.py | sed -n '/6[0-9][0-9]:/p'
```

これらのテストが検証していた振る舞い（マージ規則・集計）は `tests/unit/test_persistence_trade_diff_repository.py` と `tests/unit/test_analytics_query.py` が引き継いでいる。**削除前に、削除するテスト 1 本ずつについて対応する新テストがあることを確認すること。** 対応が無い振る舞いがあれば、削除せず新テストに足す。

- [ ] **Step 4: 旧モジュールを削除する**

```bash
cd python
git rm src/prediction/db/paper_real_diff.py
```

`python/src/prediction/db/__init__.py` から次を削除する:
- 27-29 行の `load_open_close_advantage_summary` / `load_paper_real_diff_summary` / `upsert_paper_real_diff` の import
- 49-51 行の `__all__` の同 3 エントリ
- `from src.prediction.db.paper_real_diff import (...)` の import 文そのもの

- [ ] **Step 5: プロキシから 2 名を外す**

`python/src/utils/db/__init__.py` の `_PREDICTION_DB` から `"load_paper_real_diff_summary"` と `"upsert_paper_real_diff"` の 2 行を削除する。残りは 15 要素になる。

```bash
cd python
py -c "from src.utils.db import _DbPackageProxy; print(len(_DbPackageProxy._PREDICTION_DB))"
```

Expected: `15`

- [ ] **Step 6: ガードテストの許容リストが 1 件のままであることを確認する**

`python/tests/unit/test_architecture_dynamic_import_guard.py` の `GRANDFATHERED_DYNAMIC_IMPORTS` は `{"utils/db/__init__.py"}` のまま変更しない。プロキシ本体（`importlib.import_module` 2 箇所）は束②③が残るため Phase 4c まで存続する。

```bash
cd python
py -m pytest tests/unit/test_architecture_dynamic_import_guard.py -v
```

Expected: PASS

- [ ] **Step 7: VERSION を上げる**

`python/VERSION` の中身を `2.15.0` から `2.16.0` に書き換える。新しいポート 2 本と値オブジェクト 1 本を追加した後方互換のある機能追加のため minor。

**BOM を付けないこと。** PowerShell で書いた場合は必ず確認する:

```bash
cd python
xxd VERSION | head -1
```

Expected: `00000000: 322e 3136 2e30 0a` のように `32 2e 31 36 2e 30` で始まること。`efbb bf` で始まっていたら BOM 混入なので書き直す。

- [ ] **Step 8: コミット**

```bash
git add -A python/src/prediction/db/ python/src/utils/db/__init__.py python/tests/unit/test_db_prediction.py python/VERSION
git commit -m "$(cat <<'EOF'
refactor: paper_real_diff の旧経路を撤去し VERSION を 2.16.0 に上げる

呼び出し元ゼロの load_open_close_advantage_summary を削除し、
_DbPackageProxy._PREDICTION_DB を 17 から 15 に減らす。

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01N4FK4xZ3jgfaWpZcMDv964
EOF
)"
```

---

## Task 11: 全体検証と PR

**Files:** なし（検証のみ）

- [ ] **Step 1: 契約が緑であることを確認する**

```bash
cd python
PYTHONUTF8=1 py -c "from importlinter.cli import lint_imports_command; lint_imports_command()"
```

Expected: `Contracts: 3 kept, 0 broken.`

`py -m importlinter.cli lint-imports` は無出力 exit 0 を返すため**使わないこと**。

- [ ] **Step 2: わざと壊して契約が本当に効いていることを確かめる**

`python/src/trading/execution/recording.py` の先頭に一時的に次を足す:

```python
from src.prediction.db import prediction_results  # noqa: F401
```

Step 1 のコマンドを再実行する。

Expected: `independence` 契約が broken になる（trading → prediction）。確認できたら行を削除し、再度 Step 1 を実行して緑に戻ることを確認する。

- [ ] **Step 3: unit テスト全体を通す**

CI と同じ 2 段構成で実行する（`test_predict_unified_lightgbm_alignment.py` は xdist 下で不安定なため分離する）。

```bash
cd python
py -m pytest tests/unit/ -n 2 -v \
  --ignore=tests/unit/test_predict_unified_lightgbm_alignment.py \
  --cov=src --cov-branch --cov-report= --cov-fail-under=0
py -m pytest tests/unit/test_predict_unified_lightgbm_alignment.py -v \
  --cov=src --cov-branch --cov-append --cov-report=term-missing --cov-fail-under=80
```

Expected: 全 PASS、カバレッジ 80% 以上

- [ ] **Step 4: lint / 型チェックを通す**

```bash
cd python
black .
isort .
flake8 .
mypy src/
```

Expected: いずれもエラーなし。引数の設定は `pyproject.toml` / `.flake8` が正本であり、コマンドに直接書かないこと。

**注意:** パッケージを分割・新設すると `mypy` の override 設定が `module` 指定のままではサブモジュールに効かず、CI の Lint（ブロッキング）で初めて露出することがある。`src.infrastructure.persistence` 配下に override があれば `module.*` 形式になっているか確認する。

- [ ] **Step 5: CI 相当の一括チェックを実行する**

```powershell
cd python; .\check-ci.ps1
```

Expected: lint / mypy / pylint / import-linter / unit tests / bandit / pip-audit がすべて通る。

- [ ] **Step 6: ブランチを作って push する**

```bash
git fetch
git status
```

`develop` が進んでいれば先に取り込む。

```bash
git push -u origin refactor/trade-diff-sink-port
```

- [ ] **Step 7: PR を作る**

ベースブランチは `develop`。ブランチ名は `refactor/trade-diff-sink-port`。

PR ボディ（`validate-pr-body` が必須セクションをチェックする）:

```markdown
## version_impact
minor

## version_rationale
TradeDiffSink / AnalyticsQuery の 2 ポートと TradeDiffRecord 値オブジェクトを新設し、
paper_real_diff テーブルの所有権を prediction BC から infrastructure のアダプタへ移した。
既存の呼び出し規約は変わるが（trade_diff_sink が必須引数になる）、外部インターフェース・
DB スキーマ・SQL は一切変更していないため minor とする。

## VERSION 更新
- version_update_required: yes
- version_before: 2.15.0
- version_after: 2.16.0

## VERSION 未更新理由
該当なし
```

本文末尾に次を付ける:

```
🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01N4FK4xZ3jgfaWpZcMDv964
```

- [ ] **Step 8: マージ方法を確認する**

**squash マージではなく通常 merge を使うこと**（個別コミットを保持する方針）。Phase 4a の PR#736 は squash されており、12 コミットが 1 つに潰れている。同じことを繰り返さない。

---

## 完了条件

- [ ] `src/trading/` から `upsert_paper_real_diff` の参照が消えている（module-level import 含む）
- [ ] `src/reporting/` から `load_paper_real_diff_summary` の参照が消えている
- [ ] `src/prediction/db/paper_real_diff.py` が存在しない
- [ ] `_DbPackageProxy._PREDICTION_DB` が 15 要素になっている
- [ ] `grep -rn "record_diff" --include=*.py src/ tests/ run_*.py` が空
- [ ] `grep -n "trade_diff_sink" src/trading/**/*.py | grep None` が空（任意注入が無い）
- [ ] `notifications_report.py` の `if diff_summary is None:` フォールバックが消えている
- [ ] `tests/unit/test_composition_roots.py` が存在し、`run_auto_trade.main()` と `daily.run_daily_auto_order()` の両方で本番アダプタの注入を検証している
- [ ] `tests/unit/test_architecture_dynamic_import_guard.py` が PASS し、許容リストは `utils/db/__init__.py` の 1 件のまま
- [ ] `lint-imports` が 3 kept / 0 broken
- [ ] `python/VERSION` が `2.16.0`（BOM なし）
- [ ] unit テスト全体が緑、カバレッジ 80% 以上
- [ ] 「わざと壊す」確認を 3 箇所（Task 3 Step 7 / Task 7 Step 10 / Task 11 Step 2）で実施済み

---

## 次の計画（Phase 4c）

本計画の完了後、残りを 1 冊にまとめる。

1. **束②**（accuracy / drift / weekly snapshot）— `AnalyticsQuery` に `drift_summary` / `prediction_accuracy` / `weekly_accuracy_snapshots` の 3 メソッドを足し、`dashboard.py` / `kpi.py` / `notifications_report.py` / `query_service.py` の残りの読み取りを押し上げる
2. **束③**（`prediction_results`）— 定義済みで実装の無い `PredictionResultRepository` の Postgres アダプタを新設し、`runner.py:70` の `prediction_repo: PredictionResultRepository | None = None` を必須注入に直す。
   **注意**: `python/src/trading/pre_close_alert_service.py` も関数内 import で `src.utils.db` の `load_latest_prediction_timestamp` / `load_prediction_results` をプロキシ経由で消費している（束③のスコープ一覧には未記載）。この 2 関数を `PredictionResultRepository` 経由へ配線し直す対象に **`pre_close_alert_service.py` を追加すること**。漏らすと `_PREDICTION_DB` を空にしてプロキシを撤去した時点で `ImportError` となり、`src/orchestration/jobs/daily.py` から到達する本番の引け前アラートジョブが実行時に落ちる。
3. **後片付け** — `_PREDICTION_DB` を空にしてプロキシの `importlib.import_module` 2 箇所を撤去、`.importlinter` の `allow_indirect_imports = True` を削除、`GRANDFATHERED_DYNAMIC_IMPORTS` を空にする

`src/infrastructure` のパッケージ循環（`infrastructure`→`market_data`、`backtest`/`reporting`/`quality`→`infrastructure`）は Phase 4 の対象外であり、別 Issue として扱う。
