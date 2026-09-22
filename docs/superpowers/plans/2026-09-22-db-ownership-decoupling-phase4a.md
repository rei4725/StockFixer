# DB 所有権の疎結合化 Phase 4a 実装計画（土台 + order_run_summary パイロット）

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 契約・ガード・資料の土台を固めたうえで、`order_run_summary` テーブルをポート＋アダプタ方式で BC の外へ出し、以後のテーブルに機械的に繰り返せる型を確立する。

**Architecture:** `src/domain/ports.py` に書き込みポートを定義し、`src/infrastructure/persistence/` に Postgres アダプタを置く。BC（`src/trading`）はポート型のみを知り、実体は合成ルート（`run_*.py` / `src/orchestration`）が構築して注入する。`| None = None` の任意注入は行わず、必須引数とすることで旧経路を残さない。

**Tech Stack:** Python 3.14 / pytest (+pytest-xdist) / import-linter / psycopg (Postgres) / pandas

**Spec:** `docs/superpowers/specs/2026-09-22-db-ownership-decoupling-design.md`

## Global Constraints

- **SQL は一文字も変更しない。** 本計画の全変更は挙動不変である。本番は `auto_deploy.ps1` が `develop` を監視して自動デプロイするため、挙動変化は事故に直結する。
- **`| None = None` の任意注入を導入しない。** ポートは必須引数とし、合成ルートが必ず渡す。
- コマンドは `python/` ディレクトリから実行する。Windows では `py` を使う。
- 作業開始前に `git fetch` / `git status` / 必要なら `git pull` を行う（ベースブランチは `develop`）。
- コミットメッセージは Conventional Commits（`<type>: <subject>`、日本語本文可）。
- `python/VERSION` の更新は PR 単位で行い、本計画の完了時点で `2.15.0` とする（現在 `2.14.2`）。
- **Windows では `PYTHONUTF8=1` を常に設定しておく。** `.importlinter` には日本語コメントがあり、素のまま import-linter を走らせると `'cp932' codec can't decode byte ...` で落ちる。pre-commit フックが cp932 で落ちる場合も同じ対処。PowerShell では `$env:PYTHONUTF8 = "1"`、bash では `export PYTHONUTF8=1`。
- import-linter の実挙動は `py -c "from importlinter.cli import lint_imports_command; lint_imports_command()"` でのみ確認できる。`py -m importlinter.cli lint-imports` は無出力 exit 0 を返すため使わない。緑の場合の出力は `Contracts: 2 kept, 0 broken.` である。
- `tests/unit/` は `tests/unit/conftest.py` の autouse `_isolate_db` フィクスチャによりテスト専用 Postgres に接続する。unit テストから実 DB を触ることは既存の作法であり、禁止ではない。

## 本計画のスコープと、設計書からの変更点

設計書 §9 の PR 分割は「書き側 / 読み側」で切っていたが、`paper_real_diff.py` は書き（`upsert_paper_real_diff`）と読み（`load_paper_real_diff_summary`）が同一モジュールに同居しているため、方向で切ると 1 つのテーブルの SQL が 2 箇所に分かれる期間が生じる。**テーブル単位に切り直す。**

| PR | 内容 | 本計画 |
|---|---|---|
| PR-1 | 土台: ガード + 契約 + 資料 | ✅ Task 1〜3 |
| PR-2 | `order_run_summary`（書きのみ。読みは呼び出し元ゼロ） | ✅ Task 4〜8 |
| PR-3 | `paper_real_diff`（書き＋読み＋reporting 押し上げ） | Phase 4b（別計画） |
| PR-4 | 束②（accuracy / drift / weekly）読み＋reporting | Phase 4b（別計画） |
| PR-5 | 後片付け（プロキシ縮小・`src/prediction/db/` 整理） | Phase 4b（別計画） |

`order_run_summary` を最初のパイロットに選ぶ理由は、80 行・書き手 1 箇所・越境読み手ゼロで、**ポートの型を確立するのに必要な要素だけを含み、余計な波及がない**ためである。Phase 4b は本計画で確立した型をなぞる。

---

## Task 1: 動的 import 禁止ガード

今回の抜け道の本質は「文字列による動的 import は import-linter に見えない」ことである。契約では表現できないためテストで縛る。

ただし `src/utils/db/__init__.py` の 2 箇所は Phase 4b の PR-5 まで残るため、**ratchet 方式**で既知の違反だけを許容し、新規追加を禁じる。許容リストが空になることが Phase 4 の完了条件となる。

**Files:**
- Create: `python/tests/unit/test_architecture_dynamic_import_guard.py`

**Interfaces:**
- Consumes: なし
- Produces: `GRANDFATHERED_DYNAMIC_IMPORTS: frozenset[str]`（後続 PR がここから項目を削る）

- [ ] **Step 1: 失敗するテストを書く**

`python/tests/unit/test_architecture_dynamic_import_guard.py` を新規作成する。

```python
"""アーキテクチャガード: src/ 配下での自パッケージの動的 import を禁止する。

importlib.import_module("src....") は文字列ベースであるため import-linter が
検出できず、レイヤー契約・BC 独立性契約を迂回する抜け道になる。
実際に src/utils/db/__init__.py がこの手段で src.prediction.db を参照し、
utils(最下層) -> prediction(BC) の層逆転を隠していた。

GRANDFATHERED_DYNAMIC_IMPORTS は解消途中の既知違反のみを列挙する ratchet である。
項目を増やしてはならない。空になった時点でこの定数ごと削除してよい。
"""

import re
import unittest
from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parents[2] / "src"

# Phase 4b (PR-5) でプロキシ撤去とともに空になる予定。項目を追加しないこと。
GRANDFATHERED_DYNAMIC_IMPORTS = frozenset(
    {
        "utils/db/__init__.py",
    }
)

_PATTERN = re.compile(r"""import_module\(\s*["']src[.\"']""")


def _iter_python_files():
    for path in sorted(_SRC_ROOT.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        yield path


class TestNoDynamicSelfImport(unittest.TestCase):
    def test_no_new_dynamic_self_imports(self):
        """src/ 配下で importlib.import_module("src...") を新規に使っていないこと。"""
        offenders = []
        for path in _iter_python_files():
            rel = path.relative_to(_SRC_ROOT).as_posix()
            if rel in GRANDFATHERED_DYNAMIC_IMPORTS:
                continue
            if _PATTERN.search(path.read_text(encoding="utf-8")):
                offenders.append(rel)

        self.assertEqual(
            offenders,
            [],
            "src/ 配下で src.* の動的 import を検出した。"
            "レイヤー契約を迂回するため、静的 import かポート注入に置き換えること: "
            f"{offenders}",
        )

    def test_grandfathered_entries_still_violate(self):
        """許容リストの項目が実際にまだ違反していること（解消後の削除漏れ検出）。"""
        stale = []
        for rel in sorted(GRANDFATHERED_DYNAMIC_IMPORTS):
            path = _SRC_ROOT / rel
            if not path.exists() or not _PATTERN.search(path.read_text(encoding="utf-8")):
                stale.append(rel)

        self.assertEqual(
            stale,
            [],
            "許容リストに、もう違反していない項目が残っている。"
            f"GRANDFATHERED_DYNAMIC_IMPORTS から削除すること: {stale}",
        )
```

- [ ] **Step 2: テストを実行して通ることを確認する**

Run: `cd python; py -m pytest tests/unit/test_architecture_dynamic_import_guard.py -v -p no:cacheprovider`
Expected: 2 tests PASS（現状 `utils/db/__init__.py` のみが違反であり、それは許容リストに入っているため）

- [ ] **Step 3: ガードが本当に効くか、わざと壊して確かめる**

`python/src/utils/logger.py` の末尾に一時的に次の行を足す。

```python
_tmp = __import__("importlib").import_module("src.prediction.db")
```

Run: `cd python; py -m pytest tests/unit/test_architecture_dynamic_import_guard.py::TestNoDynamicSelfImport::test_no_new_dynamic_self_imports -v -p no:cacheprovider`
Expected: FAIL（`offenders` に `utils/logger.py` が現れる）

確認できたら追加した行を削除し、再実行して PASS に戻ることを確かめる。

- [ ] **Step 4: 許容リストの stale 検出も壊して確かめる**

`GRANDFATHERED_DYNAMIC_IMPORTS` に `"utils/logger.py"` を一時的に足す。

Run: `cd python; py -m pytest tests/unit/test_architecture_dynamic_import_guard.py::TestNoDynamicSelfImport::test_grandfathered_entries_still_violate -v -p no:cacheprovider`
Expected: FAIL（`stale` に `utils/logger.py` が現れる）

確認できたら元に戻し、再実行して PASS に戻ることを確かめる。

- [ ] **Step 5: コミット**

```bash
cd /c/src/StockFixer
git add python/tests/unit/test_architecture_dynamic_import_guard.py
git commit -m "test: src 配下の動的 import を禁止するアーキテクチャガードを追加する"
```

---

## Task 2: `.importlinter` に `src.domain` を最下層として追加

`src/domain/` は `src.domain.types` 以外を一切 import しない真の最下層でありながら、`layers` に列挙されていないため契約の対象外だった。最下層として明示する。

`src.infrastructure` は追加しない。`src/infrastructure/*` が `src.market_data` を import する一方で `src/backtest/*` `src/reporting/*` `src/quality/*` が `src.infrastructure` を import しており、パッケージ単位で循環しているためである。

**Files:**
- Modify: `.importlinter`

**Interfaces:**
- Consumes: なし
- Produces: `src.domain` を最下層に含む layers 契約

- [ ] **Step 1: 変更前の契約が緑であることを確認する**

Run: `cd python; py -c "from importlinter.cli import lint_imports_command; lint_imports_command()"`
Expected: 2 contracts, 2 kept, 0 broken

- [ ] **Step 2: `.importlinter` の Contract 1 を書き換える**

`layers =` ブロックの末尾に `src.domain` を追加し、`src.infrastructure` を載せない理由をコメントに残す。

```ini
# ============================================================
# Contract 1: レイヤー階層契約
#   domain < utils < BC群 < orchestration/api
#   上位レイヤーが下位レイヤーを参照するのは合法。逆は禁止。
#
# src.domain は型とポートのみを持つ共有カーネルであり、src.domain.types 以外を
# 一切 import しない真の最下層。2026-09-22 の設計判断でこれを恒久的な位置づけとした
# （docs/superpowers/specs/2026-09-22-db-ownership-decoupling-design.md §3）。
#
# src.infrastructure は意図的に layers へ載せていない。
#   src/infrastructure/* -> src.market_data（アダプタが BC を参照）
#   src/backtest/*, src/reporting/*, src/quality/* -> src.infrastructure
# とパッケージ単位で循環しているため、層に追加した時点で契約が落ちる。
# 循環の解消は別 Issue とする。
#
# ignore_imports は両契約とも未使用（例外 0 件が現状）。
# 例外を追加する際は必ずレビューを経ること。
# ============================================================
[importlinter:contract:layers]
name = Layered architecture (domain < utils < BC < orchestration/api)
type = layers
layers =
    src.api | src.orchestration
    src.backtest | src.prediction | src.trading | src.reporting | src.watchlist | src.market_data | src.rule_engine | src.quality | src.screening
    src.utils
    src.domain
```

- [ ] **Step 3: 契約が緑のままであることを確認する**

Run: `cd python; py -c "from importlinter.cli import lint_imports_command; lint_imports_command()"`
Expected: 2 contracts, 2 kept, 0 broken

- [ ] **Step 4: 契約が本当に効くか、わざと壊して確かめる**

`python/src/domain/types.py` の先頭に一時的に次の行を足す。

```python
from src.utils.logger import get_logger  # noqa: F401
```

Run: `cd python; py -c "from importlinter.cli import lint_imports_command; lint_imports_command()"`
Expected: BROKEN（`src.domain` が `src.utils` を参照している旨の違反が出る）

確認できたら追加した行を削除し、再実行して緑に戻ることを確かめる。

- [ ] **Step 5: Contract 2 のコメントを現状に合わせる**

`[importlinter:contract:independence]` 直上のコメント中、`#338 で根本解消予定` の記述を次に差し替える。`allow_indirect_imports = True` の行自体は変更しない（Phase 4b の PR-5 まで残す）。

```ini
# utils.db が prediction.db を再輸出している間接依存は Phase 4（2026-09-22 設計書）で解消中。
# 解消完了までは indirect check を無効化して direct import のみ検証する。
# allow_indirect_imports の削除は Phase 4b (PR-5) の完了条件。
```

- [ ] **Step 6: コミット**

```bash
cd /c/src/StockFixer
git add .importlinter
git commit -m "chore: layers 契約に src.domain を最下層として追加する"
```

---

## Task 3: 資料の整合

`docs/DDD_ARCHITECTURE.md`（2026-04-27, Accepted）は「`domain/types.py` はフェーズ 4 で削除する」と宣言しているが、実装は逆向きに進んでおり（`domain/types.py` が本物の定義を持ち、`prediction/types.py` が再輸出側）、`CLAUDE.md` は `domain/` を恒久的な Shared kernel と記述している。この矛盾を解消する。

**Files:**
- Modify: `docs/DDD_ARCHITECTURE.md`
- Modify: `docs/ARCHITECTURE.md`
- Modify: `python/src/utils/db/__init__.py`（docstring のみ）
- Read/確認: `.github/copilot-instructions.md`, `CLAUDE.md`

**Interfaces:**
- Consumes: `docs/superpowers/specs/2026-09-22-db-ownership-decoupling-design.md` §3 の ADR
- Produces: なし（文書のみ）

- [ ] **Step 1: `docs/DDD_ARCHITECTURE.md` の ADR-003 に Superseded を明記する**

`### ADR-003: domain/types.py は移行期間中 re-export モジュールとして維持` の見出し直下に次を挿入する。

```markdown
> **⚠️ Superseded（2026-09-22）**
> 本 ADR とフェーズ 4 のタスク 4-1 / 4-3（`domain/types.py` の re-export 削除、`domain/` フォルダ削除）は
> 廃止された。実装は逆方向に進んでおり、`src/domain/types.py` が型の正本、
> `src/prediction/types.py` が再輸出側となっている。また `src/domain/ports.py` と
> `src/infrastructure/` の導入により、`domain` はヘキサゴナルアーキテクチャの
> **恒久的な共有カーネル**として位置づけられた。
> 詳細と根拠: [DB 所有権の疎結合化 設計書 §3](superpowers/specs/2026-09-22-db-ownership-decoupling-design.md#3-資料の不整合先に解決すべき教義問題)
```

- [ ] **Step 2: フェーズ 4 のタスク表に Superseded を明記する**

`### フェーズ 4: 仕上げ（import パス統一・ドキュメント整合）` の表の直下に次を追記する。

```markdown
> **⚠️ タスク 4-1 / 4-3 は Superseded（2026-09-22）。** ADR-003 の注記を参照。
> `domain/types.py` の re-export 削除と `domain/` の削除は行わない。
```

- [ ] **Step 3: `docs/ARCHITECTURE.md` のレイヤー節を実態に更新する**

`## レイヤーアーキテクチャ`（161 行付近）の節に、現行のレイヤー図と共有カーネルの説明を反映する。既存の記述を読んだうえで、次の内容が含まれる状態にする。

```markdown
### 現行のレイヤー構造（2026-09-22 時点）

```
run_*.py                    CLI エントリポイント（引数解析のみ）
    ↓
src/api/  src/orchestration/    合成ルート: アダプタを構築し BC へ注入する
    ↓
src/backtest/ src/prediction/ src/trading/ src/reporting/
src/watchlist/ src/market_data/ src/screening/ src/rule_engine/ src/quality/
                            Bounded Context（相互参照禁止）
    ↓
src/utils/                  DB 接続・ロギング・リトライ等の技術的ユーティリティ
    ↓
src/domain/                 共有カーネル: 型（types.py）とポート（ports.py）のみ。
                            何も import しない最下層
```

`src/infrastructure/` はポートの実装アダプタ（yfinance / Discord / LLM）を置く場所である。
現時点では `src.market_data` を参照する一方で複数の BC から参照されており、パッケージ単位で
循環しているため import-linter の layers 契約には含めていない（別 Issue で解消予定）。

レイヤー契約の正本は `.importlinter` である。
```

- [ ] **Step 4: `src/utils/db/__init__.py` の docstring から亡霊を除去する**

モジュール構成の一覧に `prediction.py - prediction_results / model_metrics / prediction_accuracy テーブル操作` という行があるが、該当モジュールは既に `src/prediction/db/` へ移動しており、`src/utils/db/prediction.py` は存在しない。この 1 行を削除する。他の行と `_DbPackageProxy` のコードには手を触れない。

- [ ] **Step 5: `.github/copilot-instructions.md` と `CLAUDE.md` を確認する**

Run: `grep -n "domain\|infrastructure\|prediction/db\|レイヤー" .github/copilot-instructions.md CLAUDE.md`

レイヤー構造や `src/prediction/db/` に関する記述があり、かつ Step 3 の内容と食い違っている場合のみ修正する。食い違いがなければ変更しない（**無理に書き換えない**）。

- [ ] **Step 6: 文書に壊れたリンクがないか確認する**

Run: `cd python; py -m pytest tests/unit/ -k "doc" -v -p no:cacheprovider`

`Doc Drift (path references in CLAUDE.md / skills)` の pre-commit フックが存在するため、コミット時にも検査される。

- [ ] **Step 7: コミット**

```bash
cd /c/src/StockFixer
git add docs/DDD_ARCHITECTURE.md docs/ARCHITECTURE.md python/src/utils/db/__init__.py
git commit -m "docs: domain を恒久的な共有カーネルとする決定を資料へ反映する"
```

ここまでが **PR-1** である。`VERSION` は変更しない（PR-2 側でまとめて上げる）。

---

## Task 4: `OrderRunSummary` 値オブジェクトを domain に定義する

`save_order_run_summary` は引数を 10 個取る。そのままポートに写すと呼び出しが読めなくなるため、値オブジェクトにまとめる。

**Files:**
- Modify: `python/src/domain/types.py`
- Test: `python/tests/unit/test_domain_types_order_run_summary.py`

**Interfaces:**
- Consumes: なし
- Produces:

```python
@dataclass(frozen=True)
class OrderRunSummary:
    run_id: str
    market: str
    mode: str
    buy_orders: int
    sell_orders: int
    short_orders: int
    skipped: int
    skipped_min_change: int
    total_turnover: float
    min_change_ratio: float
```

- [ ] **Step 1: 失敗するテストを書く**

`python/tests/unit/test_domain_types_order_run_summary.py` を新規作成する。

```python
"""ユニットテスト: OrderRunSummary 値オブジェクト。"""

import dataclasses
import unittest

from src.domain.types import OrderRunSummary


class TestOrderRunSummary(unittest.TestCase):
    def _sample(self) -> OrderRunSummary:
        return OrderRunSummary(
            run_id="abc123456789",
            market="jp",
            mode="paper",
            buy_orders=3,
            sell_orders=1,
            short_orders=0,
            skipped=2,
            skipped_min_change=1,
            total_turnover=1234567.0,
            min_change_ratio=0.005,
        )

    def test_holds_all_fields(self):
        s = self._sample()
        self.assertEqual(s.run_id, "abc123456789")
        self.assertEqual(s.market, "jp")
        self.assertEqual(s.mode, "paper")
        self.assertEqual(s.buy_orders, 3)
        self.assertEqual(s.sell_orders, 1)
        self.assertEqual(s.short_orders, 0)
        self.assertEqual(s.skipped, 2)
        self.assertEqual(s.skipped_min_change, 1)
        self.assertEqual(s.total_turnover, 1234567.0)
        self.assertEqual(s.min_change_ratio, 0.005)

    def test_is_frozen(self):
        s = self._sample()
        with self.assertRaises(dataclasses.FrozenInstanceError):
            s.buy_orders = 99  # type: ignore[misc]
```

- [ ] **Step 2: テストを実行して失敗することを確認する**

Run: `cd python; py -m pytest tests/unit/test_domain_types_order_run_summary.py -v -p no:cacheprovider`
Expected: FAIL（`ImportError: cannot import name 'OrderRunSummary' from 'src.domain.types'`）

- [ ] **Step 3: `src/domain/types.py` に dataclass を追加する**

既存の dataclass 群と同じ書式で、ファイル末尾に追加する。

```python
@dataclass(frozen=True)
class OrderRunSummary:
    """発注実行 1 回分のサマリー（order_run_summary テーブルの 1 行に対応）。

    run_id: 実行ごとに採番される短縮 UUID
    mode: "paper" または "live"
    min_change_ratio: この実行で適用された最小変化率しきい値
    """

    run_id: str
    market: str
    mode: str
    buy_orders: int
    sell_orders: int
    short_orders: int
    skipped: int
    skipped_min_change: int
    total_turnover: float
    min_change_ratio: float
```

- [ ] **Step 4: テストを実行して通ることを確認する**

Run: `cd python; py -m pytest tests/unit/test_domain_types_order_run_summary.py -v -p no:cacheprovider`
Expected: 2 tests PASS

- [ ] **Step 5: コミット**

```bash
cd /c/src/StockFixer
git add python/src/domain/types.py python/tests/unit/test_domain_types_order_run_summary.py
git commit -m "feat: OrderRunSummary 値オブジェクトを domain に追加する"
```

---

## Task 5: `OrderRunSink` ポートとインメモリ実装

**Files:**
- Modify: `python/src/domain/ports.py`
- Modify: `python/src/infrastructure/in_memory.py`
- Test: `python/tests/unit/test_in_memory_order_run_sink.py`

**Interfaces:**
- Consumes: `src.domain.types.OrderRunSummary`
- Produces:
  - `src.domain.ports.OrderRunSink`（抽象。メソッド `save(self, summary: OrderRunSummary) -> None`）
  - `src.infrastructure.in_memory.InMemoryOrderRunSink`（属性 `saved: list[OrderRunSummary]`）

- [ ] **Step 1: 失敗するテストを書く**

`python/tests/unit/test_in_memory_order_run_sink.py` を新規作成する。

```python
"""ユニットテスト: InMemoryOrderRunSink（テスト用の偽アダプタ）。"""

import unittest

from src.domain.ports import OrderRunSink
from src.domain.types import OrderRunSummary
from src.infrastructure.in_memory import InMemoryOrderRunSink


def _summary(run_id: str = "run-0001") -> OrderRunSummary:
    return OrderRunSummary(
        run_id=run_id,
        market="jp",
        mode="paper",
        buy_orders=1,
        sell_orders=0,
        short_orders=0,
        skipped=0,
        skipped_min_change=0,
        total_turnover=100.0,
        min_change_ratio=0.005,
    )


class TestInMemoryOrderRunSink(unittest.TestCase):
    def test_implements_port(self):
        self.assertIsInstance(InMemoryOrderRunSink(), OrderRunSink)

    def test_records_saved_summaries_in_order(self):
        sink = InMemoryOrderRunSink()
        sink.save(_summary("run-0001"))
        sink.save(_summary("run-0002"))

        self.assertEqual([s.run_id for s in sink.saved], ["run-0001", "run-0002"])

    def test_starts_empty(self):
        self.assertEqual(InMemoryOrderRunSink().saved, [])
```

- [ ] **Step 2: テストを実行して失敗することを確認する**

Run: `cd python; py -m pytest tests/unit/test_in_memory_order_run_sink.py -v -p no:cacheprovider`
Expected: FAIL（`ImportError: cannot import name 'OrderRunSink' from 'src.domain.ports'`）

- [ ] **Step 3: ポートを定義する**

`python/src/domain/ports.py` の `PredictionResultRepository` の直後に追加する。`from src.domain.types import OrderRunSummary` をファイル先頭の import 群に追加すること。

```python
class OrderRunSink(ABC):
    """発注実行サマリーの書き込みポート。

    trading BC は自らの実行結果を記録するが、記録先（テーブル・DB）を知らない。
    実装は src/infrastructure/persistence/ に置き、合成ルートが注入する。
    """

    @abstractmethod
    def save(self, summary: OrderRunSummary) -> None:
        """発注実行サマリーを 1 件保存する"""
```

- [ ] **Step 4: インメモリ実装を追加する**

`python/src/infrastructure/in_memory.py` の `from src.domain.ports import (...)` に `OrderRunSink` を追加し、`from src.domain.types import OrderRunSummary` を追加したうえで、ファイル末尾に追加する。

```python
class InMemoryOrderRunSink(OrderRunSink):
    """インメモリ発注サマリー Sink（テスト用）"""

    def __init__(self) -> None:
        self.saved: list[OrderRunSummary] = []

    def save(self, summary: OrderRunSummary) -> None:
        self.saved.append(summary)
```

- [ ] **Step 5: テストを実行して通ることを確認する**

Run: `cd python; py -m pytest tests/unit/test_in_memory_order_run_sink.py -v -p no:cacheprovider`
Expected: 3 tests PASS

- [ ] **Step 6: 契約が緑のままであることを確認する**

Run: `cd python; py -c "from importlinter.cli import lint_imports_command; lint_imports_command()"`
Expected: 2 contracts, 2 kept, 0 broken

- [ ] **Step 7: コミット**

```bash
cd /c/src/StockFixer
git add python/src/domain/ports.py python/src/infrastructure/in_memory.py python/tests/unit/test_in_memory_order_run_sink.py
git commit -m "feat: OrderRunSink ポートとインメモリ実装を追加する"
```

---

## Task 6: Postgres アダプタ（SQL の移設）

`src/prediction/db/order_summary.py` の `save_order_run_summary` を、SQL を一文字も変えずにアダプタへ移す。

**Files:**
- Create: `python/src/infrastructure/persistence/__init__.py`
- Create: `python/src/infrastructure/persistence/order_run_repository.py`
- Test: `python/tests/unit/test_postgres_order_run_sink.py`

**Interfaces:**
- Consumes: `src.domain.ports.OrderRunSink`, `src.domain.types.OrderRunSummary`
- Produces: `src.infrastructure.persistence.order_run_repository.PostgresOrderRunSink`（引数なしで構築可能）

- [ ] **Step 1: 失敗するテストを書く**

`python/tests/unit/test_postgres_order_run_sink.py` を新規作成する。`tests/unit/conftest.py` の autouse `_isolate_db` により、テスト専用 Postgres に接続される。

```python
"""ユニットテスト: PostgresOrderRunSink（order_run_summary テーブルへの書き込み）。"""

import unittest

from src.domain.ports import OrderRunSink
from src.domain.types import OrderRunSummary
from src.infrastructure.persistence.order_run_repository import PostgresOrderRunSink
from src.utils.db._connection import _db_connection


class TestPostgresOrderRunSink(unittest.TestCase):
    def setUp(self):
        with _db_connection() as con:
            con.execute("DELETE FROM order_run_summary")

    def test_implements_port(self):
        self.assertIsInstance(PostgresOrderRunSink(), OrderRunSink)

    def test_save_inserts_one_row(self):
        PostgresOrderRunSink().save(
            OrderRunSummary(
                run_id="run-adapter-1",
                market="jp",
                mode="paper",
                buy_orders=3,
                sell_orders=1,
                short_orders=2,
                skipped=4,
                skipped_min_change=5,
                total_turnover=987654.0,
                min_change_ratio=0.005,
            )
        )

        with _db_connection() as con:
            row = con.execute(
                "SELECT market, mode, buy_orders, sell_orders, short_orders, "
                "skipped, skipped_min_change, total_turnover, min_change_ratio "
                "FROM order_run_summary WHERE run_id = %s",
                ["run-adapter-1"],
            ).fetchone()

        self.assertIsNotNone(row)
        self.assertEqual(row[0], "jp")
        self.assertEqual(row[1], "paper")
        self.assertEqual(row[2], 3)
        self.assertEqual(row[3], 1)
        self.assertEqual(row[4], 2)
        self.assertEqual(row[5], 4)
        self.assertEqual(row[6], 5)
        self.assertAlmostEqual(float(row[7]), 987654.0, places=3)
        self.assertAlmostEqual(float(row[8]), 0.005, places=6)
```

- [ ] **Step 2: テストを実行して失敗することを確認する**

Run: `cd python; py -m pytest tests/unit/test_postgres_order_run_sink.py -v -p no:cacheprovider`
Expected: FAIL（`ModuleNotFoundError: No module named 'src.infrastructure.persistence'`）

- [ ] **Step 3: パッケージを作る**

`python/src/infrastructure/persistence/__init__.py` を新規作成する。

```python
"""永続化アダプタ — src/domain/ports.py のリポジトリ系ポートの Postgres 実装。

テーブルごとにモジュールを分ける。BC はここを import せず、合成ルート
（run_*.py / src/orchestration）が構築して BC へ注入する。
"""
```

- [ ] **Step 4: アダプタを実装する**

`python/src/infrastructure/persistence/order_run_repository.py` を新規作成する。SQL とログ文言は `src/prediction/db/order_summary.py` から**一文字も変えずに**写す。

```python
"""order_run_summary テーブル: 発注実行サマリー（R-214）のアダプタ。"""

from src.domain.ports import OrderRunSink
from src.domain.types import OrderRunSummary
from src.utils.db._connection import _db_connection
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PostgresOrderRunSink(OrderRunSink):
    """order_run_summary テーブルへ書き込む OrderRunSink 実装。"""

    def save(self, summary: OrderRunSummary) -> None:
        with _db_connection() as con:
            con.execute(
                """
                INSERT INTO order_run_summary
                    (run_id, market, mode, run_at, buy_orders, sell_orders, short_orders,
                     skipped, skipped_min_change, total_turnover, min_change_ratio)
                VALUES (%s, %s, %s, CURRENT_TIMESTAMP, %s, %s, %s, %s, %s, %s, %s)
                """,
                [
                    summary.run_id,
                    summary.market,
                    summary.mode,
                    summary.buy_orders,
                    summary.sell_orders,
                    summary.short_orders,
                    summary.skipped,
                    summary.skipped_min_change,
                    summary.total_turnover,
                    summary.min_change_ratio,
                ],
            )
        logger.info(
            f"order_run_summary 保存: run_id={summary.run_id} market={summary.market} "
            f"mode={summary.mode} "
            f"buy={summary.buy_orders} sell={summary.sell_orders} short={summary.short_orders} "
            f"skipped={summary.skipped}(min_change={summary.skipped_min_change}) "
            f"turnover={summary.total_turnover:.0f}"
        )
```

- [ ] **Step 5: テストを実行して通ることを確認する**

Run: `cd python; py -m pytest tests/unit/test_postgres_order_run_sink.py -v -p no:cacheprovider`
Expected: 2 tests PASS

- [ ] **Step 6: 契約が緑のままであることを確認する**

Run: `cd python; py -c "from importlinter.cli import lint_imports_command; lint_imports_command()"`
Expected: 2 contracts, 2 kept, 0 broken

- [ ] **Step 7: コミット**

```bash
cd /c/src/StockFixer
git add python/src/infrastructure/persistence/ python/tests/unit/test_postgres_order_run_sink.py
git commit -m "feat: order_run_summary の Postgres アダプタを infrastructure に追加する"
```

---

## Task 7: `run_daily_orders` に Sink を必須注入する

`src/trading/execution/runner.py` の `save_order_run_summary` 直呼びを撤去し、`OrderRunSink` を必須引数にする。既定値は付けない（Global Constraints）。

呼び出し元は本番 2 箇所（`src/orchestration/jobs/daily.py:209`, `run_auto_trade.py:92`）とテスト約 30 箇所である。

**Files:**
- Modify: `python/src/trading/execution/runner.py:38`（import）, `:64-71`（シグネチャ）, `:571-587`（保存処理）
- Modify: `python/src/orchestration/jobs/daily.py:209`
- Modify: `run_auto_trade.py:92`
- Modify: `python/tests/unit/test_order_execution_pipeline.py`
- Modify: `python/tests/unit/test_notification_adapters.py`
- Modify: `python/tests/unit/test_repository_pattern.py`

**Interfaces:**
- Consumes: `src.domain.ports.OrderRunSink`, `src.infrastructure.in_memory.InMemoryOrderRunSink`, `src.infrastructure.persistence.order_run_repository.PostgresOrderRunSink`
- Produces: `run_daily_orders(broker, *, order_run_sink, market="jp", mode="paper", market_data=None, notifier=None, prediction_repo=None) -> OrderExecutionStats`

- [ ] **Step 1: 失敗するテストを書く**

サマリー保存（`runner.py:571` 付近）に到達するには完全経路を通す必要がある。`run_daily_orders` には早期リターンが 3 本あり（`runner.py:114` 取引ゲート停止 / `:136` 同 / `:150` 予測が空）、いずれもサマリー保存の手前で返る。

既存の `TestRunDailyOrders`（`tests/unit/test_order_execution_pipeline.py:63`）が `_patch_pipeline` / `_start_patches` / `_stop_patches` と `_make_broker` / `_make_predictions` で完全経路を通す仕掛けを持っているため、**新しいクラスを作らず、このクラスにテストメソッドを 2 つ追加する**。

まずファイル先頭の import に追加する。

```python
from src.infrastructure.in_memory import InMemoryOrderRunSink
```

次に `TestRunDailyOrders` クラスの末尾にテストを追加する。

```python
    def test_order_run_sink_receives_summary_with_stats(self):
        """注入された OrderRunSink に stats と一致するサマリーが渡されること。"""
        broker = _make_broker()
        predictions = _make_predictions(n_buy=3)
        patches = self._patch_pipeline(predictions)
        _, patch_list = self._start_patches(patches)
        sink = InMemoryOrderRunSink()
        try:
            stats = run_daily_orders(
                broker, order_run_sink=sink, market="jp", mode="paper"
            )
        finally:
            self._stop_patches(patch_list)

        self.assertEqual(len(sink.saved), 1)
        saved = sink.saved[0]
        self.assertEqual(saved.market, "jp")
        self.assertEqual(saved.mode, "paper")
        self.assertEqual(saved.buy_orders, stats["buy_orders"])
        self.assertEqual(saved.sell_orders, stats["sell_orders"])
        self.assertEqual(saved.short_orders, stats["short_orders"])
        self.assertEqual(saved.skipped, stats["skipped"])
        self.assertEqual(saved.skipped_min_change, stats["skipped_min_change"])
        self.assertEqual(saved.total_turnover, stats["total_turnover"])
        self.assertEqual(saved.min_change_ratio, MIN_CHANGE_RATIO)
        self.assertNotEqual(saved.run_id, "")

    def test_order_run_sink_is_required(self):
        """order_run_sink を渡さない呼び出しは TypeError になること。"""
        broker = _make_broker()
        with self.assertRaises(TypeError):
            run_daily_orders(broker, market="jp", mode="paper")  # type: ignore[call-arg]
```

- [ ] **Step 2: テストを実行して失敗することを確認する**

Run: `cd python; py -m pytest tests/unit/test_order_execution_pipeline.py::TestRunDailyOrders::test_order_run_sink_receives_summary_with_stats tests/unit/test_order_execution_pipeline.py::TestRunDailyOrders::test_order_run_sink_is_required -v -p no:cacheprovider`
Expected: 1 本目が FAIL（`run_daily_orders() got an unexpected keyword argument 'order_run_sink'`）、2 本目は現状でも TypeError にならないため FAIL

- [ ] **Step 3: `runner.py` のシグネチャと保存処理を変更する**

import を差し替える。

```python
# 削除
from src.utils.db import save_order_run_summary

# 追加（既存の from src.domain.ports import (...) に OrderRunSink を足す）
from src.domain.ports import (
    AlertLevel,
    MarketDataPort,
    NotificationPort,
    OrderRunSink,
    PredictionResultRepository,
)
from src.domain.types import OrderRunSummary
```

シグネチャを変更する。`order_run_sink` はキーワード専用の必須引数とする。

```python
def run_daily_orders(
    broker: BrokerBase,
    *,
    order_run_sink: OrderRunSink,
    market: str = "jp",
    mode: str = "paper",
    market_data: MarketDataPort | None = None,
    notifier: NotificationPort | None = None,
    prediction_repo: PredictionResultRepository | None = None,
) -> OrderExecutionStats:
```

保存処理を差し替える。`try` / `except` の握り潰しとログ文言はそのまま維持する。

```python
    # 発注サマリーを保存（R-214）
    _run_id = str(uuid.uuid4())[:12]
    try:
        order_run_sink.save(
            OrderRunSummary(
                run_id=_run_id,
                market=market,
                mode=mode,
                buy_orders=stats["buy_orders"],
                sell_orders=stats["sell_orders"],
                short_orders=stats["short_orders"],
                skipped=stats["skipped"],
                skipped_min_change=stats["skipped_min_change"],
                total_turnover=stats["total_turnover"],
                min_change_ratio=MIN_CHANGE_RATIO,
            )
        )
    except Exception:
        logger.error("[exec] order_run_summary 保存失敗", exc_info=True)
    return stats
```

- [ ] **Step 4: テストを実行して通ることを確認する**

`_patch_pipeline` の中に `patch("src.trading.execution.runner.save_order_run_summary")` が残っていると、Step 3 で import を消した時点で `AttributeError` になる。この patch 項目を先に削除してから実行すること（Step 6 で扱う 5 箇所のうちの 1 つ）。

Run: `cd python; py -m pytest tests/unit/test_order_execution_pipeline.py::TestRunDailyOrders -v -p no:cacheprovider`
Expected: 新規 2 本を含め全 PASS

- [ ] **Step 5: 本番の呼び出し元 2 箇所を結線する**

`python/src/orchestration/jobs/daily.py` の関数内 import 群に追加する。

```python
    from src.infrastructure.persistence.order_run_repository import PostgresOrderRunSink
```

呼び出しを差し替える。

```python
        stats = run_daily_orders(
            broker=broker,
            order_run_sink=PostgresOrderRunSink(),
            market="jp",
            mode=mode,
            market_data=market_data,
        )
```

`run_auto_trade.py` も同様に、ファイル先頭の import 群に

```python
from src.infrastructure.persistence.order_run_repository import PostgresOrderRunSink
```

を足し、呼び出しを差し替える。

```python
            stats = run_daily_orders(
                broker=broker,
                order_run_sink=PostgresOrderRunSink(),
                market=args.market,
                mode=args.mode,
            )
```

- [ ] **Step 6: 既存テストの呼び出し約 30 箇所を更新する**

`tests/unit/test_order_execution_pipeline.py` / `test_notification_adapters.py` / `test_repository_pattern.py` の `run_daily_orders(...)` 呼び出しすべてに `order_run_sink=InMemoryOrderRunSink()` を追加し、各ファイルに `from src.infrastructure.in_memory import InMemoryOrderRunSink` を import する。

あわせて `test_order_execution_pipeline.py` の 5 箇所にある

```python
            patch("src.trading.execution.runner.save_order_run_summary"),
```

を**削除する**。内訳は `TestRunDailyOrders._patch_pipeline` の中に 1 箇所、および 739 / 790 / 836 / 970 行付近の別クラスに 4 箇所（行番号は変更前のもの）。`InMemoryOrderRunSink` が注入されるため DB に触れず、patch が不要になる。

Run: `cd python; grep -n "save_order_run_summary" tests/unit/*.py`
Expected: `test_db_prediction.py` の行のみ（これは Task 8 で削除する）。`test_order_execution_pipeline.py` は出力に現れない

Run: `cd python; grep -n "run_daily_orders(" tests/unit/*.py`
Expected: すべての呼び出しに `order_run_sink=` が含まれている

- [ ] **Step 7: trading 関連のテストをすべて実行する**

Run: `cd python; py -m pytest tests/unit/test_order_execution_pipeline.py tests/unit/test_notification_adapters.py tests/unit/test_repository_pattern.py -v -p no:cacheprovider`
Expected: 全 PASS

- [ ] **Step 8: 注入が本当に効いているか、わざと壊して確かめる**

`runner.py` の `order_run_sink.save(...)` の呼び出しを一時的にコメントアウトする。

Run: `cd python; py -m pytest tests/unit/test_order_execution_pipeline.py::TestRunDailyOrders::test_order_run_sink_receives_summary_with_stats -v -p no:cacheprovider`
Expected: FAIL（`len(sink.saved) == 1` のアサーションで落ちる）

確認できたらコメントアウトを戻し、再実行して PASS に戻ることを確かめる。

- [ ] **Step 9: 契約が緑であることを確認する**

Run: `cd python; py -c "from importlinter.cli import lint_imports_command; lint_imports_command()"`
Expected: 2 contracts, 2 kept, 0 broken

> `src/trading` は `src.domain.ports` のみを参照し、`src.infrastructure` を参照しない。
> `src/orchestration` と `run_auto_trade.py` のみがアダプタを構築する。

- [ ] **Step 10: コミット**

```bash
cd /c/src/StockFixer
git add python/src/trading/execution/runner.py python/src/orchestration/jobs/daily.py python/run_auto_trade.py python/tests/unit/test_order_execution_pipeline.py python/tests/unit/test_notification_adapters.py python/tests/unit/test_repository_pattern.py
git commit -m "refactor: 発注サマリー保存を OrderRunSink ポート経由にする"
```

---

## Task 8: 旧経路の撤去と VERSION 更新

`src/prediction/db/order_summary.py` を削除し、プロキシから `save_order_run_summary` を外す。`load_turnover_comparison` は本番コードからの呼び出し元がゼロ（参照はテストのみ）であるため、テストごと削除する。

**Files:**
- Delete: `python/src/prediction/db/order_summary.py`
- Modify: `python/src/prediction/db/__init__.py`
- Modify: `python/src/utils/db/__init__.py`
- Modify: `python/tests/unit/test_db_prediction.py`
- Modify: `python/VERSION`

**Interfaces:**
- Consumes: Task 6 の `PostgresOrderRunSink`
- Produces: なし（撤去のみ）

- [ ] **Step 1: 残存参照がないことを確認する**

Run: `cd python; grep -rn "save_order_run_summary\|load_turnover_comparison" --include=*.py src/ run_*.py tests/`
Expected: `src/prediction/db/order_summary.py`、`src/prediction/db/__init__.py`、`src/utils/db/__init__.py`、`tests/unit/test_db_prediction.py` のみ。`src/trading/` と `src/orchestration/` と `run_auto_trade.py` には現れない。

現れる場合は Task 7 の Step 5〜6 が漏れている。先に直すこと。

- [ ] **Step 2: テストを削除する**

`tests/unit/test_db_prediction.py` から次を削除する。

- ファイル先頭 import の `load_turnover_comparison` と `save_order_run_summary`
- `load_turnover_comparison` を呼ぶテストクラス／テストメソッド（800〜900 行付近）

Run: `cd python; grep -n "turnover_comparison\|order_run_summary" tests/unit/test_db_prediction.py`
Expected: 出力なし

- [ ] **Step 3: モジュールと再輸出を削除する**

`python/src/prediction/db/order_summary.py` を削除する。

`python/src/prediction/db/__init__.py` から次の行を削除する。

```python
from .order_summary import load_turnover_comparison, save_order_run_summary  # noqa: F401
```

`__all__` から次の 2 要素を削除する。

```python
    # order_run_summary
    "save_order_run_summary",
    "load_turnover_comparison",
```

`python/src/utils/db/__init__.py` の `_DbPackageProxy._PREDICTION_DB` から次の 1 行を削除する。

```python
            "save_order_run_summary",
```

- [ ] **Step 4: 削除後のテストを実行する**

Run: `cd python; py -m pytest tests/unit/test_db_prediction.py -v -p no:cacheprovider`
Expected: 全 PASS（削除したテストが消え、他は影響を受けない）

- [ ] **Step 5: unit テスト全体を実行する**

Run: `cd python; py -m pytest tests/unit/ -n 2 -q --ignore=tests/unit/test_predict_unified_lightgbm_alignment.py -p no:cacheprovider`
Expected: 全 PASS

Run: `cd python; py -m pytest tests/unit/test_predict_unified_lightgbm_alignment.py -q -p no:cacheprovider`
Expected: PASS（このテストは xdist 下で不安定になるため単独・直列で実行する）

- [ ] **Step 6: lint と型検査を通す**

```bash
cd python
py -m black .
py -m isort .
py -m flake8 .
py -m mypy src/
py -c "from importlinter.cli import lint_imports_command; lint_imports_command()"
```

Expected: いずれもエラーなし、契約は 2 kept / 0 broken

- [ ] **Step 7: VERSION を更新する**

`python/VERSION` を `2.14.2` から `2.15.0` にする。

> **注意:** PowerShell 5.1 の `Set-Content -Encoding utf8` は BOM を付ける。VERSION に BOM が
> 混入すると本番デプロイ時に壊れる。エディタか `py` で書き、書いたあと
> `py -c "print(open('python/VERSION','rb').read())"` が `b'2.15.0\n'` であることを確認すること。

Run: `cd /c/src/StockFixer; py -c "print(open('python/VERSION','rb').read())"`
Expected: `b'2.15.0\n'`（`\xef\xbb\xbf` が先頭に無いこと）

- [ ] **Step 8: コミット**

```bash
cd /c/src/StockFixer
git add python/src/prediction/db/__init__.py python/src/utils/db/__init__.py python/tests/unit/test_db_prediction.py python/VERSION
git rm python/src/prediction/db/order_summary.py
git commit -m "refactor: order_run_summary の旧経路を撤去する"
```

---

## PR の作成

PR-1（Task 1〜3）と PR-2（Task 4〜8）は別ブランチ・別 PR とする。ベースブランチはいずれも `develop`。

**PR-1:** ブランチ `chore/architecture-guards-and-docs`

```markdown
## version_impact
none

## version_rationale
アーキテクチャガードの追加と資料整合のみで、本番の挙動およびパッケージの公開 API に変更はない。

## VERSION 更新
- version_update_required: no
- version_before: 2.14.2
- version_after: 2.14.2

## VERSION 未更新理由
version_impact が none のため更新しない。
```

**PR-2:** ブランチ `refactor/order-run-sink-port`

```markdown
## version_impact
minor

## version_rationale
発注サマリーの保存経路をポート注入方式へ変更し、`run_daily_orders` のシグネチャに必須引数 `order_run_sink` を追加した。外部挙動は不変だがモジュール境界と内部 API が変わるため minor とする。

## VERSION 更新
- version_update_required: yes
- version_before: 2.14.2
- version_after: 2.15.0

## VERSION 未更新理由
該当なし
```

---

## 完了条件

- [ ] `src/trading/` と `run_auto_trade.py` から `save_order_run_summary` の参照が消えている
- [ ] `_DbPackageProxy._PREDICTION_DB` が 17 要素になっている（`save_order_run_summary` が抜けた状態）
- [ ] `src/prediction/db/order_summary.py` が存在しない
- [ ] `tests/unit/test_architecture_dynamic_import_guard.py` が PASS し、許容リストが `utils/db/__init__.py` の 1 件のみ
- [ ] `.importlinter` の layers 契約に `src.domain` が最下層として含まれ、lint-imports が 2 kept / 0 broken
- [ ] `docs/DDD_ARCHITECTURE.md` の ADR-003 とフェーズ 4-1 / 4-3 に Superseded が明記されている
- [ ] `python/VERSION` が `2.15.0`（BOM なし）
- [ ] unit テスト全体が緑
- [ ] 「わざと壊す」確認を Task 1 Step 3-4 / Task 2 Step 4 / Task 7 Step 8 の 4 箇所で実施済み

## 次の計画（Phase 4b）

本計画の完了後、`paper_real_diff`（書き＋読み＋reporting 押し上げ）、束②（accuracy / drift / weekly）、
後片付け（プロキシ全撤去・`src/prediction/db/` 整理・`allow_indirect_imports` 削除）を対象に
Phase 4b の計画を作成する。本計画で確立した「値オブジェクト → ポート → アダプタ → 合成ルート結線 → 旧経路撤去」
の 5 ステップをテーブルごとになぞる。
