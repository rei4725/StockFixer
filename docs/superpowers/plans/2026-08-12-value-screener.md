# 低PER・低配当性向・財務安定 バリュー・スクリーナー Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `stock_fundamentals` の最新スナップショットから低PER・低配当性向・財務安定（低D/E＋黒字）な jp 銘柄を抽出する前向きライブスクリーンを追加する。

**Architecture:** `market_data.fundamentals` に PER・配当性向の取得を追加し、`stock_fundamentals` テーブルに新カラムを追加、新規 `src/screening/value_screener.py` が独立に（トレンド候補を経由せず）財務スナップショットのみでスクリーンし、`run_value_screen.py` から呼び出す。バックテストは提供しない（PIT非対応のため設計上不可能と判明済み — 詳細は spec 参照）。

**Tech Stack:** Python, pandas, yfinance, PostgreSQL（`psycopg` 経由）, pytest/unittest

**Spec:** `docs/superpowers/specs/2026-08-12-value-screener-design.md`

## Global Constraints

- DuckDB/Postgres 書き込みは逐次（並列書き込み禁止）。本計画は既存の逐次書き込み経路（`_db_connection`）をそのまま使うため新たな考慮は不要
- `run_*.py` は CLI ラッパーのみ。引数パースとサービス呼び出し、表示・保存に徹し、ビジネスロジックを書かない
- 型は dataclass を使う。生 dict を上位レイヤーに渡さない
- ログは各モジュールで `get_logger(__name__)` を使う。`except: pass` は禁止（例外は `logger.error(..., exc_info=True)` または `logger.warning(...)` で残す）
- マイグレーションファイル名は `NNNN_description_postgres.sql`（`src/utils/db/migration_runner.py` の `_VERSION_RE = re.compile(r"^(\d{4})_(.+?)_postgres\.sql$")` に一致させること。合わないと無言でスキップされ、テーブルにカラムが増えない）
- `debt_to_equity` は **パーセントポイント単位**（yfinance実測準拠。例: 78.4 は D/E比 0.78 を意味する）。既存 `quality_gate.py` の `max_debt_to_equity=1.5` は単位が食い違った既知の別問題であり、本計画のコードはそれを踏襲しない
- screening BC は `utils.db.stock_fundamentals` のみを参照し、`market_data` BC を import しない（既存 `quality_gate.py` と同じレイヤー規約）
- レイヤー: `run_value_screen.py` → `src/screening/value_screener.py` → `src/utils/db/stock_fundamentals.py` → DB。上位は下位のみを import する

---

### Task 1: `FundamentalRecord` に PER・配当性向を追加し、取得ロジックを実装する

**Files:**
- Modify: `python/src/market_data/types.py`
- Modify: `python/src/market_data/fundamentals.py`
- Test: `python/tests/unit/test_fundamentals.py`

**Interfaces:**
- Consumes: なし（このタスクが起点）
- Produces: `FundamentalRecord.trailing_pe: Optional[float]`, `FundamentalRecord.payout_ratio: Optional[float]`。Task 2 がこの2フィールドを DB 保存対象に加える

- [ ] **Step 1: 失敗するテストを書く**

`python/tests/unit/test_fundamentals.py` の `TestFetchFundamentals.test_parse_from_info` を、`info` に `trailingPE` / `payoutRatio` を追加し、対応するアサーションを追加する形に書き換える（既存の他アサーションは変更しない）:

```python
    def test_parse_from_info(self):
        info = {
            "totalRevenue": 400.0,
            "netIncomeToCommon": 80.0,
            "trailingEps": 5.5,
            "returnOnEquity": 0.25,
            "operatingMargins": 0.25,
            "profitMargins": 0.20,
            "debtToEquity": 1.5,
            "totalCash": 123.0,
            "marketCap": 9999.0,
            "sharesOutstanding": 1000.0,
            "trailingPE": 8.48,
            "payoutRatio": 0.27,
        }
        with patch("src.market_data.fundamentals.yf.Ticker", return_value=_make_ticker(info)):
            rec = fetch_fundamentals("us", "AAPL")

        self.assertIsInstance(rec, FundamentalRecord)
        self.assertEqual(rec.market, "us")
        self.assertEqual(rec.symbol, "AAPL")
        self.assertEqual(rec.revenue, 400.0)
        self.assertEqual(rec.net_income, 80.0)
        self.assertEqual(rec.eps, 5.5)
        self.assertEqual(rec.roe, 0.25)
        self.assertEqual(rec.op_margin, 0.25)
        self.assertEqual(rec.net_margin, 0.20)
        self.assertEqual(rec.debt_to_equity, 1.5)
        self.assertEqual(rec.cash, 123.0)
        self.assertEqual(rec.market_cap, 9999.0)
        self.assertEqual(rec.shares_outstanding, 1000.0)
        self.assertEqual(rec.trailing_pe, 8.48)
        self.assertEqual(rec.payout_ratio, 0.27)
        # income_stmt（=financials）から営業利益
        self.assertEqual(rec.operating_income, 100.0)
```

さらに、`info` に `trailingPE` / `payoutRatio` が無いケース（欠損時に `None` になること）を確認する新規テストを `TestFetchFundamentals` クラス内に追加する:

```python
    def test_missing_pe_and_payout_ratio_are_none(self):
        """info に trailingPE / payoutRatio が無ければ None のまま（フォールバック計算はしない）。"""
        info = {"marketCap": 1.0}
        with patch("src.market_data.fundamentals.yf.Ticker", return_value=_make_ticker(info)):
            rec = fetch_fundamentals("us", "NOPE")
        self.assertIsNotNone(rec)
        self.assertIsNone(rec.trailing_pe)
        self.assertIsNone(rec.payout_ratio)
```

- [ ] **Step 2: テストが失敗することを確認する**

Run: `cd python && py -m pytest tests/unit/test_fundamentals.py::TestFetchFundamentals -v`
Expected: `test_parse_from_info` が `AttributeError: 'FundamentalRecord' object has no attribute 'trailing_pe'` で FAIL。`test_missing_pe_and_payout_ratio_are_none` も同様に FAIL

- [ ] **Step 3: `FundamentalRecord` にフィールドを追加する**

`python/src/market_data/types.py` の `revenue_cagr_3y: Optional[float] = None`（54行目）の直後に追加する:

```python
    trailing_pe: Optional[float] = None
    payout_ratio: Optional[float] = None
```

docstring の Attributes 一覧（`revenue_cagr_3y: 売上高 CAGR...` の行の直後、38行目の `"""` の前）にも追記する:

```
        trailing_pe: 実績PER（yfinance trailingPE）
        payout_ratio: 配当性向（0〜1のfraction、yfinance payoutRatio）
```

- [ ] **Step 4: `fetch_fundamentals()` に取得ロジックを追加する**

`python/src/market_data/fundamentals.py` の `revenue_cagr_3y = _revenue_cagr_3y(financials)`（165行目）の直後に追加する:

```python
    trailing_pe = _safe_float(info.get("trailingPE"))
    payout_ratio = _safe_float(info.get("payoutRatio"))
```

`record = FundamentalRecord(...)`（167行目〜）の `revenue_cagr_3y=revenue_cagr_3y,` の直後の行に追加する:

```python
        trailing_pe=trailing_pe,
        payout_ratio=payout_ratio,
```

`metric_fields` タプル（186〜194行目、全フィールド空なら失敗扱いにする判定）は変更しない。PER・配当性向は補助指標であり、他の主要指標が全て空で PER/配当性向だけ埋まっているケースは想定しないため。

- [ ] **Step 5: テストが通ることを確認する**

Run: `cd python && py -m pytest tests/unit/test_fundamentals.py::TestFetchFundamentals -v`
Expected: 全 PASS（既存5件 + 新規1件 = 6件、`test_parse_from_info` 含む）

- [ ] **Step 6: コミット**

```bash
cd python
git add src/market_data/types.py src/market_data/fundamentals.py tests/unit/test_fundamentals.py
git commit -m "feat: FundamentalRecordにPER・配当性向を追加"
```

---

### Task 2: `stock_fundamentals` テーブルに PER・配当性向カラムを追加する

**Files:**
- Create: `python/src/utils/db/migrations/0004_add_fundamentals_value_fields_postgres.sql`
- Modify: `python/src/utils/db/stock_fundamentals.py`
- Test: `python/tests/unit/test_fundamentals.py`

**Interfaces:**
- Consumes: `FundamentalRecord.trailing_pe`, `FundamentalRecord.payout_ratio`（Task 1 で追加済み）
- Produces: `load_all_fundamentals()` が返す DataFrame に `trailing_pe`, `payout_ratio` 列が含まれる。Task 3 がこの2列を読む

- [ ] **Step 1: 失敗するテストを書く**

`python/tests/unit/test_fundamentals.py` の `TestFundamentalsDb._record()` ヘルパーを、`trailing_pe` / `payout_ratio` を引数に取り渡すように書き換える:

```python
    def _record(self, symbol="AAPL", revenue=400.0, trailing_pe=8.48, payout_ratio=0.27):
        return FundamentalRecord(
            market="us",
            symbol=symbol,
            as_of=pd.Timestamp("2026-06-04 00:00:00").to_pydatetime(),
            revenue=revenue,
            operating_income=100.0,
            net_income=80.0,
            eps=5.5,
            roe=0.25,
            op_margin=0.25,
            net_margin=0.20,
            debt_to_equity=1.5,
            cash=50.0,
            market_cap=9999.0,
            shares_outstanding=1000.0,
            revenue_cagr_3y=0.5,
            trailing_pe=trailing_pe,
            payout_ratio=payout_ratio,
        )
```

`test_upsert_load_roundtrip` にアサーションを追加する:

```python
    def test_upsert_load_roundtrip(self):
        upsert_fundamentals(self._record())
        loaded = load_fundamentals("us", "AAPL")
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["revenue"], 400.0)
        self.assertEqual(loaded["eps"], 5.5)
        self.assertEqual(loaded["symbol"], "AAPL")
        self.assertEqual(loaded["trailing_pe"], 8.48)
        self.assertEqual(loaded["payout_ratio"], 0.27)
```

- [ ] **Step 2: テストが失敗することを確認する**

Run: `cd python && py -m pytest tests/unit/test_fundamentals.py::TestFundamentalsDb -v`
Expected: `TypeError: _record() got an unexpected keyword argument` ではなく（`_record` 自体は自分で定義するので通る）、`upsert_fundamentals` 実行時に `trailing_pe` 列が無く DB エラー、または `loaded["trailing_pe"]` で `KeyError` により FAIL

- [ ] **Step 3: マイグレーションファイルを作成する**

`python/src/utils/db/migrations/0004_add_fundamentals_value_fields_postgres.sql` を新規作成する:

```sql
-- 0004_add_fundamentals_value_fields: stock_fundamentals にPER・配当性向を追加
-- バリュー・スクリーナー（低PER・低配当性向・財務安定）向けのフィールド。
-- 既存行に対しては NULL で埋まる（次回 run_fetch_fundamentals.py 実行で充填される）。
ALTER TABLE stock_fundamentals ADD COLUMN IF NOT EXISTS trailing_pe DOUBLE PRECISION;
ALTER TABLE stock_fundamentals ADD COLUMN IF NOT EXISTS payout_ratio DOUBLE PRECISION;
```

- [ ] **Step 4: `stock_fundamentals.py` の `_COLUMNS` と Protocol を更新する**

`python/src/utils/db/stock_fundamentals.py` の `_COLUMNS` リスト（29〜45行目）の末尾 `"revenue_cagr_3y",` の直後に追加する:

```python
    "trailing_pe",
    "payout_ratio",
```

`_FundamentalRecordLike` Protocol（48〜65行目）の末尾 `revenue_cagr_3y: Optional[float]` の直後に追加する:

```python
    trailing_pe: Optional[float]
    payout_ratio: Optional[float]
```

- [ ] **Step 5: テストが通ることを確認する**

Run: `cd python && py -m pytest tests/unit/test_fundamentals.py -v`
Expected: 全 PASS（Task 1 の6件 + `TestFundamentalsDb` 全件）

`_db_connection()` はマイグレーション未適用のテーブルに対しても `run_migrations` を接続確立時に自動実行する（`src/utils/db/_connection.py` 参照）ため、テスト用の一時 DB でも新マイグレーションが自動適用される。もし列不足エラーが出たら、`db_module._tables_initialized = False`（`setUp` で既に設定済み）が効いているか確認すること。

- [ ] **Step 6: コミット**

```bash
cd python
git add src/utils/db/migrations/0004_add_fundamentals_value_fields_postgres.sql src/utils/db/stock_fundamentals.py tests/unit/test_fundamentals.py
git commit -m "feat: stock_fundamentalsテーブルにPER・配当性向カラムを追加"
```

---

### Task 3: バリュー・スクリーナー本体を実装する

**Files:**
- Modify: `python/src/screening/types.py`
- Create: `python/src/screening/value_screener.py`
- Test: `python/tests/unit/test_value_screener.py`

**Interfaces:**
- Consumes: `src.utils.db.stock_fundamentals.load_all_fundamentals() -> pd.DataFrame`（既存関数。Task 2 で `trailing_pe`, `payout_ratio` 列が追加済み。DataFrame は `market, symbol, trailing_pe, payout_ratio, debt_to_equity, net_income, market_cap` 等の列を持つ）
- Produces: `screen_value_candidates(market, max_per, max_payout_ratio, max_debt_to_equity, top_n) -> list[ValueCandidate]`、`save_value_candidates(candidates, market) -> str`。Task 4 がこの2関数を呼ぶ

- [ ] **Step 1: `ValueCandidate` 型を追加する**

`python/src/screening/types.py` の末尾（`MultibaggerCandidate` クラスの後）に追加する:

```python


@dataclass
class ValueCandidate:
    """バリュー・スクリーナー（低PER・低配当性向・財務安定）が返す候補銘柄。"""

    market: str
    symbol: str
    trailing_pe: float
    payout_ratio: float
    debt_to_equity: float  # パーセントポイント単位（yfinance実測準拠、例: 78.4 = D/E比0.78）
    net_income: float
    market_cap: Optional[float]  # 表示用。ゲート対象外のため欠損を許容する
```

- [ ] **Step 2: 失敗するテストを書く**

`python/tests/unit/test_value_screener.py` を新規作成する:

```python
"""value_screener のユニットテスト。"""

import unittest
from unittest.mock import patch

import pandas as pd

from src.screening.value_screener import screen_value_candidates


def _row(
    symbol,
    market="jp",
    trailing_pe=8.0,
    payout_ratio=0.20,
    debt_to_equity=50.0,
    net_income=100.0,
    market_cap=1000.0,
):
    """screen_value_candidates が読む DataFrame の1行分を辞書で作る。"""
    return {
        "market": market,
        "symbol": symbol,
        "trailing_pe": trailing_pe,
        "payout_ratio": payout_ratio,
        "debt_to_equity": debt_to_equity,
        "net_income": net_income,
        "market_cap": market_cap,
    }


def _patch_loader(rows):
    """load_all_fundamentals を rows から作った DataFrame でモックする。"""
    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    return patch("src.screening.value_screener.load_all_fundamentals", return_value=df)


class TestScreenValueCandidates(unittest.TestCase):
    def test_good_candidate_passes(self):
        rows = [_row("GOOD")]
        with _patch_loader(rows):
            result = screen_value_candidates(market="jp")
        self.assertEqual([c.symbol for c in result], ["GOOD"])
        self.assertEqual(result[0].trailing_pe, 8.0)
        self.assertEqual(result[0].payout_ratio, 0.20)
        self.assertEqual(result[0].market_cap, 1000.0)

    def test_high_per_excluded(self):
        rows = [_row("GOOD"), _row("HIPER", trailing_pe=50.0)]
        with _patch_loader(rows):
            result = screen_value_candidates(market="jp", max_per=10.0)
        codes = [c.symbol for c in result]
        self.assertIn("GOOD", codes)
        self.assertNotIn("HIPER", codes)

    def test_high_payout_ratio_excluded(self):
        rows = [_row("GOOD"), _row("HIPAYOUT", payout_ratio=0.80)]
        with _patch_loader(rows):
            result = screen_value_candidates(market="jp", max_payout_ratio=0.30)
        codes = [c.symbol for c in result]
        self.assertIn("GOOD", codes)
        self.assertNotIn("HIPAYOUT", codes)

    def test_high_debt_to_equity_excluded(self):
        rows = [_row("GOOD"), _row("HIDEBT", debt_to_equity=200.0)]
        with _patch_loader(rows):
            result = screen_value_candidates(market="jp", max_debt_to_equity=100.0)
        codes = [c.symbol for c in result]
        self.assertIn("GOOD", codes)
        self.assertNotIn("HIDEBT", codes)

    def test_unprofitable_excluded(self):
        rows = [_row("GOOD"), _row("LOSS", net_income=-10.0)]
        with _patch_loader(rows):
            result = screen_value_candidates(market="jp")
        codes = [c.symbol for c in result]
        self.assertIn("GOOD", codes)
        self.assertNotIn("LOSS", codes)

    def test_missing_field_excluded(self):
        rows = [_row("GOOD"), _row("MISSING", trailing_pe=None)]
        with _patch_loader(rows):
            result = screen_value_candidates(market="jp")
        codes = [c.symbol for c in result]
        self.assertIn("GOOD", codes)
        self.assertNotIn("MISSING", codes)

    def test_sorted_by_per_ascending(self):
        rows = [
            _row("HIGH", trailing_pe=9.0),
            _row("LOW", trailing_pe=3.0),
            _row("MID", trailing_pe=6.0),
        ]
        with _patch_loader(rows):
            result = screen_value_candidates(market="jp")
        self.assertEqual([c.symbol for c in result], ["LOW", "MID", "HIGH"])

    def test_top_n_limits_results(self):
        rows = [_row(f"S{i}", trailing_pe=float(i)) for i in range(1, 6)]
        with _patch_loader(rows):
            result = screen_value_candidates(market="jp", top_n=2)
        self.assertEqual([c.symbol for c in result], ["S1", "S2"])

    def test_other_market_excluded(self):
        rows = [_row("JPGOOD", market="jp"), _row("USGOOD", market="us")]
        with _patch_loader(rows):
            result = screen_value_candidates(market="jp")
        self.assertEqual([c.symbol for c in result], ["JPGOOD"])

    def test_empty_fundamentals_returns_empty(self):
        with _patch_loader([]):
            result = screen_value_candidates(market="jp")
        self.assertEqual(result, [])

    def test_no_candidates_pass_returns_empty(self):
        rows = [_row("HIPER", trailing_pe=999.0)]
        with _patch_loader(rows):
            result = screen_value_candidates(market="jp", max_per=10.0)
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 3: テストが失敗することを確認する**

Run: `cd python && py -m pytest tests/unit/test_value_screener.py -v`
Expected: `ModuleNotFoundError: No module named 'src.screening.value_screener'` で全件 FAIL

- [ ] **Step 4: `value_screener.py` を実装する**

`python/src/screening/value_screener.py` を新規作成する:

```python
"""低PER・低配当性向・財務安定 バリュー・スクリーナー

`stock_fundamentals` の最新スナップショットのみを使い、割安・低配当性向・
財務健全な銘柄を抽出する前向きライブスクリーン。過去のPER・配当性向は
DBに残らない（PIT非対応）ため、本スクリーンにバックテストは存在しない。

レイヤー規約: screening BC は utils（``utils.db.stock_fundamentals``）のみ
参照し、market_data BC は import しない（財務は DB 経由で読む）。
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from src.screening.types import ValueCandidate
from src.utils.data_path_utils import ensure_dir, get_results_dir
from src.utils.db.stock_fundamentals import load_all_fundamentals
from src.utils.logger import get_logger

logger = get_logger(__name__)


def screen_value_candidates(
    market: str = "jp",
    max_per: float = 10.0,
    max_payout_ratio: float = 0.30,
    max_debt_to_equity: float = 100.0,
    top_n: int = 30,
) -> list[ValueCandidate]:
    """低PER・低配当性向・財務安定な銘柄を抽出し、PER昇順でランキングする。

    ハードゲート（すべて満たす銘柄のみ残す）:
        - trailing_pe が存在し max_per 以下
        - payout_ratio が存在し max_payout_ratio 以下
        - debt_to_equity が存在し max_debt_to_equity 以下（パーセントポイント単位）
        - net_income が存在し 0 より大きい（黒字）

    ゲート対象フィールドが欠損（NaN/None）の銘柄は判定不能として除外する。

    Returns:
        trailing_pe 昇順の ValueCandidate リスト（最大 top_n 件）。
        該当なしなら空リスト。
    """
    df = load_all_fundamentals()
    if df.empty:
        logger.warning("stock_fundamentals が空です")
        return []

    universe = df[df["market"] == market]
    if universe.empty:
        logger.warning(f"財務データがありません market={market}")
        return []

    required = ["trailing_pe", "payout_ratio", "debt_to_equity", "net_income"]
    mask = pd.Series(True, index=universe.index)
    for col in required:
        mask &= universe[col].notna()
    universe = universe[mask]

    universe = universe[
        (universe["trailing_pe"] <= max_per)
        & (universe["payout_ratio"] <= max_payout_ratio)
        & (universe["debt_to_equity"] <= max_debt_to_equity)
        & (universe["net_income"] > 0)
    ]

    if universe.empty:
        logger.warning("バリュー・スクリーン通過銘柄なし")
        return []

    universe = universe.sort_values("trailing_pe", ascending=True).head(top_n)

    candidates = [
        ValueCandidate(
            market=row["market"],
            symbol=row["symbol"],
            trailing_pe=float(row["trailing_pe"]),
            payout_ratio=float(row["payout_ratio"]),
            debt_to_equity=float(row["debt_to_equity"]),
            net_income=float(row["net_income"]),
            market_cap=(
                float(row["market_cap"]) if pd.notna(row.get("market_cap")) else None
            ),
        )
        for _, row in universe.iterrows()
    ]

    logger.info(f"バリュー・スクリーン完了: {len(candidates)} 銘柄を選定 ({market})")
    return candidates


def save_value_candidates(candidates: list[ValueCandidate], market: str) -> str:
    """候補リストを CSV に保存しパスを返す。"""
    out_dir = ensure_dir(f"{get_results_dir()}/screening")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = f"{out_dir}/value_candidates_{market}_{timestamp}.csv"

    df = pd.DataFrame([c.__dict__ for c in candidates])
    df.to_csv(path, index=False)
    logger.info(f"候補を保存: {path}")
    return path
```

- [ ] **Step 5: テストが通ることを確認する**

Run: `cd python && py -m pytest tests/unit/test_value_screener.py -v`
Expected: 全11件 PASS

- [ ] **Step 6: コミット**

```bash
cd python
git add src/screening/types.py src/screening/value_screener.py tests/unit/test_value_screener.py
git commit -m "feat: バリュー・スクリーナー(screen_value_candidates)を追加"
```

---

### Task 4: CLI エントリポイントを追加する

**Files:**
- Create: `python/run_value_screen.py`

**Interfaces:**
- Consumes: `src.screening.value_screener.screen_value_candidates(market, max_per, max_payout_ratio, max_debt_to_equity, top_n) -> list[ValueCandidate]`, `save_value_candidates(candidates, market) -> str`（Task 3 で実装済み）
- Produces: なし（最終タスク）

- [ ] **Step 1: `run_value_screen.py` を作成する**

`python/run_value_screen.py` を新規作成する（`run_screen.py` と同じ構成: 引数パース → サービス呼び出し → 表示・保存、ビジネスロジックなし）:

```python
r"""低PER・低配当性向・財務安定 バリュー・スクリーナー実行スクリプト

stock_fundamentals の最新スナップショットのみを使い、割安・低配当性向・
財務健全な銘柄を抽出して表示し、CSV に保存する。
前向きライブスクリーン専用（バックテストは提供しない。PIT非対応のため）。

使用例:
    py run_value_screen.py --market jp
    py run_value_screen.py --market jp --max-per 8.0 --max-payout-ratio 0.25
"""

import argparse
import sys

from src.screening.value_screener import save_value_candidates, screen_value_candidates
from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="バリュー・スクリーナーを実行する",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--market", type=str, default="jp", help="マーケット識別子")
    parser.add_argument("--max-per", type=float, default=10.0, help="実績PER上限")
    parser.add_argument(
        "--max-payout-ratio", type=float, default=0.30, help="配当性向上限（0〜1）"
    )
    parser.add_argument(
        "--max-debt-to-equity",
        type=float,
        default=100.0,
        help="D/E上限（パーセントポイント単位、yfinance実測準拠）",
    )
    parser.add_argument("--top-n", type=int, default=30, help="返す候補数")
    return parser.parse_args()


def run(args) -> None:
    candidates = screen_value_candidates(
        market=args.market,
        max_per=args.max_per,
        max_payout_ratio=args.max_payout_ratio,
        max_debt_to_equity=args.max_debt_to_equity,
        top_n=args.top_n,
    )

    if not candidates:
        logger.warning("該当銘柄なし")
        return

    print(f"{'symbol':<10}{'PER':>8}{'配当性向':>10}{'D/E':>10}{'純利益':>16}")
    for c in candidates:
        print(
            f"{c.symbol:<10}{c.trailing_pe:>8.2f}{c.payout_ratio:>10.2%}"
            f"{c.debt_to_equity:>10.1f}{c.net_income:>16,.0f}"
        )

    path = save_value_candidates(candidates, args.market)
    print(f"\n保存先: {path}")


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"バリュー・スクリーン 異常終了: {e}", exc_info=True)
        sys.exit(1)
```

- [ ] **Step 2: 動作確認する（ユニットテストではなく実行確認）**

Run: `cd python && py run_value_screen.py --market jp --help`
Expected: 引数の使い方が表示され、エラーなく終了する（`SystemExit: 0`）

Run: `cd python && py run_value_screen.py --market jp`
Expected: `stock_fundamentals` に jp データが無ければ「該当銘柄なし」の warning ログが出て正常終了する。データがあれば候補テーブルが表示され CSV が保存される。いずれの結果でも例外で終了しないこと

- [ ] **Step 3: 全体のユニットテストとカバレッジを確認する**

Run: `cd python && py -m pytest tests/unit/test_fundamentals.py tests/unit/test_value_screener.py -v --cov=src.market_data.fundamentals --cov=src.screening.value_screener --cov=src.utils.db.stock_fundamentals --cov-report=term-missing`
Expected: 全件 PASS。新規コード（`value_screener.py`, `fundamentals.py` の追加分, `stock_fundamentals.py` の追加分）に大きな未カバー行がないこと

- [ ] **Step 4: コミット**

```bash
cd python
git add run_value_screen.py
git commit -m "feat: バリュー・スクリーナーCLI(run_value_screen.py)を追加"
```

---

## Self-Review Notes

- **spec カバレッジ:** spec の1〜7コンポーネントすべてにタスクが対応する（Task1=spec§1,2 / Task2=spec§3,4 / Task3=spec§5,6 / Task4=spec§7）。テスト方針（spec 末尾）もタスクごとの Step に反映済み
- **プレースホルダ scan:** 「TBD」「後で」等なし。全ステップに実コードあり
- **型の一貫性:** `ValueCandidate` のフィールド名（`trailing_pe`, `payout_ratio`, `debt_to_equity`, `net_income`, `market_cap`）は Task 1/2 で追加する `FundamentalRecord` / DB カラム名と完全一致させた。`screen_value_candidates` の引数名（`max_per`, `max_payout_ratio`, `max_debt_to_equity`, `top_n`）は Task 4 の CLI 引数と一致させた
