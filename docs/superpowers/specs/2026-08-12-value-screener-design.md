# 低PER・低配当性向・財務安定 バリュー・スクリーナー 設計

> **For agentic workers:** REQUIRED SUB-SKILL: 実装は superpowers:subagent-driven-development（推奨）または superpowers:executing-plans を使う。

## 背景・目的

「低PER・低配当性向・財務安定の銘柄を選定し投資、配当性向が上がったところで
売り抜ける」という戦略のリターン検証を依頼された。調査の結果、**過去の
バックテストは今のデータ基盤では不可能**と判明した:

- `stock_fundamentals` は最新スナップショットのみ（PIT 非対応）。PK が
  `(market, symbol)` で DELETE→INSERT のため過去の PER・配当性向は再現できない
- yfinance の年次 `income_stmt` は 4〜5期分しか返らず、四半期版はさらに乏しい
  （実測: 8058.T の quarterly_income_stmt は1列のみ）。配当性向の観測点は
  銘柄あたり通算4〜5点しかなく、売却シグナルの発火は銘柄あたり1〜2回に留まる
- これは #625（戦略ファクトリーのゲート修正）で実証した「銘柄あたり1〜2取引の
  Sharpe は発散する」問題と同じ構造であり、`FACTORY_GATE_MIN_TRADES_PER_SYMBOL=3`
  のようなゲートを適用すれば構造的に弾かれる標本数しか取れない

そのためユーザーの合意のもと、本タスクは**バックテストではなく前向きの
ライブスクリーン**（買い候補の抽出のみ）に絞る。売却ロジック（配当性向上昇での
利確判定）は本タスクのスコープ外。

## 決定事項（ユーザー承認済み）

- 母集団: 財務指標のみで独立スクリーン（既存 `trend_screener.py` のトレンド候補を
  母集団にしない）。低PER銘柄は上昇トレンドの前提条件と相性が悪いため
- 対象マーケット: `jp` のみ
- 「財務安定」の定義: 低D/E + 黒字（`net_income > 0`）
- 実行方法: 新規 `run_value_screen.py`（scheduler・Discord 連携は今回は行わない）
- ランキング: PER 昇順の単純ソート（合成スコアは作らない）

## 副次的な発見（本タスクではスコープ外・別 Issue 化する）

`src/screening/quality_gate.py` の `max_debt_to_equity`（既定 `1.5`）は、
yfinance の `debtToEquity` が実際には**パーセントポイント単位**（実測:
AAPL=78.445, MSFT=29.118, 7203.T=114.974 — いずれも D/E比 0.78 / 0.29 / 1.15
を意味する）で返るにもかかわらず、`1.5` という比率のつもりの値と比較しており
単位が食い違っている。実データでは `de(30〜150) > max(1.5)` が常に真となり、
D/E ゲートが実質常に不合格判定になる可能性が高い。ユニットテスト
（`tests/unit/test_quality_gate.py`）は `debt_to_equity=0.5` 等の架空値の
フィクスチャを使っており、この単位ズレを検知できていない。

`run_screen.py --with-fundamentals` は手動実行のみで scheduler 未登録のため
実害は限定的だが、修正は別 Issue で扱う。本設計の新スクリーンでは
**パーセントポイント単位を前提にした閾値**（`max_debt_to_equity=100.0` 等）を
最初から使うことで同じ罠を回避する。

## アーキテクチャ

独立スクリーンとして実装する。OHLCV には触れず、`stock_fundamentals`
テーブルのみで完結する。既存の `trend_screener.py` / `quality_gate.py` と
並ぶ第3のスクリーナーとして `src/screening/value_screener.py` を新設する。

```
run_value_screen.py (新規CLI)
    ↓
src/screening/value_screener.py (新規)
    screen_value_candidates(market, max_per, max_payout_ratio,
                             max_debt_to_equity, top_n) -> list[ValueCandidate]
    ↓
src/utils/db/stock_fundamentals.py の load_all_fundamentals()（既存・変更不要）
    ↓
stock_fundamentals テーブル（trailing_pe, payout_ratio を新規カラム追加）
    ↑
src/market_data/fundamentals.py の fetch_fundamentals()（既存関数を拡張）
    ↑
run_fetch_fundamentals.py（既存・変更不要、フィールドは自動で乗る）
```

レイヤー規約は `quality_gate.py` と同じ: screening BC は
`utils.db.stock_fundamentals` のみを参照し、`market_data` BC を import しない。

## コンポーネント詳細

### 1. `FundamentalRecord` の拡張（`src/market_data/types.py`）

新規フィールドを既存フィールド群の末尾（`revenue_cagr_3y` の後）に追加する:

```python
trailing_pe: Optional[float] = None  # 実績PER（yfinance trailingPE）
payout_ratio: Optional[float] = None  # 配当性向（0〜1のfraction。yfinance payoutRatio）
```

docstring の Attributes 一覧にも追記する。PIT 非対応の警告文はそのまま維持する
（新フィールドも同じ制約を継承するため）。

### 2. `fetch_fundamentals()` の拡張（`src/market_data/fundamentals.py`）

`info` dict から素直に取得する（他の `info.get(...)` 系フィールドと同じパターン）。
派生計算は不要（`op_margin` のような fallback 計算ロジックは持たない — PER も
配当性向も `info` に無ければ `None` のまま、他ソースからの代替計算はしない。
YAGNI: 現状 `financials` 等に相当データがなく、フォールバックを作っても
テストできないため）。

```python
trailing_pe = _safe_float(info.get("trailingPE"))
payout_ratio = _safe_float(info.get("payoutRatio"))
```

`record = FundamentalRecord(...)` の引数リストに `trailing_pe=trailing_pe,
payout_ratio=payout_ratio,` を追加する。既存の「全フィールドが空なら失敗扱い」
判定（`metric_fields` タプル）は変更しない（PER・配当性向は必須指標ではなく
補助指標のため、これらだけ埋まっていて他が空というケースは想定しない）。

### 3. DB マイグレーション（新規: `src/utils/db/migrations/0004_add_fundamentals_value_fields_postgres.sql`）

命名規則は `NNNN_description_postgres.sql`（`migration_runner.py` の
`_VERSION_RE = r"^(\d{4})_(.+?)_postgres\.sql$"` に一致させる）。

```sql
-- 0004_add_fundamentals_value_fields: stock_fundamentals にPER・配当性向を追加
-- バリュー・スクリーナー（低PER・低配当性向・財務安定）向けのフィールド。
-- 既存行に対しては NULL で埋まる（次回 run_fetch_fundamentals.py 実行で充填される）。
ALTER TABLE stock_fundamentals ADD COLUMN IF NOT EXISTS trailing_pe DOUBLE PRECISION;
ALTER TABLE stock_fundamentals ADD COLUMN IF NOT EXISTS payout_ratio DOUBLE PRECISION;
```

### 4. `utils/db/stock_fundamentals.py` の更新

`_COLUMNS` リストの末尾（`"revenue_cagr_3y"` の後）に `"trailing_pe"`,
`"payout_ratio"` を追加する。`_FundamentalRecordLike` Protocol にも同名の
`Optional[float]` 属性を追加する。`upsert_fundamentals` / `load_fundamentals` /
`load_all_fundamentals` の実装自体は `_COLUMNS` 駆動のため変更不要。

### 5. `ValueCandidate` 型（`src/screening/types.py` に追加）

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
    market_cap: Optional[float]  # 表示用。欠損時は None（ゲート対象外のため許容）
```

`market_cap` のみ Optional にする理由: ゲート条件に含めないため欠損があっても
候補から除外しない（表示用の付随情報）。他フィールドはすべてハードゲートの
判定対象であり、欠損なら候補に残らない（除外済み）ため non-Optional。

### 6. `src/screening/value_screener.py`（新規ファイル）

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
    max_per: float = 10.0,  # 実績PER上限
    max_payout_ratio: float = 0.30,  # 配当性向上限（0〜1のfraction）
    max_debt_to_equity: float = 100.0,  # D/E上限（パーセントポイント単位）
    top_n: int = 30,
) -> list[ValueCandidate]:
    """低PER・低配当性向・財務安定な銘柄を抽出し、PER昇順でランキングする。

    ハードゲート（すべて満たす銘柄のみ残す）:
        - trailing_pe が存在し max_per 以下
        - payout_ratio が存在し max_payout_ratio 以下
        - debt_to_equity が存在し max_debt_to_equity 以下
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

`load_all_fundamentals()` が返す DataFrame の列は `_COLUMNS`
（`market, symbol, as_of, revenue, ..., trailing_pe, payout_ratio`）に
準拠しており、`coerce_object_numeric_columns` で数値列は既に float 化
されている。`market_cap` は既存フィールドとして DataFrame に存在する。

### 7. `run_value_screen.py`（新規ファイル、CLIラッパーのみ）

既存 `run_screen.py` と同じ構成（引数パース → サービス呼び出し → 表示・保存、
ビジネスロジックなし）。

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

## テスト方針

- `tests/unit/test_fundamentals.py`: 既存ファイル（`market_data.fundamentals`
  と `db.stock_fundamentals` 両方を対象にした単一ファイル。他のテストも
  `tests/unit/` 直下にフラットに置かれており専用サブディレクトリは無い）。
  `fetch_fundamentals()` のモック `info` に `trailingPE` / `payoutRatio` を
  含むケースを追加して `FundamentalRecord` への反映を検証し、
  `upsert_fundamentals`/`load_fundamentals` のラウンドトリップも新フィールド
  込みで検証する
- `tests/unit/test_value_screener.py`（新規、`test_quality_gate.py` /
  `test_trend_screener.py` と同じ並びに置く）:
  `load_all_fundamentals` をモックし、
  - 全ゲート通過銘柄が候補に残ること
  - PER超過・配当性向超過・D/E超過・赤字のそれぞれ単独で除外されること
  - 欠損フィールド（例: `trailing_pe` が NaN）は除外されること
  - PER 昇順にソートされていること
  - `top_n` で件数が絞られること
  - 該当なしで空リストが返ること
- `run_value_screen.py` は CLI ラッパーのみのため専用テストは作らない
  （`run_screen.py` に既存テストが無いのと同じ扱い）

## Global Constraints（実装計画のタスク全体に適用）

- DuckDB/Postgres 書き込みは逐次（並列書き込み禁止）。本設計は書き込み経路を
  変更しないため新たな考慮は不要
- `run_*.py` はCLIラッパーのみ。ビジネスロジックは `src/screening/` に置く
- 型は dataclass（`ValueCandidate`）を使い、生 dict を上位に渡さない
- ログは `get_logger(__name__)`、`except: pass` 禁止
- マイグレーションファイル名は `NNNN_description_postgres.sql`
  （`migration_runner.py` の正規表現に一致させること）
- 既存 `quality_gate.py` の D/E 単位バグは本タスクでは修正しない（別 Issue）。
  新規コードは同じ轍を踏まないよう、コメントで単位（パーセントポイント）を
  明示する
