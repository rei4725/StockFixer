# 予測出力の健全性チェック（出力 invariant）実装計画

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 日次パイプラインの予測出力に不変条件を評価し、違反時のみ Discord へ発報する。あわせて未配線の `alert_service`（NF-303）を本番経路に繋ぐ。

**Architecture:** 判定ロジックは `src/prediction/output_invariants.py` に純関数として置き、DB もネットワークも触れない。前回ラン統計は `src/prediction/db/` の新クエリが供給する。発報は既存 `src/utils/alert_service.py` に新ルール `NF-303-5` を追加して合流させ、`src/orchestration/jobs/daily.py` が両者を配線する。

**Tech Stack:** Python 3.12 / PostgreSQL (psycopg3) / pytest + unittest / dataclass

**設計書:** `docs/superpowers/specs/2026-08-01-prediction-output-invariants-design.md`

## Global Constraints

- 作業ブランチは `feature/prediction-output-invariants`（設計書コミット済み）。ベースは `develop`
- **`src/utils/` から `src/prediction/` を import してはならない。** `.importlinter` の Contract 1（`src.utils` が最下層）に違反し、CI の import-linter が落ちる。`alert_service` の新ルールは `list[str]` と `dict` のみを受け取る
- Windows 環境では `python` ではなく **`py`** を使う（`python` は Windows Store のスタブで exit 49）
- コミットは Conventional Commits（`feat:` / `fix:` / `test:` / `chore:`）
- コミットが `UnicodeEncodeError` で弾かれたら `PYTHONUTF8=1 PYTHONIOENCODING=utf-8 git commit ...` を使う（commit-msg フックの Windows バグ）
- ロガーは各モジュールで `from src.utils.logger import get_logger` → `get_logger(__name__)`。`except: pass` は禁止。必ず `logger.error(..., exc_info=True)`
- 単体テストのカバレッジゲートは 80%
- 型は dict ではなく dataclass を使う

---

### Task 1: 出力 invariant の絶対値チェック（A-1 / A-2 / A-3）

**Files:**
- Create: `python/src/prediction/output_invariants.py`
- Test: `python/tests/unit/test_output_invariants.py`

**Interfaces:**
- Consumes: `src.prediction.types.PredictionResult`（既存。`market` / `symbol` / `current_price` / `avg_pred_price` / `diff_ratio` / `model_count` を持つ dataclass）
- Produces:
  - `PredictionRunStats(symbol_count: int, median_model_count: float, diff_ratio_stdev: float)`
  - `InvariantViolation(violation_id: str, description: str, observed: float, threshold: float)`
  - `InvariantReport`（`violations: list[InvariantViolation]`, `stats: Optional[PredictionRunStats]`, `compared_with_previous: bool`, プロパティ `has_violation` / `violation_ids` / `as_details()`）
  - `build_run_stats(model_counts: Sequence[int], diff_ratios: Sequence[float]) -> Optional[PredictionRunStats]`
  - `build_run_stats_from_results(output_rows: Sequence[PredictionResult]) -> Optional[PredictionRunStats]`
  - `evaluate_output_invariants(requested_model_names, loaded_model_names, output_rows, previous_stats=None) -> InvariantReport`
  - 定数 `DEGRADED_SYMBOL_RATIO_THRESHOLD = 0.5`

- [ ] **Step 1: 失敗するテストを書く**

`python/tests/unit/test_output_invariants.py` を新規作成する。

```python
"""予測出力の健全性チェック（出力 invariant）の単体テスト。"""

import unittest

from src.prediction.output_invariants import (
    DEGRADED_SYMBOL_RATIO_THRESHOLD,
    evaluate_output_invariants,
)
from src.prediction.types import PredictionResult

REQUESTED = ["UnifiedStockXGBoost", "UnifiedStockLightGBM"]


def make_rows(count: int, model_count: int, diff_ratio_step: float = 0.001):
    """実データと同じ形の PredictionResult を組み立てる。

    model_count を捏造した dict で代替しないこと（#615 の教訓）。
    """
    rows = []
    for i in range(count):
        price = 1000.0 + i
        diff_ratio = diff_ratio_step * (i - count / 2)
        rows.append(
            PredictionResult(
                market="jp",
                symbol=str(7000 + i),
                current_price=price,
                avg_pred_price=price * (1 + diff_ratio),
                diff_ratio=diff_ratio,
                model_count=model_count,
            )
        )
    return rows


class TestAbsoluteInvariants(unittest.TestCase):
    def test_healthy_run_has_no_violation(self):
        report = evaluate_output_invariants(
            requested_model_names=REQUESTED,
            loaded_model_names=list(REQUESTED),
            output_rows=make_rows(705, model_count=2),
        )
        self.assertFalse(report.has_violation)
        self.assertEqual(report.violation_ids, [])

    def test_a1_fires_when_model_failed_to_load(self):
        """モデルファイル欠損。A-2 では原理的に検出できないケース。"""
        report = evaluate_output_invariants(
            requested_model_names=REQUESTED,
            loaded_model_names=["UnifiedStockXGBoost"],
            output_rows=make_rows(705, model_count=1),
        )
        self.assertIn("A-1", report.violation_ids)

    def test_a2_reproduces_issue_615(self):
        """#615 の実際の状態: 2モデルともロード成功、705銘柄すべて model_count=1。"""
        report = evaluate_output_invariants(
            requested_model_names=REQUESTED,
            loaded_model_names=list(REQUESTED),
            output_rows=make_rows(705, model_count=1),
        )
        self.assertIn("A-2", report.violation_ids)

    def test_a2_silent_when_only_a_few_symbols_degrade(self):
        """数銘柄の片肺化は正常運用の揺らぎとして鳴らさない。"""
        rows = make_rows(100, model_count=2)
        for row in rows[:10]:
            row.model_count = 1
        report = evaluate_output_invariants(
            requested_model_names=REQUESTED,
            loaded_model_names=list(REQUESTED),
            output_rows=rows,
        )
        self.assertNotIn("A-2", report.violation_ids)

    def test_a2_boundary_exactly_at_threshold_fires(self):
        """縮退率ちょうど 50% は違反（>= 判定）。"""
        rows = make_rows(100, model_count=2)
        for row in rows[:50]:
            row.model_count = 1
        report = evaluate_output_invariants(
            requested_model_names=REQUESTED,
            loaded_model_names=list(REQUESTED),
            output_rows=rows,
        )
        self.assertIn("A-2", report.violation_ids)
        violation = next(v for v in report.violations if v.violation_id == "A-2")
        self.assertAlmostEqual(violation.observed, 0.5)
        self.assertAlmostEqual(violation.threshold, DEGRADED_SYMBOL_RATIO_THRESHOLD)

    def test_a3_reproduces_issue_612(self):
        """全銘柄がスキップされても例外にならず空リストになる経路。"""
        report = evaluate_output_invariants(
            requested_model_names=REQUESTED,
            loaded_model_names=list(REQUESTED),
            output_rows=[],
        )
        self.assertIn("A-3", report.violation_ids)
        self.assertIsNone(report.stats)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `cd python; py -m pytest tests/unit/test_output_invariants.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.prediction.output_invariants'`

- [ ] **Step 3: 最小実装を書く**

`python/src/prediction/output_invariants.py` を新規作成する。

```python
"""予測出力の健全性チェック（出力 invariant）。

#612 / #613 / #615 に共通する「もっともらしい出力を出しながら中身が
縮退している」失敗モードを検知する。

DB もネットワークも触れない純関数として実装する。入力は「要求したモデル名」
「ロードできたモデル名」「今回の予測結果」「前回ラン統計」の4つのみ。
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Optional, Sequence

from src.prediction.types import PredictionResult
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# 閾値定数
# ---------------------------------------------------------------------------

# A-2: model_count が期待値未満の銘柄が占める割合。これ以上で違反。
DEGRADED_SYMBOL_RATIO_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# データクラス
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PredictionRunStats:
    """1ラン分の集計値。前回ランとの比較に使う。"""

    symbol_count: int
    median_model_count: float
    diff_ratio_stdev: float


@dataclass(frozen=True)
class InvariantViolation:
    """不変条件の違反 1 件。"""

    violation_id: str
    description: str
    observed: float
    threshold: float


@dataclass
class InvariantReport:
    """1ラン分の評価結果。"""

    violations: list[InvariantViolation] = field(default_factory=list)
    stats: Optional[PredictionRunStats] = None
    compared_with_previous: bool = False

    @property
    def has_violation(self) -> bool:
        return bool(self.violations)

    @property
    def violation_ids(self) -> list[str]:
        return [v.violation_id for v in self.violations]

    def as_details(self) -> dict:
        """alert_service へ渡すプリミティブな内訳。

        alert_service は src.utils 層にあり src.prediction を import できない
        （.importlinter Contract 1）。そのため型ではなく dict で渡す。
        """
        return {
            "compared_with_previous": self.compared_with_previous,
            "symbol_count": self.stats.symbol_count if self.stats else 0,
            "median_model_count": self.stats.median_model_count if self.stats else 0.0,
            "diff_ratio_stdev": self.stats.diff_ratio_stdev if self.stats else 0.0,
            "violations": [
                {
                    "id": v.violation_id,
                    "description": v.description,
                    "observed": v.observed,
                    "threshold": v.threshold,
                }
                for v in self.violations
            ],
        }


# ---------------------------------------------------------------------------
# 集計
# ---------------------------------------------------------------------------


def build_run_stats(
    model_counts: Sequence[int], diff_ratios: Sequence[float]
) -> Optional[PredictionRunStats]:
    """ラン集計を作る。銘柄が 0 件なら None。

    標準偏差は母標準偏差（pstdev）を使う。標本標準偏差だと 1 銘柄で
    StatisticsError になるため。
    """
    if not model_counts:
        return None

    return PredictionRunStats(
        symbol_count=len(model_counts),
        median_model_count=float(statistics.median(model_counts)),
        diff_ratio_stdev=float(statistics.pstdev(diff_ratios)) if diff_ratios else 0.0,
    )


def build_run_stats_from_results(
    output_rows: Sequence[PredictionResult],
) -> Optional[PredictionRunStats]:
    """PredictionResult のリストからラン集計を作る。"""
    return build_run_stats(
        [r.model_count for r in output_rows],
        [r.diff_ratio for r in output_rows],
    )


# ---------------------------------------------------------------------------
# 絶対値チェック（前回ラン不要）
# ---------------------------------------------------------------------------


def _check_absolute(
    requested_model_names: Sequence[str],
    loaded_model_names: Sequence[str],
    output_rows: Sequence[PredictionResult],
) -> list[InvariantViolation]:
    violations: list[InvariantViolation] = []

    # A-1: モデルロード欠損
    # A-2 の期待値はロード成功数から導くため、ロード自体が減ると閾値も
    # 一緒に下がって縮退が無言化する。その穴を塞ぐのがこのルール。
    requested_count = len(requested_model_names)
    loaded_count = len(loaded_model_names)
    if loaded_count < requested_count:
        missing = sorted(set(requested_model_names) - set(loaded_model_names))
        violations.append(
            InvariantViolation(
                violation_id="A-1",
                description=f"モデルのロードに失敗: {', '.join(missing)}",
                observed=float(loaded_count),
                threshold=float(requested_count),
            )
        )

    # A-3: 予測 0 件
    # predict_all_unified の wrapper は例外を握って None を返すため、
    # 全銘柄が失敗しても空リストになるだけで例外にならない（#612 の経路）。
    if not output_rows:
        violations.append(
            InvariantViolation(
                violation_id="A-3",
                description="予測結果が 0 件",
                observed=0.0,
                threshold=1.0,
            )
        )
        return violations

    # A-2: アンサンブル縮退
    if loaded_count > 0:
        degraded = sum(1 for r in output_rows if r.model_count < loaded_count)
        ratio = degraded / len(output_rows)
        if ratio >= DEGRADED_SYMBOL_RATIO_THRESHOLD:
            violations.append(
                InvariantViolation(
                    violation_id="A-2",
                    description=(
                        f"アンサンブル縮退: {degraded}/{len(output_rows)} 銘柄が "
                        f"{loaded_count} モデル中 {loaded_count} 未満で予測されている"
                    ),
                    observed=ratio,
                    threshold=DEGRADED_SYMBOL_RATIO_THRESHOLD,
                )
            )

    return violations


# ---------------------------------------------------------------------------
# 公開 API
# ---------------------------------------------------------------------------


def evaluate_output_invariants(
    requested_model_names: Sequence[str],
    loaded_model_names: Sequence[str],
    output_rows: Sequence[PredictionResult],
    previous_stats: Optional[PredictionRunStats] = None,
) -> InvariantReport:
    """予測出力の不変条件を評価する。

    Args:
        requested_model_names: 要求したモデル名（A-1 の比較基準）
        loaded_model_names: ロードに成功したモデル名（A-2 の期待値）
        output_rows: 今回ランの予測結果
        previous_stats: 前回ラン統計。None なら急変チェックをスキップする

    Returns:
        InvariantReport
    """
    stats = build_run_stats_from_results(output_rows)
    violations = _check_absolute(requested_model_names, loaded_model_names, output_rows)

    return InvariantReport(
        violations=violations,
        stats=stats,
        compared_with_previous=False,
    )
```

- [ ] **Step 4: テストが通ることを確認**

Run: `cd python; py -m pytest tests/unit/test_output_invariants.py -v`
Expected: PASS（6 件）

- [ ] **Step 5: コミット**

```bash
git add python/src/prediction/output_invariants.py python/tests/unit/test_output_invariants.py
git commit -m "feat: 予測出力の絶対値invariant（A-1/A-2/A-3）を追加"
```

---

### Task 2: 急変チェック（B-1 / B-2 / B-3）

**Files:**
- Modify: `python/src/prediction/output_invariants.py`
- Test: `python/tests/unit/test_output_invariants.py`

**Interfaces:**
- Consumes: Task 1 の `PredictionRunStats` / `InvariantViolation` / `InvariantReport` / `evaluate_output_invariants`
- Produces: 定数 `SYMBOL_COUNT_DROP_THRESHOLD = 0.2` / `DIFF_RATIO_STDEV_SHRINK_FACTOR = 0.5` / `DIFF_RATIO_STDEV_GROW_FACTOR = 2.0`、内部関数 `_check_regression(stats, previous_stats) -> list[InvariantViolation]`

- [ ] **Step 1: 失敗するテストを書く**

`python/tests/unit/test_output_invariants.py` の末尾（`if __name__ == "__main__":` の直前）に追記する。

```python
class TestRegressionInvariants(unittest.TestCase):
    def _healthy_previous(self) -> PredictionRunStats:
        return PredictionRunStats(
            symbol_count=705, median_model_count=2.0, diff_ratio_stdev=0.01
        )

    def test_b1_fires_on_large_symbol_drop(self):
        report = evaluate_output_invariants(
            requested_model_names=REQUESTED,
            loaded_model_names=list(REQUESTED),
            output_rows=make_rows(500, model_count=2),
            previous_stats=self._healthy_previous(),
        )
        self.assertIn("B-1", report.violation_ids)
        self.assertTrue(report.compared_with_previous)

    def test_b1_silent_on_symbol_increase(self):
        """銘柄追加は正常。片側判定であることを確認する。"""
        report = evaluate_output_invariants(
            requested_model_names=REQUESTED,
            loaded_model_names=list(REQUESTED),
            output_rows=make_rows(900, model_count=2),
            previous_stats=self._healthy_previous(),
        )
        self.assertNotIn("B-1", report.violation_ids)

    def test_b1_boundary_exactly_20_percent_drop_fires(self):
        report = evaluate_output_invariants(
            requested_model_names=REQUESTED,
            loaded_model_names=list(REQUESTED),
            output_rows=make_rows(564, model_count=2),  # 705 * 0.8
            previous_stats=self._healthy_previous(),
        )
        self.assertIn("B-1", report.violation_ids)

    def test_b2_fires_when_median_model_count_drops(self):
        rows = make_rows(705, model_count=2)
        for row in rows[:400]:
            row.model_count = 1
        report = evaluate_output_invariants(
            requested_model_names=REQUESTED,
            loaded_model_names=list(REQUESTED),
            output_rows=rows,
            previous_stats=self._healthy_previous(),
        )
        self.assertIn("B-2", report.violation_ids)

    def test_b3_fires_when_stdev_shrinks_by_half(self):
        report = evaluate_output_invariants(
            requested_model_names=REQUESTED,
            loaded_model_names=list(REQUESTED),
            output_rows=make_rows(705, model_count=2, diff_ratio_step=0.00001),
            previous_stats=self._healthy_previous(),
        )
        self.assertIn("B-3", report.violation_ids)

    def test_b3_fires_on_zero_stdev_without_previous(self):
        """全銘柄が同じ予測値。前回統計が無くても違反。"""
        rows = make_rows(705, model_count=2)
        for row in rows:
            row.diff_ratio = 0.005
        report = evaluate_output_invariants(
            requested_model_names=REQUESTED,
            loaded_model_names=list(REQUESTED),
            output_rows=rows,
        )
        self.assertIn("B-3", report.violation_ids)

    def test_regression_skipped_without_previous_stats(self):
        """前回統計が無ければ急変ルールは評価されない。"""
        rows = make_rows(1, model_count=2)
        report = evaluate_output_invariants(
            requested_model_names=REQUESTED,
            loaded_model_names=list(REQUESTED),
            output_rows=rows,
        )
        self.assertFalse(report.compared_with_previous)
        for vid in ("B-1", "B-2"):
            self.assertNotIn(vid, report.violation_ids)
```

`PredictionRunStats` を import に追加する（ファイル冒頭の import を書き換える）。

```python
from src.prediction.output_invariants import (
    DEGRADED_SYMBOL_RATIO_THRESHOLD,
    PredictionRunStats,
    evaluate_output_invariants,
)
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `cd python; py -m pytest tests/unit/test_output_invariants.py -v -k Regression`
Expected: FAIL — `ImportError: cannot import name 'PredictionRunStats'` は解消済みのはずなので、`AssertionError: 'B-1' not found in []` 等で失敗する

- [ ] **Step 3: 実装を書く**

`python/src/prediction/output_invariants.py` の閾値定数ブロックに追記する。

```python
# B-1: 予測銘柄数の減少率。これ以上で違反（増加は鳴らさない）。
SYMBOL_COUNT_DROP_THRESHOLD = 0.2

# B-3: diff_ratio の標準偏差が前回比でこの倍率を下回る / 上回ると違反。
DIFF_RATIO_STDEV_SHRINK_FACTOR = 0.5
DIFF_RATIO_STDEV_GROW_FACTOR = 2.0
```

`_check_absolute` の直後（`# 公開 API` の区切りコメントの前）に急変チェックを追加する。

```python
# ---------------------------------------------------------------------------
# 急変チェック（前回ラン統計との比較）
# ---------------------------------------------------------------------------


def _check_regression(
    stats: PredictionRunStats, previous: PredictionRunStats
) -> list[InvariantViolation]:
    violations: list[InvariantViolation] = []

    # B-1: 予測銘柄数の急減（増加は正常なので片側判定）
    if previous.symbol_count > 0:
        drop_ratio = (previous.symbol_count - stats.symbol_count) / previous.symbol_count
        if drop_ratio >= SYMBOL_COUNT_DROP_THRESHOLD:
            violations.append(
                InvariantViolation(
                    violation_id="B-1",
                    description=(
                        f"予測銘柄数が急減: {previous.symbol_count} → {stats.symbol_count}"
                    ),
                    observed=drop_ratio,
                    threshold=SYMBOL_COUNT_DROP_THRESHOLD,
                )
            )

    # B-2: model_count 中央値の低下（上昇は改善なので片側判定）
    if stats.median_model_count < previous.median_model_count:
        violations.append(
            InvariantViolation(
                violation_id="B-2",
                description=(
                    f"model_count 中央値が低下: {previous.median_model_count} → "
                    f"{stats.median_model_count}"
                ),
                observed=stats.median_model_count,
                threshold=previous.median_model_count,
            )
        )

    # B-3: 予測分布の急変
    if previous.diff_ratio_stdev > 0:
        ratio = stats.diff_ratio_stdev / previous.diff_ratio_stdev
        if ratio < DIFF_RATIO_STDEV_SHRINK_FACTOR:
            violations.append(
                InvariantViolation(
                    violation_id="B-3",
                    description=(
                        f"予測分散が急縮小: {previous.diff_ratio_stdev:.6f} → "
                        f"{stats.diff_ratio_stdev:.6f}"
                    ),
                    observed=ratio,
                    threshold=DIFF_RATIO_STDEV_SHRINK_FACTOR,
                )
            )
        elif ratio > DIFF_RATIO_STDEV_GROW_FACTOR:
            violations.append(
                InvariantViolation(
                    violation_id="B-3",
                    description=(
                        f"予測分散が急拡大: {previous.diff_ratio_stdev:.6f} → "
                        f"{stats.diff_ratio_stdev:.6f}"
                    ),
                    observed=ratio,
                    threshold=DIFF_RATIO_STDEV_GROW_FACTOR,
                )
            )

    return violations
```

`evaluate_output_invariants` の本体を差し替える。

```python
    stats = build_run_stats_from_results(output_rows)
    violations = _check_absolute(requested_model_names, loaded_model_names, output_rows)

    # B-3 の絶対条件: 全銘柄が同じ予測値（分散ゼロ）。前回統計を要さない。
    # 1 銘柄しかない場合は分散 0 が自然なので除外する。
    if stats is not None and stats.symbol_count >= 2 and stats.diff_ratio_stdev == 0.0:
        violations.append(
            InvariantViolation(
                violation_id="B-3",
                description="全銘柄の予測変化率が同一（分散ゼロ）",
                observed=0.0,
                threshold=0.0,
            )
        )

    compared = False
    if stats is not None and previous_stats is not None:
        compared = True
        violations.extend(_check_regression(stats, previous_stats))

    return InvariantReport(
        violations=violations,
        stats=stats,
        compared_with_previous=compared,
    )
```

- [ ] **Step 4: テストが通ることを確認**

Run: `cd python; py -m pytest tests/unit/test_output_invariants.py -v`
Expected: PASS（13 件）

- [ ] **Step 5: コミット**

```bash
git add python/src/prediction/output_invariants.py python/tests/unit/test_output_invariants.py
git commit -m "feat: 予測出力の急変invariant（B-1/B-2/B-3）を追加"
```

---

### Task 3: `preload_models` がロード成功したモデル名を返す

**Files:**
- Modify: `python/src/prediction/predict_unified.py:70-85`
- Test: `python/tests/unit/test_output_invariants.py`（新規クラスを追記）

**Interfaces:**
- Produces: `preload_models(model_types: List[str] = None) -> List[str]`（ロードに成功したモデル名のリスト。要求順を保つ）

**Note:** もう一つの呼び出し元 `predict_all_unified_multi_horizon`（`prediction_pipeline.py:162`）は本番経路から呼ばれていない（呼び出し元はテストのみ、2026-08-01 確認済み）。戻り値の追加は既存呼び出し元に影響しない。

- [ ] **Step 1: 失敗するテストを書く**

`python/tests/unit/test_output_invariants.py` の末尾（`if __name__ == "__main__":` の直前）に追記する。

```python
class TestPreloadModelsReturnValue(unittest.TestCase):
    def test_returns_only_successfully_loaded_names(self):
        from unittest.mock import patch

        from src.prediction.predict_unified import preload_models

        def fake_get_cached_model(name):
            return object() if name == "UnifiedStockXGBoost" else None

        with patch(
            "src.prediction.predict_unified.get_cached_model",
            side_effect=fake_get_cached_model,
        ):
            loaded = preload_models(["UnifiedStockXGBoost", "UnifiedStockLightGBM"])

        self.assertEqual(loaded, ["UnifiedStockXGBoost"])

    def test_returns_all_names_when_all_load(self):
        from unittest.mock import patch

        from src.prediction.predict_unified import preload_models

        with patch(
            "src.prediction.predict_unified.get_cached_model",
            return_value=object(),
        ):
            loaded = preload_models(["UnifiedStockXGBoost", "UnifiedStockLightGBM"])

        self.assertEqual(loaded, ["UnifiedStockXGBoost", "UnifiedStockLightGBM"])
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `cd python; py -m pytest tests/unit/test_output_invariants.py -v -k Preload`
Expected: FAIL — `AssertionError: None != ['UnifiedStockXGBoost']`

- [ ] **Step 3: 実装を書く**

`python/src/prediction/predict_unified.py` の `preload_models` を書き換える。

```python
def preload_models(model_types: List[str] = None) -> List[str]:
    """
    モデルを事前にロードしてキャッシュする
    並列処理の前に呼び出すことで、スレッド間でモデルを共有できる

    Returns:
        ロードに成功したモデル名のリスト（要求順）。
        出力 invariant の A-1 / A-2 が期待値としてこれを使う。
    """
    if model_types is None:
        model_types = ["UnifiedStockXGBoost", "UnifiedStockLightGBM"]

    logger.info("モデルを事前ロード中: %s", model_types)
    loaded: List[str] = []
    for model_name in model_types:
        model = get_cached_model(model_name)
        if model is not None:
            logger.info("  - %s: ロード完了", model_name)
            loaded.append(model_name)
        else:
            logger.info("  - %s: 見つかりません", model_name)
    logger.info("モデルの事前ロード完了: %d/%d", len(loaded), len(model_types))
    return loaded
```

- [ ] **Step 4: テストが通ることを確認**

Run: `cd python; py -m pytest tests/unit/test_output_invariants.py tests/unit/test_prediction_pipeline.py -v`
Expected: PASS（既存の `predict_all_unified_multi_horizon` テストも回帰なし）

- [ ] **Step 5: コミット**

```bash
git add python/src/prediction/predict_unified.py python/tests/unit/test_output_invariants.py
git commit -m "feat: preload_models がロード成功したモデル名を返すようにする"
```

---

### Task 4: 前回ラン統計を引く DB クエリ

**Files:**
- Modify: `python/src/prediction/db/prediction_results.py`
- Modify: `python/src/prediction/db/__init__.py`
- Test: `python/tests/unit/test_output_invariants.py`（新規クラスを追記）

**Interfaces:**
- Produces: `load_previous_run_stats(exclude_predicted_at: str, model_version: str = "production") -> Optional[tuple[list[int], list[float]]]`
  - 戻り値は `(model_counts, diff_ratios)`。前回ランが存在しなければ `None`
  - 集計そのものは行わない。`build_run_stats` が Python 側で計算する（SQL 方言差を避けるため）

- [ ] **Step 1: 失敗するテストを書く**

`python/tests/unit/test_output_invariants.py` の末尾（`if __name__ == "__main__":` の直前）に追記する。

```python
class TestLoadPreviousRunStats(unittest.TestCase):
    def _patched_connection(self, rows_for_timestamp, rows_for_data):
        from unittest.mock import MagicMock, patch

        cursor = MagicMock()
        cursor.fetchone.return_value = rows_for_timestamp
        cursor.fetchall.return_value = rows_for_data

        con = MagicMock()
        con.execute.return_value = cursor

        ctx = MagicMock()
        ctx.__enter__.return_value = con
        ctx.__exit__.return_value = False

        return patch(
            "src.prediction.db.prediction_results._db_connection", return_value=ctx
        )

    def test_returns_none_when_no_previous_run(self):
        from src.prediction.db.prediction_results import load_previous_run_stats

        with self._patched_connection(None, []):
            result = load_previous_run_stats("20260801_073000")

        self.assertIsNone(result)

    def test_returns_model_counts_and_diff_ratios(self):
        from src.prediction.db.prediction_results import load_previous_run_stats

        with self._patched_connection(
            ("20260731_073000",), [(2, 0.01), (2, -0.02), (1, 0.005)]
        ):
            result = load_previous_run_stats("20260801_073000")

        self.assertEqual(result, ([2, 2, 1], [0.01, -0.02, 0.005]))

    def test_returns_none_on_db_error(self):
        from unittest.mock import patch

        from src.prediction.db.prediction_results import load_previous_run_stats

        with patch(
            "src.prediction.db.prediction_results._db_connection",
            side_effect=RuntimeError("connection refused"),
        ):
            result = load_previous_run_stats("20260801_073000")

        self.assertIsNone(result)
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `cd python; py -m pytest tests/unit/test_output_invariants.py -v -k LoadPrevious`
Expected: FAIL — `ImportError: cannot import name 'load_previous_run_stats'`

- [ ] **Step 3: 実装を書く**

`python/src/prediction/db/prediction_results.py` の `load_latest_prediction_timestamp`（80行目付近）の直後に追加する。

```python
def load_previous_run_stats(
    exclude_predicted_at: str, model_version: str = "production"
) -> Optional[tuple[list[int], list[float]]]:
    """今回を除く直近ランの model_count / diff_ratio を返す。

    出力 invariant の急変チェック（B-1/B-2/B-3）が使う。集計は行わず生の値を
    返す。中央値・標準偏差の計算は Python 側で行い、SQL 方言差を持ち込まない。

    Args:
        exclude_predicted_at: 今回ランの predicted_at（これより前を対象にする）
        model_version: 対象のモデルバージョン

    Returns:
        (model_counts, diff_ratios) のタプル。前回ランが無ければ None。
    """
    try:
        with _db_connection() as con:
            row = con.execute(
                "SELECT predicted_at FROM prediction_results "
                "WHERE model_version = %s AND predicted_at < %s "
                "ORDER BY predicted_at DESC LIMIT 1",
                (model_version, exclude_predicted_at),
            ).fetchone()
            if not row:
                logger.info("前回ラン統計なし（比較をスキップ）")
                return None

            previous_at = row[0]
            rows = con.execute(
                "SELECT model_count, diff_ratio FROM prediction_results "
                "WHERE predicted_at = %s AND model_version = %s",
                (previous_at, model_version),
            ).fetchall()

        model_counts = [int(r[0]) for r in rows if r[0] is not None]
        diff_ratios = [float(r[1]) for r in rows if r[1] is not None]
        logger.info("前回ラン統計を取得: predicted_at=%s 件数=%d", previous_at, len(model_counts))
        return model_counts, diff_ratios
    except Exception as e:
        logger.error(f"前回ラン統計の取得失敗: {e}", exc_info=True)
        return None
```

`python/src/prediction/db/__init__.py` の `from .prediction_results import (...)` ブロックに `load_previous_run_stats,` を追加し、同ファイル末尾の `__all__` リストにも `"load_previous_run_stats",` を追加する。

- [ ] **Step 4: テストが通ることを確認**

Run: `cd python; py -m pytest tests/unit/test_output_invariants.py tests/unit/test_db_prediction.py -v`
Expected: PASS

- [ ] **Step 5: コミット**

```bash
git add python/src/prediction/db/prediction_results.py python/src/prediction/db/__init__.py python/tests/unit/test_output_invariants.py
git commit -m "feat: 前回ラン統計を取得するクエリを追加"
```

---

### Task 5: `alert_service` に NF-303-5 を追加し、サマリー送信をやめる

**Files:**
- Modify: `python/src/utils/alert_service.py`
- Modify: `python/tests/unit/test_alert_service.py:293`（および `run_conditional_notification` のテスト群）
- Test: `python/tests/unit/test_alert_service.py`

**Interfaces:**
- Consumes: なし（`src.prediction` を import してはならない。`.importlinter` Contract 1 違反になる）
- Produces:
  - `check_prediction_output_rule(violation_ids: Optional[list[str]], details: Optional[dict] = None) -> AlertResult`
  - `evaluate_alert_conditions(state_file_path=None, prediction_violation_ids=None, prediction_details=None) -> list[AlertResult]`
  - 定数 `PREDICTION_OUTPUT_THRESHOLD = 1`

**設計上の判断:** `violation_ids=None` は「評価が実行されなかった」を意味し、**違反として扱う**（`triggered=True`、違反ID `A-0`）。空リスト `[]` は「評価した結果、違反なし」を意味する。両者を区別しないと、評価が例外で落ちたときに正常と見分けがつかなくなり、本設計が防ごうとしている失敗モードそのものを再現してしまう。

- [ ] **Step 1: 失敗するテストを書く**

`python/tests/unit/test_alert_service.py` の `TestEvaluateAlertConditions` クラスの直前に追記する。

```python
class TestPredictionOutputRule(unittest.TestCase):
    def test_no_violation_is_not_triggered(self):
        from src.utils.alert_service import check_prediction_output_rule

        result = check_prediction_output_rule([], {"symbol_count": 705})
        self.assertFalse(result.triggered)
        self.assertEqual(result.rule_id, "NF-303-5")

    def test_single_violation_triggers_immediately(self):
        """ストリークを使わない。1 回の違反で即発報する。"""
        from src.utils.alert_service import check_prediction_output_rule

        result = check_prediction_output_rule(["A-2"], {"symbol_count": 705})
        self.assertTrue(result.triggered)
        self.assertEqual(result.consecutive_count, 1)
        self.assertEqual(result.threshold, 1)

    def test_unevaluated_is_treated_as_violation(self):
        """評価が実行されなかった場合を正常と見分けられるようにする。"""
        from src.utils.alert_service import check_prediction_output_rule

        result = check_prediction_output_rule(None)
        self.assertTrue(result.triggered)
        self.assertIn("A-0", result.details.get("violation_ids", []))
```

続けて `TestEvaluateAlertConditions.test_returns_four_results` を書き換える。既存の 4 つの `patch` に 5 つ目を足し、メソッド名と期待値を変える。

```python
    def test_returns_five_results(self):
        with (
            patch(
                "src.utils.alert_service.check_pipeline_fail_rule",
                return_value=AlertResult("NF-303-1", "A", False, 0, 2),
            ),
            patch(
                "src.utils.alert_service.check_loss_limit_rule",
                return_value=AlertResult("NF-303-2", "B", False, 0, 3),
            ),
            patch(
                "src.utils.alert_service.check_drift_warn_rule",
                return_value=AlertResult("NF-303-3", "C", False, 0, 2),
            ),
            patch(
                "src.utils.alert_service.check_health_degraded_rule",
                return_value=AlertResult("NF-303-4", "D", False, 0, 2),
            ),
            patch(
                "src.utils.alert_service.check_prediction_output_rule",
                return_value=AlertResult("NF-303-5", "E", False, 0, 1),
            ),
        ):
            results = evaluate_alert_conditions(prediction_violation_ids=[])
        self.assertEqual(len(results), 5)
```

`TestEvaluateAlertConditions._all_ok_results` にも 5 件目を足す。

```python
            AlertResult("NF-303-5", "E", False, 0, 1),
```

さらに `test_sends_summary_when_all_ok`（366 行目付近）を「送らない」挙動に書き換える。

```python
    def test_does_not_send_summary_when_all_ok(self):
        """条件非成立時は Discord へ何も送らない（違反時のみ発報する）。"""
        results = self._all_ok_results()
        notifier = MagicMock(return_value=True)
        with (
            patch("src.utils.alert_service._send_alert_detail") as mock_detail,
            patch("src.utils.alert_service._send_daily_summary") as mock_summary,
        ):
            ok = run_conditional_notification(results=results, notifier=notifier)
        self.assertFalse(ok)
        mock_detail.assert_not_called()
        mock_summary.assert_not_called()
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `cd python; py -m pytest tests/unit/test_alert_service.py -v`
Expected: FAIL — `ImportError: cannot import name 'check_prediction_output_rule'` および `len(results) == 4 != 5`

- [ ] **Step 3: 実装を書く**

`python/src/utils/alert_service.py` の閾値定数ブロックに追記する。

```python
PREDICTION_OUTPUT_THRESHOLD = 1  # 予測出力 invariant はストリークを使わず即発報
```

`# アラート条件の一括評価` の区切りコメントの直前に新ルールを追加する。

```python
# ---------------------------------------------------------------------------
# ルール 5: 予測出力の健全性（出力 invariant）
# ---------------------------------------------------------------------------


def check_prediction_output_rule(
    violation_ids: list[str] | None,
    details: dict | None = None,
) -> AlertResult:
    """予測出力の不変条件違反を評価する。

    判定ロジック本体は src/prediction/output_invariants.py にある。本モジュールは
    src.utils 層にあり src.prediction を import できない（.importlinter Contract 1）
    ため、プリミティブのみを受け取る。

    Args:
        violation_ids: 違反 ID のリスト。None は「評価が実行されなかった」を意味し、
            違反として扱う。空リストは「評価した結果、違反なし」を意味する。
        details: 評価の内訳（銘柄数・中央値など）

    Returns:
        AlertResult（rule_id="NF-303-5"）
    """
    payload = dict(details or {})

    if violation_ids is None:
        payload["violation_ids"] = ["A-0"]
        payload["note"] = "出力 invariant の評価が実行されなかった"
        logger.warning("予測出力チェック: 評価が実行されていない（違反として扱う）")
        return AlertResult(
            rule_id="NF-303-5",
            name="予測出力の健全性",
            triggered=True,
            consecutive_count=1,
            threshold=PREDICTION_OUTPUT_THRESHOLD,
            details=payload,
        )

    payload["violation_ids"] = list(violation_ids)
    count = len(violation_ids)
    triggered = count >= PREDICTION_OUTPUT_THRESHOLD
    logger.info("予測出力チェック: violations=%d triggered=%s", count, triggered)
    return AlertResult(
        rule_id="NF-303-5",
        name="予測出力の健全性",
        triggered=triggered,
        consecutive_count=count,
        threshold=PREDICTION_OUTPUT_THRESHOLD,
        details=payload,
    )
```

`evaluate_alert_conditions` を書き換える。

```python
def evaluate_alert_conditions(
    state_file_path: str | None = None,
    prediction_violation_ids: list[str] | None = None,
    prediction_details: dict | None = None,
) -> list[AlertResult]:
    """
    全アラートルールを評価し、結果リストを返す。

    Args:
        state_file_path: scheduler_queue_state.json のパス（None = デフォルト）
        prediction_violation_ids: 予測出力 invariant の違反 ID。None は未評価を意味する
        prediction_details: 予測出力 invariant の内訳

    Returns:
        AlertResult リスト（triggered=True が優先アラート対象）
    """
    results = [
        check_pipeline_fail_rule(state_file_path),
        check_loss_limit_rule(),
        check_drift_warn_rule(),
        check_health_degraded_rule(),
        check_prediction_output_rule(prediction_violation_ids, prediction_details),
    ]
    triggered_count = sum(1 for r in results if r.triggered)
    logger.info("アラート評価完了: %d/%d 条件成立", triggered_count, len(results))
    return results
```

`_send_alert_detail` に違反の内訳を出せるよう、`AlertResult.as_discord_lines` を拡張する（63-68 行目）。

```python
    def as_discord_lines(self) -> list[str]:
        status = "🚨 **アラート**" if self.triggered else "✅ 正常"
        lines = [
            f"{status} [{self.rule_id}] {self.name}",
            f"  連続回数: {self.consecutive_count} / 閾値: {self.threshold}",
        ]
        for violation in self.details.get("violations", []):
            lines.append(
                f"  - [{violation.get('id')}] {violation.get('description')} "
                f"(実測 {violation.get('observed')} / 閾値 {violation.get('threshold')})"
            )
        return lines
```

`run_conditional_notification` の末尾を書き換え、非成立時はサマリーを送らないようにする。

```python
    if any_triggered:
        logger.warning("アラート条件成立 — 詳細通知を送信します")
        return _send_alert_detail(results, notifier)

    logger.info("アラート条件非成立 — 通知は送信しません")
    return False
```

`_send_daily_summary` は削除しない。`tests/unit/test_alert_service_extra_unit.py:73` が直接テストしており、将来サマリーを復活させる余地も残すためである。未使用警告を避けるため、docstring に「現在は未使用（違反時のみ発報する運用）」と明記する。

- [ ] **Step 4: テストが通ることを確認**

Run: `cd python; py -m pytest tests/unit/test_alert_service.py tests/unit/test_alert_service_extra_unit.py -v`
Expected: PASS

- [ ] **Step 5: import-linter でレイヤー違反がないことを確認**

Run: `cd python; py -m importlinter.cli lint`
Expected: `Contracts: 2 kept, 0 broken.`

- [ ] **Step 6: コミット**

```bash
git add python/src/utils/alert_service.py python/tests/unit/test_alert_service.py
git commit -m "feat: 予測出力の健全性ルール(NF-303-5)を追加し非成立時のサマリー送信を廃止"
```

---

### Task 6: 日次パイプラインへの配線

**Files:**
- Modify: `python/src/orchestration/jobs/daily.py:49-66`（[2/5] 予測ステージ）と `:114-121`（[5/5] 通知ステージ）
- Test: `python/tests/unit/test_daily_jobs_invariants.py`（新規）

**Interfaces:**
- Consumes:
  - `src.prediction.output_invariants.evaluate_output_invariants` / `build_run_stats`（Task 1・2）
  - `src.prediction.predict_unified.preload_models`（Task 3、戻り値を使う）
  - `src.prediction.db.load_previous_run_stats`（Task 4）
  - `src.utils.alert_service.evaluate_alert_conditions` / `run_conditional_notification`（Task 5）
  - `src.reporting.discord.webhook_sender.send_webhook_notification`（既存。`(title, message, color) -> bool` で `NotifierFn` と一致する）
- Produces: なし（最終配線）

**Note:** `predict_all_unified()` は内部で `preload_models` を呼んでおり、ロード成功リストを外へ返さない。`daily.py` 側で `preload_models` を先に呼んでからロード結果を受け取る。`get_cached_model` はキャッシュ済みモデルを返すため、二重ロードのコストは発生しない。

- [ ] **Step 1: 失敗するテストを書く**

`python/tests/unit/test_daily_jobs_invariants.py` を新規作成する。

```python
"""日次パイプラインへの出力 invariant 配線のテスト。"""

import unittest
from unittest.mock import MagicMock, patch

from src.prediction.types import PredictionResult
from src.utils.alert_service import AlertResult


def _rows(count: int, model_count: int):
    return [
        PredictionResult(
            market="jp",
            symbol=str(7000 + i),
            current_price=1000.0 + i,
            avg_pred_price=1010.0 + i,
            diff_ratio=0.001 * (i - count / 2),
            model_count=model_count,
        )
        for i in range(count)
    ]


class TestDailyPipelineInvariantWiring(unittest.TestCase):
    def _run_pipeline(self, output_rows, loaded_models, previous):
        """日次パイプラインを最小のモックで走らせ、通知に渡った results を返す。"""
        captured = {}

        def fake_run_conditional_notification(results=None, **kwargs):
            captured["results"] = results
            return True

        with (
            patch("src.watchlist.batch_runner.load_target_symbols", return_value=[]),
            patch("src.market_data.pipeline.run_batch_pipeline"),
            patch(
                "src.infrastructure.discord_notification_adapter."
                "DiscordNotificationAdapter",
                MagicMock(),
            ),
            patch(
                "src.prediction.predict_unified.preload_models",
                return_value=loaded_models,
            ),
            patch(
                "src.prediction.prediction_pipeline.predict_all_unified",
                return_value=output_rows,
            ),
            patch("src.prediction.prediction_pipeline.output_top_worst_results"),
            patch(
                "src.prediction.db.prediction_results.load_previous_run_stats",
                return_value=previous,
            ),
            patch("src.prediction.shadow_evaluation.predict_with_challenger_unified", return_value=[]),
            patch("src.prediction.prediction_pipeline.run_accuracy_check", return_value={}),
            patch("src.reporting.discord.discord_utils.send_accuracy_summary"),
            patch("src.orchestration.jobs.daily.run_daily_drift_check"),
            patch("src.reporting.discord.discord_utils.send_daily_pipeline_completion"),
            patch(
                "src.utils.alert_service.run_conditional_notification",
                side_effect=fake_run_conditional_notification,
            ),
            # 既存 4 ルールは DB / 状態ファイルを触るためテストから遮断する。
            # 密閉性が崩れると CI とデプロイが落ちる（#516 / #517 の前例）。
            patch(
                "src.utils.alert_service.check_pipeline_fail_rule",
                return_value=AlertResult("NF-303-1", "A", False, 0, 2),
            ),
            patch(
                "src.utils.alert_service.check_loss_limit_rule",
                return_value=AlertResult("NF-303-2", "B", False, 0, 3),
            ),
            patch(
                "src.utils.alert_service.check_drift_warn_rule",
                return_value=AlertResult("NF-303-3", "C", False, 0, 2),
            ),
            patch(
                "src.utils.alert_service.check_health_degraded_rule",
                return_value=AlertResult("NF-303-4", "D", False, 0, 2),
            ),
        ):
            from src.orchestration.jobs.daily import run_daily_pipeline

            run_daily_pipeline()

        return captured.get("results")

    def test_degraded_ensemble_reaches_notification(self):
        """#615 の状態が NF-303-5 の triggered として通知に届く。"""
        results = self._run_pipeline(
            output_rows=_rows(705, model_count=1),
            loaded_models=["UnifiedStockXGBoost", "UnifiedStockLightGBM"],
            previous=None,
        )
        self.assertIsNotNone(results)
        rule = next(r for r in results if r.rule_id == "NF-303-5")
        self.assertTrue(rule.triggered)
        self.assertIn("A-2", rule.details["violation_ids"])

    def test_healthy_run_is_not_triggered(self):
        results = self._run_pipeline(
            output_rows=_rows(705, model_count=2),
            loaded_models=["UnifiedStockXGBoost", "UnifiedStockLightGBM"],
            previous=None,
        )
        rule = next(r for r in results if r.rule_id == "NF-303-5")
        self.assertFalse(rule.triggered)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `cd python; py -m pytest tests/unit/test_daily_jobs_invariants.py -v`
Expected: FAIL — `StopIteration`（`NF-303-5` が results に存在しない）もしくは `results is None`

- [ ] **Step 3: 実装を書く**

`python/src/orchestration/jobs/daily.py` の [2/5] ステージ（49-66 行目）を書き換える。

```python
    # 2. 予測（CRITICAL: 失敗時はパイプライン停止 + Discord通知）
    logger.info("[2/5] 予測開始 (production)")
    from src.prediction.prediction_pipeline import output_top_worst_results, predict_all_unified

    # 出力 invariant 用。None は「評価が実行されなかった」を意味する（#615 対策）
    prediction_violation_ids: list[str] | None = None
    prediction_details: dict | None = None

    try:
        from src.prediction.predict_unified import preload_models

        requested_models = ["UnifiedStockXGBoost", "UnifiedStockLightGBM"]
        loaded_models = preload_models(requested_models)

        output_rows = predict_all_unified()
        output_top_worst_results(
            output_rows, mode="unified", shadow_mode=True, model_version="production"
        )
        logger.info("[2/5] 予測完了 (production): %d 銘柄", len(output_rows))
    except Exception as e:
        if _handle_stage_error(
            PipelineStage.CRITICAL,
            "[2/5] 予測 (production)",
            e,
            send_daily_pipeline_error,
        ):
            raise

    # 2.1. 出力 invariant 評価（NON_CRITICAL: 健全性チェックが本体を止めてはならない）
    logger.info("[2.1/5] 出力 invariant 評価開始")
    try:
        from src.prediction.db import load_latest_prediction_timestamp
        from src.prediction.db.prediction_results import load_previous_run_stats
        from src.prediction.output_invariants import (
            build_run_stats,
            evaluate_output_invariants,
        )

        previous_stats = None
        current_at = load_latest_prediction_timestamp()
        if current_at:
            previous_raw = load_previous_run_stats(current_at)
            if previous_raw is not None:
                previous_stats = build_run_stats(previous_raw[0], previous_raw[1])

        report = evaluate_output_invariants(
            requested_model_names=requested_models,
            loaded_model_names=loaded_models,
            output_rows=output_rows,
            previous_stats=previous_stats,
        )
        prediction_violation_ids = report.violation_ids
        prediction_details = report.as_details()
        logger.info("[2.1/5] 出力 invariant 評価完了: %s", prediction_details)
    except Exception as e:
        _handle_stage_error(PipelineStage.NON_CRITICAL, "[2.1/5] 出力 invariant 評価", e)
```

続けて [5/5] ステージ（114-121 行目）を書き換える。

```python
    # 5. Discord通知（CRITICAL: 失敗時はパイプライン停止）
    logger.info("[5/5] Discord通知送信")
    try:
        send_daily_pipeline_completion()
        logger.info("[5/5] Discord通知完了")
    except Exception as e:
        if _handle_stage_error(PipelineStage.CRITICAL, "[5/5] Discord通知", e):
            raise

    # 6. 運用アラート評価（NON_CRITICAL: 条件成立時のみ Discord へ発報する）
    logger.info("[6/6] 運用アラート評価開始")
    try:
        from src.reporting.discord.webhook_sender import send_webhook_notification
        from src.utils.alert_service import (
            evaluate_alert_conditions,
            run_conditional_notification,
        )

        alert_results = evaluate_alert_conditions(
            prediction_violation_ids=prediction_violation_ids,
            prediction_details=prediction_details,
        )
        run_conditional_notification(
            results=alert_results, notifier=send_webhook_notification
        )
        logger.info("[6/6] 運用アラート評価完了")
    except Exception as e:
        _handle_stage_error(PipelineStage.NON_CRITICAL, "[6/6] 運用アラート評価", e)

    logger.info("=== 日次パイプライン完了 ===")
```

docstring（15-25 行目）の流れの記述も更新する。

```python
    """
    毎日実行: データ取得 → 予測 → 出力invariant評価 → Challenger shadow 予測
             → 精度チェック → ドリフト監視 → Discord通知 → 運用アラート評価

    流れ:
        1. 全マーケットのデータを取得（バッチ）
        2. Top10/Worst10の予測を実行（production）
        2.1. 出力 invariant 評価（非致命的。結果は 6 で通知する）
        2.5. Challenger shadow 予測（モデルが存在する場合のみ、非致命的）
        3. 前日予測の精度チェック: production / challenger それぞれ記録（非致命的）
        4. 日次ドリフトチェック（閾値超過銘柄を自動再学習）
        5. Discord通知
        6. 運用アラート評価（NF-303。条件成立時のみ発報）
    """
```

- [ ] **Step 4: テストが通ることを確認**

Run: `cd python; py -m pytest tests/unit/test_daily_jobs_invariants.py -v`
Expected: PASS（2 件）

- [ ] **Step 5: 既存の日次ジョブテストに回帰がないことを確認**

Run: `cd python; py -m pytest tests/unit/ -k "daily or scheduler or alert" -v`
Expected: PASS

- [ ] **Step 6: コミット**

```bash
git add python/src/orchestration/jobs/daily.py python/tests/unit/test_daily_jobs_invariants.py
git commit -m "feat: 日次パイプラインに出力invariant評価と運用アラート通知を配線"
```

---

### Task 7: VERSION 更新と CI 一括チェック

**Files:**
- Modify: `python/VERSION`

**Interfaces:**
- Consumes: Task 1〜6 のすべて
- Produces: なし

- [ ] **Step 1: `develop` の最新 VERSION を確認する**

Run: `git fetch; git show origin/develop:python/VERSION`
Expected: `2.2.1`（異なる場合はその値を基準に +1 する）

- [ ] **Step 2: VERSION を更新する**

`python/VERSION` の内容を `2.3.0` にする（後方互換の新機能追加なので minor）。

- [ ] **Step 3: CI 相当の一括チェックを実行する**

Run: `cd python; py -m pytest tests/unit/ -v --cov=src --cov-branch --cov-report=term-missing --cov-fail-under=80`
Expected: PASS、カバレッジ 80% 以上

**注意:** ローカルの unit テストは空 DB で実行すること。開発用 DB を指したままだと大量データでタイムアウトする。

- [ ] **Step 4: lint と型チェックを通す**

```bash
cd python
py -m black .
py -m isort .
py -m flake8 .
py -m mypy src/
py -m importlinter.cli lint
py scripts/check_file_size.py
```
Expected: flake8 と import-linter と file size がクリーン（mypy は本リポジトリでは非ブロッキングだが、本ブランチで新規・変更したファイルはクリーンにする）

- [ ] **Step 5: コミット**

```bash
git add python/VERSION
git commit -m "chore: VERSION を 2.3.0 に更新"
```

- [ ] **Step 6: PR を作成する**

PR ボディには CI の `validate-pr-body` が要求する以下のセクションを必ず含める。

```markdown
## version_impact
minor

## version_rationale
既存動作を変えずに予測出力の健全性チェックと運用アラート通知を追加する後方互換の機能追加。

## VERSION 更新
- version_update_required: yes
- version_before: 2.2.1
- version_after: 2.3.0

## VERSION 未更新理由
該当なし
```

ベースブランチは `develop`。

---

## 実装後の確認（人が行う）

- 最初の営業日 07:30 JST の `daily_pipeline` 実行後、コンテナログに `[2.1/5] 出力 invariant 評価完了` の構造化ログが出ていること
- 同ランの `prediction_results` で `model_count=2` になっていること（#616 の効果確認と兼ねる）
- 配線により既存 4 ルールが本番で初めて動作する。NF-303-1（パイプライン連続失敗）は 2026-08-01 時点で連続失敗 0 を実測確認済みのため、初日の誤発報懸念はない
