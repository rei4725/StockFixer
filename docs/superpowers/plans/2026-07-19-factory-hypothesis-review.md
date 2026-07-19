# 戦略ファクトリー: 仮説単位の自動批判的レビュー Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 戦略ファクトリーのゲートを通過した個々の仮説について、Claude に窓別リターン・PBO・DSR 等を渡して批判的レビューを行わせ、その結果を Issue 本文へ自動で埋め込む。

**Architecture:** 新規モジュール `src/backtest/hypothesis_review.py` が単一仮説の情報を Claude（`get_text_review_port()`）へ渡し、`{risk_level, assessment, concerns}` の構造化 JSON を返す。`src/backtest/factory.py` の `run_factory_batch` がゲート通過時にこれを呼び出し、`write_report` の `issue_body` に追記してから1回で書き出す。レビュー失敗時は `None` を返し、レポートは通常通りレビューなしで書かれる（ゲート判定はレビューに依存しない）。

**Tech Stack:** Python 3.11 / `src.infrastructure.llm.factory.get_text_review_port()`（既存の SDK/CLI 切替ポート） / pydantic-settings / unittest + `unittest.mock`

## Global Constraints

- `FACTORY_HYPOTHESIS_REVIEW_ENABLED` の既定値は `False`（無効ロールアウト、既存の `BACKTEST_REVIEW_ENABLED` 等と同じ思想）。
- レビューはゲート判定（合格/不合格）に一切関与しない。失敗時は `logger.error(..., exc_info=True)` を出し `None` を返す（`except: pass` 禁止）。
- 過去の仮説履歴との比較は行わない。Claude に渡すのは当該仮説の情報のみ。
- `auto-ok` ラベルは付与しない（既存方針を維持、変更対象外）。
- import は `python/` からの絶対パスで統一（例: `from src.backtest.hypothesis_review import review_hypothesis`）。
- ロガーは `from src.utils.logger import get_logger; logger = get_logger(__name__)` を使用。

---

## File Structure

| ファイル | 種別 | 責務 |
|---|---|---|
| `python/config/settings.py` | 変更 | `FACTORY_HYPOTHESIS_REVIEW_*` の3設定を追加（Field定義＋モジュール平坦化エクスポート） |
| `python/src/backtest/hypothesis_review.py` | 新規 | 単一仮説の批判的レビュー（Claude 呼び出し・スキーマ検証・graceful degradation） |
| `python/src/backtest/factory.py` | 変更 | `_build_issue_body`/`write_report` にレビュー埋め込みを追加、`run_factory_batch` からレビュー呼び出し |
| `python/tests/unit/test_hypothesis_review.py` | 新規 | `hypothesis_review.py` のユニットテスト |
| `python/tests/unit/test_strategy_factory.py` | 変更 | レビュー埋め込み・レビュー失敗時の graceful degradation のテストケース追加 |

---

### Task 1: `hypothesis_review.py` の新規実装（設定追加込み）

**Files:**
- Modify: `python/config/settings.py:87-104`（`Settings` クラス内、`BACKTEST_REVIEW_*` ブロックの直後に追加）
- Modify: `python/config/settings.py:205-207`（モジュール平坦化エクスポートブロック、`BACKTEST_REVIEW_*` の直後に追加）
- Create: `python/src/backtest/hypothesis_review.py`
- Test: `python/tests/unit/test_hypothesis_review.py`

**Interfaces:**
- Consumes:
  - `src.backtest.types.FactoryEvaluation`（既存。フィールド: `hypothesis: FactoryHypothesis`, `sharpe_ratio: float`, `dsr: float`, `pbo: float`, `num_trades: int`, `max_drawdown: float`, `win_rate: float`, `total_return: float`, `window_returns: list[float]`, `n_symbols: int`）
  - `src.backtest.types.FactoryHypothesis`（既存。フィールド: `rule_spec: dict`, `market: str`, `lookback_years: int`, `hypothesis_hash: str` プロパティ）
  - `src.infrastructure.llm.factory.get_text_review_port() -> TextReviewPort`（既存。`port.complete(*, system: str, user: str, model: str, max_tokens: int, schema: Optional[dict] = None) -> str`）
- Produces:
  - `review_hypothesis(evaluation: FactoryEvaluation, champion_sharpe: float) -> Optional[dict]`
    - 戻り値の dict 形状: `{"risk_level": "low"|"medium"|"high", "assessment": str, "concerns": list[str]}`
    - 無効時・生成/解析失敗時は `None`
  - Task 2 はこの関数を `factory.py` から呼び出す。

- [ ] **Step 1: `config/settings.py` に設定3項目を追加（Field定義）**

`python/config/settings.py` の92行目（`BACKTEST_REVIEW_MAX_TOKENS: int = Field(default=4096)`）の直後に挿入:

```python
    # ---------- Claude 仮説単位レビュー（backtest/hypothesis_review.py） ----------
    # 戦略ファクトリーのゲート通過仮説1件ごとに Claude が過学習/偶然性リスクを評価する。
    # 既定無効。読み取り専用（ゲート判定・コード変更・発注には一切関与しない）。
    FACTORY_HYPOTHESIS_REVIEW_ENABLED: bool = Field(default=False)
    FACTORY_HYPOTHESIS_REVIEW_MODEL: str = Field(default="claude-opus-4-8")
    FACTORY_HYPOTHESIS_REVIEW_MAX_TOKENS: int = Field(default=2048)
```

- [ ] **Step 2: `config/settings.py` のモジュール平坦化エクスポートに追加**

207行目（`BACKTEST_REVIEW_MAX_TOKENS: int = settings.BACKTEST_REVIEW_MAX_TOKENS`）の直後に挿入:

```python
FACTORY_HYPOTHESIS_REVIEW_ENABLED: bool = settings.FACTORY_HYPOTHESIS_REVIEW_ENABLED
FACTORY_HYPOTHESIS_REVIEW_MODEL: str = settings.FACTORY_HYPOTHESIS_REVIEW_MODEL
FACTORY_HYPOTHESIS_REVIEW_MAX_TOKENS: int = settings.FACTORY_HYPOTHESIS_REVIEW_MAX_TOKENS
```

- [ ] **Step 3: 失敗系・無効時のテストを先に書く（Red）**

Create `python/tests/unit/test_hypothesis_review.py`:

```python
"""ユニットテスト: 仮説単位の批判的レビュー（src/backtest/hypothesis_review.py）

anthropic クライアントは MagicMock で差し替え、API 呼び出しは行わない。
"""

import json
import unittest
from unittest.mock import MagicMock, patch

from src.backtest import hypothesis_review
from src.backtest.types import FactoryEvaluation, FactoryHypothesis

_SPEC = {"type": "atomic", "rule": "ema_momentum", "params": {"fast_window": 8, "slow_window": 21}}


def _make_evaluation(**kwargs):
    defaults = dict(
        hypothesis=FactoryHypothesis(rule_spec=_SPEC, market="jp"),
        sharpe_ratio=1.8,
        dsr=0.97,
        pbo=0.25,
        num_trades=45,
        max_drawdown=-0.12,
        win_rate=0.55,
        total_return=0.30,
        window_returns=[0.02, -0.01, 0.03, 0.01, 0.00, -0.02, 0.04, 0.01],
        n_symbols=5,
    )
    defaults.update(kwargs)
    return FactoryEvaluation(**defaults)


def _mock_anthropic(review: dict):
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = json.dumps(review)
    response = MagicMock()
    response.content = [text_block]
    client = MagicMock()
    client.messages.create.return_value = response
    module = MagicMock()
    module.Anthropic.return_value = client
    return module, client


_SAMPLE_REVIEW = {
    "risk_level": "medium",
    "assessment": "窓7のみリターンが突出しており、他窓は横ばい。",
    "concerns": ["窓7への依存度が高い", "取引数に対しSharpeがやや高い"],
}


# 環境変数 LLM_BACKEND に依存させず、anthropic モックが効く SDK 経路に固定する
@patch("src.infrastructure.llm.factory.LLM_BACKEND", "sdk")
class TestReviewHypothesis(unittest.TestCase):
    @patch("src.backtest.hypothesis_review.FACTORY_HYPOTHESIS_REVIEW_ENABLED", False)
    def test_disabled_returns_none(self):
        ev = _make_evaluation()
        self.assertIsNone(hypothesis_review.review_hypothesis(ev, champion_sharpe=1.0))

    @patch("src.backtest.hypothesis_review.FACTORY_HYPOTHESIS_REVIEW_ENABLED", True)
    def test_enabled_returns_parsed_review(self):
        ev = _make_evaluation()
        module, client = _mock_anthropic(_SAMPLE_REVIEW)
        with patch.dict("sys.modules", {"anthropic": module}):
            result = hypothesis_review.review_hypothesis(ev, champion_sharpe=1.0)
        self.assertEqual(result, _SAMPLE_REVIEW)
        # 構造化出力を要求していること
        _, kwargs = client.messages.create.call_args
        self.assertIn("output_config", kwargs)

    @patch("src.backtest.hypothesis_review.FACTORY_HYPOTHESIS_REVIEW_ENABLED", True)
    def test_api_error_returns_none(self):
        ev = _make_evaluation()
        module = MagicMock()
        module.Anthropic.side_effect = RuntimeError("API down")
        with patch.dict("sys.modules", {"anthropic": module}):
            self.assertIsNone(hypothesis_review.review_hypothesis(ev, champion_sharpe=1.0))

    @patch("src.backtest.hypothesis_review.FACTORY_HYPOTHESIS_REVIEW_ENABLED", True)
    def test_malformed_schema_returns_none(self):
        ev = _make_evaluation()
        module, _client = _mock_anthropic({"unexpected": "shape"})
        with patch.dict("sys.modules", {"anthropic": module}):
            self.assertIsNone(hypothesis_review.review_hypothesis(ev, champion_sharpe=1.0))

    @patch("src.backtest.hypothesis_review.FACTORY_HYPOTHESIS_REVIEW_ENABLED", True)
    def test_champion_nan_does_not_raise(self):
        ev = _make_evaluation()
        module, _client = _mock_anthropic(_SAMPLE_REVIEW)
        with patch.dict("sys.modules", {"anthropic": module}):
            result = hypothesis_review.review_hypothesis(ev, champion_sharpe=float("nan"))
        self.assertEqual(result, _SAMPLE_REVIEW)


class TestBuildReviewContext(unittest.TestCase):
    def test_context_includes_spec_and_metrics(self):
        ev = _make_evaluation()
        context = hypothesis_review._build_review_context(ev, champion_sharpe=1.0)
        self.assertIn("ema_momentum", context)
        self.assertIn("DSR", context.replace(" ", "").replace("(", "").replace(")", "") + "DSR")
        self.assertIn("0.970", context)  # dsr
        self.assertIn("窓1", context)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 4: テストを実行し FAIL を確認**

Run: `cd python && python -m pytest tests/unit/test_hypothesis_review.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.backtest.hypothesis_review'`

- [ ] **Step 5: `hypothesis_review.py` を実装（Green）**

Create `python/src/backtest/hypothesis_review.py`:

```python
"""
戦略ファクトリー: 仮説単位の批判的レビュー

ゲートを通過した個々の仮説について、Claude に窓別リターン・PBO・DSR 等を渡し、
過学習や偶然性のリスクを短く評価させる。人間の最終採否判断を補助するだけであり、
ゲート判定（合格/不合格）には一切関与しない。

バックエンドは LLM_BACKEND で選択する（sdk=API 課金 / cli=サブスク認証）。
FACTORY_HYPOTHESIS_REVIEW_ENABLED=False（既定）/生成・解析失敗時は None を返し、
呼び出し元（factory.py）はレビューなしでレポートを書き出す（graceful degradation）。
読み取り専用のレビューのみ（コード変更・発注判断には一切関与しない）。
過去の仮説履歴との比較は行わない（当該仮説の情報のみを渡す）。
"""

from __future__ import annotations

import json
import math
from typing import Any, Optional

from config.settings import (
    FACTORY_HYPOTHESIS_REVIEW_ENABLED,
    FACTORY_HYPOTHESIS_REVIEW_MAX_TOKENS,
    FACTORY_HYPOTHESIS_REVIEW_MODEL,
)
from src.backtest.types import FactoryEvaluation
from src.utils.logger import get_logger

logger = get_logger(__name__)

_SYSTEM_PROMPT = (
    "あなたはクオンツトレーディングシステムの厳格なレビュアーです。"
    "戦略ファクトリーが夜間バッチで生成した単一のルール仮説について、"
    "窓別リターン・Sharpe・Deflated Sharpe（DSR）・PBO・取引数などのメトリクスから、"
    "過学習や偶然性のリスクを評価してください。"
    "重点観点: 窓間のリターンのばらつき（一部の窓だけに依存していないか）、"
    "パラメータが探索グリッドの端に位置していないか、取引数に対して"
    "Sharpe が不自然に高くないか、PBO/DSR とリターン分布の整合性。"
    "与えられた情報のみを根拠とし、推測の数値を作らないこと。"
    "確証が薄い懸念は risk_level を下げること。"
)

_REVIEW_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "risk_level": {"type": "string", "enum": ["low", "medium", "high"]},
        "assessment": {"type": "string"},
        "concerns": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["risk_level", "assessment", "concerns"],
    "additionalProperties": False,
}


def _build_review_context(evaluation: FactoryEvaluation, champion_sharpe: float) -> str:
    h = evaluation.hypothesis
    window_lines = "\n".join(
        f"- 窓{i + 1}: {r:+.2%}" for i, r in enumerate(evaluation.window_returns)
    )
    champion_line = (
        "対照群（チャンピオン）Sharpe: なし"
        if math.isnan(champion_sharpe)
        else f"対照群（チャンピオン）Sharpe: {champion_sharpe:.3f}"
    )
    return f"""## 仮説スペック
```json
{json.dumps(h.rule_spec, ensure_ascii=False, indent=2)}
```
マーケット: {h.market} / 評価期間: {h.lookback_years}年 / 対象銘柄数: {evaluation.n_symbols}

## メトリクス
- Sharpe（銘柄平均）: {evaluation.sharpe_ratio:.3f}
- Deflated Sharpe (DSR): {evaluation.dsr:.3f}
- PBO: {evaluation.pbo:.3f}
- 取引数（合計）: {evaluation.num_trades}
- 最大DD（最悪銘柄）: {evaluation.max_drawdown:.2%}
- 勝率（銘柄平均）: {evaluation.win_rate:.2%}
- リターン（銘柄平均）: {evaluation.total_return:.2%}
- {champion_line}

## 窓別リターン（銘柄平均）
{window_lines}
"""


def review_hypothesis(evaluation: FactoryEvaluation, champion_sharpe: float) -> Optional[dict]:
    """仮説単位の批判的レビューを実行する。

    無効時・生成/解析失敗時は None を返す（呼び出し元はレビューなしでレポートを書き出す）。
    """
    if not FACTORY_HYPOTHESIS_REVIEW_ENABLED:
        return None

    from src.infrastructure.llm.factory import get_text_review_port  # noqa: PLC0415

    try:
        context = _build_review_context(evaluation, champion_sharpe)
        port = get_text_review_port()
        text = port.complete(
            system=_SYSTEM_PROMPT,
            user=context,
            model=FACTORY_HYPOTHESIS_REVIEW_MODEL,
            max_tokens=FACTORY_HYPOTHESIS_REVIEW_MAX_TOKENS,
            schema=_REVIEW_SCHEMA,
        )
        data = json.loads(text)
    except Exception:
        logger.error(
            "[hypothesis_review] レビュー生成でエラー: %s",
            evaluation.hypothesis.hypothesis_hash,
            exc_info=True,
        )
        return None

    if (
        not isinstance(data, dict)
        or "risk_level" not in data
        or "assessment" not in data
        or "concerns" not in data
    ):
        logger.warning(
            "[hypothesis_review] レビュー結果のスキーマ不正: %s",
            evaluation.hypothesis.hypothesis_hash,
        )
        return None

    return data
```

- [ ] **Step 6: テストを実行し PASS を確認**

Run: `cd python && python -m pytest tests/unit/test_hypothesis_review.py -v`
Expected: PASS（全ケース）

- [ ] **Step 7: Lint/型チェック**

Run: `cd python && black src/backtest/hypothesis_review.py config/settings.py tests/unit/test_hypothesis_review.py && isort src/backtest/hypothesis_review.py config/settings.py tests/unit/test_hypothesis_review.py && flake8 src/backtest/hypothesis_review.py config/settings.py && mypy src/backtest/hypothesis_review.py`
Expected: エラーなし

- [ ] **Step 8: Commit**

```bash
git add python/config/settings.py python/src/backtest/hypothesis_review.py python/tests/unit/test_hypothesis_review.py
git commit -m "feat: 戦略ファクトリー仮説単位の自動批判的レビューを追加"
```

---

### Task 2: `factory.py` への統合（Issue本文への埋め込み）

**Files:**
- Modify: `python/src/backtest/factory.py:377-425`（`_build_issue_body`）
- Modify: `python/src/backtest/factory.py:428-456`（`write_report`）
- Modify: `python/src/backtest/factory.py:536-555`（`run_factory_batch` の候補ループ）
- Test: `python/tests/unit/test_strategy_factory.py`

**Interfaces:**
- Consumes: `review_hypothesis(evaluation: FactoryEvaluation, champion_sharpe: float) -> Optional[dict]`（Task 1 で実装済み。戻り値は `{"risk_level": str, "assessment": str, "concerns": list[str]}` または `None`）
- Produces: `write_report(evaluation, champion_sharpe, period, review=None)` — `review` を受け取り `issue_body` に反映する（既存呼び出し元は `review` 省略で従来通り動作）

- [ ] **Step 1: レビュー埋め込みの失敗テストを先に書く（Red）**

`python/tests/unit/test_strategy_factory.py` の `TestWriteReport` クラスに以下のテストを追加（`test_high_batch_pbo_adds_warning_to_body` の直後）:

```python
    def test_review_section_embedded_when_provided(self):
        ev = FactoryEvaluation(
            hypothesis=FactoryHypothesis(rule_spec=_ATOMIC_SPEC, market="jp"),
            sharpe_ratio=1.5,
            dsr=0.96,
            pbo=0.2,
            num_trades=40,
            max_drawdown=-0.1,
            window_returns=[0.01] * 8,
            n_symbols=3,
        )
        review = {
            "risk_level": "high",
            "assessment": "窓1のみに依存した成績。",
            "concerns": ["窓1以外はほぼ横ばい"],
        }
        with tempfile.TemporaryDirectory() as tmp:
            with patch("src.backtest.factory.get_results_dir", return_value=tmp):
                path = write_report(
                    ev, champion_sharpe=1.0, period=("2024-01-01", "2026-01-01"), review=review
                )
            with open(path, encoding="utf-8") as f:
                report = json.load(f)
        self.assertIn("Claude批判的レビュー", report["issue_body"])
        self.assertIn("窓1のみに依存した成績。", report["issue_body"])
        self.assertIn("窓1以外はほぼ横ばい", report["issue_body"])
        self.assertIn("risk_level=high", report["issue_body"])
        self.assertEqual(report["review"], review)

    def test_review_section_omitted_when_none(self):
        ev = FactoryEvaluation(
            hypothesis=FactoryHypothesis(rule_spec=_ATOMIC_SPEC, market="jp"),
            sharpe_ratio=1.5,
            dsr=0.96,
            pbo=0.2,
            num_trades=40,
            max_drawdown=-0.1,
            window_returns=[0.01] * 8,
            n_symbols=3,
        )
        with tempfile.TemporaryDirectory() as tmp:
            with patch("src.backtest.factory.get_results_dir", return_value=tmp):
                path = write_report(ev, champion_sharpe=1.0, period=("2024-01-01", "2026-01-01"))
            with open(path, encoding="utf-8") as f:
                report = json.load(f)
        self.assertNotIn("Claude批判的レビュー", report["issue_body"])
        self.assertIsNone(report["review"])
```

`TestRunFactoryBatch` クラスに以下のテストを追加（`test_batch_aborts_without_symbol_data` の直後）:

```python
    @patch("src.backtest.factory.review_hypothesis")
    @patch("src.backtest.factory.save_factory_run")
    @patch("src.backtest.factory.count_factory_runs", return_value=0)
    @patch("src.backtest.factory.load_factory_hashes", return_value=set())
    @patch("src.backtest.factory.get_backtest_data_port")
    def test_batch_calls_review_only_for_passed_hypotheses(
        self, mock_port, mock_hashes, mock_count, mock_save, mock_review
    ):
        mock_port.return_value = self._fake_port()
        mock_review.return_value = None

        with tempfile.TemporaryDirectory() as tmp:
            with patch("src.backtest.factory.get_results_dir", return_value=tmp):
                result = run_factory_batch(
                    market="jp", symbols=["AAA", "BBB"], budget=4, n_windows=6, seed=123
                )

        self.assertEqual(mock_review.call_count, len(result.passed))

    @patch("src.backtest.factory.review_hypothesis", return_value=None)
    @patch("src.backtest.factory.save_factory_run")
    @patch("src.backtest.factory.count_factory_runs", return_value=0)
    @patch("src.backtest.factory.load_factory_hashes", return_value=set())
    @patch("src.backtest.factory.get_backtest_data_port")
    def test_review_none_still_writes_report(
        self, mock_port, mock_hashes, mock_count, mock_save, mock_review
    ):
        # review_hypothesis はグレースフルデグラデーション契約により失敗時 None を返す
        # （例外を投げない）。None が返ってもレポート書き込みは通常通り完了する。
        mock_port.return_value = self._fake_port()

        with tempfile.TemporaryDirectory() as tmp:
            with patch("src.backtest.factory.get_results_dir", return_value=tmp):
                result = run_factory_batch(
                    market="jp", symbols=["AAA", "BBB"], budget=4, n_windows=6, seed=123
                )

        for evaluation in result.passed:
            self.assertIsNotNone(evaluation.report_path)
            self.assertTrue(os.path.exists(evaluation.report_path))
            with open(evaluation.report_path, encoding="utf-8") as f:
                report = json.load(f)
            self.assertIsNone(report["review"])
```

- [ ] **Step 2: テストを実行し FAIL を確認**

Run: `cd python && python -m pytest tests/unit/test_strategy_factory.py -v`
Expected: FAIL — `TypeError: write_report() got an unexpected keyword argument 'review'` および `AttributeError: <module 'src.backtest.factory'> does not have the attribute 'review_hypothesis'`

- [ ] **Step 3: `factory.py` を修正（Green）**

`python/src/backtest/factory.py` の import ブロック（31行目付近、`from src.backtest.types import ...` の直前）に追加:

```python
from src.backtest.hypothesis_review import review_hypothesis
```

`_build_issue_body` を以下に置き換え（377-425行目）:

```python
def _build_review_section(review: Optional[dict]) -> str:
    """レビュー結果を Markdown セクション化する。review が None なら空文字を返す。"""
    if not review:
        return ""
    risk_level = review.get("risk_level", "low")
    assessment = review.get("assessment", "")
    concerns = review.get("concerns") or []
    banner = (
        f"\n> ⚠️ **Claude批判的レビュー: risk_level={risk_level}**\n" if risk_level == "high" else ""
    )
    concerns_block = ""
    if concerns:
        concern_lines = "\n".join(f"- {c}" for c in concerns)
        concerns_block = f"\n**懸念点:**\n{concern_lines}\n"
    return f"""
### Claude批判的レビュー
{banner}
{assessment}
{concerns_block}"""


def _build_issue_body(
    evaluation: FactoryEvaluation,
    champion_sharpe: float,
    period: tuple[str, str],
    review: Optional[dict] = None,
) -> str:
    h = evaluation.hypothesis
    window_rows = "\n".join(
        f"| {i + 1} | {r:+.2%} |" for i, r in enumerate(evaluation.window_returns)
    )
    champion_cell = f"> チャンピオン {champion_sharpe:.3f} × {FACTORY_GATE_CHAMPION_MARGIN}"
    pbo_warning = (
        f"\n> ⚠️ **バッチPBO={evaluation.pbo:.3f} > {FACTORY_GATE_MAX_PBO}**: "
        "この夜のバッチは選択過程の過学習リスクが高い。OOS 劣化に注意してレビューすること。\n"
        if not math.isnan(evaluation.pbo) and evaluation.pbo > FACTORY_GATE_MAX_PBO
        else ""
    )
    review_section = _build_review_section(review)
    return f"""## 戦略仮説（自動生成）

夜間ファクトリーのゲートを通過した仮説です。`hypothesis_hash={h.hypothesis_hash}`
{pbo_warning}

### スペック

```json
{json.dumps(h.rule_spec, ensure_ascii=False, indent=2)}
```

- マーケット: {h.market}
- 評価期間: {period[0]} 〜 {period[1]}（{h.lookback_years}年、銘柄数 {evaluation.n_symbols}）

### メトリクス

| 指標 | 値 | ゲート |
|---|---|---|
| Sharpe（銘柄平均） | {evaluation.sharpe_ratio:.3f} | {champion_cell} |
| Deflated Sharpe | {evaluation.dsr:.3f} | >= {FACTORY_GATE_MIN_DSR} |
| PBO | {evaluation.pbo:.3f} | <= {FACTORY_GATE_MAX_PBO} |
| 取引数（合計） | {evaluation.num_trades} | >= {FACTORY_GATE_MIN_TRADES} |
| 最大DD（最悪銘柄） | {evaluation.max_drawdown:.2%} | >= {FACTORY_GATE_MAX_DRAWDOWN:.0%} |
| 勝率（銘柄平均） | {evaluation.win_rate:.2%} | - |
| リターン（銘柄平均） | {evaluation.total_return:.2%} | - |

### 窓別リターン（銘柄平均）

| 窓 | リターン |
|---|---|
{window_rows}
{review_section}
---
*この Issue は StockFixer 戦略ファクトリー（#369 Phase 1）が自動生成したレポートです。*
"""
```

`write_report` を以下に置き換え（428-456行目）:

```python
def write_report(
    evaluation: FactoryEvaluation,
    champion_sharpe: float,
    period: tuple[str, str],
    review: Optional[dict] = None,
) -> str:
    """ゲート合格仮説の不変 JSON レポートを原子的に書き出してパスを返す。"""
    h = evaluation.hypothesis
    report = {
        "schema_version": _REPORT_SCHEMA_VERSION,
        "hypothesis_hash": h.hypothesis_hash,
        "created_at": datetime.now().astimezone().isoformat(),
        "issue_title": f"[factory:{h.hypothesis_hash}] {h.label} ({h.market})",
        "issue_body": _build_issue_body(evaluation, champion_sharpe, period, review=review),
        "labels": ["strategy-factory"],
        "gate": {
            "sharpe_ratio": evaluation.sharpe_ratio,
            "dsr": evaluation.dsr,
            "pbo": evaluation.pbo,
            "num_trades": evaluation.num_trades,
            "max_drawdown": evaluation.max_drawdown,
            "champion_sharpe": champion_sharpe,
        },
        "spec": h.rule_spec,
        "market": h.market,
        "review": review,
    }
    path = os.path.join(_reports_dir(), f"{h.hypothesis_hash}.json")
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)
    return path
```

`run_factory_batch` の候補ループを以下に置き換え（536-555行目付近、`for evaluation in result_candidates(evaluations):` から始まるブロック）:

```python
    for evaluation in result_candidates(evaluations):
        if evaluation.gate_passed:
            review = review_hypothesis(evaluation, champion_sharpe)
            evaluation.report_path = write_report(
                evaluation, champion_sharpe, (start, end), review=review
            )
        save_factory_run(
            hypothesis_hash=evaluation.hypothesis.hypothesis_hash,
            market=market,
            spec_json=json.dumps(evaluation.hypothesis.rule_spec, ensure_ascii=False),
            sharpe_ratio=evaluation.sharpe_ratio,
            win_rate=evaluation.win_rate,
            num_trades=evaluation.num_trades,
            max_drawdown=evaluation.max_drawdown,
            total_return=evaluation.total_return,
            dsr=evaluation.dsr,
            pbo=evaluation.pbo,
            gate_passed=evaluation.gate_passed,
            gate_reasons="; ".join(evaluation.gate_reasons) or None,
            report_path=evaluation.report_path,
        )
```

- [ ] **Step 4: テストを実行し PASS を確認**

Run: `cd python && python -m pytest tests/unit/test_strategy_factory.py tests/unit/test_hypothesis_review.py -v`
Expected: PASS（全ケース）

- [ ] **Step 5: 既存の関連テストに回帰がないことを確認**

Run: `cd python && python -m pytest tests/unit/test_critical_review.py tests/unit/ -k "factory or review" -v`
Expected: PASS（全ケース。既存の `test_critical_review.py` も無関係な変更で壊れていないことを確認）

- [ ] **Step 6: Lint/型チェック**

Run: `cd python && black src/backtest/factory.py tests/unit/test_strategy_factory.py && isort src/backtest/factory.py tests/unit/test_strategy_factory.py && flake8 src/backtest/factory.py && mypy src/backtest/factory.py`
Expected: エラーなし

- [ ] **Step 7: import-linter でレイヤー違反がないことを確認**

Run: `cd python && pre-commit run import-linter --files src/backtest/factory.py src/backtest/hypothesis_review.py`
Expected: PASS（`hypothesis_review.py` は同一 BC 内 `backtest/` の兄弟モジュールであり、`critical_review.py` と同じ import パターンのため違反なし）

- [ ] **Step 8: Commit**

```bash
git add python/src/backtest/factory.py python/tests/unit/test_strategy_factory.py
git commit -m "feat: 戦略ファクトリーのレポートに仮説単位レビューを埋め込み"
```

---

### Task 3: カバレッジと CI 一括チェック

**Files:**
- なし（検証のみ、コード変更なし）

**Interfaces:**
- Consumes: Task 1・Task 2 で作成した全ファイル
- Produces: なし（完了確認のみ）

- [ ] **Step 1: カバレッジ込みで対象テストを実行**

Run: `cd python && python -m pytest tests/unit/test_hypothesis_review.py tests/unit/test_strategy_factory.py -v --cov=src.backtest --cov-branch --cov-report=term-missing`
Expected: PASS。`src/backtest/hypothesis_review.py` の未カバー行がないか確認し、あれば Task 1/2 のテストに追記して埋める。

- [ ] **Step 2: CI 相当の一括チェックを実行**

Run: `cd python && .\check-ci.ps1`
Expected: lint / mypy / pylint / import-linter / unit tests (cov>=80%) / bandit / pip-audit（未インストール時はスキップ）すべて PASS

- [ ] **Step 3: 完了確認**

`docs/superpowers/specs/2026-07-19-factory-hypothesis-review-design.md` の「非目標」「スコープ外」に反する変更が紛れ込んでいないか（`auto-ok` ラベル付与・ゲート判定への関与・履歴比較の追加がないか）を diff で確認する。

Run: `git diff docs/superpowers/plans/../../.. --stat` の代わりに `git log --oneline -3` と `git diff HEAD~2 -- python/src/backtest python/config/settings.py` を確認。
