"""
戦略ファクトリー Phase 1（#369）: レポート出力（不変 JSON、原子書き込み）

ゲート合格仮説の Issue 起票用 JSON レポートを組み立てて書き出す。
仮説単位レビュー（hypothesis_review.py）の結果を受け取り Issue 本文に埋め込むが、
レビューの生成自体はこのモジュールの責務ではない（呼び出し元が渡す）。

GitHub への Issue 起票は IssueAgent 側（--factory-intake）の責務であり、
本モジュールは GitHub トークンを一切扱わない。
"""

from __future__ import annotations

import json
import math
import os
from datetime import datetime
from typing import Optional

from config.settings import (
    FACTORY_GATE_CHAMPION_MARGIN,
    FACTORY_GATE_MAX_DRAWDOWN,
    FACTORY_GATE_MAX_PBO,
    FACTORY_GATE_MIN_DSR,
    FACTORY_GATE_MIN_EFFECTIVE_SYMBOLS,
    FACTORY_GATE_MIN_TRADES,
    FACTORY_GATE_MIN_TRADES_PER_SYMBOL,
)
from src.backtest.types import FactoryEvaluation
from src.utils.data_path_utils import ensure_dir, get_results_dir

_REPORT_SCHEMA_VERSION = 1


def _reports_dir() -> str:
    path = os.path.join(get_results_dir(), "factory", "reports")
    ensure_dir(path)
    return path


def _build_review_section(review: Optional[dict]) -> str:
    """レビュー結果を Markdown セクション化する。review が None なら空文字を返す。"""
    if not review:
        return ""
    risk_level = review.get("risk_level", "low")
    assessment = review.get("assessment", "")
    concerns = review.get("concerns") or []
    banner = (
        f"\n> ⚠️ **Claude批判的レビュー: risk_level={risk_level}**\n"
        if risk_level == "high"
        else ""
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


def _build_spec_section(spec: dict) -> str:
    """ルールスペックを Markdown セクション化する。generated_code は python フェンスで表示。"""
    if spec.get("type") == "generated_code":
        return f"""### 生成ルール（Claude提案）

**{spec.get('rule_name', '?')}**: {spec.get('description', '')}

```python
{spec.get('source_code', '')}
```
"""
    return f"""### スペック

```json
{json.dumps(spec, ensure_ascii=False, indent=2)}
```
"""


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
    spec_section = _build_spec_section(h.rule_spec)
    effective_symbols_row = (
        f"| 有効銘柄（{FACTORY_GATE_MIN_TRADES_PER_SYMBOL}取引以上） "
        f"| {evaluation.n_effective_symbols} | >= {FACTORY_GATE_MIN_EFFECTIVE_SYMBOLS} |"
    )
    return f"""## 戦略仮説（自動生成）

夜間ファクトリーのゲートを通過した仮説です。`hypothesis_hash={h.hypothesis_hash}`
{pbo_warning}

{spec_section}
- マーケット: {h.market}
- 評価期間: {period[0]} 〜 {period[1]}（{h.lookback_years}年、データ取得銘柄数 {evaluation.n_symbols}）

### メトリクス

| 指標 | 値 | ゲート |
|---|---|---|
| Sharpe（有効銘柄平均） | {evaluation.sharpe_ratio:.3f} | {champion_cell} |
| Deflated Sharpe | {evaluation.dsr:.3f} | >= {FACTORY_GATE_MIN_DSR} |
| PBO | {evaluation.pbo:.3f} | <= {FACTORY_GATE_MAX_PBO} |
| 取引数（有効銘柄合計） | {evaluation.num_trades} | >= {FACTORY_GATE_MIN_TRADES} |
| シグナル発生銘柄 | {evaluation.n_symbols_with_signal} | - |
{effective_symbols_row}
| 銘柄あたり平均取引数（シグナル発生銘柄基準） | {evaluation.avg_trades_per_symbol:.2f} | - |
| 最大DD（最悪銘柄） | {evaluation.max_drawdown:.2%} | >= {FACTORY_GATE_MAX_DRAWDOWN:.0%} |
| 勝率（有効銘柄平均） | {evaluation.win_rate:.2%} | - |
| リターン（有効銘柄平均） | {evaluation.total_return:.2%} | - |

### 窓別リターン（銘柄平均）

| 窓 | リターン |
|---|---|
{window_rows}
{review_section}
---
*この Issue は StockFixer 戦略ファクトリー（#369 Phase 1）が自動生成したレポートです。*
"""


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
            "n_symbols_with_signal": evaluation.n_symbols_with_signal,
            "n_effective_symbols": evaluation.n_effective_symbols,
            "avg_trades_per_symbol": evaluation.avg_trades_per_symbol,
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
