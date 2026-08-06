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
マーケット: {h.market} / 評価期間: {h.lookback_years}年 / データ取得銘柄数: {evaluation.n_symbols}

## メトリクス
- Sharpe（有効銘柄平均）: {evaluation.sharpe_ratio:.3f}
- Deflated Sharpe (DSR): {evaluation.dsr:.3f}
- PBO: {evaluation.pbo:.3f}
- 取引数（有効銘柄合計）: {evaluation.num_trades}
- シグナル発生銘柄数: {evaluation.n_symbols_with_signal}
- 有効銘柄数（集計母数）: {evaluation.n_effective_symbols}
- 銘柄あたり平均取引数: {evaluation.avg_trades_per_symbol:.2f}
- 最大DD（最悪銘柄）: {evaluation.max_drawdown:.2%}
- 勝率（有効銘柄平均）: {evaluation.win_rate:.2%}
- リターン（有効銘柄平均）: {evaluation.total_return:.2%}
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
