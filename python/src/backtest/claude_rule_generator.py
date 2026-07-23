"""Claudeに新しい TradingRule 実装を発想させ、機械的な失敗のみ修復リトライする。

ゲート判定（DSR/PBO等）への修復リトライは意図的に実装しない。ゲート不合格は
「バグ」ではなく「良いルールではなかった」という正当な判定結果であり、ここに
修復ループを回すと過学習対策そのものを破壊するため（p-hacking化）。
"""

from __future__ import annotations

import json
from typing import Any, Optional

from config.settings import (
    FACTORY_CLAUDE_RULEGEN_COUNT,
    FACTORY_CLAUDE_RULEGEN_ENABLED,
    FACTORY_CLAUDE_RULEGEN_MAX_REPAIR_ATTEMPTS,
    FACTORY_CLAUDE_RULEGEN_MAX_TOKENS,
    FACTORY_CLAUDE_RULEGEN_MODEL,
)
from src.backtest.sandbox_executor import SandboxRunResult, run_sandboxed_evaluation
from src.backtest.types import FactoryEvaluation, FactoryHypothesis
from src.infrastructure.llm.factory import get_text_review_port
from src.utils.logger import get_logger

logger = get_logger(__name__)

_SYSTEM_PROMPT = (
    "あなたはクオンツトレーディングシステムのルール開発者です。"
    "OHLCV + テクニカル指標（macd, macd_signal, macd_diff, ema_fast, ema_slow, atr, rsi, "
    "bb_upper, bb_middle, bb_lower, bb_width, stoch_k, stoch_d, obv, volume_ratio, "
    "volume_price_trend, volume_ma_deviation, day_of_week, month, is_month_end, "
    "および w_/m_ 接頭辞の週足/月足マルチタイムフレーム特徴量）を受け取り、"
    "1=buy, -1=sell, 0=hold を返す generate_signal(self, df) を実装した新しい "
    "売買ルールクラスを1つ提案してください。"
    "既存の手書きルール（出来高ブレイクアウト・EMAモメンタム・RSI逆張り・"
    "ボリンジャーバンド・MACD+RSI・ボラティリティブレイクアウト）とは異なる着眼点を"
    "選んでください。単一銘柄OHLCVと上記指標列のみが利用可能です（他銘柄・マクロ指標・"
    "ネットワークアクセスは一切使えません）。"
    "importはpandas/numpy/taのみ許可されます。os/subprocess/socket等は一切使わないこと。"
    '{"rule_name": str, "class_name": str, "description": str, "source_code": str} '
    "のJSONのみを出力してください（説明文やMarkdownのコードフェンスは不要）。"
)

_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "rule_name": {"type": "string"},
        "class_name": {"type": "string"},
        "description": {"type": "string"},
        "source_code": {"type": "string"},
    },
    "required": ["rule_name", "class_name", "description", "source_code"],
    "additionalProperties": False,
}


def _parse_response(text: str) -> Optional[dict]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    required = {"rule_name", "class_name", "description", "source_code"}
    if not isinstance(data, dict) or not required.issubset(data):
        return None
    return data


def _generate_one_candidate(
    market: str, repair_context: Optional[str] = None
) -> Optional[FactoryHypothesis]:
    port = get_text_review_port()
    user_prompt = f"マーケット: {market}\n新しいルールを1つ提案してください。"
    if repair_context:
        user_prompt += (
            f"\n\n直前の提案には以下の問題がありました。修正して同じJSON形式で再提案して"
            f"ください:\n{repair_context}"
        )
    try:
        text = port.complete(
            system=_SYSTEM_PROMPT,
            user=user_prompt,
            model=FACTORY_CLAUDE_RULEGEN_MODEL,
            max_tokens=FACTORY_CLAUDE_RULEGEN_MAX_TOKENS,
            schema=_RESPONSE_SCHEMA,
        )
    except Exception:
        logger.error("[claude_rule_generator] Claude呼び出し失敗", exc_info=True)
        return None

    data = _parse_response(text)
    if data is None:
        logger.warning("[claude_rule_generator] 応答JSONのスキーマ不正: %s", text[:500])
        return None

    return FactoryHypothesis(
        rule_spec={
            "type": "generated_code",
            "source_code": data["source_code"],
            "class_name": data["class_name"],
            "rule_name": data["rule_name"],
            "description": data["description"],
        },
        market=market,
    )


def _generate_and_evaluate_with_repair(
    market: str, shared_data_dir: str, windows_file: str
) -> Optional[FactoryEvaluation]:
    """1候補につき初回生成＋最大 FACTORY_CLAUDE_RULEGEN_MAX_REPAIR_ATTEMPTS 回の修復。

    修復対象は「機械的な壊れ方」（静的検査違反・実行時例外・応答JSON不正）のみ。
    ゲート判定（DSR/PBO等）が絡む結果はここでは判定せず、gate_evaluated ならそのまま返す
    （合否は呼び出し元の apply_gate に委ねる）。
    """
    repair_context: Optional[str] = None
    attempts = FACTORY_CLAUDE_RULEGEN_MAX_REPAIR_ATTEMPTS + 1

    for attempt in range(attempts):
        hypothesis = _generate_one_candidate(market, repair_context=repair_context)
        if hypothesis is None:
            # 応答JSON不正 or Claude呼び出し失敗も「機械的な壊れ方」として修復対象にする
            repair_context = (
                "前回の応答がJSONとして解析できませんでした。厳密なJSONで再提案してください。"
            )
            continue

        result: SandboxRunResult = run_sandboxed_evaluation(
            hypothesis, shared_data_dir, windows_file
        )
        if result.kind == "gate_evaluated":
            return result.evaluation
        if result.kind == "infra_error":
            logger.warning(
                "[claude_rule_generator] インフラ起因の失敗のためこの候補をスキップ: %s",
                result.infra_detail,
            )
            return None
        # kind == "repairable"
        logger.info(
            "[claude_rule_generator] 修復リトライ %d/%d: %s",
            attempt + 1,
            attempts - 1,
            result.repair_detail,
        )
        repair_context = result.repair_detail

    logger.warning("[claude_rule_generator] 修復予算を使い切ったためこの候補を諦めます")
    return None


def generate_claude_hypotheses(
    market: str,
    champion_sharpe: float,
    shared_data_dir: str,
    windows_file: str,
) -> list[FactoryEvaluation]:
    """1晩分のClaude生成候補を生成・サンドボックス評価する。

    既定無効（FACTORY_CLAUDE_RULEGEN_ENABLED=False）。champion_sharpe は現状
    プロンプトへの直接利用はしていない（将来のプロンプト改善用に引数として残す）。
    """
    del champion_sharpe  # 将来のプロンプト改善用に予約（現状は未使用）
    if not FACTORY_CLAUDE_RULEGEN_ENABLED:
        return []

    evaluations: list[FactoryEvaluation] = []
    for i in range(FACTORY_CLAUDE_RULEGEN_COUNT):
        logger.info(
            "[claude_rule_generator] 候補 %d/%d 生成中...", i + 1, FACTORY_CLAUDE_RULEGEN_COUNT
        )
        evaluation = _generate_and_evaluate_with_repair(market, shared_data_dir, windows_file)
        if evaluation is not None:
            evaluations.append(evaluation)
    return evaluations
