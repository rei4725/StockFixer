"""
LLM 推薦理由生成モジュール

Ollama ローカル LLM を使って予測結果に「なぜ買い/売りか」の
自然言語説明（1文）を添付する。

OLLAMA_URL が未設定または接続できない場合は None を返し、
呼び出し元は理由なしで通知を送る。
"""

from typing import Optional

from src.domain.types import PredictionResult
from src.utils.logger import get_logger

logger = get_logger(__name__)

_REASON_PROMPT = """\
あなたは株式アナリストです。以下の予測データをもとに、この銘柄が注目される理由を
投資家向けに日本語で1文だけ答えてください。余計な前置きや説明は不要です。

銘柄コード: {symbol}（市場: {market}）
予測方向: {direction}
予測変化率: {diff_ratio:+.1%}
モデル信頼度: {confidence}
"""

_REASON_PROMPT_EN = """\
You are a stock analyst. Based on the prediction data below, explain in one Japanese sentence \
why this stock is notable for investors. Answer with the sentence only, no preamble.

Symbol: {symbol} (Market: {market})
Predicted direction: {direction}
Predicted change: {diff_ratio:+.1%}
Model confidence: {confidence}
"""


def generate_reason(result: PredictionResult) -> Optional[str]:
    """予測結果から推薦理由を1文生成する。

    Args:
        result: 対象銘柄の予測結果

    Returns:
        推薦理由の1文。Ollama 未接続・失敗時は None。
    """
    from src.utils.ollama_client import OllamaClient  # noqa: PLC0415

    client = OllamaClient()
    if not client.is_available:
        logger.debug("Ollama 未接続のため理由生成をスキップ: %s", result.symbol)
        return None

    direction = "上昇予測 📈" if result.diff_ratio > 0 else "下落予測 📉"
    confidence = f"{result.confidence_ratio:.0%}" if result.confidence_ratio is not None else "不明"

    prompt = _REASON_PROMPT.format(
        symbol=result.symbol,
        market=result.market,
        direction=direction,
        diff_ratio=result.diff_ratio,
        confidence=confidence,
    )

    reason = client.generate_text(prompt, temperature=0.3)
    if reason:
        logger.debug("理由生成完了: %s → %s", result.symbol, reason[:40])
    return reason


def generate_reasons_for_top(
    results: list[PredictionResult],
    max_count: int = 3,
) -> dict[str, str]:
    """上位銘柄の推薦理由を一括生成する。

    Args:
        results: 予測結果リスト（ランク順）
        max_count: 理由を生成する最大銘柄数

    Returns:
        {symbol: reason} の辞書。Ollama 未接続時は空辞書。
    """
    from src.utils.ollama_client import OllamaClient  # noqa: PLC0415

    client = OllamaClient()
    if not client.is_available:
        return {}

    reasons: dict[str, str] = {}
    for result in results[:max_count]:
        reason = generate_reason(result)
        if reason:
            reasons[result.symbol] = reason

    return reasons
