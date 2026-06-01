"""
OllamaClient — ローカルLLM汎用クライアント

複数の BC（market_data / reporting 等）から利用される共通クライアント。
utils 層に置くことで BC 間の循環依存を回避する。
"""

import json
import os
from typing import Optional

import requests

from src.utils.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_MODEL = "qwen2.5:3b"

_SCORE_PROMPT = """\
You are a financial sentiment analyst.
Analyze the sentiment of the following news headlines or disclosure text about a stock.
Return ONLY a valid JSON object with a single key "score" (float between -1.0 and 1.0).
  -1.0 = very bearish, 0.0 = neutral, 1.0 = very bullish

Text:
{texts}
"""


class OllamaClient:
    """Ollama ローカルLLM クライアント。

    センチメントスコア算出（score）と自由文生成（generate_text）の2機能を提供する。
    OLLAMA_URL 未設定・接続失敗時は None を返し呼び出し元がフォールバックする。
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = 30,
    ) -> None:
        raw_url = base_url if base_url is not None else os.environ.get("OLLAMA_URL", "")
        self._base_url = (raw_url or "").rstrip("/")
        self._model = model or os.environ.get("OLLAMA_MODEL", _DEFAULT_MODEL)
        self._timeout = timeout

    @property
    def is_available(self) -> bool:
        """OLLAMA_URL が設定されており、サーバーが応答するか確認する。"""
        if not self._base_url:
            return False
        try:
            resp = requests.get(f"{self._base_url}/api/tags", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    def score(self, texts: list[str]) -> Optional[float]:
        """テキストリストのセンチメントスコアを返す。

        Args:
            texts: ニュースタイトルまたは開示文のリスト

        Returns:
            -1.0〜1.0 のスコア。LLM 呼び出し失敗時は None。
        """
        if not self._base_url:
            logger.debug("OLLAMA_URL 未設定のため LLM センチメントをスキップ")
            return None

        combined = "\n".join(f"- {t}" for t in texts if t)
        if not combined:
            return None
        prompt = _SCORE_PROMPT.format(texts=combined)

        try:
            resp = requests.post(
                f"{self._base_url}/api/generate",
                json={
                    "model": self._model,
                    "prompt": prompt,
                    "format": "json",
                    "stream": False,
                    "options": {"temperature": 0.0},
                },
                timeout=self._timeout,
            )
            resp.raise_for_status()
            raw = resp.json().get("response", "")
            parsed = json.loads(raw)
            score = float(parsed["score"])
            return max(-1.0, min(1.0, round(score, 4)))
        except Exception as e:
            logger.warning("Ollama センチメント推論失敗: %s", e)
            return None

    def generate_text(self, prompt: str, temperature: float = 0.3) -> Optional[str]:
        """自由文を生成して返す。

        Args:
            prompt: LLM に渡すプロンプト
            temperature: 生成の多様性（0.0=決定的、1.0=創造的）

        Returns:
            生成テキスト。LLM 呼び出し失敗時は None。
        """
        if not self._base_url:
            logger.debug("OLLAMA_URL 未設定のため LLM テキスト生成をスキップ")
            return None

        try:
            resp = requests.post(
                f"{self._base_url}/api/generate",
                json={
                    "model": self._model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": temperature},
                },
                timeout=self._timeout,
            )
            resp.raise_for_status()
            text = resp.json().get("response", "").strip()
            return text if text else None
        except Exception as e:
            logger.warning("Ollama テキスト生成失敗: %s", e)
            return None
