"""
OllamaClient — ローカルLLMによるセンチメントスコア算出

OLLAMA_URL 環境変数が設定されている場合に Ollama の /api/generate を呼び出し、
ニュース/開示テキストのリストを -1.0〜1.0 のセンチメントスコアに変換する。

未設定・接続失敗時は None を返し、呼び出し元がキーワードマッチにフォールバックする。
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
    """Ollama ローカルLLM へのセンチメント推論クライアント。"""

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
        """
        テキストリストのセンチメントスコアを返す。

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
