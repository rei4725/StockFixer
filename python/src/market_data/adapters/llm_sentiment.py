"""
OllamaClient の後方互換 re-export。

実装は src.utils.ollama_client に移動済み。
既存の import パスを壊さないために本モジュールを維持する。
"""

from src.utils.ollama_client import OllamaClient  # noqa: F401
