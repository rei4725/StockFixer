# ---------------------------------------------------------------------------
# 後方互換 shim（Backward-compatibility shim）
#
# 実装は src.services.trading.risk_manager に移動しました。
# 新規コードは新パスを使用してください:
#   from src.services.trading.risk_manager import ...
#
# このファイルを直接編集しないでください。
# ---------------------------------------------------------------------------
import importlib
import sys

sys.modules[__name__] = importlib.import_module("src.services.trading.risk_manager")
