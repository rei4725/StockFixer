# ---------------------------------------------------------------------------
# 後方互換 shim（Backward-compatibility shim）
#
# 実装は src.reporting.query_service に移動しました。
# 新規コードは新パスを使用してください:
#   from src.reporting.query_service import ...
#
# このファイルを直接編集しないでください。
# ---------------------------------------------------------------------------
import importlib
import sys

sys.modules[__name__] = importlib.import_module("src.reporting.query_service")
