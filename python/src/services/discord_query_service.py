# ---------------------------------------------------------------------------
# 後方互換 shim（Backward-compatibility shim）
#
# 実装は src.services.reporting.discord_query_service に移動しました。
# 新規コードは新パスを使用してください:
#   from src.services.reporting.discord_query_service import ...
#
# このファイルを直接編集しないでください。
# ---------------------------------------------------------------------------
import importlib
import sys

sys.modules[__name__] = importlib.import_module("src.services.reporting.discord_query_service")
