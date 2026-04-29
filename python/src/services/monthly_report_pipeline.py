# ---------------------------------------------------------------------------
# 後方互換 shim（Backward-compatibility shim）
#
# 実装は src.reporting.monthly に移動しました。
# 新規コードは新パスを使用してください:
#   from src.reporting.monthly import ...
#
# このファイルを直接編集しないでください。
# ---------------------------------------------------------------------------
import importlib
import sys

sys.modules[__name__] = importlib.import_module("src.reporting.monthly")
