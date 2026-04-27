# ---------------------------------------------------------------------------
# 後方互換 shim（Backward-compatibility shim）
#
# 実装は src.backtest.walk_forward_report に移動しました。
# 新規コードは新パスを使用してください:
#   from src.backtest.walk_forward_report import ...
#
# このファイルを直接編集しないでください。
# ---------------------------------------------------------------------------
import importlib
import sys

sys.modules[__name__] = importlib.import_module("src.backtest.walk_forward_report")
