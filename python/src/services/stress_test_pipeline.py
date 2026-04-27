# ---------------------------------------------------------------------------
# 後方互換 shim（Backward-compatibility shim）
#
# 実装は src.backtest.stress_test に移動しました。
# 新規コードは新パスを使用してください:
#   from src.backtest.stress_test import ...
#
# このファイルを直接編集しないでください。
# ---------------------------------------------------------------------------
import importlib
import sys

sys.modules[__name__] = importlib.import_module("src.backtest.stress_test")
