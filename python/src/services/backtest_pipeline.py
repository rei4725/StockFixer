# ---------------------------------------------------------------------------
# 後方互換 shim（Backward-compatibility shim）
#
# 実装は src.backtest.pipeline に移動しました（DDD フェーズ2）。
# 新規コードは新パスを使用してください:
#   from src.backtest.pipeline import ...
#
# このファイルを直接編集しないでください。
# ---------------------------------------------------------------------------
import importlib
import sys

sys.modules[__name__] = importlib.import_module("src.backtest.pipeline")
