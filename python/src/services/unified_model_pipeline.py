# ---------------------------------------------------------------------------
# 後方互換 shim（Backward-compatibility shim）
#
# 実装は src.prediction.unified_model_pipeline に移動しました。
# 新規コードは新パスを使用してください:
#   from src.prediction.unified_model_pipeline import ...
#
# このファイルを直接編集しないでください。
# ---------------------------------------------------------------------------
import importlib
import sys

sys.modules[__name__] = importlib.import_module("src.prediction.unified_model_pipeline")
