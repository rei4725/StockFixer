# ---------------------------------------------------------------------------
# 後方互換 shim（Backward-compatibility shim）
#
# 実装は src.services.training.model_training_pipeline に移動しました。
# 新規コードは新パスを使用してください:
#   from src.services.training.model_training_pipeline import ...
#
# このファイルを直接編集しないでください。
# ---------------------------------------------------------------------------
import importlib
import sys

sys.modules[__name__] = importlib.import_module("src.services.training.model_training_pipeline")
