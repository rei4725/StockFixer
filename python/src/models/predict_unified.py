# 後方互換 shim — フェーズ4で削除予定
# 実装本体は src.prediction.predict_unified に移動済み
import importlib
import sys

sys.modules[__name__] = importlib.import_module("src.prediction.predict_unified")
