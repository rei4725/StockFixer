# 後方互换 shim — フェーズ4で削除予定
# 実装本体は src.reporting.monthly に移動済み
import importlib
import sys

sys.modules[__name__] = importlib.import_module("src.reporting.monthly")
