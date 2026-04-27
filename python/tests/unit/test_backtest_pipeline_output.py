"""Unit Test: backtest_pipeline.print_backtest_metrics output"""

import sys
import types
from importlib import util
from pathlib import Path

_MODULE_PATH = Path(__file__).resolve().parents[2] / "src" / "services" / "backtest_pipeline.py"
# services.__init__ の重い import を避けつつ、coverage が同名モジュールとして認識できるようにする
if "src.services" not in sys.modules:
    pkg = types.ModuleType("src.services")
    pkg.__path__ = [str(_MODULE_PATH.parent)]
    sys.modules["src.services"] = pkg

_SPEC = util.spec_from_file_location("src.services.backtest_pipeline", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = util.module_from_spec(_SPEC)
sys.modules["src.services.backtest_pipeline"] = _MODULE
_SPEC.loader.exec_module(_MODULE)
# shim が sys.modules を実モジュールに差し替える場合を考慮して再取得
_MODULE = sys.modules["src.services.backtest_pipeline"]
print_backtest_metrics = _MODULE.print_backtest_metrics


def test_print_backtest_metrics_with_gross(capsys):
    metrics = {
        "final_cash": 1010000.0,
        "total_return": 0.01,
        "num_trades": 3,
        "win_rate": 0.6667,
        "profit_factor": 1.5,
        "sharpe_ratio": 1.2,
        "max_drawdown": -0.03,
        "gross_final_cash": 1015000.0,
        "gross_total_return": 0.015,
        "gross_sharpe_ratio": 1.4,
        "gross_max_drawdown": -0.025,
        "cost_impact_cash": 5000.0,
        "cost_impact_return": 0.005,
    }

    print_backtest_metrics(metrics, label="jp/7203")
    out = capsys.readouterr().out

    assert "[NET]" in out
    assert "[GROSS]" in out
    assert "cost_impact_cash" in out
