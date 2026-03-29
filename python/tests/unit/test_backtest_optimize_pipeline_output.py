"""Unit Test: backtest_optimize_pipeline print output"""

import sys
import types
from importlib import util
from pathlib import Path

import pandas as pd

_MODULE_PATH = (
    Path(__file__).resolve().parents[2] / "src" / "services" / "backtest_optimize_pipeline.py"
)
if "src.services" not in sys.modules:
    pkg = types.ModuleType("src.services")
    pkg.__path__ = [str(_MODULE_PATH.parent)]
    sys.modules["src.services"] = pkg

_SPEC = util.spec_from_file_location("src.services.backtest_optimize_pipeline", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = util.module_from_spec(_SPEC)
sys.modules["src.services.backtest_optimize_pipeline"] = _MODULE
_SPEC.loader.exec_module(_MODULE)
print_optimization_results = _MODULE.print_optimization_results


def test_print_optimization_results_includes_gross_and_cost_columns(capsys):
    df = pd.DataFrame(
        [
            {
                "threshold": 0.001,
                "total_return": 0.010,
                "gross_total_return": 0.015,
                "cost_impact_return": 0.005,
                "sharpe_ratio": 1.1,
                "gross_sharpe_ratio": 1.3,
                "max_drawdown": -0.030,
                "gross_max_drawdown": -0.028,
                "win_rate": 0.55,
                "profit_factor": 1.4,
                "num_trades": 12,
            }
        ]
    )

    print_optimization_results(df, sort_by="gross_total_return")
    out = capsys.readouterr().out

    assert "gross_total_return" in out
    assert "cost_impact_return" in out
    assert "gross_sharpe_ratio" in out
    assert "gross_max_drawdown" in out


def test_print_optimization_results_cost_impact_is_ascending(capsys):
    df = pd.DataFrame(
        [
            {
                "threshold": 0.001,
                "total_return": 0.011,
                "cost_impact_return": 0.006,
                "sharpe_ratio": 1.2,
                "max_drawdown": -0.025,
                "win_rate": 0.5,
                "profit_factor": 1.2,
                "num_trades": 10,
            },
            {
                "threshold": 0.002,
                "total_return": 0.010,
                "cost_impact_return": 0.002,
                "sharpe_ratio": 1.1,
                "max_drawdown": -0.022,
                "win_rate": 0.5,
                "profit_factor": 1.1,
                "num_trades": 10,
            },
        ]
    )

    print_optimization_results(df, sort_by="cost_impact_return")
    out = capsys.readouterr().out

    assert "ベスト（cost_impact_return基準）" in out
    assert "閾値: 0.002" in out
