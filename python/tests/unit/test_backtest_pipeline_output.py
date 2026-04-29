"""Unit Test: backtest/pipeline.print_backtest_metrics output"""

from src.backtest.pipeline import print_backtest_metrics


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
