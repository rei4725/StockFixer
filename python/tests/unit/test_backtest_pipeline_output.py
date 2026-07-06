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


def test_print_backtest_metrics_with_monte_carlo(capsys):
    metrics = {
        "final_cash": 1010000.0,
        "total_return": 0.01,
        "num_trades": 3,
        "win_rate": 0.6667,
        "profit_factor": 1.5,
        "sharpe_ratio": 1.2,
        "max_drawdown": -0.03,
        "mc_max_drawdown_mean": -0.05,
        "mc_max_drawdown_p95": -0.09,
        "mc_final_cash_p05": 980000.0,
        "mc_final_cash_p50": 1010000.0,
        "mc_final_cash_p95": 1040000.0,
    }

    print_backtest_metrics(metrics, label="jp/7203")
    out = capsys.readouterr().out

    assert "[Monte Carlo]" in out
    assert "final_cash_p05" in out


def test_print_backtest_metrics_without_monte_carlo_omits_section(capsys):
    metrics = {
        "final_cash": 1010000.0,
        "total_return": 0.01,
        "num_trades": 3,
        "win_rate": 0.6667,
        "profit_factor": 1.5,
        "sharpe_ratio": 1.2,
        "max_drawdown": -0.03,
    }

    print_backtest_metrics(metrics, label="jp/7203")
    out = capsys.readouterr().out

    assert "[Monte Carlo]" not in out
