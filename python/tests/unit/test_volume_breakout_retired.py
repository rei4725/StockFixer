"""ユニットテスト: volume_breakout を週次のルール評価の候補から外す（#1614・R-002）"""

from __future__ import annotations

from datetime import date
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.backtest import factory_sampling
from src.backtest.rule_selector import evaluate_all_symbols
from src.backtest.rules import ALL_RULES
from src.rule_engine.pipeline import _RULE_INSTANCES, generate_rule_signals

_RULE = "volume_breakout"
_SIGNAL_DATE = date(2024, 3, 22)


class _Port:
    """OHLCVWithIndicatorsPort の最小の実装"""

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    def get_ohlcv_with_indicators(self, ticker: str, lookback_days: int) -> pd.DataFrame:
        return self._df


def _bars(last_close: float) -> pd.DataFrame:
    """2024-03-22 が最後の足になる60本の足を返す。"""
    index = pd.date_range("2024-01-01", periods=60, freq="B")
    close = np.linspace(1000.0, 1100.0, 60)
    close[-1] = last_close
    return pd.DataFrame(
        {"Close": close, "Volume": np.full(60, 500_000, dtype=float)},
        index=index,
    )


def _generate(df: pd.DataFrame) -> tuple[Any, Any]:
    row = {"symbol": "7203", "best_rule": _RULE, "win_rate": 0.6, "net_profit": 1000.0}
    with (
        patch(
            "src.rule_engine.pipeline.load_effective_rules",
            return_value=pd.DataFrame([row]),
        ),
        patch("src.rule_engine.pipeline.upsert_rule_signal") as upsert,
    ):
        batch = generate_rule_signals("jp", signal_date=_SIGNAL_DATE, market_data_port=_Port(df))
    return batch, upsert


def _results_with_retired_rule_on_top(**kwargs: Any) -> list[dict[str, Any]]:
    """候補の全部を有効にし、volume_breakout の成績を一番良くしたバックテストの結果を返す。"""
    return [
        {
            "rule": rule.name,
            "win_rate": 0.9 if rule.name == _RULE else 0.6,
            "net_profit": 9000.0 if rule.name == _RULE else 1000.0,
            "num_trades": 10,
            "profit_factor": 1.5,
            "max_drawdown": 0.1,
        }
        for rule in kwargs["rules"]
    ]


def test_retired_from_candidates_but_signal_generation_keeps_working() -> None:
    """候補からは外れ、rule_best に行が残る銘柄のシグナルは次の評価まで出続ける。"""
    assert _RULE not in {rule.name for rule in ALL_RULES}
    assert _RULE in _RULE_INSTANCES

    batch, upsert = _generate(_bars(last_close=1100.0))

    assert batch.unavailable == []
    assert [signal["rule"] for signal in batch.signals] == [_RULE]
    upsert.assert_called_once()


def test_held_symbol_row_still_produces_sell() -> None:
    """保有中に行が残る銘柄は、値が崩れたら SELL が出る（PR 1612 で行は残る）。"""
    batch, upsert = _generate(_bars(last_close=900.0))

    assert batch.unavailable == []
    assert [signal["signal_label"] for signal in batch.signals] == ["SELL"]
    assert upsert.call_args.args[:5] == (_SIGNAL_DATE, "jp", "7203", _RULE, -1)


def test_weekly_evaluation_never_selects_volume_breakout() -> None:
    """週次のルール評価は、実際の ALL_RULES で評価して volume_breakout を選ばない。"""
    with (
        patch(
            "src.backtest.rule_selector.run_rule_backtest",
            side_effect=_results_with_retired_rule_on_top,
        ),
        patch("src.backtest.rule_selector.load_held_symbols", return_value=set()),
        patch("src.backtest.rule_selector.upsert_rule_best") as upsert,
    ):
        summary = evaluate_all_symbols("jp", "2024-01-01", "2024-12-31", symbols=["7203"])

    assert summary["effective"] == 1
    upsert.assert_called_once()
    assert upsert.call_args.kwargs["best_rule"] != _RULE


def test_strategy_factory_keeps_volume_breakout() -> None:
    """戦略ファクトリーの部品からは外さない。"""
    assert factory_sampling._RULE_CLASSES[_RULE].__name__ == "VolumeBreakoutRule"
    assert _RULE in factory_sampling._PARAM_GRID
