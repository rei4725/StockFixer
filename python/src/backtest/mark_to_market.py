"""バックテストの日々の時価評価（mark-to-market）と資産曲線の構築（#492）。"""

from typing import Any

import pandas as pd

from src.backtest.execution import ExecutionModel


def mark_to_market(
    execution: ExecutionModel,
    date: Any,
    cash: float,
    cash_gross: float,
    position: int,
    price: float,
    short_position: int,
    short_price: float,
    short_entry_vol: float,
    exec_vol: float,
) -> tuple[Any, float, float]:
    """各バーの mark-to-market equity（net / gross）を返す（#492）。

    保有中のロング時価とショート未実現損益を現金に加算し、決済を待たずに
    含み損益を資産曲線へ反映する。これにより max_drawdown の過小評価を防ぐ。
    - long_value : 保有株の時価（コスト無視。清算コストは決済時に計上）
    - short_net  : 現時点でカバーした場合の純損益（往復コスト込み）。#1231: 返す時の
      損益の式とそろえ、建てた時の売りは short_entry_vol（建てた時点の平均出来高）、
      返す買いは exec_vol（その足で分かっている前の足までの平均出来高）で動的スリッページを入れる
    - short_gross: コスト無視のショート未実現損益
    """
    long_value = position * price
    if short_position > 0:
        short_net = execution.sell_proceeds(
            short_position, short_price, short_entry_vol
        ) - execution.buy_cost(short_position, price, exec_vol)
        short_gross = (short_price - price) * short_position
    else:
        short_net = 0.0
        short_gross = 0.0
    equity_net = cash + long_value + short_net
    equity_gross = cash_gross + long_value + short_gross
    return date, equity_net, equity_gross


def build_equity_curves(
    equity_records: list[tuple[Any, float, float]],
) -> tuple[pd.Series, pd.Series]:
    """記録した (date, equity_net, equity_gross) 列から日次 equity Series を構築する。

    記録が空の場合は空 Series を返し、metrics 側で従来の決済時点キャッシュ
    ベースの equity にフォールバックさせる（後方互換）。
    """
    if not equity_records:
        empty = pd.Series(dtype=float)
        return empty, empty
    dates = [r[0] for r in equity_records]
    net = pd.Series([r[1] for r in equity_records], index=dates)
    gross = pd.Series([r[2] for r in equity_records], index=dates)
    return net, gross
