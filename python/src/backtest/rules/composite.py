"""
複合ルール（AND / OR 組み合わせ）

複数のルールシグナルを論理演算で組み合わせる。
- AndRule: 全ルールが同じシグナルを出した場合のみシグナル発生（高精度・低頻度）
- OrRule: いずれかのルールがシグナルを出した場合にシグナル発生（高頻度・低精度）
"""

from __future__ import annotations

import pandas as pd

from src.backtest.rules.base import TradingRule


class AndRule:
    """
    全ルールが一致した場合のみシグナルを発生させる複合ルール。

    Buy: 全ルールの buy シグナルが同時に発生
    Sell: 任意ルールが sell シグナルを発生（早期撤退）
    """

    def __init__(self, rules: list[TradingRule], name: str | None = None):
        self.rules = rules
        self.name = name or "and_" + "_".join(r.name for r in rules)
        self.description = "AND複合: " + " & ".join(r.description for r in rules)

    def generate_signal(self, df: pd.DataFrame) -> pd.Series:
        signals = [r.generate_signal(df) for r in self.rules]

        buy_all = pd.Series(True, index=df.index)
        sell_any = pd.Series(False, index=df.index)
        for sig in signals:
            buy_all &= sig == 1
            sell_any |= sig == -1

        result = pd.Series(0, index=df.index)
        result[buy_all] = 1
        result[sell_any] = -1
        return result


class OrRule:
    """
    いずれかのルールがシグナルを出した場合にシグナルを発生させる複合ルール。

    Buy: 過半数のルールが buy シグナルを発生（単純 OR より保守的）
    Sell: 過半数のルールが sell シグナルを発生
    """

    def __init__(self, rules: list[TradingRule], name: str | None = None, threshold: float = 0.5):
        self.rules = rules
        self.threshold = threshold
        self.name = name or "or_" + "_".join(r.name for r in rules)
        self.description = f"OR複合(閾値{threshold:.0%}): " + " | ".join(
            r.description for r in rules
        )

    def generate_signal(self, df: pd.DataFrame) -> pd.Series:
        signals = [r.generate_signal(df) for r in self.rules]
        n = len(signals)

        buy_count = sum((s == 1).astype(int) for s in signals)
        sell_count = sum((s == -1).astype(int) for s in signals)

        result = pd.Series(0, index=df.index)
        result[buy_count >= max(1, round(n * self.threshold))] = 1
        result[sell_count >= max(1, round(n * self.threshold))] = -1
        return result
