"""複合ルール（AND / OR 組み合わせ）"""

from __future__ import annotations

import pandas as pd

from src.rule_engine.rules.base import TradingRule


class AndRule:
    """全ルールが一致した場合のみシグナルを発生させる複合ルール"""

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
    """いずれかのルールがシグナルを出した場合にシグナルを発生させる複合ルール"""

    def __init__(self, rules: list[TradingRule], name: str | None = None, threshold: float = 0.5):
        self.rules = rules
        self.threshold = threshold
        self.name = name or "or_" + "_".join(r.name for r in rules)
        self.description = f"OR複合(閾値{threshold:.0%}): " + " | ".join(r.description for r in rules)

    def generate_signal(self, df: pd.DataFrame) -> pd.Series:
        signals = [r.generate_signal(df) for r in self.rules]
        n = len(signals)

        buy_count = sum((s == 1).astype(int) for s in signals)
        sell_count = sum((s == -1).astype(int) for s in signals)

        result = pd.Series(0, index=df.index)
        result[buy_count >= max(1, round(n * self.threshold))] = 1
        result[sell_count >= max(1, round(n * self.threshold))] = -1
        return result
