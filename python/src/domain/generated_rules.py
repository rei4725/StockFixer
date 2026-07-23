"""Claude生成ルールの共有レジストリ（shared kernel）。

人間がレビュー・マージしたPRがここに1エントリを追加する。backtest/rules/technical.py
の ALL_RULES（週次ルール評価対象）と rule_engine/pipeline.py の _RULE_INSTANCES
（本番日次シグナル生成）の両方がこのレジストリを合成して参照することで、「PRマージ＝
バックテストと本番の両方に同じコードが反映される」ことを保証する。

backtest BC と rule_engine BC は import-linter の BC independence contract
（python/.importlinter）により相互import禁止のため、両者から見て中立な domain/
（shared kernel）にこのレジストリを置く。

新規ルールを追加する場合は、このファイルへの1エントリ追加のみで完結させる。
個々のルールクラス定義は python/src/rule_engine/rules/generated/ 配下に置く。
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class GeneratedRule(Protocol):
    name: str
    description: str

    def generate_signal(self, df: pd.DataFrame) -> pd.Series:
        """OHLCV + テクニカル指標 DataFrame からシグナルを生成する。

        Returns:
            pd.Series: 1=buy, -1=sell, 0=hold（インデックスは df と同じ）
        """
        ...


GENERATED_RULES: dict[str, GeneratedRule] = {}
