from __future__ import annotations

import pandas as pd

from src.domain.generated_rules import GENERATED_RULES


class _DummyRule:
    name = "dummy_generated"
    description = "test"

    def generate_signal(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(0, index=df.index)


def test_generated_rules_empty_by_default():
    assert GENERATED_RULES == {}


def test_backtest_all_rules_includes_generated_entries():
    """ALL_RULES はモジュール読み込み時に一度だけ評価されるリストのため、
    GENERATED_RULES への追加を反映させるには importlib.reload が必要
    （本番では PR マージ＝ソースコード編集のため、プロセス起動時の import で
    自然に反映される。この reload はテストでそれを模擬している）。
    """
    import importlib

    from src.backtest.rules import technical

    GENERATED_RULES["dummy_generated"] = _DummyRule()
    try:
        importlib.reload(technical)
        names = {r.name for r in technical.ALL_RULES}
        assert "dummy_generated" in names
    finally:
        del GENERATED_RULES["dummy_generated"]
        importlib.reload(technical)


def test_rule_engine_instances_includes_generated_entries():
    import importlib

    from src.rule_engine import pipeline

    GENERATED_RULES["dummy_generated"] = _DummyRule()
    try:
        importlib.reload(pipeline)
        assert "dummy_generated" in pipeline._RULE_INSTANCES
    finally:
        del GENERATED_RULES["dummy_generated"]
        importlib.reload(pipeline)
