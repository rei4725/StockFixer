"""バックテストの黄金のテストを CI の unit の段で流す（#2030）。

CI の unit の段は tests/unit/ だけを流すため、tests/golden/ の黄金のテストのクラスを
ここに取り込み、同じテストをここでも集めさせる。判定と golden のファイルは
tests/golden/test_backtest_golden.py のものをそのまま使う。

golden ファイルの更新方法は tests/golden/test_backtest_golden.py の docstring を見る。
"""

from tests.golden.test_backtest_golden import TestBacktestGolden

__all__ = ["TestBacktestGolden"]
