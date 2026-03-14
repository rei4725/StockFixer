"""
NOTE: このファイルのテストは test_backtester_unit.py に統合されました。

重複していたクラス:
  TestSimulateTradingBasic          → TestBacktesterSimulateTradingBasic (unit)
  TestSimulateTradingFeeSlippage    → TestBacktesterProfitLoss / TestBacktesterSlippage (unit)
  TestSimulateTradingMetrics        → TestBacktesterMetricsShape (unit)
  TestBacktesterWithTask            → TestBacktesterWithTask (unit)

テスト実行:
  python -m pytest tests/unit/test_backtester_unit.py -v
"""
