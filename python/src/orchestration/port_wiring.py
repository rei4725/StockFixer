"""
ポート合成ルート（composition root）

BC（backtest / prediction）が定義する market_data アクセスポートに、
具体アダプタ（market_data BC 側の実装）を注入する。

レイヤー規約上、`market_data` を import してよいのは上位レイヤー
（orchestration / api）のみ。各 BC は自身の ports.py / data_port.py で
プロトコルだけを定義し、ここで初めて具体実装と結線する。

各エントリポイント（run_*.py / scheduler / api 起動）は、BC のサービスを
呼ぶ前に一度だけ `wire_ports()` を呼ぶこと。未注入のまま get_*_port() を
呼ぶと RuntimeError になる。
"""

from __future__ import annotations

from src.utils.logger import get_logger

logger = get_logger(__name__)

_wired = False


def wire_ports(force: bool = False) -> None:
    """BC ポートにデフォルトの market_data アダプタを注入する（冪等）。

    Args:
        force: True のとき、既に注入済みでも再注入する（テスト用途）。
    """
    global _wired
    if _wired and not force:
        return

    from src.backtest.data_port import set_backtest_data_port
    from src.market_data.backtest_adapter import BacktestMarketDataAdapter
    from src.market_data.prediction_adapter import PredictionMarketDataAdapter
    from src.prediction.ports import set_market_data_port

    set_backtest_data_port(BacktestMarketDataAdapter())
    set_market_data_port(PredictionMarketDataAdapter())
    _wired = True
    logger.debug("ポート注入完了: BacktestDataPort / MarketDataPort")
