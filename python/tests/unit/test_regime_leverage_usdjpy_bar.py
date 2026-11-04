"""レジームレバレッジのドル円が SPY の足と同じ日の足であること（#1554）"""

import os
import sys
from datetime import datetime, timezone
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.infrastructure.yfinance_market_data_adapter import (  # noqa: E402
    YFinanceMarketDataAdapter,
)
from src.trading.regime_leverage_strategy.service import (  # noqa: E402
    run_regime_leverage_weekly_check,
)

_SERVICE = "src.trading.regime_leverage_strategy.service"
_JST = ZoneInfo("Asia/Tokyo")
# 2026-11-06 は金曜。土曜 06:30 JST は UTC の金曜 21:30
_SATURDAY_SCHEDULED = datetime(2026, 11, 7, 6, 30, tzinfo=_JST)
_SUNDAY_RECOVERY = datetime(2026, 11, 8, 12, 0, tzinfo=_JST)
_THURSDAY_RATE = 151.0
_FRIDAY_RATE = 152.0


def _spy_bars() -> pd.DataFrame:
    """金曜 2026-11-06 の足で終わる、上昇の SPY の足"""
    idx = pd.bdate_range(end="2026-11-06", periods=260)
    prices = pd.Series(np.linspace(400.0, 500.0, 260), index=idx)
    return pd.DataFrame(
        {
            "Open": prices,
            "High": prices + 2,
            "Low": prices - 2,
            "Close": prices,
            "Volume": 1_000_000,
        },
        index=idx,
    )


def _download_end_exclusive(pair: str, start: str, end: str, auto_adjust: bool) -> pd.DataFrame:
    """yfinance と同じく、end の日の足を含めないドル円の足を返す"""
    bars = pd.DataFrame(
        {"Close": [_THURSDAY_RATE, _FRIDAY_RATE]},
        index=pd.to_datetime(["2026-11-05", "2026-11-06"]),
    )
    return bars[bars.index < pd.Timestamp(end)]


def _frozen_datetime(moment: datetime) -> type[datetime]:
    """now() が moment を返す datetime。コンテナの時計は UTC なので、naive の now は UTC にする"""

    class _Frozen(datetime):
        @classmethod
        def now(cls, tz=None):  # type: ignore[no-untyped-def, override]
            if tz is not None:
                return moment.astimezone(tz)
            return moment.astimezone(timezone.utc).replace(tzinfo=None)

    return _Frozen


@pytest.mark.parametrize("moment", [_SATURDAY_SCHEDULED, _SUNDAY_RECOVERY])
def test_weekly_uses_friday_rate_with_friday_spy_bar(moment):
    """土曜の定時も日曜の補完も、SPY の足と同じ金曜の足のドル円を使うこと"""
    with (
        patch("src.market_data.yfinance_source.yf_client") as yf_client,
        patch(f"{_SERVICE}.datetime", _frozen_datetime(moment)),
        patch(f"{_SERVICE}.get_latest_snapshot", return_value=None),
        patch(f"{_SERVICE}.insert_snapshot"),
    ):
        yf_client.ticker_history.return_value = _spy_bars()
        yf_client.download.side_effect = _download_end_exclusive

        decision = run_regime_leverage_weekly_check(YFinanceMarketDataAdapter())

    assert decision.action == "entry"
    assert decision.usdjpy_rate == _FRIDAY_RATE
