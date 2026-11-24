"""ユニットテスト: レジームレバレッジ戦略の日次の回は、途中の足の noop の行を書かない(#2144)。

損切りの線は65,000円・ドル円146円(約445.2ドル)。安値300ドルは維持率0.20を割る。
"""

import unittest
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from src.trading.regime_leverage_strategy.service import run_regime_leverage_daily_margin_check
from src.trading.regime_leverage_strategy.types import RegimeLeverageSnapshot

_JST = ZoneInfo("Asia/Tokyo")
_SERVICE = "src.trading.regime_leverage_strategy.service"
_NORMAL_VOLUME = 1_000_000.0


def _port(last_bar: str, lows=None, volumes=None) -> MagicMock:
    """last_bar の日までの日足(安値は既定で498ドル、出来高は100万株)を返すポート。"""
    idx = pd.bdate_range(end=last_bar, periods=260)
    df = pd.DataFrame(
        {"Open": 500.0, "High": 502.0, "Low": 498.0, "Close": 500.0, "Volume": _NORMAL_VOLUME},
        index=idx,
    )
    for day, value in (lows or {}).items():
        df.loc[pd.Timestamp(day), "Low"] = value
    for day, value in (volumes or {}).items():
        df.loc[pd.Timestamp(day), "Volume"] = value
    port = MagicMock()
    port.get_stock_data.return_value = df
    port.get_forex_data.return_value = pd.DataFrame({"Close": [146.0]}, index=[idx[-1]])
    return port


class TestDailyCheckSkipsPartialBarNoop(unittest.TestCase):
    _DAY = "2026-07-15"
    _NEXT_DAY = "2026-07-16"
    _RUN = datetime(2026, 7, 16, 7, 30, tzinfo=_JST)
    _NEXT_RUN = datetime(2026, 7, 17, 7, 30, tzinfo=_JST)
    _PARTIAL = _NORMAL_VOLUME * 0.15

    def setUp(self):
        # 前の回は JST 7/15 07:30(7/14 の足まで判定)
        self.rows: list = []
        self.snapshot = self._snapshot(datetime(2026, 7, 14, 22, 30))

    @staticmethod
    def _snapshot(executed_at: datetime) -> RegimeLeverageSnapshot:
        return RegimeLeverageSnapshot(
            id=1,
            executed_at=executed_at,
            action="noop",
            reason="daily_noop",
            spy_price_usd=500.0,
            usdjpy_rate=146.0,
            shares=27.0,
            entry_date=datetime(2026, 1, 2),
            entry_price_jpy=72500.0,
            entry_commission_jpy=0.0,
            equity_at_entry_jpy=1_000_000.0,
            stop_price_jpy=65000.0,
            equity_now_jpy=1_000_000.0,
            maintenance_ratio=None,
        )

    def _run(self, now: datetime, port: MagicMock):
        """日次の回を1回流す。行を書いたら、次の回の「前の回」をその時刻へ進める。"""

        class _FixedDatetime(datetime):
            @classmethod
            def now(cls, tz=None):
                return now.replace(tzinfo=None) if tz is None else now.astimezone(tz)

        def _insert(decision):
            self.rows.append(decision)
            self.snapshot = self._snapshot(now.astimezone(timezone.utc).replace(tzinfo=None))

        with patch(f"{_SERVICE}.get_latest_snapshot", return_value=self.snapshot), patch(
            f"{_SERVICE}.insert_snapshot", side_effect=_insert
        ), patch(f"{_SERVICE}.datetime", _FixedDatetime):
            return run_regime_leverage_daily_margin_check(port)

    def _run_partial(self):
        return self._run(self._RUN, _port(self._DAY, volumes={self._DAY: self._PARTIAL}))

    def test_partial_bar_without_hit_writes_no_row_and_warns_once(self):
        with self.assertLogs(_SERVICE, level="WARNING") as logs:
            decision = self._run(self._RUN, _port(self._DAY, volumes={self._DAY: self._PARTIAL}))
        self.assertEqual(decision.action, "noop")
        self.assertEqual(self.rows, [])
        self.assertEqual(len(logs.output), 1)
        self.assertIn("途中の足", logs.output[0])

    def test_zero_volume_bar_is_partial(self):
        self._run(self._RUN, _port(self._DAY, volumes={self._DAY: 0.0}))
        self.assertEqual(self.rows, [])

    def test_next_run_exits_on_corrected_low(self):
        for low, reason in ((440.0, "initial_stop"), (300.0, "margin_call")):
            with self.subTest(reason=reason):
                self.setUp()
                self._run_partial()
                decision = self._run(self._NEXT_RUN, _port(self._NEXT_DAY, lows={self._DAY: low}))
                self.assertEqual(decision.reason, reason)
                self.assertEqual(decision.spy_price_usd, low)
                self.assertEqual(self.rows, [decision])

    def test_partial_bar_with_hit_exits_in_the_same_run(self):
        for low, reason in ((440.0, "initial_stop"), (300.0, "margin_call")):
            with self.subTest(reason=reason):
                self.setUp()
                decision = self._run(
                    self._RUN,
                    _port(self._DAY, lows={self._DAY: low}, volumes={self._DAY: self._PARTIAL}),
                )
                self.assertEqual(decision.reason, reason)
                self.assertEqual(self.rows, [decision])

    def test_unchecked_bar_before_partial_bar_exits(self):
        decision = self._run(
            self._NEXT_RUN,
            _port(self._NEXT_DAY, lows={self._DAY: 440.0}, volumes={self._NEXT_DAY: self._PARTIAL}),
        )
        self.assertEqual(decision.reason, "initial_stop")
        self.assertEqual(self.rows, [decision])

    def test_normal_and_half_day_volume_write_row(self):
        for ratio in (0.4, 0.6, 1.0, 2.0):
            with self.subTest(ratio=ratio):
                self.setUp()
                with self.assertNoLogs(_SERVICE, level="WARNING"):
                    decision = self._run(
                        self._RUN, _port(self._DAY, volumes={self._DAY: _NORMAL_VOLUME * ratio})
                    )
                self.assertEqual(self.rows, [decision])
                self.assertEqual(decision.reason, "daily_noop")

    def test_missing_volume_writes_row(self):
        decision = self._run(self._RUN, _port(self._DAY, volumes={self._DAY: np.nan}))
        self.assertEqual(self.rows, [decision])

    def test_missing_previous_volumes_write_row(self):
        port = _port(self._DAY, volumes={self._DAY: self._PARTIAL})
        df = port.get_stock_data.return_value
        df.iloc[:-1, df.columns.get_loc("Volume")] = np.nan
        decision = self._run(self._RUN, port)
        self.assertEqual(self.rows, [decision])

    def test_no_volume_column_writes_row(self):
        port = _port(self._DAY)
        port.get_stock_data.return_value = port.get_stock_data.return_value.drop(columns="Volume")
        decision = self._run(self._RUN, port)
        self.assertEqual(self.rows, [decision])

    def test_unclosed_today_bar_volume_is_not_used(self):
        # JST 7/16 23:00 は 7/16 の場中。出来高の少ない 7/16 の足は見ず、7/15 の足で行を書く
        decision = self._run(
            datetime(2026, 7, 16, 23, 0, tzinfo=_JST),
            _port(self._NEXT_DAY, volumes={self._NEXT_DAY: self._PARTIAL}),
        )
        self.assertEqual(self.rows, [decision])

    def test_bar_still_thin_in_next_run_does_not_block(self):
        self._run_partial()
        with self.assertNoLogs(_SERVICE, level="WARNING"):
            decision = self._run(
                self._NEXT_RUN, _port(self._NEXT_DAY, volumes={self._DAY: self._PARTIAL})
            )
        self.assertEqual(decision.reason, "daily_noop")
        self.assertEqual(self.rows, [decision])

    def test_bar_still_thin_in_next_run_is_checked(self):
        # 本当に出来高の少ない日でも、次の回はその足の安値で判定する
        self._run_partial()
        decision = self._run(
            self._NEXT_RUN,
            _port(self._NEXT_DAY, lows={self._DAY: 440.0}, volumes={self._DAY: self._PARTIAL}),
        )
        self.assertEqual(decision.reason, "initial_stop")


if __name__ == "__main__":
    unittest.main()
