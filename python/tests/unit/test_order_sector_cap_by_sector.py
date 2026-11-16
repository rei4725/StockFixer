"""
ユニットテスト: 8:50 の新規買いで業種の上限が落とした候補の業種ごとの数（#1908）

業種の上限で落とした候補の業種ごとの数を stats の sector_cap_by_sector に入れ、
results/orders/<日付>.json に写す。和は skip_reasons の sector_cap と一致する。
選ばれる候補・skip_reasons・skipped は変わらず、get_symbol_sector を呼ぶ回数も増えない。
"""

import json
import unittest
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pandas as pd

from src.trading.execution.gate_record import save_order_gate_record
from src.trading.execution.selection import _apply_buy_sector_limit, _select_buy_signals
from src.trading.execution.stats import _new_stats

_SECTORS = {
    # 保有
    "9001": "A",
    "9002": "UNKNOWN",
    # 候補
    "1001": "A",
    "1002": "A",
    "1003": "A",
    "2001": "B",
    "2002": "B",
    "2003": "B",
    "2004": "B",
    "5001": "UNKNOWN",
    "5002": "UNKNOWN",
    "5003": "UNKNOWN",
}
_HELD = {"9001", "9002"}
_CANDIDATES = [
    ("1001", 0.30),
    ("2001", 0.29),
    ("5001", 0.28),
    ("1002", 0.27),
    ("2002", 0.26),
    ("5002", 0.25),
    ("1003", 0.24),  # A は保有1 + 1001・1002 で3件埋まり落ちる
    ("2003", 0.23),
    ("5003", 0.22),  # UNKNOWN は保有1 + 5001・5002 で3件埋まり落ちる
    ("2004", 0.21),  # B は 2001・2002・2003 で3件埋まり落ちる
]
_CAP = 3


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        [{"market": "jp", "symbol": s, "multi_horizon_score": v} for s, v in _CANDIDATES]
    )


def _sector(market, symbol):
    return _SECTORS[symbol]


class TestSectorCapBySector(unittest.TestCase):
    def test_breakdown_sums_to_sector_cap(self):
        """2つの業種と UNKNOWN が混ざる朝は、業種ごとの数の和が sector_cap と一致する"""
        stats = _new_stats()
        with (
            patch(
                "src.trading.execution.selection.get_symbol_sector", side_effect=_sector
            ) as getter,
            patch("src.trading.execution.selection.MAX_ORDERS_PER_RUN", 100),
        ):
            selected = _select_buy_signals_with_cap(stats)
        self.assertEqual(stats["sector_cap_by_sector"], {"A": 1, "B": 1, "UNKNOWN": 1})
        self.assertEqual(sum(stats["sector_cap_by_sector"].values()), 3)
        self.assertEqual(stats["skip_reasons"], {"sector_cap": 3})
        self.assertEqual(stats["skipped"], 3)
        self.assertEqual(
            list(selected["symbol"]),
            ["1001", "2001", "5001", "1002", "2002", "5002", "2003"],
        )
        # 保有2件 + 候補10件。落とした行の業種は既に引いたものを使う
        self.assertEqual(getter.call_count, len(_HELD) + len(_CANDIDATES))

    def test_selection_unchanged_with_counter(self):
        """業種ごとの数を受け取っても、選ばれる候補は受け取らないときと同じ"""
        with patch("src.trading.execution.selection.get_symbol_sector", side_effect=_sector):
            plain = _apply_buy_sector_limit(_frame(), max_sector_positions=_CAP, held_symbols=_HELD)
            dropped: dict[str, int] = {}
            counted = _apply_buy_sector_limit(
                _frame(), max_sector_positions=_CAP, held_symbols=_HELD, dropped_by_sector=dropped
            )
        pd.testing.assert_frame_equal(plain, counted)
        self.assertEqual(dropped, {"A": 1, "B": 1, "UNKNOWN": 1})

    def test_no_drop_leaves_empty_breakdown(self):
        """落ちる候補が無い朝は sector_cap_by_sector が空で、sector_cap も出ない"""
        stats = _new_stats()
        with patch("src.trading.execution.selection.get_symbol_sector", side_effect=_sector):
            _select_buy_signals(_frame().head(2), set(), "jp", stats)
        self.assertEqual(stats["sector_cap_by_sector"], {})
        self.assertEqual(stats["skip_reasons"], {})

    def test_record_adds_breakdown_key_only(self):
        """記録は既存の鍵と値はそのままで、sector_cap_by_sector が足されるだけ"""
        stats = _new_stats()
        stats["skipped"] = 3
        stats["skip_reasons"] = {"sector_cap": 3}
        stats["sector_cap_by_sector"] = {"A": 1, "B": 1, "UNKNOWN": 1}
        now = datetime(2026, 11, 17, 8, 50, tzinfo=ZoneInfo("Asia/Tokyo"))
        with TemporaryDirectory() as tmp:
            with (
                patch("src.trading.execution.gate_record.get_results_dir", return_value=tmp),
                patch("src.trading.execution.gate_record.now_jst", return_value=now),
            ):
                path = save_order_gate_record(stats, market="jp", mode="paper")
            assert path is not None
            record = json.loads(Path(path).read_text(encoding="utf-8"))
        self.assertEqual(record["sector_cap_by_sector"], {"A": 1, "B": 1, "UNKNOWN": 1})
        self.assertEqual(record["skip_reasons"], {"sector_cap": 3})
        self.assertEqual(record["skipped"], 3)


def _select_buy_signals_with_cap(stats):
    """_select_buy_signals は既定の上限を使うので、上限を _CAP にして呼ぶ"""
    with patch(
        "src.trading.execution.selection._apply_buy_sector_limit",
        side_effect=lambda c, **kw: _apply_buy_sector_limit(c, max_sector_positions=_CAP, **kw),
    ):
        return _select_buy_signals(_frame(), _HELD, "jp", stats)


if __name__ == "__main__":
    unittest.main()
