"""入金のある週の投資比率と、その週の古いファイルの計算し直しのユニットテスト（#2177）"""

import json
from datetime import date
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import pytest

from src.trading import paper_equity
from src.trading.brokers.base import OrderSide
from src.trading.paper_equity import compute_weekly_results, write_paper_weekly_results

_INITIAL = 1_000_000.0
_MARK = "exposure_deposit_basis"
_ORDERS = pd.DataFrame(
    [("jp", "7203", int(OrderSide.BUY), 5000, 100.0, pd.Timestamp("2026-09-16"))],
    columns=["market", "symbol", "side", "qty", "fill_price", "filled_at"],
)


def _prices(market: str, symbol: str) -> pd.Series:
    """2026-10-15 までは 100 円、2026-10-16 から 110 円。"""
    prices = pd.Series(100.0, index=pd.bdate_range("2026-09-01", "2026-11-06"))
    prices[prices.index >= pd.Timestamp("2026-10-16")] = 110.0
    return prices


def _fetcher(ticker: str, start: str, end: str) -> dict[str, Any]:
    return {"ticker": ticker, "total_return": 0.10, "start": start, "end": end}


def _failing_fetcher(ticker: str, start: str, end: str) -> dict[str, Any]:
    return {"ticker": ticker, "total_return": None, "start": start, "end": end}


def _deposits(*days: str, amount: float = 9_000_000.0) -> list[tuple[pd.Timestamp, float]]:
    return [(pd.Timestamp(day), amount) for day in days]


def _compute(deposits: list[tuple[pd.Timestamp, float]]) -> dict[str, Any]:
    """2026-10-12〜16 の週（基準の日は 10-09）を計算する。"""
    return compute_weekly_results(
        _ORDERS, _prices, _INITIAL, date(2026, 10, 17), _fetcher, deposits=deposits
    )


# 土曜・日曜・月曜（東証の休場の日）・週の途中・金曜の入金
@pytest.mark.parametrize(
    "day", ["2026-10-10", "2026-10-11", "2026-10-12", "2026-10-14", "2026-10-15", "2026-10-16"]
)
def test_days_before_deposit_are_divided_by_base_after_deposit(day: str) -> None:
    jp = _compute(_deposits(day))["markets"]["jp"]
    # 保有 50 万円 ÷ 入金後の元手 1000 万円。保有はベンチマークと同じ +10%
    assert jp["exposure"] == pytest.approx(0.05)
    assert jp["weekly_return"] == pytest.approx(0.005)
    assert jp["excess_return"] == pytest.approx(0.0)


def test_two_deposits_in_one_week_are_added_once_each() -> None:
    jp = _compute(_deposits("2026-10-11", "2026-10-14", amount=4_500_000.0))["markets"]["jp"]
    assert jp["exposure"] == pytest.approx(0.05)
    assert jp["excess_return"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    "day, exposure",
    [
        ("2026-10-09", 0.05),  # 基準の日の入金は、評価額にもう入っている
        ("2026-10-03", 0.05),
        ("2026-10-17", 0.5),  # 週の終わりより後の入金
    ],
)
def test_deposit_outside_week_is_not_added(day: str, exposure: float) -> None:
    result = _compute(_deposits(day))
    assert result["markets"]["jp"]["exposure"] == pytest.approx(exposure)
    assert _MARK not in result


def test_mark_is_written_only_in_deposit_week() -> None:
    assert _compute(_deposits("2026-10-14"))[_MARK] == "after_deposits"
    assert _MARK not in _compute([])


# 木曜の回。対象の週は W44、前の3週は W41〜W43
_AS_OF = date(2026, 11, 5)


def _patch(monkeypatch: pytest.MonkeyPatch, deposits: list[tuple[pd.Timestamp, float]]) -> None:
    monkeypatch.setattr(paper_equity, "PAPER_INITIAL_BALANCE", _INITIAL)
    monkeypatch.setattr(paper_equity, "_load_filled_orders", lambda: _ORDERS)
    monkeypatch.setattr(paper_equity, "_default_price_loader", _prices)
    monkeypatch.setattr(paper_equity, "_default_split_loader", lambda market, symbol: None)
    monkeypatch.setattr(paper_equity, "load_cash_events", lambda: deposits)


def _seed(directory: Path, week: str, excess: Optional[float] = 0.123) -> Path:
    """印の無い、今までの計算の週のファイルを書く。"""
    values = {
        "weekly_return": 0.2,
        "benchmark_return": 0.1,
        "excess_return": excess,
        "exposure": 0.77,
        "excess_basis": "exposure_scaled",
        "partial_week": False,
    }
    path = directory / f"{week}.json"
    path.write_text(
        json.dumps({"week": week, "markets": {"jp": values, "us": dict(values)}}), encoding="utf-8"
    )
    return path


# 週の途中の入金、週の前の土曜の入金、見る範囲（前の3週）の外の週の入金
@pytest.mark.parametrize(
    "day, week",
    [("2026-10-14", "2026-W42"), ("2026-10-10", "2026-W42"), ("2026-09-23", "2026-W39")],
)
def test_old_deposit_week_is_recomputed_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, day: str, week: str
) -> None:
    _patch(monkeypatch, _deposits(day))
    path = _seed(tmp_path, week)
    other = _seed(tmp_path, "2026-W43")  # 入金の無い、切り替えより後の週
    other_before = other.read_bytes()

    write_paper_weekly_results(_AS_OF, _fetcher, str(tmp_path))

    written = json.loads(path.read_text(encoding="utf-8"))
    assert written["markets"]["jp"]["exposure"] == pytest.approx(0.05)
    assert written["markets"]["jp"]["excess_basis"] == "exposure_scaled"
    assert written[_MARK] == "after_deposits"
    assert other.read_bytes() == other_before

    # 2回目の回では書き直さない（書き直せば消える欄を足して確かめる）
    written["touched"] = True
    path.write_text(json.dumps(written), encoding="utf-8")
    before = path.read_bytes()
    write_paper_weekly_results(_AS_OF, _fetcher, str(tmp_path))
    assert path.read_bytes() == before


def test_old_deposit_week_is_kept_without_benchmark_and_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch(monkeypatch, _deposits("2026-10-14"))
    path = _seed(tmp_path, "2026-W42")
    before = path.read_bytes()

    write_paper_weekly_results(_AS_OF, _failing_fetcher, str(tmp_path))
    assert path.read_bytes() == before

    write_paper_weekly_results(_AS_OF, _fetcher, str(tmp_path))
    assert json.loads(path.read_text(encoding="utf-8"))[_MARK] == "after_deposits"


# 2026-10-17 は W42 の終わりの次の土曜。W43 の週の入金として扱う
@pytest.mark.parametrize("day", ["2026-10-17", "2026-10-21"])
def test_weeks_before_switch_are_not_rewritten(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, day: str
) -> None:
    _patch(monkeypatch, _deposits(day))
    frozen = [_seed(tmp_path, "2026-W39"), _seed(tmp_path, "2026-W42", excess=None)]
    before = [path.read_bytes() for path in frozen]
    switch_week = _seed(tmp_path, "2026-W43")

    write_paper_weekly_results(_AS_OF, _fetcher, str(tmp_path))

    assert [path.read_bytes() for path in frozen] == before
    assert json.loads(switch_week.read_text(encoding="utf-8"))[_MARK] == "after_deposits"
