"""相場の指標の列が止まった銘柄を、回に1通の知らせにまとめるテスト（#1634）"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from tests.unit.test_daily_fetch_error_notice import _run_daily_pipeline

from src.domain.types import BatchResult, MacroStaleSkip, SymbolTask
from src.market_data import pipeline
from src.orchestration.jobs.unavailable_gates import _raise_if_fetch_mostly_unavailable

_GATES = "src.orchestration.jobs.unavailable_gates"
_NEW_BAR = "2026-10-21"  # 水曜。^VIX の最後の日より平日で4日新しい（線の外）
_OLD_BAR = "2026-10-20"  # 火曜。平日で3日新しい（線ちょうど）
_VIX_LAST = "2026-10-15"


def _bars(last_day: str) -> pd.DataFrame:
    dates = pd.bdate_range("2026-06-01", last_day)
    n = len(dates)
    return pd.DataFrame(
        {
            "Open": np.linspace(100, 120, n),
            "High": np.linspace(101, 121, n),
            "Low": np.linspace(99, 119, n),
            "Close": np.linspace(100, 120, n),
            "Volume": np.full(n, 3000),
        },
        index=dates,
    )


def _history(ticker, start, end):
    """^VIX だけ止まった相場の指標の足を返す ticker_history の代わり"""
    dates = pd.bdate_range("2026-05-01", _VIX_LAST if ticker == "^VIX" else _NEW_BAR)
    return pd.DataFrame({"Close": np.linspace(10, 20, len(dates))}, index=dates)


def _run_batch(last_bars: dict[str, str]) -> BatchResult:
    """銘柄ごとの最後の足を DB から読む形で、run_batch_pipeline を本物のまま流す"""

    def _db_bars(market, symbol, start, end):
        return _bars(last_bars[symbol])

    tasks = [SymbolTask(market="us", symbol=symbol) for symbol in last_bars]
    with (
        patch("src.market_data.yf_client.ticker_history", side_effect=_history),
        patch(f"{pipeline.__name__}.get_raw_ohlcv_from_db", side_effect=_db_bars),
        patch(f"{pipeline.__name__}.should_fetch_fresh_data", return_value=False),
        patch(f"{pipeline.__name__}.fetch_news_sentiment_with_llm", return_value=None),
        patch(f"{pipeline.__name__}.fetch_pageview_features", return_value=None),
        patch(f"{pipeline.__name__}.resolve_article", return_value=None),
        patch(f"{pipeline.__name__}.save_features_to_db"),
        patch(f"{pipeline.__name__}.save_raw_ohlcv"),
    ):
        return pipeline.run_batch_pipeline(tasks)


def _stale(symbol: str, market: str = "us", **columns: str) -> MacroStaleSkip:
    return MacroStaleSkip(market=market, symbol=symbol, columns=columns)


def test_few_stale_symbols_send_one_notice_with_column_day_and_count():
    """線に届かない少数の銘柄だけが止まった回は、列と最後の日と銘柄の数の知らせが1通出る"""
    last_bars = {"A": _NEW_BAR, "B": _NEW_BAR, "C": _OLD_BAR, "D": _OLD_BAR, "E": _OLD_BAR}
    batch = _run_batch(last_bars)

    assert sorted(s.symbol for s in batch.macro_stale) == ["A", "B"]
    assert len(batch.skipped) == 2
    with patch("src.reporting.discord.discord_utils.send_webhook_text") as webhook:
        _raise_if_fetch_mostly_unavailable(len(last_bars), batch, {"us": len(last_bars)})

    webhook.assert_called_once()
    text = webhook.call_args.args[0]
    assert "[1/6]" in text
    assert "vix_close" in text
    assert _VIX_LAST in text
    assert "2 件" in text
    assert "A" not in text  # 銘柄は並べない


def test_second_run_does_not_carry_over_the_previous_record():
    """回の記録は次の回に持ち越さない"""
    _run_batch({"A": _NEW_BAR, "B": _OLD_BAR, "C": _OLD_BAR})

    batch = _run_batch({"A": _OLD_BAR, "B": _OLD_BAR, "C": _OLD_BAR})

    assert batch.macro_stale == []
    assert batch.skipped == []


def test_no_stale_symbol_sends_no_notice():
    batch = _run_batch({"A": _OLD_BAR, "B": _OLD_BAR})

    with patch("src.reporting.discord.discord_utils.send_webhook_text") as webhook:
        _raise_if_fetch_mostly_unavailable(2, batch)

    webhook.assert_not_called()


def test_every_symbol_stale_names_the_column_in_the_stage_notice():
    """全銘柄が止まった回の [1/6] の知らせに原因の列が出て、知らせは増えない"""
    batch = BatchResult(
        skipped=[{"market": "us", "symbol": f"S{i}"} for i in range(4)],
        macro_stale=[_stale(f"S{i}", tnx_close=_VIX_LAST) for i in range(4)],
    )

    mocks = _run_daily_pipeline(batch, total=4)

    assert isinstance(mocks["raised"], RuntimeError)
    mocks["error"].assert_called_once()
    text = mocks["error"].call_args.args[0]
    assert "[1/6] データ取得 失敗" in text
    assert "多すぎる" in text
    assert "tnx_close" in text and _VIX_LAST in text and "4 件" in text
    mocks["webhook"].assert_not_called()


def test_order_market_gate_names_the_column():
    """jp だけが線を超えて止まった回も、その知らせに原因の列が出る"""
    batch = BatchResult(
        succeeded=[object()] * 8,
        skipped=[{"market": "jp", "symbol": f"S{i}"} for i in range(2)],
        macro_stale=[_stale(f"S{i}", market="jp", gold_close="なし") for i in range(2)],
    )

    with pytest.raises(RuntimeError, match="jp の取得できなかった.*gold_close") as raised:
        _raise_if_fetch_mostly_unavailable(10, batch, {"jp": 3, "us": 7})

    assert "なし" in str(raised.value)


def test_notice_groups_columns_and_days_and_counts_symbols_once():
    """1銘柄に2列止まっても銘柄の数は1で、列と最後の日の組ごとに数える"""
    batch = BatchResult(
        succeeded=[object()] * 8,
        skipped=[{"market": "us", "symbol": s} for s in ("A", "B")],
        macro_stale=[
            _stale("A", vix_close=_VIX_LAST, dxy_close="2026-10-02"),
            _stale("B", vix_close="2026-10-14"),
        ],
    )

    with patch(f"{_GATES}._notify_stage_failure") as notify:
        _raise_if_fetch_mostly_unavailable(10, batch)

    notify.assert_called_once()
    message = str(notify.call_args.args[1])
    assert "銘柄 2 件" in message
    assert f"vix_close（最後の日 {_VIX_LAST}、1 件）" in message
    assert "vix_close（最後の日 2026-10-14、1 件）" in message
    assert "dxy_close（最後の日 2026-10-02、1 件）" in message


def test_stage_notice_stays_short_when_the_groups_are_many():
    """Discord の欄の長さに収まるよう、列と最後の日の組は先頭だけを並べる"""
    batch = BatchResult(
        skipped=[{"market": "us", "symbol": f"S{i}"} for i in range(40)],
        macro_stale=[_stale(f"S{i}", vix_close=f"2026-10-{i % 28 + 1:02d}") for i in range(40)],
    )

    with pytest.raises(RuntimeError) as raised:
        _raise_if_fetch_mostly_unavailable(40, batch)

    assert "ほか" in str(raised.value)
    assert len(str(raised.value)) < 1000


def test_missing_column_is_not_counted_as_stale():
    """欠けた列は止まった列の知らせに数えない（線と倒し方は今までどおり）"""
    with (
        patch("src.market_data.yf_client.ticker_history", return_value=pd.DataFrame()),
        patch(f"{pipeline.__name__}.get_raw_ohlcv_from_db", return_value=_bars(_NEW_BAR)),
        patch(f"{pipeline.__name__}.should_fetch_fresh_data", return_value=False),
    ):
        batch = pipeline.run_batch_pipeline([SymbolTask(market="us", symbol="A")])

    assert len(batch.skipped) == 1
    assert batch.macro_stale == []
