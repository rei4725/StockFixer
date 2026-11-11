"""相場の指標の列が欠けた銘柄も、回に1通の知らせと [1/6] の文に載せるテスト（#1642）"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from tests.unit.test_daily_fetch_error_notice import _run_daily_pipeline
from tests.unit.test_macro_stale_notice import _bars

from src.domain.types import BatchResult, MacroMissingSkip, MacroStaleSkip, SymbolTask
from src.market_data import pipeline
from src.orchestration.jobs.unavailable_gates import (
    _macro_stale_summary,
    _raise_if_fetch_mostly_unavailable,
)

_GATES = "src.orchestration.jobs.unavailable_gates"
_LAST_BAR = "2026-10-21"
_ALL_COLUMNS = (
    "vix_close",
    "usdjpy_close",
    "tnx_close",
    "sp500_close",
    "dxy_close",
    "gold_close",
)


def _history_empty_for(*empty_tickers: str):
    """empty_tickers だけ空を返し、ほかは終値の足を返す ticker_history の代わり"""

    def _history(ticker, start, end):
        if ticker in empty_tickers:
            return pd.DataFrame()
        dates = pd.bdate_range("2026-05-01", _LAST_BAR)
        return pd.DataFrame({"Close": np.linspace(10, 20, len(dates))}, index=dates)

    return _history


def _run_batch(history, symbols: tuple[str, ...], market: str = "us") -> BatchResult:
    """run_batch_pipeline を本物のまま流す"""
    tasks = [SymbolTask(market=market, symbol=symbol) for symbol in symbols]
    with (
        patch("src.market_data.yf_client.ticker_history", side_effect=history),
        patch(f"{pipeline.__name__}.get_raw_ohlcv_from_db", return_value=_bars(_LAST_BAR)),
        patch(f"{pipeline.__name__}.should_fetch_fresh_data", return_value=False),
        patch(f"{pipeline.__name__}.fetch_news_sentiment_with_llm", return_value=None),
        patch(f"{pipeline.__name__}.fetch_pageview_features", return_value=None),
        patch(f"{pipeline.__name__}.resolve_article", return_value=None),
        patch(f"{pipeline.__name__}.save_features_to_db"),
        patch(f"{pipeline.__name__}.save_raw_ohlcv"),
    ):
        return pipeline.run_batch_pipeline(tasks)


def test_empty_ticker_sends_one_notice_with_column_and_count():
    """線の下でも、ティッカーが丸ごと空の回は欠けた列の名と銘柄の数の知らせが1通出る"""
    batch = _run_batch(_history_empty_for("^VIX"), ("A", "B"))

    assert len(batch.skipped) == 2
    with patch("src.reporting.discord.discord_utils.send_webhook_text") as webhook:
        _raise_if_fetch_mostly_unavailable(100, batch)

    webhook.assert_called_once()
    text = webhook.call_args.args[0]
    assert "[1/6]" in text
    assert "vix_close" in text
    assert "欠けて" in text
    assert "2 件" in text
    assert "sp500_close" not in text  # 取れた列は出さない


def test_every_symbol_missing_names_the_columns_in_the_stage_notice():
    """全銘柄が欠けで落ちた回の [1/6] の文に全部の欠けた列が出て、知らせは増えない"""
    batch = _run_batch(lambda ticker, start, end: pd.DataFrame(), ("A", "B", "C", "D"))

    mocks = _run_daily_pipeline(batch, total=4)

    assert isinstance(mocks["raised"], RuntimeError)
    mocks["error"].assert_called_once()
    text = mocks["error"].call_args.args[0]
    assert "[1/6] データ取得 失敗" in text
    assert "多すぎる" in text
    assert "欠けて" in text and "4 件" in text
    assert all(col in text for col in _ALL_COLUMNS)
    mocks["webhook"].assert_not_called()


def test_order_market_gate_names_the_missing_column():
    """jp だけが線を超えて落ちた回も、その文に欠けた列が出る"""
    batch = _run_batch(_history_empty_for("GC=F"), ("A", "B"), market="jp")

    with pytest.raises(RuntimeError, match="jp の取得できなかった.*gold_close"):
        _raise_if_fetch_mostly_unavailable(100, batch, {"jp": 3, "us": 97})


def test_second_run_does_not_carry_over_the_previous_record():
    """回の記録は次の回に持ち越さない"""
    _run_batch(_history_empty_for("^VIX"), ("A", "B"))

    batch = _run_batch(_history_empty_for(), ("A", "B"))

    assert batch.macro_missing == []
    assert batch.skipped == []


def test_no_missing_or_stale_symbol_sends_no_notice():
    batch = _run_batch(_history_empty_for(), ("A", "B"))

    with patch("src.reporting.discord.discord_utils.send_webhook_text") as webhook:
        _raise_if_fetch_mostly_unavailable(2, batch)

    webhook.assert_not_called()


def test_stale_and_missing_share_one_notice_and_keep_the_stale_text():
    """同じ回に止まった列と欠けた列があっても知らせは1通で、止まった列の文は変わらない"""
    stale_batch = BatchResult(
        macro_stale=[MacroStaleSkip("us", "A", {"tnx_close": "2026-10-15"})],
    )
    batch = BatchResult(
        macro_stale=stale_batch.macro_stale,
        macro_missing=[MacroMissingSkip("us", "B", ["vix_close"])],
    )

    with patch(f"{_GATES}._notify_stage_failure") as notify:
        _raise_if_fetch_mostly_unavailable(10, batch)

    notify.assert_called_once()
    message = str(notify.call_args.args[1])
    assert _macro_stale_summary(stale_batch) in message
    assert "銘柄 1 件: tnx_close（最後の日 2026-10-15、1 件）" in message
    assert "欠けて特徴量を作らなかった銘柄 1 件: vix_close（欠けた、1 件）" in message


def test_stage_notice_stays_short_when_every_column_is_missing_and_stale():
    """止まった列の組が上限まであり、6列全部が欠けても Discord の欄の長さに収まる"""
    batch = BatchResult(
        skipped=[{"market": "us", "symbol": f"S{i}"} for i in range(1000)],
        macro_stale=[
            MacroStaleSkip("us", f"S{i}", {"usdjpy_close": f"2026-10-{i + 1:02d}"})
            for i in range(28)
        ],
        macro_missing=[MacroMissingSkip("us", f"M{i}", list(_ALL_COLUMNS)) for i in range(972)],
    )

    with pytest.raises(RuntimeError) as raised:
        _raise_if_fetch_mostly_unavailable(1000, batch)

    assert "欠けて" in str(raised.value)
    assert len(str(raised.value)) < 1000
