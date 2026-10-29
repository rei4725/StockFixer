"""日次の取得で market_data_raw の保存が落ちた銘柄を failed に数えるテスト（#1377）"""

from unittest.mock import patch

import pandas as pd

from src.domain.types import SymbolTask

_PIPELINE = "src.market_data.pipeline"

# 足の保存が落ちる銘柄（OK は成功の道、BAD は成功の道、SKIPBAD はスキップの道）
_RAW_FAIL = {"BAD", "SKIPBAD"}


def _bars() -> pd.DataFrame:
    return pd.DataFrame(
        {"Open": [1.0], "High": [1.0], "Low": [1.0], "Close": [1.0], "Volume": [100]},
        index=pd.bdate_range(start="2026-10-28", periods=1),
    )


def _fake_fetch(market, symbol, **_kwargs):
    if symbol.startswith("SKIP"):
        return (market, symbol, None, _bars(), None)
    return (market, symbol, pd.DataFrame({"x": [1]}), _bars(), None)


def _fake_save_raw(market, symbol, _data):
    if symbol in _RAW_FAIL:
        raise RuntimeError("volume の型が合わない")


def _run(symbols):
    from src.market_data.pipeline import run_batch_pipeline

    tasks = [SymbolTask(market="us", symbol=s) for s in symbols]
    with (
        patch(f"{_PIPELINE}.fetch_stock_data_with_features", side_effect=_fake_fetch),
        patch(f"{_PIPELINE}.save_raw_ohlcv", side_effect=_fake_save_raw) as save_raw,
        patch(f"{_PIPELINE}.save_features_to_db") as save_features,
    ):
        result = run_batch_pipeline(tasks)
    return result, save_raw, save_features


def _succeeded_symbols(result) -> set:
    return {r["symbol"] for r in result.succeeded}


def _failed_symbols(result) -> set:
    return {f.symbol for f in result.failed}


class TestRawSaveFailureCountsAsFailed:
    def test_success_path_raw_save_failure_is_failed(self):
        """特徴量は保存できても足の保存が落ちた銘柄は failed に入り succeeded に入らないこと"""
        result, _, _ = _run(["OK", "BAD"])

        assert _failed_symbols(result) == {"BAD"}
        assert _succeeded_symbols(result) == {"OK"}
        failure = next(f for f in result.failed if f.symbol == "BAD")
        assert "market_data_raw の保存に失敗" in failure.error
        assert "volume の型が合わない" in failure.error

    def test_skip_path_raw_save_failure_is_failed(self):
        """特徴量をスキップした銘柄でも足の保存が落ちたら failed に入ること"""
        result, _, _ = _run(["OK", "SKIPOK", "SKIPBAD"])

        assert _failed_symbols(result) == {"SKIPBAD"}
        assert _succeeded_symbols(result) == {"OK"}
        assert {r["symbol"] for r in result.skipped} == {"SKIPOK"}
        failure = next(f for f in result.failed if f.symbol == "SKIPBAD")
        assert "market_data_raw の保存に失敗" in failure.error

    def test_other_symbols_and_features_still_saved(self):
        """足の保存が落ちても、ほかの銘柄の足とすべての成功銘柄の特徴量は保存を続けること"""
        result, save_raw, save_features = _run(["SKIPBAD", "BAD", "OK", "SKIPOK"])

        raw_saved = {c.args[1] for c in save_raw.call_args_list}
        assert raw_saved == {"SKIPBAD", "BAD", "OK", "SKIPOK"}
        features_saved = {c.args[1] for c in save_features.call_args_list}
        assert features_saved == {"BAD", "OK"}
        assert _succeeded_symbols(result) & _failed_symbols(result) == set()
        assert len(result.failed) == 2

    def test_gate_lists_raw_save_failure_in_notice(self):
        """_raise_if_fetch_mostly_unavailable が足の保存の落ちた銘柄を知らせに並べること"""
        from src.orchestration.jobs.unavailable_gates import _raise_if_fetch_mostly_unavailable

        symbols = [f"OK{i}" for i in range(20)] + ["BAD"]
        result, _, _ = _run(symbols)

        with patch("src.orchestration.jobs.unavailable_gates._notify_stage_failure") as notify:
            _raise_if_fetch_mostly_unavailable(len(symbols), result)

        notify.assert_called_once()
        assert "us_BAD" in str(notify.call_args.args[1])
