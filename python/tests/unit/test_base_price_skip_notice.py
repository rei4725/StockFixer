"""[3/6] の基準の値段の飛ばしを理由つきで jp の知らせと止める文に出すテスト（#1662）"""

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import run_predict
from pandas.tseries.offsets import BDay
from tests.unit.test_predict_unified_current_price_date import (
    _feature_frame,
    _history,
    _latest_weekday,
    _mock_model,
)
from tests.unit.test_prediction_gap_jp_gate import _rows, _run, _tasks

from src.orchestration.job_errors import NonRetryableJobError
from src.orchestration.jobs.unavailable_gates import (
    _raise_if_order_market_predictions_missing,
    order_market_predictions_mostly_missing,
)
from src.prediction.prediction_pipeline import predict_all_unified
from src.prediction.shadow_evaluation import predict_with_challenger_unified

_GATES = "src.orchestration.jobs.unavailable_gates"
_MARKER = "src.orchestration.jobs.prediction_saved_marker"
_PIPE = "src.prediction.prediction_pipeline"
_SHADOW = "src.prediction.shadow_evaluation"
_UNIFIED = "src.prediction.predict_unified"

_FEATURE_DAY = _latest_weekday()
_OLD_BAR = _FEATURE_DAY - BDay(1)
_NETWORK_ERROR = RuntimeError("network error")
_NO_BARS = pd.DataFrame()
_NAN_CLOSE = _history(_FEATURE_DAY, close=float("nan"))
_GOOD = _history(_FEATURE_DAY)
_OLD = _history(_OLD_BAR)


def _make_ticker(plan: dict):
    """銘柄ごとの history の結果（例外か足の表）を返す yf.Ticker の代わり"""

    def _ticker(key: tuple[str, str]) -> MagicMock:
        ticker = MagicMock()
        outcome = plan[key]
        if isinstance(outcome, Exception):
            ticker.history.side_effect = outcome
        else:
            ticker.history.return_value = outcome
        return ticker

    return _ticker


@contextmanager
def _unified_world(plan: dict):
    """特徴量とモデルを差し替え、基準の値段の取得だけ本物の道を通す"""
    with (
        patch(f"{_UNIFIED}.preload_models", return_value=[]),
        patch(f"{_PIPE}.get_active_symbols", return_value=list(plan)),
        patch("src.utils.db.get_active_symbols", return_value=list(plan)),
        patch(
            f"{_UNIFIED}.load_feature_data", side_effect=lambda m, s: _feature_frame(_FEATURE_DAY)
        ),
        patch(f"{_UNIFIED}.get_cached_model", return_value=_mock_model()),
        patch(f"{_UNIFIED}.get_ticker", side_effect=lambda m, s: (m, s)),
        patch(f"{_UNIFIED}.load_model_weights", return_value=[0.5, 0.5]),
        patch(f"{_UNIFIED}.get_service_url", return_value=None),
        patch(f"{_UNIFIED}.yf.Ticker", side_effect=_make_ticker(plan)),
    ):
        yield


def _run_production(plan: dict) -> list:
    with _unified_world(plan):
        return predict_all_unified(max_workers=2)


def _run_challenger(plan: dict) -> list:
    with (
        _unified_world(plan),
        patch(f"{_SHADOW}.available_unified_challengers", return_value=["Challenger"]),
    ):
        return predict_with_challenger_unified()


def _jp(n: int, outcome=_GOOD, prefix: str = "J") -> dict:
    return {("jp", f"{prefix}{i}"): outcome for i in range(n)}


def _gate(totals: dict[str, int], rows: list) -> tuple[bool, MagicMock]:
    with patch(f"{_GATES}._notify_stage_failure") as notify:
        kept = order_market_predictions_mostly_missing(totals, rows)
    return kept, notify


@pytest.fixture(autouse=True)
def _empty_record_after_test():
    """控えを次のテストへ残さない"""
    yield
    _run_production({})


def test_four_reasons_over_the_line_appear_in_notice_and_stop_message():
    """線を超えた回の知らせと止める文に、4つの理由が件数で出る"""
    rows = _run_production(
        {
            ("jp", "A"): _NETWORK_ERROR,
            ("jp", "B"): _NO_BARS,
            ("jp", "C"): _OLD,
            ("jp", "D"): _OLD,
            ("jp", "E"): _NAN_CLOSE,
        }
    )
    assert rows == []

    missing, notify = _gate({"jp": 5}, rows)
    with pytest.raises(NonRetryableJobError) as raised:
        _raise_if_order_market_predictions_missing({"jp": 5}, rows)

    assert missing is True
    notify.assert_called_once()
    assert "の欠け" in notify.call_args.args[0]
    for text in (str(notify.call_args.args[1]), str(raised.value)):
        assert "多すぎる: 5 / jp の対象 5" in text
        assert "基準の値段が取れず飛ばした jp の銘柄 5 件" in text
        assert "取得の例外 1 件" in text
        assert "足が空 1 件" in text
        assert "足の日付の不一致 2 件" in text
        assert "終値の欠け 1 件" in text


def test_date_mismatch_names_bar_day_and_feature_day():
    """足の日付の不一致の文に、足の日と特徴量の日が出る"""
    rows = _run_production(_jp(2, _OLD))

    _, notify = _gate({"jp": 2}, rows)

    text = str(notify.call_args.args[1])
    bar = _OLD_BAR.strftime("%Y-%m-%d")
    feature = _FEATURE_DAY.strftime("%Y-%m-%d")
    assert f"足の日 {bar}・特徴量の日 {feature} が 2 件" in text


def test_many_date_pairs_stay_short():
    """足の日の組が上限を超えると「ほか」で切り、全部の組は並べない"""
    bars = [_FEATURE_DAY - BDay(n) for n in range(1, 6)]
    plan = {("jp", f"J{n}"): _history(bar) for n, bar in enumerate(bars)}
    rows = _run_production(plan)

    _, notify = _gate({"jp": 5}, rows)

    text = str(notify.call_args.args[1])
    assert "足の日付の不一致 5 件" in text
    assert " ほか" in text
    assert text.count("足の日 ") == 3


def test_under_the_line_sends_one_notice_and_does_not_stop():
    """線の下で jp の飛ばしが1件ある回は1通だけ知らせ、止めない"""
    plan = _jp(9) | {("jp", "X"): _NO_BARS, ("us", "U"): _OLD}
    rows = _run_production(plan)

    missing, notify = _gate({"jp": 10, "us": 1}, rows)
    with patch(f"{_GATES}._notify_stage_failure") as second:
        _raise_if_order_market_predictions_missing({"jp": 10, "us": 1}, rows)

    assert missing is False
    notify.assert_called_once()
    stage, error = notify.call_args.args
    assert "[3/6]" in stage
    assert "基準の値段" in stage
    text = str(error)
    assert "jp の銘柄 1 件" in text
    assert "足が空 1 件" in text
    assert "足の日付の不一致" not in text  # us の飛ばしは数えない
    assert "jp の対象 10" in text
    second.assert_not_called()


def test_us_only_skips_send_no_notice():
    rows = _run_production(_jp(3) | {("us", "U"): _NETWORK_ERROR})

    missing, notify = _gate({"jp": 3, "us": 1}, rows)

    assert missing is False
    notify.assert_not_called()


def test_no_skips_send_no_notice():
    rows = _run_production(_jp(3))

    missing, notify = _gate({"jp": 3}, rows)

    assert missing is False
    notify.assert_not_called()


def test_challenger_skips_stay_out_of_the_production_record():
    """[3.5/6] の飛ばしは [3/6] の集計に入らない。保存の後の止める文も同じ"""
    rows = _run_production({("jp", "A"): _NO_BARS, ("jp", "B"): _GOOD})
    _run_challenger({("jp", "A"): _OLD, ("jp", "B"): _OLD, ("jp", "C"): _NAN_CLOSE})

    _, notify = _gate({"jp": 2}, rows)
    with pytest.raises(NonRetryableJobError) as raised:
        _raise_if_order_market_predictions_missing({"jp": 1}, [])

    notify.assert_called_once()
    for text in (str(notify.call_args.args[1]), str(raised.value)):
        assert "jp の銘柄 1 件" in text
        assert "足が空 1 件" in text
        assert "足の日付の不一致" not in text
        assert "終値の欠け" not in text


def test_next_run_does_not_carry_over_the_previous_record():
    _run_production(_jp(2, _NO_BARS))

    rows = _run_production(_jp(2))
    missing, notify = _gate({"jp": 2}, rows)

    assert missing is False
    notify.assert_not_called()


def test_run_predict_path_uses_the_same_message_as_the_daily_path():
    """run_predict.py の道の知らせと、日次の道の知らせが同じ文になる"""
    plan = {("jp", "A"): _NO_BARS, ("jp", "B"): _OLD}
    targets = [SimpleNamespace(market=m, symbol=s) for m, s in plan]
    with (
        _unified_world(plan),
        patch("run_predict.output_top_worst_results"),
        patch("src.watchlist.batch_runner.load_target_symbols", return_value=targets),
        patch(f"{_MARKER}.clear_prediction_saved_marker"),
        patch(f"{_MARKER}.mark_prediction_saved") as mark,
        patch(f"{_GATES}._notify_stage_failure") as notify,
    ):
        run_predict.run_top10()

    mark.assert_not_called()
    notify.assert_called_once()
    text = str(notify.call_args.args[1])
    assert "足が空 1 件" in text and "足の日付の不一致 1 件" in text

    daily = _run(_rows("us", 1), _tasks(2, 1))

    assert str(daily["notify"].call_args.args[1]) == text
    assert "足が空 1 件" in str(daily["raised"])
