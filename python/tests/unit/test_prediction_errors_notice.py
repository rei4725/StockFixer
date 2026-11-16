"""notify_prediction_errors の知らせの文を直に確かめる（#1901）。

送り先の _notify_stage_failure は差し替え、外へは送らない。
"""

from unittest.mock import MagicMock

import pytest

from src.orchestration.jobs import prediction_errors
from src.orchestration.jobs.prediction_errors import notify_prediction_errors


@pytest.fixture
def notify(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    mock = MagicMock()
    monkeypatch.setattr(prediction_errors, "_notify_stage_failure", mock)
    return mock


def _sent(notify: MagicMock) -> tuple[str, str]:
    notify.assert_called_once()
    title, exc = notify.call_args.args
    assert isinstance(exc, RuntimeError)
    return title, str(exc)


def test_empty_sends_nothing(notify: MagicMock) -> None:
    notify_prediction_errors([])
    notify.assert_not_called()


def test_six_keys_show_total_and_sorted_first_five_with_hoka(
    notify: MagicMock,
) -> None:
    keys = [
        ("us", "MSFT"),
        ("jp", "9984"),
        ("jp", "1301"),
        ("us", "AAPL"),
        ("jp", "7203"),
        ("jp", "6758"),
    ]
    notify_prediction_errors(keys)

    _, message = _sent(notify)
    assert message == (
        "例外で予想を出せなかった銘柄が 6 件あった: "
        "jp_1301, jp_6758, jp_7203, jp_9984, us_AAPL ほか"
    )
    assert "us_MSFT" not in message


def test_exactly_five_keys_have_no_hoka(notify: MagicMock) -> None:
    keys = [
        ("us", "AAPL"),
        ("jp", "9984"),
        ("jp", "1301"),
        ("jp", "7203"),
        ("jp", "6758"),
    ]
    notify_prediction_errors(keys)

    _, message = _sent(notify)
    assert message == (
        "例外で予想を出せなかった銘柄が 5 件あった: jp_1301, jp_6758, jp_7203, jp_9984, us_AAPL"
    )
    assert "ほか" not in message


def test_default_stage_in_title(notify: MagicMock) -> None:
    notify_prediction_errors([("jp", "7203")])

    title, _ = _sent(notify)
    assert title == "[3/6] 予測 (production) の一部の銘柄"


def test_given_stage_in_title(notify: MagicMock) -> None:
    notify_prediction_errors([("jp", "7203")], stage="[3.5/6] challenger 影の予想")

    title, message = _sent(notify)
    assert title == "[3.5/6] challenger 影の予想 の一部の銘柄"
    assert message == "例外で予想を出せなかった銘柄が 1 件あった: jp_7203"
