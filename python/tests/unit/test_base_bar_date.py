"""base_bar_date（採点の基準の日と分割の倍率）の部品のテスト"""

from datetime import datetime

import pytest

from src.prediction.base_bar_date import _base_bar_date, _split_adjust_scale


@pytest.mark.parametrize(
    ("predicted_at", "market", "expected"),
    [
        # jp の引けは 15:30 JST = 06:30 UTC。引けちょうどはその日
        pytest.param("20261124_063000", "jp", datetime(2026, 11, 24), id="jp-at-close"),
        # 11-23 は祝日だが、営業日には丸めない
        pytest.param("20261124_062959", "jp", datetime(2026, 11, 23), id="jp-before-close"),
        # us の冬時間の引けは 21:00 UTC
        pytest.param("20261124_210000", "us", datetime(2026, 11, 24), id="us-winter-at-close"),
        pytest.param("20261124_205959", "us", datetime(2026, 11, 23), id="us-winter-before-close"),
        # 夏時間の引け（20:00 UTC）は過ぎているが、冬時間ではまだ引けていない
        pytest.param(
            "20261124_203000", "us", datetime(2026, 11, 23), id="us-winter-past-dst-close"
        ),
        # us の夏時間の引けは 20:00 UTC
        pytest.param("20260715_200000", "us", datetime(2026, 7, 15), id="us-summer-at-close"),
        pytest.param("20260715_195959", "us", datetime(2026, 7, 14), id="us-summer-before-close"),
        # 市場が不明・時刻が読めないときは UTC の日付
        pytest.param("20261124_000000", "xx", datetime(2026, 11, 24), id="unknown-market"),
        pytest.param("20261124", "jp", datetime(2026, 11, 24), id="jp-no-time"),
        pytest.param("20261124", "us", datetime(2026, 11, 24), id="us-no-time"),
        pytest.param("20261124_xxxxxx", "jp", datetime(2026, 11, 24), id="jp-bad-time"),
        pytest.param("20261124_xxxxxx", "us", datetime(2026, 11, 24), id="us-bad-time"),
    ],
)
def test_base_bar_date(predicted_at: str, market: str, expected: datetime) -> None:
    assert _base_bar_date(predicted_at, market) == expected


@pytest.mark.parametrize("market", ["jp", "us", "xx"])
def test_base_bar_date_unreadable_date_raises(market: str) -> None:
    with pytest.raises(ValueError):
        _base_bar_date("invalid", market)


@pytest.mark.parametrize(
    ("current_price", "base_close", "expected"),
    [
        pytest.param(1000.0, 1300.0, 1.0, id="gap-tolerance-edge"),
        pytest.param(1000.0, 1340.0, None, id="outside-split-ratio-tolerance"),
        pytest.param(1000.0, 1360.0, 1.5, id="inside-split-ratio-tolerance"),
        # 基準の終値が高い側（gap > 1）は倍率をそのまま返す
        pytest.param(1000.0, 5000.0, 5.0, id="gap-above-one"),
        # 基準の終値が低い側は逆数を返す
        pytest.param(1000.0, 200.0, 0.2, id="gap-below-one"),
        pytest.param(0.0, 1000.0, None, id="zero-current"),
        pytest.param(1000.0, 0.0, None, id="zero-base"),
        pytest.param(-1000.0, 1000.0, None, id="negative-current"),
        pytest.param(1000.0, -1000.0, None, id="negative-base"),
        pytest.param(-1000.0, -1000.0, None, id="negative-both"),
    ],
)
def test_split_adjust_scale(
    current_price: float, base_close: float, expected: float | None
) -> None:
    result = _split_adjust_scale(current_price, base_close)
    if expected is None:
        assert result is None
    else:
        assert result == pytest.approx(expected)
