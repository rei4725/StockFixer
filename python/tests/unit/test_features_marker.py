"""features_marker（特徴量の版の印と作れた割合の線）のテスト（#1835）。

save_features_version と prediction_code_version は、features_marker が関数の中で
import する先（src.prediction.backtest_promotion）の上で差し替える。
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.domain.types import BatchFailure, BatchResult
from src.orchestration.jobs import features_marker as fm

_PKG = "src.prediction.backtest_promotion"


def _skips(market: str, n: int, *, as_dict: bool) -> list:
    """スキップの結果を n 件作る。dict（market の鍵）か属性の形。"""
    if as_dict:
        return [{"market": market, "symbol": f"{market}{i}", "status": "skip"} for i in range(n)]
    return [SimpleNamespace(market=market, symbol=f"{market}{i}") for i in range(n)]


def _fails(market: str, n: int) -> list[BatchFailure]:
    return [BatchFailure(market=market, symbol=f"{market}f{i}", error="boom") for i in range(n)]


def _batch(skipped: list | None = None, failed: list | None = None) -> BatchResult:
    return BatchResult(skipped=skipped or [], failed=failed or [])


# (名前, total, BatchResult, market_totals, 文を返すか)
_GAP_CASES = [
    ("全体も jp も0件", 10, _batch(), {"jp": 5, "us": 5}, False),
    ("全体の線の内", 10, _batch(failed=_fails("us", 1)), {"us": 10}, False),
    (
        "全体の線を超える（dict のスキップ+エラー）",
        10,
        _batch(skipped=_skips("us", 2, as_dict=True), failed=_fails("us", 1)),
        {"us": 10},
        True,
    ),
    (
        "全体の線を超える（属性のスキップ）",
        10,
        _batch(skipped=_skips("us", 3, as_dict=False)),
        {"us": 10},
        True,
    ),
    (
        "全体は線の内で jp だけ超える（dict のスキップ）",
        100,
        _batch(skipped=_skips("jp", 3, as_dict=True)),
        {"jp": 10, "us": 90},
        True,
    ),
    (
        "全体は線の内で jp だけ超える（属性のスキップ+エラー）",
        100,
        _batch(skipped=_skips("jp", 2, as_dict=False), failed=_fails("jp", 1)),
        {"jp": 10, "us": 90},
        True,
    ),
    (
        "us のスキップは jp の数に入らない",
        100,
        _batch(skipped=_skips("us", 3, as_dict=True)),
        {"jp": 10, "us": 90},
        False,
    ),
    ("対象0件で jp の対象も無い", 0, _batch(failed=_fails("jp", 1)), {}, False),
    (
        "全体の対象0件でも jp の対象があれば jp で見る",
        0,
        _batch(failed=_fails("jp", 3)),
        {"jp": 10},
        True,
    ),
    # ちょうど線の上（2/10）は線の内として書く側（#1843）。前は line = 1 - 0.8 が浮動小数で
    # 0.19999999999999996 になり、0.2 > line が真のため線の上も書かなかった（#1835 で固定していた）
    (
        "全体のちょうど線の上",
        10,
        _batch(skipped=_skips("us", 1, as_dict=True), failed=_fails("us", 1)),
        {"us": 10},
        False,
    ),
    (
        "jp のちょうど線の上",
        100,
        _batch(skipped=_skips("jp", 1, as_dict=False), failed=_fails("jp", 1)),
        {"jp": 10, "us": 90},
        False,
    ),
]


@pytest.mark.parametrize(
    "total,batch,market_totals,expect_gap",
    [c[1:] for c in _GAP_CASES],
    ids=[c[0] for c in _GAP_CASES],
)
def test_coverage_gap_table(total, batch, market_totals, expect_gap):
    gap = fm._coverage_gap(total, batch, market_totals)
    assert (gap is not None) == expect_gap, gap


def test_coverage_gap_message_names_jp_counts():
    batch = _batch(skipped=_skips("jp", 2, as_dict=True), failed=_fails("jp", 1))
    gap = fm._coverage_gap(100, batch, {"jp": 10, "us": 90})
    assert gap == "jp のスキップ 2 + エラー 1 / jp の対象 10（線 20%）"


def test_coverage_gap_message_names_total_counts():
    batch = _batch(skipped=_skips("us", 2, as_dict=False), failed=_fails("us", 1))
    gap = fm._coverage_gap(10, batch, {"us": 10})
    assert gap == "スキップ 2 + エラー 1 / 対象 10（線 20%）"


@pytest.fixture
def save_spy(monkeypatch) -> MagicMock:
    spy = MagicMock()
    monkeypatch.setattr(f"{_PKG}.save_features_version", spy)
    return spy


def test_if_covered_writes_once_inside_line(save_spy):
    batch = _batch(skipped=_skips("jp", 1, as_dict=True), failed=_fails("us", 1))
    fm.record_features_version_if_covered("v1", 100, batch, {"jp": 10, "us": 90})
    save_spy.assert_called_once_with("v1")


@pytest.mark.parametrize(
    "total,batch,market_totals",
    [
        (10, _batch(skipped=_skips("us", 1, as_dict=True), failed=_fails("us", 1)), {"us": 10}),
        (
            100,
            _batch(skipped=_skips("jp", 1, as_dict=False), failed=_fails("jp", 1)),
            {"jp": 10, "us": 90},
        ),
    ],
    ids=["全体のちょうど線の上", "jp のちょうど線の上"],
)
def test_if_covered_writes_on_exact_line(save_spy, total, batch, market_totals):
    """ちょうど 20% の回は線を超えていないので印を書く（#1843）。"""
    fm.record_features_version_if_covered("v1", total, batch, market_totals)
    save_spy.assert_called_once_with("v1")


def test_if_covered_skips_when_version_none(save_spy):
    fm.record_features_version_if_covered(None, 10, _batch(), {"jp": 10})
    save_spy.assert_not_called()


def test_if_covered_skips_when_over_total_line(save_spy):
    batch = _batch(skipped=_skips("us", 3, as_dict=False))
    fm.record_features_version_if_covered("v1", 10, batch, {"us": 10})
    save_spy.assert_not_called()


def test_if_covered_skips_when_over_jp_line(save_spy):
    batch = _batch(skipped=_skips("jp", 3, as_dict=True))
    fm.record_features_version_if_covered("v1", 100, batch, {"jp": 10, "us": 90})
    save_spy.assert_not_called()


def test_if_covered_skips_when_counting_fails(save_spy, monkeypatch):
    def _boom(*_args):
        raise RuntimeError("count failed")

    monkeypatch.setattr(fm, "_coverage_gap", _boom)
    fm.record_features_version_if_covered("v1", 10, _batch(), {"jp": 10})
    save_spy.assert_not_called()


def test_if_covered_skips_when_batch_cannot_be_counted(save_spy):
    """skipped が数えられない形なら、数えるのが落ちた回として書かない。"""
    broken = BatchResult(skipped=None)  # type: ignore[arg-type]
    fm.record_features_version_if_covered("v1", 10, broken, {"jp": 10})
    save_spy.assert_not_called()


def test_if_covered_does_not_raise_when_save_fails(monkeypatch):
    spy = MagicMock(side_effect=RuntimeError("db down"))
    monkeypatch.setattr(f"{_PKG}.save_features_version", spy)
    fm.record_features_version_if_covered("v1", 10, _batch(), {"jp": 10})
    spy.assert_called_once_with("v1")


def test_record_features_version_none_does_not_save(save_spy):
    fm.record_features_version(None)
    save_spy.assert_not_called()


def test_current_code_version_returns_version(monkeypatch):
    monkeypatch.setattr(f"{_PKG}.prediction_code_version", lambda: "abc123")
    assert fm.current_code_version() == "abc123"


def test_current_code_version_returns_none_on_error(monkeypatch):
    def _boom():
        raise RuntimeError("no git")

    monkeypatch.setattr(f"{_PKG}.prediction_code_version", _boom)
    assert fm.current_code_version() is None
