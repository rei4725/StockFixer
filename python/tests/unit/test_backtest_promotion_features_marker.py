"""日次の道が stock_features を作り直した版の印を書くテスト（仮の入れ替えの特徴量の待ち）

結果の置き場は conftest がテストごとの一時フォルダに向ける。
"""

from types import SimpleNamespace
from unittest.mock import patch

from tests.unit.test_daily_fetch_error_notice import _batch, _run_daily_pipeline

from src.domain.types import BatchResult, SymbolTask
from src.prediction.backtest_promotion import state as st


def test_daily_pipeline_writes_marker_after_successful_rebuild():
    mocks = _run_daily_pipeline(_batch(783, 0, 1), total=784)

    assert mocks["raised"] is None
    assert st.load_features_version() == st.prediction_code_version()


def test_daily_pipeline_keeps_marker_when_rebuild_fails():
    st.save_features_version("oldcode00000")

    mocks = _run_daily_pipeline(_batch(300, 483, 1), total=784)

    assert isinstance(mocks["raised"], RuntimeError)
    assert st.load_features_version() == "oldcode00000"


def test_daily_pipeline_keeps_marker_when_overall_coverage_below_min(caplog):
    """全体の欠けが 1 - MIN_COVERAGE を超え取得の線の下なら、道は進み印は前の版のまま（#1710）"""
    st.save_features_version("oldcode00000")

    with caplog.at_level("WARNING"):
        mocks = _run_daily_pipeline(_batch(584, 0, 200), total=784)

    assert mocks["raised"] is None
    mocks["predict"].assert_called_once()
    assert st.load_features_version() == "oldcode00000"
    lines = [r.getMessage() for r in caplog.records if "特徴量の版の印を書かない" in r.getMessage()]
    assert len(lines) == 1
    assert "200" in lines[0] and "784" in lines[0] and "20%" in lines[0]


def test_daily_pipeline_keeps_marker_when_jp_coverage_below_min(caplog):
    """全体は線の下でも jp の欠けが 1 - MIN_COVERAGE を超えたら印は前の版のまま（#1710）"""
    st.save_features_version("oldcode00000")
    tasks = [SymbolTask(market="us", symbol=f"U{i}") for i in range(584)]
    tasks += [SymbolTask(market="jp", symbol=f"{1000 + i}") for i in range(200)]
    batch = BatchResult(
        succeeded=[object()] * 734,
        failed=[],
        skipped=[{"market": "jp", "symbol": f"{1000 + i}"} for i in range(50)],
    )

    with caplog.at_level("WARNING"):
        rows = [SimpleNamespace(market="jp")] * 150  # [3/6] の jp の欠けの線は通す
        mocks = _run_daily_pipeline(batch, total=784, tasks=tasks, rows=rows)

    assert mocks["raised"] is None
    mocks["predict"].assert_called_once()
    assert st.load_features_version() == "oldcode00000"
    lines = [r.getMessage() for r in caplog.records if "特徴量の版の印を書かない" in r.getMessage()]
    assert len(lines) == 1
    assert "jp" in lines[0] and "50" in lines[0] and "200" in lines[0] and "20%" in lines[0]


def test_marker_write_failure_does_not_stop_daily_pipeline():
    with patch.object(st, "_write_json_atomic", side_effect=OSError("disk full")):
        mocks = _run_daily_pipeline(_batch(783, 0, 1), total=784)

    assert mocks["raised"] is None
    mocks["predict"].assert_called_once()
    assert st.load_features_version() is None
