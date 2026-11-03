"""日次の道が stock_features を作り直した版の印を書くテスト（仮の入れ替えの特徴量の待ち）

結果の置き場は conftest がテストごとの一時フォルダに向ける。
"""

from unittest.mock import patch

from tests.unit.test_daily_fetch_error_notice import _batch, _run_daily_pipeline

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


def test_marker_write_failure_does_not_stop_daily_pipeline():
    with patch.object(st, "_write_json_atomic", side_effect=OSError("disk full")):
        mocks = _run_daily_pipeline(_batch(783, 0, 1), total=784)

    assert mocks["raised"] is None
    mocks["predict"].assert_called_once()
    assert st.load_features_version() is None
