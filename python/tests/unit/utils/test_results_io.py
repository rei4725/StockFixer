"""Unit Test: 結果 CSV 保存ヘルパー（Phase 1 / PR-2）

results/ 配下への「ensure_dir + タイムスタンプ + to_csv + パス返却」を一本化した
ヘルパーのテスト。タイムゾーンは CLAUDE.md の「内部 UTC」に揃える。
"""

import re
from unittest.mock import patch

import pandas as pd
import pytest

from src.utils.results_io import results_timestamp, save_csv, save_result_csvs

_TS_PATTERN = re.compile(r"^\d{8}_\d{6}$")


@pytest.fixture
def frame():
    return pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})


class TestResultsTimestamp:
    def test_format(self):
        assert _TS_PATTERN.match(results_timestamp())

    def test_is_utc_not_local(self):
        """ローカル時刻ではなく UTC であること（JST なら 9 時間ずれる）。"""
        from datetime import datetime, timezone

        expected = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
        assert results_timestamp().startswith(expected)


class TestSaveCsv:
    def test_creates_file_and_returns_path(self, tmp_path, frame):
        path = save_csv(frame, str(tmp_path / "nested"), "out.csv")
        assert path.endswith("out.csv")
        assert pd.read_csv(path).equals(frame)

    def test_creates_missing_directory(self, tmp_path, frame):
        target = tmp_path / "a" / "b" / "c"
        save_csv(frame, str(target), "out.csv")
        assert target.is_dir()

    def test_does_not_write_index(self, tmp_path, frame):
        path = save_csv(frame, str(tmp_path), "out.csv")
        with open(path, encoding="utf-8") as fh:
            assert fh.readline().strip() == "a,b"

    def test_encoding_is_honored(self, tmp_path):
        """stress_test は utf-8-sig（BOM 付き）を要求する。"""
        df = pd.DataFrame({"名前": ["あ"]})
        path = save_csv(df, str(tmp_path), "sig.csv", encoding="utf-8-sig")
        with open(path, "rb") as fh:
            assert fh.read(3) == b"\xef\xbb\xbf"

    def test_default_encoding_has_no_bom(self, tmp_path, frame):
        path = save_csv(frame, str(tmp_path), "plain.csv")
        with open(path, "rb") as fh:
            assert fh.read(3) != b"\xef\xbb\xbf"


class TestSaveResultCsvs:
    def test_saves_each_frame_under_results_subdir(self, tmp_path, frame):
        with patch("src.utils.results_io.get_results_dir", return_value=str(tmp_path)):
            paths = save_result_csvs({"equity_us": frame, "trades_us": frame}, "backtest/longterm")

        assert set(paths) == {"equity_us", "trades_us"}
        for key, path in paths.items():
            assert (tmp_path / "backtest" / "longterm").as_posix() in path.replace("\\", "/")
            assert key in path

    def test_filename_is_stem_plus_timestamp(self, tmp_path, frame):
        with patch("src.utils.results_io.get_results_dir", return_value=str(tmp_path)):
            paths = save_result_csvs({"equity_us": frame}, "backtest/longterm")

        name = paths["equity_us"].replace("\\", "/").rsplit("/", 1)[-1]
        m = re.match(r"^(?P<stem>.+)_(?P<ts>\d{8}_\d{6})\.csv$", name)
        assert m is not None, name
        assert m.group("stem") == "equity_us"

    def test_all_frames_share_one_timestamp(self, tmp_path, frame):
        """同じ保存呼び出しの CSV 群は同一タイムスタンプで揃うこと。"""
        with patch("src.utils.results_io.get_results_dir", return_value=str(tmp_path)):
            paths = save_result_csvs({"equity": frame, "trades": frame}, "backtest/longterm")

        stamps = {
            re.search(r"_(\d{8}_\d{6})\.csv$", p).group(1)  # type: ignore[union-attr]
            for p in paths.values()
        }
        assert len(stamps) == 1

    def test_empty_frames_mapping_returns_empty(self, tmp_path):
        with patch("src.utils.results_io.get_results_dir", return_value=str(tmp_path)):
            assert save_result_csvs({}, "backtest/longterm") == {}

    def test_writes_readable_content(self, tmp_path, frame):
        with patch("src.utils.results_io.get_results_dir", return_value=str(tmp_path)):
            paths = save_result_csvs({"equity": frame}, "screening")
        assert pd.read_csv(paths["equity"]).equals(frame)
