"""unit テストの本番 DB 隔離ガードの検証（#548 再発防止）。

conftest.py の autouse fixture（_isolate_db / _forbid_production_duckdb_connect）が
すべての get_db_path 経路を一時 DB へ向け、本番 data ディレクトリへの
duckdb.connect を拒否することを確認する。
"""

import os

import duckdb
import pytest


class TestIsolateDbPaths:
    """_isolate_db: 全 import 経路の get_db_path が一時 DB を返すこと"""

    def test_data_path_utils_get_db_path_is_isolated(self):
        """data_path_utils 直 import 経路（週次 compact 等が使う）が隔離されること"""
        from src.utils.data_path_utils import get_db_path

        assert os.path.basename(get_db_path()) == "unit.duckdb"

    def test_db_package_get_db_path_is_isolated(self):
        """src.utils.db プロキシ経由（_db_connection 等が使う）が隔離されること"""
        import src.utils.db as db_module

        assert os.path.basename(db_module.get_db_path()) == "unit.duckdb"

    def test_both_paths_point_to_same_file(self):
        import src.utils.db as db_module
        from src.utils.data_path_utils import get_db_path

        assert get_db_path() == db_module.get_db_path()


class TestForbidProductionDuckdbConnect:
    """_forbid_production_duckdb_connect: 本番 data ディレクトリへの接続拒否"""

    def test_connect_to_production_db_is_blocked(self):
        """本番 DB パスへの直接 connect が RuntimeError になること"""
        from src.utils.data_path_utils import get_data_dir

        prod_db = os.path.join(get_data_dir(), "stockfixer.duckdb")
        with pytest.raises(RuntimeError, match="禁止"):
            duckdb.connect(prod_db)

    def test_connect_to_production_data_subpath_is_blocked(self):
        """本番 data 配下の別ファイル（.compact 等）への connect も拒否されること"""
        from src.utils.data_path_utils import get_data_dir

        compact_path = os.path.join(get_data_dir(), "stockfixer.duckdb.compact")
        with pytest.raises(RuntimeError, match="禁止"):
            duckdb.connect(compact_path)

    def test_connect_to_tmp_path_is_allowed(self, tmp_path):
        """一時ディレクトリへの connect は通常どおり成功すること"""
        con = duckdb.connect(str(tmp_path / "ok.duckdb"))
        try:
            assert con.execute("SELECT 1").fetchone() == (1,)
        finally:
            con.close()

    def test_connect_in_memory_is_allowed(self):
        """インメモリ接続は許可されること"""
        con = duckdb.connect(":memory:")
        try:
            assert con.execute("SELECT 1").fetchone() == (1,)
        finally:
            con.close()
