"""VERSION ファイルの読み取りが BOM に汚染されないことのテスト。

PowerShell 5.1 の `Set-Content -Encoding utf8` は BOM 付きで書くため VERSION に
BOM が混入しうる。PowerShell 側（auto_deploy.ps1 の Get-Content）は BOM を剥がすので
デプロイは通ってしまい、Python 側だけが "﻿2.10.0" をレポートの産地情報に
書き込む——という気づきにくい壊れ方をする。
"""

from __future__ import annotations

import unittest
import unittest.mock

from src.backtest import factory_report


class TestImageVersion(unittest.TestCase):
    def _version_from(self, raw: bytes, tmp) -> str | None:
        path = tmp / "VERSION"
        path.write_bytes(raw)
        with unittest.mock.patch.object(factory_report.os.path, "join", return_value=str(path)):
            return factory_report._image_version()

    def test_strips_utf8_bom(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(self._version_from(b"\xef\xbb\xbf2.10.1\n", Path(d)), "2.10.1")

    def test_reads_plain_utf8(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(self._version_from(b"2.10.1\n", Path(d)), "2.10.1")

    def test_repository_version_file_has_no_bom(self):
        """リポジトリ実体の VERSION に BOM が入っていないことを固定する。"""
        import os

        python_root = os.path.dirname(os.path.dirname(os.path.dirname(factory_report.__file__)))
        with open(os.path.join(python_root, "VERSION"), "rb") as f:
            self.assertFalse(f.read(3).startswith(b"\xef\xbb\xbf"))


if __name__ == "__main__":
    unittest.main()
