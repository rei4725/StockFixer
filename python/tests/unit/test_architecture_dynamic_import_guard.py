"""アーキテクチャガード: src/ 配下での自パッケージの動的 import を禁止する。

importlib.import_module("src....") は文字列ベースであるため import-linter が
検出できず、レイヤー契約・BC 独立性契約を迂回する抜け道になる。
実際に src/utils/db/__init__.py がこの手段で src.prediction.db を参照し、
utils(最下層) -> prediction(BC) の層逆転を隠していた。

既知違反を許容する ratchet（GRANDFATHERED_DYNAMIC_IMPORTS）は、プロキシ撤去で
空になったため撤去した。例外を再導入してはならない。
"""

import re
import unittest
from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parents[2] / "src"

_PATTERN = re.compile(r"""import_module\(\s*["']src[.\"']""")


def _iter_python_files():
    for path in sorted(_SRC_ROOT.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        yield path


class TestNoDynamicSelfImport(unittest.TestCase):
    def test_no_new_dynamic_self_imports(self):
        """src/ 配下で importlib.import_module("src...") を新規に使っていないこと。"""
        offenders = []
        for path in _iter_python_files():
            if _PATTERN.search(path.read_text(encoding="utf-8")):
                offenders.append(path.relative_to(_SRC_ROOT).as_posix())

        self.assertEqual(
            offenders,
            [],
            "src/ 配下で src.* の動的 import を検出した。"
            "レイヤー契約を迂回するため、静的 import かポート注入に置き換えること: "
            f"{offenders}",
        )
