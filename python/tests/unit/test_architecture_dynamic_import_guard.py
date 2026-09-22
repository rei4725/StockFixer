"""アーキテクチャガード: src/ 配下での自パッケージの動的 import を禁止する。

importlib.import_module("src....") は文字列ベースであるため import-linter が
検出できず、レイヤー契約・BC 独立性契約を迂回する抜け道になる。
実際に src/utils/db/__init__.py がこの手段で src.prediction.db を参照し、
utils(最下層) -> prediction(BC) の層逆転を隠していた。

GRANDFATHERED_DYNAMIC_IMPORTS は解消途中の既知違反のみを列挙する ratchet である。
項目を増やしてはならない。空になった時点でこの定数ごと削除してよい。
"""

import re
import unittest
from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parents[2] / "src"

# Phase 4b (PR-5) でプロキシ撤去とともに空になる予定。項目を追加しないこと。
GRANDFATHERED_DYNAMIC_IMPORTS = frozenset(
    {
        "utils/db/__init__.py",
    }
)

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
            rel = path.relative_to(_SRC_ROOT).as_posix()
            if rel in GRANDFATHERED_DYNAMIC_IMPORTS:
                continue
            if _PATTERN.search(path.read_text(encoding="utf-8")):
                offenders.append(rel)

        self.assertEqual(
            offenders,
            [],
            "src/ 配下で src.* の動的 import を検出した。"
            "レイヤー契約を迂回するため、静的 import かポート注入に置き換えること: "
            f"{offenders}",
        )

    def test_grandfathered_entries_still_violate(self):
        """許容リストの項目が実際にまだ違反していること（解消後の削除漏れ検出）。"""
        stale = []
        for rel in sorted(GRANDFATHERED_DYNAMIC_IMPORTS):
            path = _SRC_ROOT / rel
            if not path.exists() or not _PATTERN.search(path.read_text(encoding="utf-8")):
                stale.append(rel)

        self.assertEqual(
            stale,
            [],
            "許容リストに、もう違反していない項目が残っている。"
            f"GRANDFATHERED_DYNAMIC_IMPORTS から削除すること: {stale}",
        )
