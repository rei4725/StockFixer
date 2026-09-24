"""アーキテクチャガード: bounded context から src.infrastructure への import を禁止する。

BC はポート（src/domain/ports.py）にのみ依存し、アダプタ（src/infrastructure/）は
合成ルート（run_*.py / orchestration / api）が注入する。

src.infrastructure は infrastructure -> market_data / reporting の逆参照があるため
.importlinter の layers 契約に載せられず、この違反は import-linter では検出できない（#741）。
ゆえに AST でここで検出する。関数内の遅延 import も対象とする。

GRANDFATHERED_BC_INFRA_IMPORTS は解消途中の既知違反のみを列挙する ratchet である。
項目を増やしてはならない。空になった時点でこの定数ごと削除してよい。
"""

import ast
import unittest
from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parents[2] / "src"

_BOUNDED_CONTEXTS = (
    "backtest",
    "market_data",
    "prediction",
    "quality",
    "reporting",
    "rule_engine",
    "screening",
    "trading",
    "watchlist",
)

# discord_bot.py は Discord コマンドのハンドラ自身が合成ルートを兼ねている。
# 項目を追加しないこと。
GRANDFATHERED_BC_INFRA_IMPORTS = frozenset(
    {
        "reporting/discord/discord_bot.py",
    }
)


def _imports_infrastructure(path: Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module == "src.infrastructure" or node.module.startswith("src.infrastructure."):
                return True
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "src.infrastructure" or alias.name.startswith(
                    "src.infrastructure."
                ):
                    return True
    return False


def _iter_bc_files():
    for bc in _BOUNDED_CONTEXTS:
        for path in sorted((_SRC_ROOT / bc).rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            yield path


class TestNoBoundedContextInfrastructureImport(unittest.TestCase):
    def test_no_new_bc_infrastructure_imports(self):
        """BC 配下で src.infrastructure を新規に import していないこと。"""
        offenders = []
        for path in _iter_bc_files():
            rel = path.relative_to(_SRC_ROOT).as_posix()
            if rel in GRANDFATHERED_BC_INFRA_IMPORTS:
                continue
            if _imports_infrastructure(path):
                offenders.append(rel)

        self.assertEqual(
            offenders,
            [],
            "BC から src.infrastructure への import を検出した。"
            "domain のポートを引数で受け取り、合成ルートで注入すること: "
            f"{offenders}",
        )

    def test_grandfathered_entries_still_violate(self):
        """許容リストの項目が実際にまだ違反していること（解消後の削除漏れ検出）。"""
        stale = []
        for rel in sorted(GRANDFATHERED_BC_INFRA_IMPORTS):
            path = _SRC_ROOT / rel
            if not path.exists() or not _imports_infrastructure(path):
                stale.append(rel)

        self.assertEqual(
            stale,
            [],
            "許容リストに、もう違反していない項目が残っている。"
            f"GRANDFATHERED_BC_INFRA_IMPORTS から削除すること: {stale}",
        )
