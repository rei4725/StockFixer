"""quality BC の共有型。"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class CoverageTarget:
    """テスト穴埋めの対象ファイル1件。

    coverage JSON の1ファイル分エントリを正規化したもの。
    """

    path: str
    percent_covered: float
    missing_lines: list[int] = field(default_factory=list)

    @property
    def missing_count(self) -> int:
        return len(self.missing_lines)
