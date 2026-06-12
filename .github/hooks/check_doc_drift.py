#!/usr/bin/env python3
"""
ドキュメントドリフト（実在しないパス参照）を検出する静的チェックスクリプト

対象: CLAUDE.md / .claude/skills/**/SKILL.md
ドキュメント内の Markdown リンクとバッククォート参照からリポジトリ相対パスを抽出し、
ファイル・ディレクトリの実在を検証する。コード移動時にドキュメントが取り残される
「ドリフト」をコミット時点で検出する。

判定ルール:
- Markdown リンク [text](path): http(s)/mailto/アンカーのみを除き、md ファイル基準と
  リポジトリルート基準で解決を試みる
- バッククォート `path`: "/" を含み、拡張子を持つか "/" で終わるものだけをパスとみなす
  （ブランチ名・コマンド・glob は除外）
- `src/...` は python/src/... としても解決を試みる（ドキュメント慣行）
- runtime 生成物（results/ Logs/ data/ models/ 等）は対象外
"""

import re
import sys
from pathlib import Path
from typing import List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# runtime 生成物や環境依存パス（実在チェック対象外）
_SKIP_PREFIXES = (
    "results/",
    "logs/",
    "data/",
    "models/",
    "python/results/",
    "python/logs/",
    "python/data/",
    "python/models/",
)
_SKIP_EXACT = {".env", "optimal_params.json", "config/optimal_params.json"}

_MD_LINK_RE = re.compile(r"\[[^\]]*\]\(([^)\s#]+)(?:#[^)]*)?\)")
_BACKTICK_RE = re.compile(r"`([^`\n]+)`")
# パスらしさ: 空白・glob・テンプレート記号を含まない
_PATHISH_RE = re.compile(r"^[\w@.\-/\\]+$")


def _is_external(target: str) -> bool:
    return target.startswith(("http://", "https://", "mailto:", "#"))


def _looks_like_path(candidate: str) -> bool:
    """バッククォート内文字列がリポジトリパス参照らしいかを判定する。"""
    if not _PATHISH_RE.match(candidate):
        return False
    normalized = candidate.replace("\\", "/")
    if "/" not in normalized:
        return False
    # 拡張子を持つファイル参照、または "/" 終端のディレクトリ参照のみ対象
    last = normalized.rstrip("/").rsplit("/", maxsplit=1)[-1]
    if normalized.endswith("/"):
        return True
    return "." in last and not last.startswith(".")


def _is_skipped(rel: str) -> bool:
    normalized = rel.replace("\\", "/").lstrip("./").lower()
    if normalized in _SKIP_EXACT:
        return True
    return any(normalized.startswith(p) for p in _SKIP_PREFIXES)


def _resolves(candidate: str, doc_dir: Path) -> bool:
    """md / リポジトリルート / python/ / python/src/ のいずれかの基準で実在すれば True。

    python/src/ 基準は `backtest/` `prediction/types.py` のような BC 短縮表記
    （ドキュメント慣行）を解決するためのもの。
    """
    normalized = candidate.replace("\\", "/")
    bases = [doc_dir, REPO_ROOT, REPO_ROOT / "python", REPO_ROOT / "python" / "src"]
    for base in bases:
        try:
            if (base / normalized).exists():
                return True
        except OSError:
            continue
    return False


def check_file(filepath: str) -> List[Tuple[int, str]]:
    """ドキュメント内の解決不能なパス参照を返す。

    Returns:
        [(line_num, message), ...]
    """
    path = Path(filepath)
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception as e:  # noqa: BLE001
        return [(0, f"ファイル読込エラー: {e}")]

    doc_dir = path.resolve().parent
    problems: List[Tuple[int, str]] = []
    in_code_block = False

    for i, line in enumerate(lines, 1):
        # フェンスコードブロック内はコマンド例が多くノイズになるためスキップ
        if line.lstrip().startswith("```"):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            continue

        candidates: set[str] = set()
        for m in _MD_LINK_RE.finditer(line):
            target = m.group(1)
            if not _is_external(target) and _looks_like_path(target):
                candidates.add(target)
        for m in _BACKTICK_RE.finditer(line):
            inner = m.group(1).strip()
            if _looks_like_path(inner):
                candidates.add(inner)

        for candidate in candidates:
            if "*" in candidate or "{" in candidate:
                continue
            # プレースホルダ例（test_xxx.py 等）は実在チェック対象外
            if "xxx" in candidate.lower():
                continue
            if _is_skipped(candidate):
                continue
            if not _resolves(candidate, doc_dir):
                problems.append((i, f"[doc-drift] 参照先が存在しません: {candidate}"))

    return problems


def main() -> None:
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

    if len(sys.argv) < 2:
        print("使用方法: check_doc_drift.py <file1.md> [file2.md] ...")
        sys.exit(0)

    exit_code = 0
    total = 0
    for filepath in sys.argv[1:]:
        if not Path(filepath).exists():
            continue
        for line_num, message in check_file(filepath):
            print(f"{filepath}:{line_num}: {message}")
            total += 1
            exit_code = 1

    if total:
        print(
            f"\n[ERROR] ドキュメントドリフト {total} 件。参照先パスを現状に合わせて更新してください。"
        )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
