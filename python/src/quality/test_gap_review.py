"""テスト穴埋めボット（A-3）

coverage の JSON レポートから低カバレッジ・未カバー行を抽出し、その箇所の
ソース抜粋を Claude に渡して「不足している不足テストの追加案」を構造化出力で
得る。発見事項を factory-intake と同じスキーマの JSON として
results/factory/reports/ に書き出す。IssueAgent の --factory-intake が
GitHub Issue 草案として起票する（局所的・test 担保の auto-ok 適格な小粒を狙う）。

バックエンドは LLM_BACKEND で選択する（sdk=API 課金 / cli=サブスク認証）。
運用既定は CLI（CLAUDE_CODE_OAUTH_TOKEN）で ANTHROPIC_API_KEY は不要。
TEST_GAP_ENABLED=False（既定）/生成・解析失敗時は空リストを返し、何も
書き込まない（graceful degradation）。
テストコード本体は生成せず Issue 草案のみ（誤生成コードの混入を避ける）。
人手レビュー必須のため auto-ok ラベルは付与しない。

⚠️ CLI バックエンドはプロンプトを `claude -p "<prompt>"` の argv で渡すため、
ソースを丸ごと渡さず未カバー行＋前後数行の抜粋に絞り、
TEST_GAP_CONTEXT_CHAR_BUDGET でプロンプト全体を上限管理する
（Windows の CreateProcess コマンドライン上限 32,767 文字対策）。
"""

import hashlib
import json
import os
from typing import Any, Optional

from config.settings import (
    TEST_GAP_CONTEXT_CHAR_BUDGET,
    TEST_GAP_ENABLED,
    TEST_GAP_MAX_FILES,
    TEST_GAP_MAX_TOKENS,
    TEST_GAP_MIN_COVERAGE,
    TEST_GAP_MODEL,
)
from src.quality.types import CoverageTarget
from src.utils.data_path_utils import get_python_root, get_results_dir
from src.utils.logger import get_logger

logger = get_logger(__name__)

_REPORT_SCHEMA_VERSION = 1
_CONTEXT_WINDOW = 3  # 未カバー行の前後に含めるソース行数

_SYSTEM_PROMPT = (
    "あなたは Python のテストエンジニアです。"
    "与えられた未カバー（テスト未到達）のコード抜粋を読み、その分岐・例外・境界を"
    "実行する不足ユニットテストの追加案を提案してください。"
    "各対象ファイルについて、target_path（与えられたパスのまま）・"
    "uncovered_summary（未カバー箇所が何をする処理か）・"
    "test_outline（追加すべきテストケースの概要：入力・期待結果・モック対象を箇条書き）・"
    "priority（low/medium/high）を挙げてください。"
    "与えられた抜粋のみを根拠とし、存在しない関数や引数を捏造しないこと。"
    "テストコード本体は書かず、概要のみを示すこと。"
)

_SUGGESTIONS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "suggestions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "target_path": {"type": "string"},
                    "uncovered_summary": {"type": "string"},
                    "test_outline": {"type": "string"},
                    "priority": {"type": "string", "enum": ["low", "medium", "high"]},
                },
                "required": ["target_path", "uncovered_summary", "test_outline", "priority"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["suggestions"],
    "additionalProperties": False,
}

# priority → GitHub ラベル
_PRIORITY_LABELS = {"high": ["test", "P2"], "medium": ["test", "P3"], "low": ["test"]}


def _reports_dir() -> str:
    path = os.path.join(get_results_dir(), "factory", "reports")
    os.makedirs(path, exist_ok=True)
    return path


def _resolve_path(path: str) -> str:
    """coverage JSON 内の相対パスを python ルート基準の絶対パスへ解決する。"""
    if os.path.isabs(path):
        return path
    return os.path.join(get_python_root(), path)


def _load_coverage(coverage_json: str) -> dict:
    """coverage JSON を読み込む。"""
    path = coverage_json if os.path.exists(coverage_json) else _resolve_path(coverage_json)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _select_targets(data: dict) -> list[CoverageTarget]:
    """低カバレッジ・未カバー行を持つファイルを抽出し、未カバー行数の多い順に絞る。"""
    files = data.get("files", {})
    if not isinstance(files, dict):
        return []
    targets: list[CoverageTarget] = []
    for path, entry in files.items():
        if not isinstance(entry, dict):
            continue
        summary = entry.get("summary", {})
        missing = entry.get("missing_lines", [])
        if not isinstance(missing, list) or not missing:
            continue
        percent = float(summary.get("percent_covered", 100.0))
        if percent >= TEST_GAP_MIN_COVERAGE:
            continue
        targets.append(
            CoverageTarget(
                path=path,
                percent_covered=percent,
                missing_lines=[int(ln) for ln in missing],
            )
        )
    targets.sort(key=lambda t: t.missing_count, reverse=True)
    return targets[:TEST_GAP_MAX_FILES]


def _read_source_excerpt(target: CoverageTarget, window: int = _CONTEXT_WINDOW) -> str:
    """未カバー行＋前後 window 行の抜粋を生成する（>> 印が未カバー行）。"""
    abs_path = _resolve_path(target.path)
    try:
        with open(abs_path, encoding="utf-8") as f:
            lines = f.readlines()
    except OSError:
        return ""
    total = len(lines)
    missing_set = set(target.missing_lines)
    wanted: set[int] = set()
    for ln in target.missing_lines:
        for i in range(ln - window, ln + window + 1):
            if 1 <= i <= total:
                wanted.add(i)
    if not wanted:
        return ""
    out: list[str] = []
    prev: Optional[int] = None
    for i in sorted(wanted):
        if prev is not None and i != prev + 1:
            out.append("    ...")
        marker = ">>" if i in missing_set else "  "
        out.append(f"{marker}{i:5d}: {lines[i - 1].rstrip()}")
        prev = i
    return "\n".join(out)


def _gather_targets_context(targets: list[CoverageTarget]) -> str:
    """全対象を1つのプロンプトテキストに集約する（文字数バジェットで上限管理）。"""
    parts = [
        "以下は StockFixer のユニットテストで未カバーのコード抜粋です。",
        "各ファイルについて、未カバー行（>> 印）を実行する不足テストの追加案を提案してください。",
        "",
    ]
    used = sum(len(s) + 1 for s in parts)
    header_len = len(parts)
    for t in targets:
        excerpt = _read_source_excerpt(t)
        block = "\n".join(
            [
                f"## {t.path}（カバレッジ {t.percent_covered:.1f}%・未カバー {t.missing_count} 行）",
                "```python",
                excerpt or "（ソース抜粋を取得できませんでした）",
                "```",
                "",
            ]
        )
        # 1件目は必ず入れる。2件目以降は予算を超えたら打ち切る（重要度順ゆえ末尾が落ちる）。
        if len(parts) > header_len and used + len(block) > TEST_GAP_CONTEXT_CHAR_BUDGET:
            logger.info("[test_gap] 文字数バジェット到達。残りの対象を打ち切り")
            break
        parts.append(block)
        used += len(block) + 1
    return "\n".join(parts)


def _finding_hash(target_path: str) -> str:
    """提案の冪等キー（test-gap- 名前空間）。同一ファイルは同一 Issue に集約。"""
    digest = hashlib.sha256(target_path.encode("utf-8")).hexdigest()[:12]
    return f"test-gap-{digest}"


def _build_issue_body(suggestion: dict) -> str:
    return f"""## テスト穴埋め提案（Claude 自動生成）

- **優先度**: {suggestion.get('priority', 'low')}
- **対象**: `{suggestion.get('target_path', '')}`

### 未カバー箇所
{suggestion.get('uncovered_summary', '')}

### 追加すべきテストの概要
{suggestion.get('test_outline', '')}

---
*この Issue は StockFixer のテスト穴埋めボット（A-3）が Claude で自動生成した草案です。\
テスト内容の妥当性は人手で確認してください（auto-ok 非対象）。*
"""


def _write_suggestion_report(suggestion: dict) -> Optional[str]:
    """提案を factory-intake スキーマの JSON として書き出す。"""
    target_path = str(suggestion.get("target_path", "")).strip()
    priority = str(suggestion.get("priority", "low")).strip()
    if not target_path:
        return None

    h = _finding_hash(target_path)
    report = {
        "schema_version": _REPORT_SCHEMA_VERSION,
        "hypothesis_hash": h,
        "issue_title": f"[factory:{h}] [テスト穴埋め] {target_path}",
        "issue_body": _build_issue_body(suggestion),
        "labels": _PRIORITY_LABELS.get(priority, ["test"]),
    }
    path = os.path.join(_reports_dir(), f"{h}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return path


def _request_suggestions(context: str) -> list[dict]:
    """Claude に構造化出力でテスト追加案を要求する。"""
    from src.infrastructure.llm.factory import get_text_review_port  # noqa: PLC0415

    port = get_text_review_port()
    text = port.complete(
        system=_SYSTEM_PROMPT,
        user=context,
        model=TEST_GAP_MODEL,
        max_tokens=TEST_GAP_MAX_TOKENS,
        schema=_SUGGESTIONS_SCHEMA,
    )
    data = json.loads(text)
    suggestions = data.get("suggestions", [])
    return suggestions if isinstance(suggestions, list) else []


def run_test_gap_review(coverage_json: str = "coverage.json", dry_run: bool = False) -> list[dict]:
    """coverage の未カバー行を Claude でレビューし、テスト追加案を Issue 草案 JSON 化する。

    Args:
        coverage_json: coverage JSON レポートのパス（既定 coverage.json）。
        dry_run: True なら JSON を書き込まず、得た提案のみ返す。

    Returns:
        テスト追加案の dict リスト（無効・失敗・対象なし時は空リスト）。
    """
    if not TEST_GAP_ENABLED:
        logger.info("[test_gap] TEST_GAP_ENABLED=false のためスキップ")
        return []

    try:
        data = _load_coverage(coverage_json)
    except (OSError, json.JSONDecodeError):
        logger.error("[test_gap] coverage JSON の読み込みに失敗", exc_info=True)
        return []

    targets = _select_targets(data)
    if not targets:
        logger.info("[test_gap] 閾値未満の対象なし")
        return []

    context = _gather_targets_context(targets)
    try:
        suggestions = _request_suggestions(context)
    except Exception:
        logger.error("[test_gap] 提案生成でエラー", exc_info=True)
        return []

    if not suggestions:
        logger.info("[test_gap] 提案なし")
        return []

    if dry_run:
        logger.info("[test_gap] dry-run: %d 件の提案（書き込みなし）", len(suggestions))
        return suggestions

    written = 0
    for suggestion in suggestions:
        try:
            if _write_suggestion_report(suggestion):
                written += 1
        except Exception:
            logger.error(
                "[test_gap] レポート書き込み失敗: %s",
                suggestion.get("target_path"),
                exc_info=True,
            )
    logger.info("[test_gap] %d/%d 件のレポートを書き込み", written, len(suggestions))
    return suggestions
