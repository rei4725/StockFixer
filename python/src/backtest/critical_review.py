"""
バックテスト批判的レビュー自動化（B-6）

Claude にバックテストの方法論・設定・最適化メトリクスを渡し、楽観バイアスや
方法論的欠陥（先読みリーク・survivorship・スリッページ過小評価・過学習等）を
検出させ、発見事項を factory-intake と同じスキーマの JSON として
results/factory/reports/ に書き出す。IssueAgent の --factory-intake が
GitHub Issue 草案として起票する（#492-497 を生んだ批判的レビューの自動化）。

BACKTEST_REVIEW_ENABLED=False（既定）/ANTHROPIC_API_KEY 未設定/anthropic 未導入/
API 失敗時は空リストを返し、何も書き込まない（graceful degradation）。
読み取り専用のレビューのみ（コード変更・発注判断には一切関与しない）。
人手レビュー必須のため auto-ok ラベルは付与しない。
"""

import hashlib
import json
import os
from typing import Any, Optional

from config.settings import (
    BACKTEST_REVIEW_ENABLED,
    BACKTEST_REVIEW_MAX_TOKENS,
    BACKTEST_REVIEW_MODEL,
)
from src.utils.data_path_utils import get_results_dir
from src.utils.logger import get_logger

logger = get_logger(__name__)

_REPORT_SCHEMA_VERSION = 1

_SYSTEM_PROMPT = (
    "あなたはクオンツトレーディングシステムの厳格なレビュアーです。"
    "与えられたバックテストの方法論・設定・最適化メトリクスを精査し、"
    "報告数値に楽観バイアスを生む金融的・方法論的欠陥を検出してください。"
    "重点観点: 先読み（look-ahead）リーク、survivorship bias、スリッページ/取引コストの過小評価、"
    "同バー約定、過学習（in-sample 最適化・過剰な好成績）、データスヌーピング、生存銘柄のみ評価。"
    "各欠陥について title（簡潔）・severity（low/medium/high）・category・"
    "rationale（なぜ問題か）・suggestion（是正案）を挙げてください。"
    "与えられた情報のみを根拠とし、推測の数値を作らないこと。"
    "実在しない欠陥を捏造せず、確証が薄いものは severity を下げること。"
)

_FINDINGS_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "severity": {"type": "string", "enum": ["low", "medium", "high"]},
                    "category": {"type": "string"},
                    "rationale": {"type": "string"},
                    "suggestion": {"type": "string"},
                },
                "required": ["title", "severity", "category", "rationale", "suggestion"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["findings"],
    "additionalProperties": False,
}

# severity → GitHub ラベル
_SEVERITY_LABELS = {"high": ["backtest", "P2"], "medium": ["backtest", "P3"], "low": ["backtest"]}


def _reports_dir() -> str:
    path = os.path.join(get_results_dir(), "factory", "reports")
    os.makedirs(path, exist_ok=True)
    return path


def _gather_config_snapshot() -> list[str]:
    """バックテスト関連の設定定数を収集する（名前に既知キーワードを含むもの）。"""
    from config import settings as cfg

    keywords = ("SLIPPAGE", "COST", "COMMISSION", "FEE", "THRESHOLD", "POSITION", "LOSS")
    lines: list[str] = []
    for name in sorted(dir(cfg)):
        if not name.isupper():
            continue
        if not any(kw in name for kw in keywords):
            continue
        value = getattr(cfg, name, None)
        if isinstance(value, (int, float, str, bool)):
            lines.append(f"- {name} = {value}")
    return lines


def _summarize_optimal_params() -> list[str]:
    """optimal_params.json から銘柄数とメトリクス分布を要約する（過学習の兆候検出用）。"""
    params_path = os.path.join(get_results_dir(), "optimal_params.json")
    if not os.path.exists(params_path):
        params_path = os.path.join(
            os.path.dirname(get_results_dir()), "config", "optimal_params.json"
        )
    if not os.path.exists(params_path):
        return ["optimal_params.json は見つかりませんでした。"]

    try:
        with open(params_path, encoding="utf-8") as f:
            params = json.load(f)
    except (OSError, json.JSONDecodeError):
        return ["optimal_params.json の読み込みに失敗しました。"]

    if not isinstance(params, dict) or not params:
        return ["optimal_params.json は空です。"]

    lines = [f"最適化済み銘柄数: {len(params)}"]
    sharpes: list[float] = []
    win_rates: list[float] = []
    for entry in params.values():
        metrics = entry.get("metrics", {}) if isinstance(entry, dict) else {}
        for key, bucket in (("sharpe", sharpes), ("win_rate", win_rates)):
            try:
                bucket.append(float(metrics[key]))
            except (KeyError, TypeError, ValueError):
                pass
    if sharpes:
        lines.append(
            f"Sharpe 分布: 最小={min(sharpes):.2f}, 平均={sum(sharpes) / len(sharpes):.2f}, "
            f"最大={max(sharpes):.2f}（n={len(sharpes)}）"
        )
    if win_rates:
        lines.append(
            f"勝率分布: 最小={min(win_rates):.2%}, 平均={sum(win_rates) / len(win_rates):.2%}, "
            f"最大={max(win_rates):.2%}（n={len(win_rates)}）"
        )
    return lines


def _gather_review_context() -> str:
    """Claude に渡すレビュー材料テキストを組み立てる。"""
    sections = [
        "## バックテストの方法論（概要）",
        "ML 予測（XGBoost/LightGBM）の diff_ratio が閾値を超えた銘柄を売買し、"
        "SL/TP・確信度連動の分割発注を行う。最適パラメータは銘柄別グリッドサーチで決定。",
        "",
        "## バックテスト関連の設定定数",
        *(_gather_config_snapshot() or ["（該当する設定が見つかりませんでした）"]),
        "",
        "## 最適化メトリクスの分布",
        *_summarize_optimal_params(),
    ]
    return "\n".join(sections)


def _finding_hash(category: str, title: str) -> str:
    """発見事項の冪等キー（review- 名前空間）。同一カテゴリ・タイトルは同一 Issue に集約。"""
    digest = hashlib.sha256(f"{category}|{title}".encode("utf-8")).hexdigest()[:12]
    return f"review-{digest}"


def _build_issue_body(finding: dict) -> str:
    return f"""## バックテスト批判的レビュー（Claude 自動生成）

- **重大度**: {finding['severity']}
- **カテゴリ**: {finding['category']}

### 指摘
{finding['rationale']}

### 是正案
{finding['suggestion']}

---
*この Issue は StockFixer のバックテスト批判的レビュー（B-6）が Claude で自動生成した\
草案です。妥当性は人手で確認してください（auto-ok 非対象）。*
"""


def _write_finding_report(finding: dict) -> Optional[str]:
    """発見事項を factory-intake スキーマの JSON として書き出す。"""
    title = str(finding.get("title", "")).strip()
    category = str(finding.get("category", "")).strip()
    severity = str(finding.get("severity", "low")).strip()
    if not title:
        return None

    h = _finding_hash(category, title)
    report = {
        "schema_version": _REPORT_SCHEMA_VERSION,
        "hypothesis_hash": h,
        "issue_title": f"[factory:{h}] [BTレビュー] {title}",
        "issue_body": _build_issue_body(finding),
        "labels": _SEVERITY_LABELS.get(severity, ["backtest"]),
    }
    path = os.path.join(_reports_dir(), f"{h}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return path


def _request_findings(context: str) -> list[dict]:
    """Claude に構造化出力でレビュー結果を要求する。"""
    import anthropic  # noqa: PLC0415  遅延 import（未インストール環境を許容）

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=BACKTEST_REVIEW_MODEL,
        max_tokens=BACKTEST_REVIEW_MAX_TOKENS,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": context}],
        output_config={"format": {"type": "json_schema", "schema": _FINDINGS_SCHEMA}},
    )
    text = next((b.text for b in response.content if b.type == "text"), "")
    data = json.loads(text)
    findings = data.get("findings", [])
    return findings if isinstance(findings, list) else []


def run_backtest_review(dry_run: bool = False) -> list[dict]:
    """バックテストを Claude で批判的レビューし、発見事項を Issue 草案 JSON 化する。

    Args:
        dry_run: True なら JSON を書き込まず、検出した発見事項のみ返す。

    Returns:
        検出した発見事項の dict リスト（無効・失敗時は空リスト）。
    """
    if not BACKTEST_REVIEW_ENABLED:
        logger.info("[bt_review] BACKTEST_REVIEW_ENABLED=false のためスキップ")
        return []

    try:
        import anthropic  # noqa: F401, PLC0415  存在確認
    except ImportError:
        logger.warning("[bt_review] anthropic 未インストールのためスキップ")
        return []

    context = _gather_review_context()
    try:
        findings = _request_findings(context)
    except Exception:
        logger.error("[bt_review] レビュー生成でエラー", exc_info=True)
        return []

    if not findings:
        logger.info("[bt_review] 発見事項なし")
        return []

    if dry_run:
        logger.info("[bt_review] dry-run: %d 件の発見事項（書き込みなし）", len(findings))
        return findings

    written = 0
    for finding in findings:
        try:
            if _write_finding_report(finding):
                written += 1
        except Exception:
            logger.error(
                "[bt_review] レポート書き込み失敗: %s", finding.get("title"), exc_info=True
            )
    logger.info("[bt_review] %d/%d 件のレポートを書き込み", written, len(findings))
    return findings
