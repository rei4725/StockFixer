"""
戦略ファクトリー自動昇格ループ: GitHub PR/Issue から昇格対象を特定する純粋関数群。

ネットワーク I/O は行わない。PR 本文・Issue タイトルの文字列解析と、
既存のローカル JSON レポート（results/factory/reports/<hash>.json）の読み込みのみ。
"""

from __future__ import annotations

import json
import os
import re
from typing import Optional

from src.utils.data_path_utils import get_results_dir

_CLOSING_KEYWORDS = r"close[sd]?|fix(?:e[sd])?|resolve[sd]?"
_CLOSING_PATTERN = re.compile(rf"\b(?:{_CLOSING_KEYWORDS})\s+#(\d+)", re.IGNORECASE)
_FACTORY_HASH_PATTERN = re.compile(r"\[factory:([0-9a-f]+)\]")


def extract_closing_issue_numbers(pr_body: Optional[str]) -> list[int]:
    """PR本文の GitHub クローズキーワード（Closes/Fixes/Resolves #N）から Issue 番号を抽出する。"""
    if not pr_body:
        return []
    return [int(n) for n in _CLOSING_PATTERN.findall(pr_body)]


def extract_factory_hash(issue_title: Optional[str]) -> Optional[str]:
    """Issue タイトルの `[factory:<hash>]` マーカーから仮説ハッシュを抽出する。"""
    if not issue_title:
        return None
    m = _FACTORY_HASH_PATTERN.search(issue_title)
    return m.group(1) if m else None


def load_gate_baseline(hypothesis_hash: str) -> Optional[float]:
    """既存の不変レポート（results/factory/reports/<hash>.json）から
    ゲート判定時のチャンピオン Sharpe を読み取る。昇格直前のベースラインとして使う。
    """
    path = os.path.join(get_results_dir(), "factory", "reports", f"{hypothesis_hash}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        report = json.load(f)
    return report.get("gate", {}).get("champion_sharpe")
