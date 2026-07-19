"""
戦略ファクトリー自動昇格ループ: GitHub REST API の薄いクライアント。

StockFixer 本体からの GitHub API 呼び出しはこのモジュールに限定する
（他モジュールが直接 requests で api.github.com を叩かない）。
"""

from __future__ import annotations

from datetime import datetime

import requests

from config.settings import GITHUB_REPO, GITHUB_TOKEN

_API_BASE = "https://api.github.com"
_TIMEOUT_SECONDS = 30


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def list_recently_merged_pull_requests(since: datetime) -> list[dict]:
    """直近更新の closed PR のうち、`since` 以降に更新されマージ済みのものを返す。"""
    params: dict[str, str | int] = {
        "state": "closed",
        "sort": "updated",
        "direction": "desc",
        "per_page": 50,
    }
    response = requests.get(
        f"{_API_BASE}/repos/{GITHUB_REPO}/pulls",
        headers=_headers(),
        params=params,
        timeout=_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    since_iso = since.isoformat()
    return [
        {
            "number": pr["number"],
            "body": pr.get("body") or "",
            "merge_commit_sha": pr["merge_commit_sha"],
        }
        for pr in response.json()
        if pr.get("merged_at") is not None and pr["updated_at"] >= since_iso
    ]


def get_issue(issue_number: int) -> dict:
    """指定 Issue のタイトル・ラベルを取得する。"""
    response = requests.get(
        f"{_API_BASE}/repos/{GITHUB_REPO}/issues/{issue_number}",
        headers=_headers(),
        timeout=_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    data = response.json()
    return {
        "number": data["number"],
        "title": data["title"],
        "labels": [label["name"] for label in data.get("labels", [])],
    }
