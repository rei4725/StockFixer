# 戦略ファクトリー: 昇格記録基盤 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** マージ済みの戦略ファクトリー由来 PR を検出し、`strategy_promotions` DuckDB テーブルに記録する。これは後続の「CIガード」計画・「ロールバック監視」計画・「LLMアイデア発想」計画すべてが土台にする、挙動変化のない記録基盤。

**Architecture:** 2時間ごとのオーケストレーションジョブが GitHub REST API で直近マージ済み PR を取得し、PR 本文の `Closes #N` 記法から連携 Issue を特定、`strategy-factory`/`strategy-factory-idea` ラベルと `[factory:<hash>]` タイトルマーカーで戦略ファクトリー由来と確認し、既存のローカル JSON レポート（`results/factory/reports/<hash>.json`）から昇格直前のベースライン Sharpe を読み取って DuckDB に記録する。

**Tech Stack:** Python, DuckDB, requests（GitHub REST API 呼び出し）, APScheduler（既存オーケストレーション基盤）, unittest + unittest.mock。

## Global Constraints

- 本計画の実装時点では、どの PR にも `auto-ok` ラベルは付与されていない（LLMアイデア発想計画は未実装）。したがって本ジョブが検出するのは、既に存在する `strategy-factory` ラベル付き Issue を**人間が手動でレビューし実装・マージした** PR のみ。挙動を変える機能ではなく、あくまで記録基盤。
- 新規モジュールはすべて `FeatureName_ENABLED` 相当の kill-switch 設定を持ち、既定値 `False`（安全側）。
- `.env` ファイル自体の作成・編集はこの計画のスコープ外（禁止ファイル）。`GITHUB_TOKEN` の実際の値設定はユーザー自身が行う。
- 既存コードのレイヤリング（`run_*.py` は CLI ラッパーのみ、BC は `orchestration/`・`api/` を import しない）を破らない。

---

### Task 1: `strategy_promotions` DuckDB テーブルと CRUD 関数

**Files:**
- Create: `python/src/utils/db/strategy_promotions.py`
- Test: `python/tests/unit/test_db_strategy_promotions.py`

**Interfaces:**
- Produces:
  - `ensure_strategy_promotions_table() -> None`
  - `save_strategy_promotion(pr_number: int, merge_commit_hash: str, rule_or_feature_id: str, pre_promotion_baseline: float, promoted_at: Optional[datetime] = None) -> None`
  - `promotion_exists(pr_number: int) -> bool`
  - `load_active_promotions() -> pd.DataFrame`
  - `mark_promotion_rolled_back(pr_number: int) -> None`

- [ ] **Step 1: Write the failing test**

`python/tests/unit/test_db_strategy_promotions.py` を新規作成する。既存の `python/tests/unit/test_db_experiment.py` と同じ `_TmpDbTestCase`（一時 DuckDB ファイルへの向け替え）パターンを使う。

```python
import os
import tempfile
import unittest
from datetime import datetime

import src.utils.data_path_utils as path_utils
import src.utils.db as db_module
from src.utils.db.strategy_promotions import (
    load_active_promotions,
    mark_promotion_rolled_back,
    promotion_exists,
    save_strategy_promotion,
)


class _TmpDbTestCase(unittest.TestCase):
    def setUp(self):
        db_module.close_connection()
        self.tmp_dir = tempfile.mkdtemp()
        self.tmp_db = os.path.join(self.tmp_dir, "test.duckdb")
        self._orig_get_db_path = path_utils.get_db_path
        path_utils.get_db_path = lambda: self.tmp_db
        db_module.get_db_path = lambda: self.tmp_db
        db_module._tables_initialized = False

    def tearDown(self):
        db_module.close_connection()
        path_utils.get_db_path = self._orig_get_db_path
        db_module.get_db_path = self._orig_get_db_path


class TestStrategyPromotionsDb(_TmpDbTestCase):
    def test_save_and_load_roundtrip(self):
        save_strategy_promotion(
            pr_number=101,
            merge_commit_hash="abc123",
            rule_or_feature_id="fb44f0011174",
            pre_promotion_baseline=1.25,
        )
        df = load_active_promotions()
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["pr_number"], 101)
        self.assertEqual(df.iloc[0]["merge_commit_hash"], "abc123")
        self.assertEqual(df.iloc[0]["rule_or_feature_id"], "fb44f0011174")
        self.assertAlmostEqual(float(df.iloc[0]["pre_promotion_baseline"]), 1.25)
        self.assertEqual(df.iloc[0]["status"], "active")

    def test_promotion_exists(self):
        self.assertFalse(promotion_exists(202))
        save_strategy_promotion(
            pr_number=202,
            merge_commit_hash="def456",
            rule_or_feature_id="hash2",
            pre_promotion_baseline=0.9,
        )
        self.assertTrue(promotion_exists(202))

    def test_mark_rolled_back_excludes_from_active(self):
        save_strategy_promotion(
            pr_number=303,
            merge_commit_hash="ghi789",
            rule_or_feature_id="hash3",
            pre_promotion_baseline=1.1,
        )
        mark_promotion_rolled_back(303)
        df = load_active_promotions()
        self.assertEqual(len(df), 0)

    def test_promoted_at_defaults_to_now(self):
        before = datetime.now()
        save_strategy_promotion(
            pr_number=404,
            merge_commit_hash="jkl012",
            rule_or_feature_id="hash4",
            pre_promotion_baseline=1.0,
        )
        df = load_active_promotions()
        promoted_at = df.iloc[0]["promoted_at"]
        self.assertGreaterEqual(promoted_at, before)

    def test_duplicate_pr_number_is_replaced_not_duplicated(self):
        save_strategy_promotion(
            pr_number=505,
            merge_commit_hash="first",
            rule_or_feature_id="hash5",
            pre_promotion_baseline=1.0,
        )
        save_strategy_promotion(
            pr_number=505,
            merge_commit_hash="second",
            rule_or_feature_id="hash5",
            pre_promotion_baseline=1.0,
        )
        df = load_active_promotions()
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["merge_commit_hash"], "second")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd python && python -m pytest tests/unit/test_db_strategy_promotions.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.utils.db.strategy_promotions'`

- [ ] **Step 3: Write minimal implementation**

`python/src/utils/db/strategy_promotions.py` を新規作成する（`python/src/utils/db/factory_runs.py` と同じ自己完結パターン: モジュール冒頭の DDL 定数 + `ensure_*_table()` を各関数が自前で呼ぶ）。

```python
"""
戦略ファクトリー自動昇格ループ: 昇格記録テーブル操作

テーブル:
    strategy_promotions … マージされた戦略ファクトリー由来 PR の昇格台帳。
    ロールバック監視ジョブが実績を追跡する際の対象一覧としても使う。
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import pandas as pd

from src.utils.db._connection import _db_connection
from src.utils.logger import get_logger

logger = get_logger(__name__)

_DDL_STRATEGY_PROMOTIONS = """
CREATE TABLE IF NOT EXISTS strategy_promotions (
    pr_number               INTEGER   NOT NULL,
    merge_commit_hash       VARCHAR   NOT NULL,
    rule_or_feature_id      VARCHAR   NOT NULL,
    promoted_at              TIMESTAMP NOT NULL,
    pre_promotion_baseline  DOUBLE    NOT NULL,
    status                  VARCHAR   NOT NULL DEFAULT 'active',
    PRIMARY KEY (pr_number)
)
"""


def ensure_strategy_promotions_table() -> None:
    with _db_connection() as con:
        con.execute(_DDL_STRATEGY_PROMOTIONS)


def save_strategy_promotion(
    pr_number: int,
    merge_commit_hash: str,
    rule_or_feature_id: str,
    pre_promotion_baseline: float,
    promoted_at: Optional[datetime] = None,
) -> None:
    """マージ検出した戦略昇格を記録する（同一 pr_number は置換）。"""
    ensure_strategy_promotions_table()
    if promoted_at is None:
        promoted_at = datetime.now()
    with _db_connection() as con:
        con.execute(
            """
            INSERT OR REPLACE INTO strategy_promotions (
                pr_number, merge_commit_hash, rule_or_feature_id,
                promoted_at, pre_promotion_baseline, status
            )
            VALUES (?, ?, ?, ?, ?, 'active')
            """,
            [pr_number, merge_commit_hash, rule_or_feature_id, promoted_at, pre_promotion_baseline],
        )
    logger.info(
        "戦略昇格記録: pr=%s hash=%s baseline=%.3f",
        pr_number,
        rule_or_feature_id,
        pre_promotion_baseline,
    )


def promotion_exists(pr_number: int) -> bool:
    """指定 PR がすでに記録済みかどうかを返す（マージ検知ジョブの重複防止用）。"""
    ensure_strategy_promotions_table()
    with _db_connection() as con:
        row = con.execute(
            "SELECT 1 FROM strategy_promotions WHERE pr_number = ?", [pr_number]
        ).fetchone()
    return row is not None


def load_active_promotions() -> pd.DataFrame:
    """status='active' の昇格レコードを返す（ロールバック監視ジョブの対象一覧）。"""
    ensure_strategy_promotions_table()
    with _db_connection() as con:
        try:
            return con.execute(
                "SELECT * FROM strategy_promotions WHERE status = 'active' ORDER BY promoted_at DESC"
            ).fetchdf()
        except Exception as e:
            logger.error("strategy_promotions 読み込み失敗: %s", e, exc_info=True)
            return pd.DataFrame()


def mark_promotion_rolled_back(pr_number: int) -> None:
    """指定 PR の昇格をロールバック済みとしてマークする。"""
    ensure_strategy_promotions_table()
    with _db_connection() as con:
        con.execute(
            "UPDATE strategy_promotions SET status = 'rolled_back' WHERE pr_number = ?",
            [pr_number],
        )
    logger.info("戦略昇格をロールバック済みにマーク: pr=%s", pr_number)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd python && python -m pytest tests/unit/test_db_strategy_promotions.py -v`
Expected: PASS（5 tests）

- [ ] **Step 5: Commit**

```bash
git add python/src/utils/db/strategy_promotions.py python/tests/unit/test_db_strategy_promotions.py
git commit -m "feat: strategy_promotionsテーブルとCRUD関数を追加"
```

---

### Task 2: 識別子抽出とベースライン読み込み（純粋関数）

**Files:**
- Create: `python/src/backtest/promotion_detection.py`
- Test: `python/tests/unit/test_promotion_detection.py`

**Interfaces:**
- Consumes: なし（純粋関数、ファイルシステムの `results/factory/reports/<hash>.json` のみ読む）
- Produces:
  - `extract_closing_issue_numbers(pr_body: str) -> list[int]`
  - `extract_factory_hash(issue_title: str) -> Optional[str]`
  - `load_gate_baseline(hypothesis_hash: str) -> Optional[float]`

これらは Task 5 のマージ検知ジョブが GitHub API から取得した PR 本文・Issue タイトルを解釈するために使う。`extract_factory_hash` が読み取る `[factory:<hash>]` マーカーは `python/src/backtest/factory_report.py:129` の `issue_title = f"[factory:{h.hypothesis_hash}] {h.label} ({h.market})"` で既に生成されている既存の規約。

- [ ] **Step 1: Write the failing test**

```python
import json
import os
import tempfile
import unittest
from unittest.mock import patch

from src.backtest.promotion_detection import (
    extract_closing_issue_numbers,
    extract_factory_hash,
    load_gate_baseline,
)


class TestExtractClosingIssueNumbers(unittest.TestCase):
    def test_closes_keyword(self):
        self.assertEqual(extract_closing_issue_numbers("Closes #564"), [564])

    def test_fixes_keyword_case_insensitive(self):
        self.assertEqual(extract_closing_issue_numbers("fixes #12"), [12])

    def test_resolved_keyword(self):
        self.assertEqual(extract_closing_issue_numbers("This resolved #99 finally"), [99])

    def test_multiple_keywords(self):
        self.assertEqual(
            sorted(extract_closing_issue_numbers("Closes #1\n\nAlso fixes #2")), [1, 2]
        )

    def test_no_keyword_returns_empty(self):
        self.assertEqual(extract_closing_issue_numbers("See #564 for context"), [])

    def test_empty_body_returns_empty(self):
        self.assertEqual(extract_closing_issue_numbers(""), [])
        self.assertEqual(extract_closing_issue_numbers(None), [])


class TestExtractFactoryHash(unittest.TestCase):
    def test_extracts_hash_from_marker(self):
        title = "[factory:fb44f0011174] AND合成ルール (jp)"
        self.assertEqual(extract_factory_hash(title), "fb44f0011174")

    def test_no_marker_returns_none(self):
        self.assertIsNone(extract_factory_hash("普通のタイトル"))

    def test_empty_title_returns_none(self):
        self.assertIsNone(extract_factory_hash(""))
        self.assertIsNone(extract_factory_hash(None))


class TestLoadGateBaseline(unittest.TestCase):
    def test_reads_champion_sharpe_from_report(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            reports_dir = os.path.join(tmp_dir, "factory", "reports")
            os.makedirs(reports_dir)
            report_path = os.path.join(reports_dir, "abc123.json")
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump({"gate": {"champion_sharpe": 1.42}}, f)

            with patch(
                "src.backtest.promotion_detection.get_results_dir", return_value=tmp_dir
            ):
                self.assertAlmostEqual(load_gate_baseline("abc123"), 1.42)

    def test_missing_report_returns_none(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            with patch(
                "src.backtest.promotion_detection.get_results_dir", return_value=tmp_dir
            ):
                self.assertIsNone(load_gate_baseline("does-not-exist"))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd python && python -m pytest tests/unit/test_promotion_detection.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.backtest.promotion_detection'`

- [ ] **Step 3: Write minimal implementation**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd python && python -m pytest tests/unit/test_promotion_detection.py -v`
Expected: PASS（11 tests）

- [ ] **Step 5: Commit**

```bash
git add python/src/backtest/promotion_detection.py python/tests/unit/test_promotion_detection.py
git commit -m "feat: PR/Issueから戦略ファクトリー識別子を抽出する純粋関数を追加"
```

---

### Task 3: GitHub REST API クライアント

**Files:**
- Create: `python/src/utils/github_api.py`
- Modify: `python/config/settings.py:99`（`FACTORY_HYPOTHESIS_REVIEW_MAX_TOKENS` の直後に追加）
- Modify: `python/config/settings.py:218`（フラット化ブロックの直後に追加）
- Test: `python/tests/unit/test_github_api.py`

**Interfaces:**
- Consumes: `config.settings.GITHUB_TOKEN`, `config.settings.GITHUB_REPO`
- Produces:
  - `list_recently_merged_pull_requests(since: datetime) -> list[dict]`（各 dict は `{"number": int, "body": str, "merge_commit_sha": str}`）
  - `get_issue(issue_number: int) -> dict`（`{"number": int, "title": str, "labels": list[str]}`）

- [ ] **Step 1: 設定を追加**

`python/config/settings.py:99` の直後（`FACTORY_HYPOTHESIS_REVIEW_MAX_TOKENS: int = Field(default=2048)` の次の行）に追加:

```python
    GITHUB_TOKEN: str = Field(default="")
    GITHUB_REPO: str = Field(default="rei4725/StockFixer")
```

`python/config/settings.py:218` の直後（フラット化ブロックの `FACTORY_HYPOTHESIS_REVIEW_MAX_TOKENS` 行の次）に追加:

```python
GITHUB_TOKEN: str = settings.GITHUB_TOKEN
GITHUB_REPO: str = settings.GITHUB_REPO
```

- [ ] **Step 2: Write the failing test**

```python
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from src.utils.github_api import get_issue, list_recently_merged_pull_requests


class TestListRecentlyMergedPullRequests:
    @patch("src.utils.github_api.requests.get")
    def test_returns_only_merged_prs(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {
                "number": 101,
                "body": "Closes #564",
                "merge_commit_sha": "abc123",
                "merged_at": "2026-07-19T01:00:00Z",
                "updated_at": "2026-07-19T01:00:00Z",
            },
            {
                "number": 102,
                "body": "WIP",
                "merge_commit_sha": None,
                "merged_at": None,
                "updated_at": "2026-07-19T01:00:00Z",
            },
        ]
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = list_recently_merged_pull_requests(since=datetime(2026, 7, 1))

        assert len(result) == 1
        assert result[0] == {"number": 101, "body": "Closes #564", "merge_commit_sha": "abc123"}

    @patch("src.utils.github_api.requests.get")
    def test_filters_by_since(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {
                "number": 201,
                "body": "Closes #1",
                "merge_commit_sha": "old",
                "merged_at": "2026-01-01T00:00:00Z",
                "updated_at": "2026-01-01T00:00:00Z",
            },
        ]
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = list_recently_merged_pull_requests(since=datetime(2026, 7, 1))

        assert result == []

    @patch("src.utils.github_api.requests.get")
    def test_raises_on_http_error(self, mock_get):
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = Exception("HTTP 401")
        mock_get.return_value = mock_response

        with pytest.raises(Exception):
            list_recently_merged_pull_requests(since=datetime(2026, 7, 1))


class TestGetIssue:
    @patch("src.utils.github_api.requests.get")
    def test_returns_number_title_labels(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "number": 564,
            "title": "[factory:fb44f0011174] AND合成ルール (jp)",
            "labels": [{"name": "strategy-factory"}, {"name": "auto-ok"}],
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        result = get_issue(564)

        assert result == {
            "number": 564,
            "title": "[factory:fb44f0011174] AND合成ルール (jp)",
            "labels": ["strategy-factory", "auto-ok"],
        }
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd python && python -m pytest tests/unit/test_github_api.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.utils.github_api'`

- [ ] **Step 4: Write minimal implementation**

```python
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
    response = requests.get(
        f"{_API_BASE}/repos/{GITHUB_REPO}/pulls",
        headers=_headers(),
        params={"state": "closed", "sort": "updated", "direction": "desc", "per_page": 50},
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
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd python && python -m pytest tests/unit/test_github_api.py -v`
Expected: PASS（4 tests）

- [ ] **Step 6: Commit**

```bash
git add python/config/settings.py python/src/utils/github_api.py python/tests/unit/test_github_api.py
git commit -m "feat: GitHub REST APIクライアントを追加（マージ検知ジョブ用）"
```

---

### Task 4: Discord 通知関数

**Files:**
- Modify: `python/src/reporting/discord/notifications_model.py`
- Modify: `python/src/reporting/discord/discord_utils.py:35-43`（re-export ブロック）
- Test: `python/tests/unit/test_notifications_model_promotion.py`

**Interfaces:**
- Consumes: `src.reporting.discord.webhook_sender.send_webhook_text_chunked`
- Produces: `send_strategy_promotion_detected(pr_number: int, rule_or_feature_id: str, pre_promotion_baseline: float) -> bool`

- [ ] **Step 1: Write the failing test**

```python
from unittest.mock import patch

from src.reporting.discord.notifications_model import send_strategy_promotion_detected


class TestSendStrategyPromotionDetected:
    @patch("src.reporting.discord.notifications_model.send_webhook_text_chunked")
    def test_sends_message_with_pr_and_hash(self, mock_send):
        mock_send.return_value = True

        result = send_strategy_promotion_detected(
            pr_number=564, rule_or_feature_id="fb44f0011174", pre_promotion_baseline=1.25
        )

        assert result is True
        mock_send.assert_called_once()
        message = mock_send.call_args[0][0]
        assert "564" in message
        assert "fb44f0011174" in message
        assert "1.25" in message

    @patch("src.reporting.discord.notifications_model.send_webhook_text_chunked")
    def test_returns_false_on_send_failure(self, mock_send):
        mock_send.return_value = False

        result = send_strategy_promotion_detected(
            pr_number=1, rule_or_feature_id="hash", pre_promotion_baseline=0.5
        )

        assert result is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd python && python -m pytest tests/unit/test_notifications_model_promotion.py -v`
Expected: FAIL with `ImportError: cannot import name 'send_strategy_promotion_detected'`

- [ ] **Step 3: Write minimal implementation**

`python/src/reporting/discord/notifications_model.py` の末尾（`send_factory_completion` 関数の後）に追加:

```python
def send_strategy_promotion_detected(
    pr_number: int, rule_or_feature_id: str, pre_promotion_baseline: float
) -> bool:
    """
    戦略ファクトリー由来 PR のマージ検出（＝昇格）を Discord Webhook に通知する。

    Args:
        pr_number: マージされた PR 番号
        rule_or_feature_id: 対象仮説/アイデアの識別子（factory hash）
        pre_promotion_baseline: 昇格直前のチャンピオン Sharpe

    Returns:
        成功時 True、失敗時 False
    """
    lines = [
        "**🏭⬆️ 戦略ファクトリー: 新規昇格を検出**",
        f"時刻: {format_jst(fmt=DISCORD_DATETIME_FORMAT)}",
        f"PR: #{pr_number}",
        f"識別子: `{rule_or_feature_id}`",
        f"昇格直前ベースライン Sharpe: {pre_promotion_baseline:.3f}",
    ]
    return send_webhook_text_chunked("\n".join(lines))
```

`python/src/reporting/discord/discord_utils.py:35-43` の re-export ブロックに `send_strategy_promotion_detected` を追加（アルファベット順を維持）:

```python
from src.reporting.discord.notifications_model import (  # noqa: F401  # re-export（#497 第4弾）
    send_factory_completion,
    send_feature_suggestion_notification,
    send_optimization_completion,
    send_promotion_result,
    send_shadow_evaluation_notification,
    send_shap_batch_summary,
    send_shap_notification,
    send_strategy_promotion_detected,
)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd python && python -m pytest tests/unit/test_notifications_model_promotion.py -v`
Expected: PASS（2 tests）

- [ ] **Step 5: Commit**

```bash
git add python/src/reporting/discord/notifications_model.py python/src/reporting/discord/discord_utils.py python/tests/unit/test_notifications_model_promotion.py
git commit -m "feat: 戦略昇格検出のDiscord通知関数を追加"
```

---

### Task 5: マージ検知ジョブ本体

**Files:**
- Modify: `python/src/orchestration/jobs/periodic.py`
- Modify: `python/config/settings.py:99`（Task 3 で追加した設定の直後）
- Modify: `python/config/settings.py:218`（同上フラット化ブロック）
- Test: `python/tests/unit/test_strategy_promotion_check.py`

**Interfaces:**
- Consumes:
  - `src.backtest.promotion_detection.extract_closing_issue_numbers`
  - `src.backtest.promotion_detection.extract_factory_hash`
  - `src.backtest.promotion_detection.load_gate_baseline`
  - `src.utils.db.strategy_promotions.promotion_exists`
  - `src.utils.db.strategy_promotions.save_strategy_promotion`
  - `src.utils.github_api.list_recently_merged_pull_requests`
  - `src.utils.github_api.get_issue`
  - `src.reporting.discord.discord_utils.send_strategy_promotion_detected`
- Produces: `run_strategy_promotion_check(force: bool = False) -> None`

- [ ] **Step 1: 設定を追加**

`python/config/settings.py:99` の直後（Task 3 で追加した `GITHUB_REPO` 行の次）に追加:

```python
    STRATEGY_PROMOTION_CHECK_ENABLED: bool = Field(default=False)
```

フラット化ブロック（`python/config/settings.py:218` 付近、Task 3 で追加した `GITHUB_REPO` 行の次）に追加:

```python
STRATEGY_PROMOTION_CHECK_ENABLED: bool = settings.STRATEGY_PROMOTION_CHECK_ENABLED
```

- [ ] **Step 2: Write the failing test**

```python
from datetime import datetime
from unittest.mock import patch

from src.orchestration.jobs.periodic import run_strategy_promotion_check


class TestRunStrategyPromotionCheck:
    @patch("config.settings.STRATEGY_PROMOTION_CHECK_ENABLED", False)
    @patch("src.utils.github_api.list_recently_merged_pull_requests")
    def test_skips_when_disabled_and_not_forced(self, mock_list):
        run_strategy_promotion_check(force=False)
        mock_list.assert_not_called()

    @patch("src.reporting.discord.discord_utils.send_strategy_promotion_detected")
    @patch("src.utils.db.strategy_promotions.save_strategy_promotion")
    @patch("src.utils.db.strategy_promotions.promotion_exists", return_value=False)
    @patch("src.backtest.promotion_detection.load_gate_baseline", return_value=1.25)
    @patch("src.utils.github_api.get_issue")
    @patch("src.utils.github_api.list_recently_merged_pull_requests")
    def test_detects_and_records_factory_pr(
        self, mock_list, mock_get_issue, mock_baseline, mock_exists, mock_save, mock_notify
    ):
        mock_list.return_value = [
            {"number": 564, "body": "Closes #999", "merge_commit_sha": "abc123"}
        ]
        mock_get_issue.return_value = {
            "number": 999,
            "title": "[factory:fb44f0011174] AND合成ルール (jp)",
            "labels": ["strategy-factory"],
        }

        run_strategy_promotion_check(force=True)

        mock_save.assert_called_once_with(
            pr_number=564,
            merge_commit_hash="abc123",
            rule_or_feature_id="fb44f0011174",
            pre_promotion_baseline=1.25,
        )
        mock_notify.assert_called_once_with(
            pr_number=564, rule_or_feature_id="fb44f0011174", pre_promotion_baseline=1.25
        )

    @patch("src.utils.db.strategy_promotions.save_strategy_promotion")
    @patch("src.utils.db.strategy_promotions.promotion_exists", return_value=False)
    @patch("src.utils.github_api.get_issue")
    @patch("src.utils.github_api.list_recently_merged_pull_requests")
    def test_ignores_pr_not_linked_to_factory_issue(
        self, mock_list, mock_get_issue, mock_exists, mock_save
    ):
        mock_list.return_value = [{"number": 1, "body": "Closes #2", "merge_commit_sha": "x"}]
        mock_get_issue.return_value = {"number": 2, "title": "普通のバグ修正", "labels": ["bug"]}

        run_strategy_promotion_check(force=True)

        mock_save.assert_not_called()

    @patch("src.utils.db.strategy_promotions.save_strategy_promotion")
    @patch("src.utils.db.strategy_promotions.promotion_exists", return_value=True)
    @patch("src.utils.github_api.get_issue")
    @patch("src.utils.github_api.list_recently_merged_pull_requests")
    def test_skips_already_recorded_pr(self, mock_list, mock_get_issue, mock_exists, mock_save):
        mock_list.return_value = [
            {"number": 564, "body": "Closes #999", "merge_commit_sha": "abc123"}
        ]

        run_strategy_promotion_check(force=True)

        mock_get_issue.assert_not_called()
        mock_save.assert_not_called()

    @patch("src.utils.github_api.list_recently_merged_pull_requests")
    def test_does_not_raise_on_github_api_failure(self, mock_list):
        mock_list.side_effect = Exception("network error")
        run_strategy_promotion_check(force=True)  # 例外を送出しないことを確認
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd python && python -m pytest tests/unit/test_strategy_promotion_check.py -v`
Expected: FAIL with `ImportError: cannot import name 'run_strategy_promotion_check'`

- [ ] **Step 4: Write minimal implementation**

`python/src/orchestration/jobs/periodic.py` の末尾（`run_nightly_strategy_factory` 関数の後）に追加:

```python
_FACTORY_LABELS = frozenset({"strategy-factory", "strategy-factory-idea"})


def run_strategy_promotion_check(force: bool = False) -> None:
    """
    2時間ごと実行: マージ済み戦略ファクトリー由来 PR を検出し strategy_promotions に記録する。

    Args:
        force: True なら STRATEGY_PROMOTION_CHECK_ENABLED=false でも実行（CLI 手動実行用）
    """
    from datetime import datetime, timedelta

    from config.settings import STRATEGY_PROMOTION_CHECK_ENABLED
    from src.backtest.promotion_detection import (
        extract_closing_issue_numbers,
        extract_factory_hash,
        load_gate_baseline,
    )
    from src.utils.db.strategy_promotions import promotion_exists, save_strategy_promotion
    from src.utils.github_api import get_issue, list_recently_merged_pull_requests

    if not STRATEGY_PROMOTION_CHECK_ENABLED and not force:
        logger.info(
            "戦略昇格チェックはスキップ（STRATEGY_PROMOTION_CHECK_ENABLED=false）"
        )
        return

    logger.info("=== 戦略昇格チェック開始 ===")
    detected: list[dict] = []
    try:
        since = datetime.now().astimezone() - timedelta(days=7)
        prs = list_recently_merged_pull_requests(since=since)
        for pr in prs:
            if promotion_exists(pr["number"]):
                continue
            for issue_number in extract_closing_issue_numbers(pr["body"]):
                try:
                    issue = get_issue(issue_number)
                except Exception as e:
                    logger.warning("Issue取得失敗: #%s: %s", issue_number, e)
                    continue
                if not _FACTORY_LABELS.intersection(issue["labels"]):
                    continue
                hypothesis_hash = extract_factory_hash(issue["title"])
                if hypothesis_hash is None:
                    continue
                baseline = load_gate_baseline(hypothesis_hash)
                if baseline is None:
                    logger.warning(
                        "baseline未発見のためスキップ: hash=%s", hypothesis_hash
                    )
                    continue
                save_strategy_promotion(
                    pr_number=pr["number"],
                    merge_commit_hash=pr["merge_commit_sha"],
                    rule_or_feature_id=hypothesis_hash,
                    pre_promotion_baseline=baseline,
                )
                detected.append(
                    {
                        "pr_number": pr["number"],
                        "rule_or_feature_id": hypothesis_hash,
                        "pre_promotion_baseline": baseline,
                    }
                )
                break  # 1PRにつき1件のみ記録（複数Issueをcloseする稀なPRは最初の一致のみ）
        logger.info("=== 戦略昇格チェック完了: 新規検出=%s ===", len(detected))
    except Exception as e:
        logger.error("戦略昇格チェック失敗: %s", e, exc_info=True)
        return

    for d in detected:
        try:
            from src.reporting.discord.discord_utils import send_strategy_promotion_detected

            send_strategy_promotion_detected(**d)
        except Exception as e:
            logger.error("戦略昇格通知失敗: %s", e, exc_info=True)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd python && python -m pytest tests/unit/test_strategy_promotion_check.py -v`
Expected: PASS（5 tests）

- [ ] **Step 6: Commit**

```bash
git add python/src/orchestration/jobs/periodic.py python/config/settings.py python/tests/unit/test_strategy_promotion_check.py
git commit -m "feat: マージ検知ジョブ run_strategy_promotion_check を追加"
```

---

### Task 6: スケジューラー配線

**Files:**
- Modify: `python/src/orchestration/scheduler.py:25`, `:75`
- Modify: `python/run_scheduler.py:156-160`, `:353-363`, `:529-530`, `:554-571`
- Test: `python/tests/unit/test_scheduler_pipeline_unit.py`（既存ファイルに追記）

**Interfaces:**
- Consumes: `src.orchestration.jobs.periodic.run_strategy_promotion_check`
- Produces: SCHEDULE_CONFIG エントリ `"strategy_promotion_check"`（2時間ごと、毎時30分オフセットで実行。IssueAgent 自体が毎正時±2時間で PR をマージするため、正時を避けてマージ直後のタイムラグを縮める）

- [ ] **Step 1: `scheduler.py` に re-export を追加**

`python/src/orchestration/scheduler.py:25` を変更:

```python
from src.orchestration.jobs.periodic import (
    run_monthly_report_job,
    run_nightly_strategy_factory,
    run_strategy_promotion_check,
)
```

`python/src/orchestration/scheduler.py:75` の `__all__` リストに追加:

```python
    "run_nightly_strategy_factory",
    "run_strategy_promotion_check",
]
```

- [ ] **Step 2: `run_scheduler.py` にジョブラッパー関数を追加**

`python/run_scheduler.py:156-160` の `job_nightly_strategy_factory()` の直後に追加:

```python
def job_strategy_promotion_check():
    """2時間ごと(毎時30分) - 戦略ファクトリー由来マージPRの昇格記録（STRATEGY_PROMOTION_CHECK_ENABLEDに従う）"""
    from src.orchestration.scheduler import run_strategy_promotion_check

    run_strategy_promotion_check()
```

- [ ] **Step 3: `SCHEDULE_CONFIG` にエントリを追加**

`python/run_scheduler.py:353-363` の `"nightly_strategy_factory"` エントリの直後に追加:

```python
    "strategy_promotion_check": {
        "func": job_strategy_promotion_check,
        "trigger": "cron",
        "period": "daily",
        "day_of_week": "mon-sun",
        "hour": "*/2",
        "minute": 30,
        "recovery_delay_minutes": 30,
        "max_executions_per_period": 12,
        "description": "2時間ごと(毎時30分) - 戦略ファクトリー由来マージPRの昇格記録",
    },
```

- [ ] **Step 4: CLI 手動実行分岐と選択肢一覧に追加**

`python/run_scheduler.py:529-530` の `elif pipeline == "factory":` ブロックの直後に追加:

```python
    elif pipeline == "promotion_check":
        queue_manager.run_job("strategy_promotion_check", reason="manual", force=True)
```

`python/run_scheduler.py:554-571` の `choices=[...]` リストの `"factory",` の直後に追加:

```python
            "promotion_check",
```

- [ ] **Step 5: Write and run a config-shape test**

`python/tests/unit/test_scheduler_pipeline_unit.py` に追記（このファイルは既存の `SCHEDULE_CONFIG`/`run_now` 関連テストを含むファイル。既存のテストクラスの末尾に新しいテストケースを追加する形。ファイルの既存インポート文を確認し、`SCHEDULE_CONFIG` と `run_now` が既にインポートされていればそれを使う）:

```python
class TestStrategyPromotionCheckSchedule(unittest.TestCase):
    def test_schedule_config_has_strategy_promotion_check(self):
        from run_scheduler import SCHEDULE_CONFIG

        self.assertIn("strategy_promotion_check", SCHEDULE_CONFIG)
        config = SCHEDULE_CONFIG["strategy_promotion_check"]
        self.assertEqual(config["trigger"], "cron")
        self.assertEqual(config["hour"], "*/2")
        self.assertEqual(config["minute"], 30)

    @patch("src.orchestration.scheduler.run_strategy_promotion_check")
    def test_run_now_promotion_check_invokes_job(self, mock_run):
        from run_scheduler import run_now

        run_now("promotion_check")
        mock_run.assert_called_once()
```

（`unittest.mock.patch` が未 import ならファイル冒頭に `from unittest.mock import patch` を追加する）

Run: `cd python && python -m pytest tests/unit/test_scheduler_pipeline_unit.py -v -k StrategyPromotionCheck`
Expected: PASS（2 tests）

- [ ] **Step 6: 手動スモークテスト**

Run: `cd python && python run_scheduler.py --run-now promotion_check`
Expected: ログに `戦略昇格チェックはスキップ（STRATEGY_PROMOTION_CHECK_ENABLED=false）` は出ない（`force=True` のため実行される）。`GITHUB_TOKEN` が未設定の場合は GitHub API 呼び出しで 401 が発生し `戦略昇格チェック失敗` ログで握りつぶされて正常終了することを確認する（クラッシュしないことが重要）。

- [ ] **Step 7: Commit**

```bash
git add python/src/orchestration/scheduler.py python/run_scheduler.py python/tests/unit/test_scheduler_pipeline_unit.py
git commit -m "feat: 戦略昇格チェックジョブをスケジューラーに配線"
```

---

## 完了条件

- [ ] 全6タスクのテストが green
- [ ] `cd python && .\check-ci.ps1` が通る
- [ ] `STRATEGY_PROMOTION_CHECK_ENABLED=false`（既定）の状態で `python run_scheduler.py` を起動しても新規ジョブは実行されない
- [ ] `.env` に `GITHUB_TOKEN`（repo 権限を持つ PAT）と、必要であれば `GITHUB_REPO`（既定 `rei4725/StockFixer` のままでよい）をユーザー自身が設定すれば、`--run-now promotion_check` で既存の `strategy-factory` ラベル付き Issue から人間が手動マージした PR があれば検出できる状態になる

## 次の計画への申し送り

- CIガード計画（`strategy-scope-guard.yml` / `pit-integrity-check.yml` / `backtest-gate-check.yml`）は本計画に依存しない。並行または後続で実施可能。
- ロールバック監視計画は本計画の `load_active_promotions()` / `mark_promotion_rolled_back()` を直接利用する。本計画完了後に着手すること。
- LLMアイデア発想＋auto-ok付与計画は、CIガード計画とロールバック監視計画の両方が完了してから着手する（安全網が揃った後に自動マージのスイッチを入れる、という合意事項サマリー通りの順序）。
