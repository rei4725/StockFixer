# Claude駆動ルール生成・検証パイプライン Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 戦略ファクトリー（#369）にClaudeが新しい `TradingRule` 実装を発想する仮説生成源を追加し、既存の過学習ゲート（DSR/PBO/#372）にそのまま通しつつ、生成コードはDockerサンドボックスで隔離実行する。合格候補は人間のPRレビューを経て初めて本番反映される。

**Architecture:** 3つの新規コンポーネント（`claude_rule_generator.py` / `sandbox_executor.py` / `scripts/sandbox_evaluate_rule.py`）+ 1つの共有レジストリ（`domain/generated_rules.py`）。既存の `evaluate_hypothesis()` / `apply_gate()` はサンドボックスコンテナ内から再利用し、変更しない。

**Tech Stack:** Python 3.12 / Docker（`docker run --network none`） / pandas / 既存 `TextReviewPort`（`LLM_BACKEND=cli`）

## Global Constraints

- 設計書: `docs/superpowers/specs/2026-07-22-claude-rule-generation-design.md`（本プランはこの設計に厳密に従う）
- 既定は無効: `FACTORY_CLAUDE_RULEGEN_ENABLED=False`（既存 `FACTORY_HYPOTHESIS_REVIEW_ENABLED` と同じ安全側既定）
- **サンドボックス境界は絶対**: Claude生成コードのexecは、`STOCKFIXER_SANDBOX=1` が設定されたDockerコンテナ内でのみ許可する。信頼された本体プロセス（夜間バッチ本体）が生成コードを直接execすることは、いかなる理由でも禁止
- **修復リトライはゲート不合格には使わない**: 静的検査違反・サンドボックス実行時例外・Claude応答JSON不正のみ修復対象（合計最大2回）。DSR/PBO等のゲート判定結果へのリトライは実装しない（過学習対策を自ら破壊するため）
- 昇格は人間のPRレビュー必須。`auto-ok` ラベルは付与しない
- BC独立性契約（`python/.importlinter` の `[importlinter:contract:independence]`）を破らない: `src.backtest` と `src.rule_engine` は相互import禁止。両方から参照する共有レジストリは `src.domain` に置く
- 新テーブルは作らない。既存 `factory_runs.spec_json`（VARCHAR、実質無制限）に生成コード全文を埋め込む
- 全ての新規Pythonファイルは `from __future__ import annotations` を先頭に置く（既存コードの慣例）
- コミットメッセージは Conventional Commits（`feat:` / `test:` 等）

---

### Task 1: Dockerサンドボックス基盤

**Files:**
- Modify: `python/Dockerfile`
- Modify: `docker-compose.yml`

**Interfaces:**
- Produces: `stockfixer` コンテナ内で `docker` CLI が使え、`/var/run/docker.sock` にアクセスできる。環境変数 `FACTORY_SANDBOX_IMAGE` が `stockfixer:${VERSION:-dev}` に展開される

- [ ] **Step 1: Dockerfile に docker CLI（静的バイナリ）を追加**

`python/Dockerfile` の「Claude Code CLI」インストールブロックの直後に追加する:

```dockerfile
# Docker CLI（静的バイナリ、サンドボックス実行の docker run に使用）
# docker.sock をマウントして兄弟コンテナを起動する Docker-outside-of-Docker 構成。
# daemon 自体は不要なため CLI のみをダウンロードする。
RUN curl -fsSL https://download.docker.com/linux/static/stable/x86_64/docker-27.3.1.tgz -o /tmp/docker.tgz \
    && tar xzf /tmp/docker.tgz -C /tmp \
    && mv /tmp/docker/docker /usr/local/bin/docker \
    && rm -rf /tmp/docker.tgz /tmp/docker
```

- [ ] **Step 2: docker-compose.yml に docker.sock マウントと SANDBOX_IMAGE を追加**

`docker-compose.yml` の `stockfixer` サービスの `environment:` ブロックに1行、`volumes:` ブロックに1行追加する:

```yaml
    environment:
      - LOG_FORMAT=json
      - OLLAMA_URL=http://ollama:11434
      - DATABASE_URL=postgresql://stockfixer:${POSTGRES_PASSWORD:-stockfixer_dev}@postgres:5432/stockfixer
      - FACTORY_SANDBOX_IMAGE=stockfixer:${VERSION:-dev}
    volumes:
      - ./python/data:/app/data
      - ./python/models:/app/models
      - ./python/results:/app/results
      - ./Logs:/app/logs
      - /var/run/docker.sock:/var/run/docker.sock
```

- [ ] **Step 3: ビルド確認**

Run: `docker build -t stockfixer:dev ./python`
Expected: ビルド成功、`docker run --rm stockfixer:dev docker --version` が正常にバージョン文字列を返す

- [ ] **Step 4: Commit**

```bash
git add python/Dockerfile docker-compose.yml
git commit -m "feat: サンドボックス実行用のDocker CLIとsocketマウントを追加"
```

---

### Task 2: 設定追加

**Files:**
- Modify: `python/config/settings.py`

**Interfaces:**
- Produces: `FACTORY_CLAUDE_RULEGEN_ENABLED`, `FACTORY_CLAUDE_RULEGEN_MODEL`, `FACTORY_CLAUDE_RULEGEN_MAX_TOKENS`, `FACTORY_CLAUDE_RULEGEN_COUNT`, `FACTORY_CLAUDE_RULEGEN_MAX_REPAIR_ATTEMPTS`, `FACTORY_SANDBOX_IMAGE`, `FACTORY_SANDBOX_TIMEOUT_SECONDS`, `FACTORY_SANDBOX_MEMORY_LIMIT`, `FACTORY_SANDBOX_CPU_LIMIT`（すべて `config.settings` からimport可能なモジュールレベル定数）

- [ ] **Step 1: Settings クラスに新規フィールドを追加**

`python/config/settings.py` の `FACTORY_HYPOTHESIS_REVIEW_MAX_TOKENS: int = Field(default=2048)` の直後に追加する:

```python
    # ---------- Claudeルール生成（backtest/claude_rule_generator.py） ----------
    # 既定無効。夜間バッチでClaudeに新しいTradingRule実装を発想させ、Dockerサンドボックス
    # で隔離実行してから既存の戦略ファクトリーのゲートに通す。ゲート合格分のみIssue化され、
    # 人間のPRレビューを経て初めて本番反映される（Claudeは合否判定・本番反映に一切関与しない）。
    FACTORY_CLAUDE_RULEGEN_ENABLED: bool = Field(default=False)
    FACTORY_CLAUDE_RULEGEN_MODEL: str = Field(default="claude-opus-4-8")
    FACTORY_CLAUDE_RULEGEN_MAX_TOKENS: int = Field(default=4096)
    FACTORY_CLAUDE_RULEGEN_COUNT: int = Field(default=3)
    # ゲート判定への修復リトライは行わない。静的検査違反・実行時例外・応答JSON不正のみ対象。
    FACTORY_CLAUDE_RULEGEN_MAX_REPAIR_ATTEMPTS: int = Field(default=2)
    # サンドボックス実行設定（docker run --network none で使用）
    FACTORY_SANDBOX_IMAGE: str = Field(default="")
    FACTORY_SANDBOX_TIMEOUT_SECONDS: int = Field(default=120)
    FACTORY_SANDBOX_MEMORY_LIMIT: str = Field(default="1g")
    FACTORY_SANDBOX_CPU_LIMIT: str = Field(default="1")
```

- [ ] **Step 2: モジュールレベル再エクスポートを追加**

`FACTORY_HYPOTHESIS_REVIEW_MAX_TOKENS: int = settings.FACTORY_HYPOTHESIS_REVIEW_MAX_TOKENS` の直後に追加する:

```python
FACTORY_CLAUDE_RULEGEN_ENABLED: bool = settings.FACTORY_CLAUDE_RULEGEN_ENABLED
FACTORY_CLAUDE_RULEGEN_MODEL: str = settings.FACTORY_CLAUDE_RULEGEN_MODEL
FACTORY_CLAUDE_RULEGEN_MAX_TOKENS: int = settings.FACTORY_CLAUDE_RULEGEN_MAX_TOKENS
FACTORY_CLAUDE_RULEGEN_COUNT: int = settings.FACTORY_CLAUDE_RULEGEN_COUNT
FACTORY_CLAUDE_RULEGEN_MAX_REPAIR_ATTEMPTS: int = settings.FACTORY_CLAUDE_RULEGEN_MAX_REPAIR_ATTEMPTS
FACTORY_SANDBOX_IMAGE: str = settings.FACTORY_SANDBOX_IMAGE
FACTORY_SANDBOX_TIMEOUT_SECONDS: int = settings.FACTORY_SANDBOX_TIMEOUT_SECONDS
FACTORY_SANDBOX_MEMORY_LIMIT: str = settings.FACTORY_SANDBOX_MEMORY_LIMIT
FACTORY_SANDBOX_CPU_LIMIT: str = settings.FACTORY_SANDBOX_CPU_LIMIT
```

- [ ] **Step 3: インポート確認**

Run: `cd python && python -c "from config.settings import FACTORY_CLAUDE_RULEGEN_ENABLED, FACTORY_SANDBOX_IMAGE; print(FACTORY_CLAUDE_RULEGEN_ENABLED, repr(FACTORY_SANDBOX_IMAGE))"`
Expected: `False ''`

- [ ] **Step 4: Commit**

```bash
git add python/config/settings.py
git commit -m "feat: Claudeルール生成・サンドボックス設定を追加"
```

---

### Task 3: AST安全性チェッカー

**Files:**
- Create: `python/src/backtest/ast_safety_check.py`
- Test: `python/tests/unit/backtest/test_ast_safety_check.py`

**Interfaces:**
- Produces: `check_source_safety(source_code: str) -> SafetyCheckResult`（`SafetyCheckResult.passed: bool`, `.violations: list[SafetyViolation]`、`SafetyViolation.line: int` / `.reason: str`）
- Consumes: なし（純粋関数）

- [ ] **Step 1: 失敗するテストを書く**

`python/tests/unit/backtest/test_ast_safety_check.py`:

```python
from src.backtest.ast_safety_check import check_source_safety


def test_safe_source_passes():
    source = """
import pandas as pd


class SafeRule:
    name = "safe_rule"
    description = "test"

    def generate_signal(self, df):
        return (df["Close"] > df["Close"].rolling(20).mean()).astype(int)
"""
    result = check_source_safety(source)
    assert result.passed is True
    assert result.violations == []


def test_banned_import_os_rejected():
    source = "import os\n\nclass R:\n    pass\n"
    result = check_source_safety(source)
    assert result.passed is False
    assert any("os" in v.reason for v in result.violations)
    assert result.violations[0].line == 1


def test_banned_import_from_subprocess_rejected():
    source = "from subprocess import run\n\nclass R:\n    pass\n"
    result = check_source_safety(source)
    assert result.passed is False
    assert any("subprocess" in v.reason for v in result.violations)


def test_banned_call_open_rejected():
    source = "class R:\n    def f(self):\n        open('/etc/passwd')\n"
    result = check_source_safety(source)
    assert result.passed is False
    assert any("open" in v.reason for v in result.violations)


def test_banned_call_eval_rejected():
    source = "class R:\n    def f(self):\n        eval('1+1')\n"
    result = check_source_safety(source)
    assert result.passed is False
    assert any("eval" in v.reason for v in result.violations)


def test_banned_dunder_subclasses_rejected():
    source = "class R:\n    def f(self):\n        object.__subclasses__()\n"
    result = check_source_safety(source)
    assert result.passed is False
    assert any("__subclasses__" in v.reason for v in result.violations)


def test_syntax_error_rejected():
    source = "def broken(:\n"
    result = check_source_safety(source)
    assert result.passed is False
    assert "構文エラー" in result.violations[0].reason
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `cd python && python -m pytest tests/unit/backtest/test_ast_safety_check.py -v`
Expected: `ModuleNotFoundError: No module named 'src.backtest.ast_safety_check'`

- [ ] **Step 3: 実装**

`python/src/backtest/ast_safety_check.py`:

```python
"""Claude生成ルールコードの静的安全性検査。

execする前に構文木を走査し、危険なimport/呼び出しを拒否する。これは早期に
明らかな不正コードを弾く高速フィルタであり、主たる安全境界ではない
（Dockerサンドボックスの --network none / 読み取り専用マウントが主たる境界）。
迂回可能であることを前提に、過信しないこと。
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field

_BANNED_IMPORTS = frozenset(
    {
        "os",
        "subprocess",
        "socket",
        "sys",
        "shutil",
        "pathlib",
        "importlib",
        "ctypes",
        "multiprocessing",
        "threading",
        "urllib",
        "requests",
        "http",
        "ftplib",
        "smtplib",
    }
)
_BANNED_CALLS = frozenset({"eval", "exec", "compile", "open", "__import__", "input"})
_BANNED_DUNDER_ATTRS = frozenset(
    {"__subclasses__", "__globals__", "__builtins__", "__import__", "__loader__"}
)


@dataclass
class SafetyViolation:
    line: int
    reason: str


@dataclass
class SafetyCheckResult:
    passed: bool
    violations: list[SafetyViolation] = field(default_factory=list)


def check_source_safety(source_code: str) -> SafetyCheckResult:
    """生成コードの構文木を走査し、危険な構文を検出する。

    構文エラー自体も違反として扱う。
    """
    try:
        tree = ast.parse(source_code)
    except SyntaxError as exc:
        return SafetyCheckResult(
            passed=False,
            violations=[SafetyViolation(line=exc.lineno or 1, reason=f"構文エラー: {exc.msg}")],
        )

    violations: list[SafetyViolation] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root in _BANNED_IMPORTS:
                    violations.append(
                        SafetyViolation(line=node.lineno, reason=f"禁止importです: {alias.name}")
                    )
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root in _BANNED_IMPORTS:
                violations.append(
                    SafetyViolation(line=node.lineno, reason=f"禁止importです: {node.module}")
                )
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in _BANNED_CALLS:
                violations.append(
                    SafetyViolation(line=node.lineno, reason=f"禁止呼び出しです: {func.id}(...)")
                )
        elif isinstance(node, ast.Attribute):
            if node.attr in _BANNED_DUNDER_ATTRS:
                violations.append(
                    SafetyViolation(line=node.lineno, reason=f"禁止属性アクセスです: .{node.attr}")
                )

    return SafetyCheckResult(passed=not violations, violations=violations)
```

- [ ] **Step 4: テストが通ることを確認**

Run: `cd python && python -m pytest tests/unit/backtest/test_ast_safety_check.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add python/src/backtest/ast_safety_check.py python/tests/unit/backtest/test_ast_safety_check.py
git commit -m "feat: 生成コードのAST静的安全性チェッカーを追加"
```

---

### Task 4: 共有レジストリ（domain/generated_rules.py）+ 配線

**Files:**
- Create: `python/src/domain/generated_rules.py`
- Modify: `python/src/backtest/rules/technical.py`
- Modify: `python/src/rule_engine/pipeline.py`
- Test: `python/tests/unit/domain/test_generated_rules.py`

**Interfaces:**
- Produces: `GENERATED_RULES: dict[str, GeneratedRule]`（`domain.generated_rules`）。`GeneratedRule` は `name: str` / `description: str` / `generate_signal(df) -> pd.Series` を持つ Protocol
- Consumes: `src.backtest.rules.ALL_RULES`（list、末尾に `GENERATED_RULES.values()` を追加）、`src.rule_engine.pipeline._RULE_INSTANCES`（dict、`GENERATED_RULES` をマージ）

**重要（BC独立性契約）**: `src.backtest` と `src.rule_engine` は `python/.importlinter` の independence contract により相互import禁止。レジストリは両BCから見て中立な `src.domain` に置く（layers/independence どちらの契約にも `src.domain` は列挙されておらず、参照は自由）。

- [ ] **Step 1: 失敗するテストを書く**

`python/tests/unit/domain/test_generated_rules.py`:

```python
import pandas as pd

from src.domain.generated_rules import GENERATED_RULES


class _DummyRule:
    name = "dummy_generated"
    description = "test"

    def generate_signal(self, df: pd.DataFrame) -> pd.Series:
        return pd.Series(0, index=df.index)


def test_generated_rules_empty_by_default():
    assert GENERATED_RULES == {}


def test_backtest_all_rules_includes_generated_entries(monkeypatch):
    """ALL_RULES はモジュール読み込み時に一度だけ評価されるリストのため、
    GENERATED_RULES への追加を反映させるには importlib.reload が必要
    （本番では PR マージ＝ソースコード編集のため、プロセス起動時の import で
    自然に反映される。この reload はテストでそれを模擬している）。
    """
    import importlib

    from src.backtest.rules import technical

    monkeypatch.setitem(GENERATED_RULES, "dummy_generated", _DummyRule())
    try:
        importlib.reload(technical)
        names = {r.name for r in technical.ALL_RULES}
        assert "dummy_generated" in names
    finally:
        importlib.reload(technical)


def test_rule_engine_instances_includes_generated_entries(monkeypatch):
    import importlib

    from src.rule_engine import pipeline

    monkeypatch.setitem(GENERATED_RULES, "dummy_generated", _DummyRule())
    try:
        importlib.reload(pipeline)
        assert "dummy_generated" in pipeline._RULE_INSTANCES
    finally:
        importlib.reload(pipeline)
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `cd python && python -m pytest tests/unit/domain/test_generated_rules.py -v`
Expected: `ModuleNotFoundError: No module named 'src.domain.generated_rules'`

- [ ] **Step 3: レジストリを実装**

`python/src/domain/generated_rules.py`:

```python
"""Claude生成ルールの共有レジストリ（shared kernel）。

人間がレビュー・マージしたPRがここに1エントリを追加する。backtest/rules/technical.py
の ALL_RULES（週次ルール評価対象）と rule_engine/pipeline.py の _RULE_INSTANCES
（本番日次シグナル生成）の両方がこのレジストリを合成して参照することで、「PRマージ＝
バックテストと本番の両方に同じコードが反映される」ことを保証する。

backtest BC と rule_engine BC は import-linter の BC independence contract
（python/.importlinter）により相互import禁止のため、両者から見て中立な domain/
（shared kernel）にこのレジストリを置く。

新規ルールを追加する場合は、このファイルへの1エントリ追加のみで完結させる。
個々のルールクラス定義は python/src/rule_engine/rules/generated/ 配下に置く。
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class GeneratedRule(Protocol):
    name: str
    description: str

    def generate_signal(self, df: pd.DataFrame) -> pd.Series:
        """OHLCV + テクニカル指標 DataFrame からシグナルを生成する。

        Returns:
            pd.Series: 1=buy, -1=sell, 0=hold（インデックスは df と同じ）
        """
        ...


GENERATED_RULES: dict[str, GeneratedRule] = {}
```

- [ ] **Step 4: `backtest/rules/technical.py` の `ALL_RULES` を拡張**

`python/src/backtest/rules/technical.py` の末尾（`ALL_RULES` 定義）を以下に置き換える:

```python
from src.domain.generated_rules import GENERATED_RULES

ALL_RULES: list = [
    VolumeBreakoutRule(),
    EMAMomentumRule(),
    RSIContrarianRule(),
    BollingerBandRule(),
    MACDRSIRule(),
    VolatilityBreakoutRule(),
    *GENERATED_RULES.values(),
]
```

`import` はファイル先頭のimport群に移し、`ALL_RULES` 定義部分のみをこの形に変更する。

- [ ] **Step 5: `rule_engine/pipeline.py` の `_RULE_INSTANCES` を拡張**

`python/src/rule_engine/pipeline.py` の `_RULE_INSTANCES` 定義を以下に置き換える:

```python
from src.domain.generated_rules import GENERATED_RULES

_RULE_INSTANCES: dict[str, TradingRule] = {
    "volume_breakout": VolumeBreakoutRule(),
    "ema_momentum": EMAMomentumRule(),
    "rsi_contrarian": RSIContrarianRule(),
    "bollinger_band": BollingerBandRule(),
    "macd_rsi": MACDRSIRule(),
    "volatility_breakout": VolatilityBreakoutRule(),
    **GENERATED_RULES,
}
```

`import` 文をファイル先頭のimport群に追加する。

- [ ] **Step 6: テストが通ることを確認**

Run: `cd python && python -m pytest tests/unit/domain/test_generated_rules.py -v`
Expected: 3 passed

- [ ] **Step 7: 既存のルール関連テストに回帰がないことを確認**

Run: `cd python && python -m pytest tests/unit/backtest/ tests/unit/rule_engine/ -v`
Expected: 全件 PASS（既存6ルールの挙動は不変）

- [ ] **Step 8: Commit**

```bash
git add python/src/domain/generated_rules.py python/src/backtest/rules/technical.py python/src/rule_engine/pipeline.py python/tests/unit/domain/test_generated_rules.py
git commit -m "feat: Claude生成ルールの共有レジストリを追加しバックテスト/本番双方に配線"
```

---

### Task 5: `build_rule()` の generated_code 対応（サンドボックス限定ガード付き）

**Files:**
- Modify: `python/src/backtest/factory.py`
- Test: `python/tests/unit/backtest/test_factory_build_rule_generated.py`

**Interfaces:**
- Consumes: `os.environ["STOCKFIXER_SANDBOX"]`（`"1"` の場合のみ generated_code のexecを許可）
- Produces: `build_rule({"type": "generated_code", "source_code": ..., "class_name": ...})` が `TradingRule` インスタンスを返す（サンドボックス環境変数が無い場合は `RuntimeError`）

- [ ] **Step 1: 失敗するテストを書く**

`python/tests/unit/backtest/test_factory_build_rule_generated.py`:

```python
import os

import pandas as pd
import pytest

from src.backtest.factory import build_rule

_VALID_SOURCE = """
class GeneratedTestRule:
    name = "generated_test_rule"
    description = "test rule"

    def generate_signal(self, df):
        import pandas as pd
        return pd.Series(1, index=df.index)
"""


def test_generated_code_rejected_without_sandbox_flag(monkeypatch):
    monkeypatch.delenv("STOCKFIXER_SANDBOX", raising=False)
    spec = {
        "type": "generated_code",
        "source_code": _VALID_SOURCE,
        "class_name": "GeneratedTestRule",
        "rule_name": "generated_test_rule",
        "description": "test",
    }
    with pytest.raises(RuntimeError, match="サンドボックス"):
        build_rule(spec)


def test_generated_code_builds_rule_inside_sandbox(monkeypatch):
    monkeypatch.setenv("STOCKFIXER_SANDBOX", "1")
    spec = {
        "type": "generated_code",
        "source_code": _VALID_SOURCE,
        "class_name": "GeneratedTestRule",
        "rule_name": "generated_test_rule",
        "description": "test",
    }
    rule = build_rule(spec)
    assert rule.name == "generated_test_rule"
    df = pd.DataFrame({"Close": [1.0, 2.0, 3.0]})
    signal = rule.generate_signal(df)
    assert list(signal) == [1, 1, 1]


def test_generated_code_missing_class_raises(monkeypatch):
    monkeypatch.setenv("STOCKFIXER_SANDBOX", "1")
    spec = {
        "type": "generated_code",
        "source_code": _VALID_SOURCE,
        "class_name": "NoSuchClass",
        "rule_name": "generated_test_rule",
        "description": "test",
    }
    with pytest.raises(ValueError, match="NoSuchClass"):
        build_rule(spec)
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `cd python && python -m pytest tests/unit/backtest/test_factory_build_rule_generated.py -v`
Expected: `ValueError: 未知の spec type: generated_code`（generated_code 未対応のため）

- [ ] **Step 3: `build_rule()` を拡張**

`python/src/backtest/factory.py` の `import json` の下に `import os` を追加し、`build_rule()` を以下に置き換える:

```python
_SANDBOX_ENV_FLAG = "STOCKFIXER_SANDBOX"


def build_rule(spec: dict) -> TradingRule:
    """rule_spec（再帰構造）から TradingRule インスタンスを構築する。"""
    spec_type = spec.get("type")
    if spec_type == "atomic":
        rule_name = spec["rule"]
        if rule_name not in _RULE_CLASSES:
            raise ValueError(f"未知のルール名: {rule_name}")
        return _RULE_CLASSES[rule_name](**(spec.get("params") or {}))
    if spec_type in ("and", "or"):
        children = [build_rule(s) for s in spec.get("rules", [])]
        if len(children) < 2:
            raise ValueError(f"合成ルールには2つ以上の子が必要: {spec}")
        return AndRule(children) if spec_type == "and" else OrRule(children)
    if spec_type == "generated_code":
        if os.environ.get(_SANDBOX_ENV_FLAG) != "1":
            raise RuntimeError(
                "generated_code スペックはサンドボックスコンテナ内でのみ構築できます"
                f"（環境変数 {_SANDBOX_ENV_FLAG}=1 が必要）。信頼された本体プロセスから"
                "未検証の生成コードをexecすることを防ぐガードです。"
            )
        return _build_generated_rule(spec)
    raise ValueError(f"未知の spec type: {spec_type}")


def _build_generated_rule(spec: dict) -> TradingRule:
    """generated_code spec からクラスをexecして TradingRule インスタンスを構築する。

    呼び出し元（build_rule）がサンドボックス環境変数を検証済みであることが前提。
    """
    source_code = spec["source_code"]
    class_name = spec["class_name"]
    namespace: dict = {}
    exec(compile(source_code, "<generated_rule>", "exec"), namespace)  # nosec B102
    if class_name not in namespace:
        raise ValueError(f"生成コードにクラス '{class_name}' が見つかりません")
    rule_cls = namespace[class_name]
    return rule_cls()
```

- [ ] **Step 4: テストが通ることを確認**

Run: `cd python && python -m pytest tests/unit/backtest/test_factory_build_rule_generated.py -v`
Expected: 3 passed

- [ ] **Step 5: 既存の factory テストに回帰がないことを確認**

Run: `cd python && python -m pytest tests/unit/backtest/test_factory.py -v`
Expected: 全件 PASS

- [ ] **Step 6: Commit**

```bash
git add python/src/backtest/factory.py python/tests/unit/backtest/test_factory_build_rule_generated.py
git commit -m "feat: build_rule()にgenerated_code対応を追加（サンドボックス限定ガード付き）"
```

---

### Task 6: サンドボックス内実行スクリプト

**Files:**
- Create: `python/scripts/sandbox_evaluate_rule.py`
- Test: `python/tests/unit/scripts/test_sandbox_evaluate_rule.py`

**Interfaces:**
- Consumes: CLI引数（`--source-file` / `--class-name` / `--rule-name` / `--description` / `--market` / `--lookback-years` / `--data-dir` / `--windows-file`）、環境変数 `STOCKFIXER_SANDBOX=1`
- Produces: stdout に1行JSON（`{"status": "ok", "evaluation": {...}}` または `{"status": "error", "error_type": ..., "traceback": ...}`）、終了コード 0（成功）/ 1（失敗）

- [ ] **Step 1: 失敗するテストを書く**

`python/tests/unit/scripts/test_sandbox_evaluate_rule.py`:

```python
import json
import os

import pandas as pd
import pytest

import scripts.sandbox_evaluate_rule as sandbox_script


def test_main_rejects_without_sandbox_flag(monkeypatch, capsys):
    monkeypatch.delenv("STOCKFIXER_SANDBOX", raising=False)
    monkeypatch.setattr(
        "sys.argv",
        [
            "sandbox_evaluate_rule.py",
            "--source-file",
            "dummy.py",
            "--class-name",
            "X",
            "--rule-name",
            "x",
            "--description",
            "x",
            "--market",
            "us",
            "--lookback-years",
            "2",
            "--data-dir",
            "dummy_dir",
            "--windows-file",
            "dummy.json",
        ],
    )
    rc = sandbox_script.main()
    assert rc == 1
    out = json.loads(capsys.readouterr().out.strip())
    assert out["status"] == "error"
    assert "STOCKFIXER_SANDBOX" in out["traceback"]


def test_main_success_path(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("STOCKFIXER_SANDBOX", "1")

    source_file = tmp_path / "candidate.py"
    source_file.write_text(
        "class GeneratedTestRule:\n"
        "    name = 'generated_test_rule'\n"
        "    description = 'test'\n"
        "    def generate_signal(self, df):\n"
        "        import pandas as pd\n"
        "        return pd.Series(0, index=df.index)\n",
        encoding="utf-8",
    )

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    dates = pd.date_range("2024-01-01", periods=60, freq="B")
    df = pd.DataFrame(
        {
            "Open": 100.0,
            "High": 101.0,
            "Low": 99.0,
            "Close": 100.0,
            "Volume": 1_000_000,
        },
        index=dates,
    )
    df.to_parquet(data_dir / "TEST.parquet")

    windows_file = tmp_path / "windows.json"
    windows_file.write_text(
        json.dumps([["2024-01-01", "2024-02-01"], ["2024-02-01", "2024-03-01"]]),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "sandbox_evaluate_rule.py",
            "--source-file",
            str(source_file),
            "--class-name",
            "GeneratedTestRule",
            "--rule-name",
            "generated_test_rule",
            "--description",
            "test",
            "--market",
            "us",
            "--lookback-years",
            "2",
            "--data-dir",
            str(data_dir),
            "--windows-file",
            str(windows_file),
        ],
    )
    rc = sandbox_script.main()
    assert rc == 0
    out = json.loads(capsys.readouterr().out.strip())
    assert out["status"] == "ok"
    assert "sharpe_ratio" in out["evaluation"]
    assert out["evaluation"]["n_symbols"] == 1
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `cd python && python -m pytest tests/unit/scripts/test_sandbox_evaluate_rule.py -v`
Expected: `ModuleNotFoundError: No module named 'scripts.sandbox_evaluate_rule'`

- [ ] **Step 3: 実装**

`python/scripts/sandbox_evaluate_rule.py`:

```python
"""サンドボックスコンテナ内で実行される、Claude生成ルールの隔離バックテスト実行スクリプト。

このスクリプトは stockfixer イメージから `docker run --network none` 経由でのみ
起動される想定であり、ホスト側の信頼されたプロセスから直接importして呼ばれることはない。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

import pandas as pd


def _load_data_by_symbol(data_dir: str) -> dict[str, pd.DataFrame]:
    data: dict[str, pd.DataFrame] = {}
    for name in sorted(os.listdir(data_dir)):
        if not name.endswith(".parquet"):
            continue
        symbol = name[: -len(".parquet")]
        data[symbol] = pd.read_parquet(os.path.join(data_dir, name))
    return data


def _load_windows(windows_file: str) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    with open(windows_file, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [(pd.Timestamp(w[0]), pd.Timestamp(w[1])) for w in raw]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-file", required=True)
    parser.add_argument("--class-name", required=True)
    parser.add_argument("--rule-name", required=True)
    parser.add_argument("--description", required=True)
    parser.add_argument("--market", required=True)
    parser.add_argument("--lookback-years", type=int, required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--windows-file", required=True)
    return parser.parse_args(argv)


def main() -> int:
    args = _parse_args()

    if os.environ.get("STOCKFIXER_SANDBOX") != "1":
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_type": "EnvironmentError",
                    "traceback": "STOCKFIXER_SANDBOX=1 が設定されていません。"
                    "このスクリプトはサンドボックスコンテナ内専用です。",
                }
            )
        )
        return 1

    from src.backtest.factory import evaluate_hypothesis
    from src.backtest.types import FactoryHypothesis

    try:
        with open(args.source_file, "r", encoding="utf-8") as f:
            source_code = f.read()

        hypothesis = FactoryHypothesis(
            rule_spec={
                "type": "generated_code",
                "source_code": source_code,
                "class_name": args.class_name,
                "rule_name": args.rule_name,
                "description": args.description,
            },
            market=args.market,
            lookback_years=args.lookback_years,
        )
        data_by_symbol = _load_data_by_symbol(args.data_dir)
        windows = _load_windows(args.windows_file)

        evaluation = evaluate_hypothesis(hypothesis, data_by_symbol, windows)

        print(
            json.dumps(
                {
                    "status": "ok",
                    "evaluation": {
                        "sharpe_ratio": evaluation.sharpe_ratio,
                        "sharpe_per_trade": evaluation.sharpe_per_trade,
                        "win_rate": evaluation.win_rate,
                        "num_trades": evaluation.num_trades,
                        "max_drawdown": evaluation.max_drawdown,
                        "total_return": evaluation.total_return,
                        "window_returns": evaluation.window_returns,
                        "n_symbols": evaluation.n_symbols,
                    },
                }
            )
        )
        return 0
    except Exception:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_type": "RuntimeError",
                    "traceback": traceback.format_exc(),
                }
            )
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: テストが通ることを確認**

Run: `cd python && python -m pytest tests/unit/scripts/test_sandbox_evaluate_rule.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add python/scripts/sandbox_evaluate_rule.py python/tests/unit/scripts/test_sandbox_evaluate_rule.py
git commit -m "feat: サンドボックス内でClaude生成ルールを評価するスクリプトを追加"
```

---

### Task 7: サンドボックス起動オーケストレーター

**Files:**
- Create: `python/src/backtest/sandbox_executor.py`
- Test: `python/tests/integration/backtest/test_sandbox_executor.py`（Docker必須、`shutil.which("docker")` が None なら skip）

**Interfaces:**
- Consumes: `FactoryHypothesis`（`rule_spec["type"] == "generated_code"`）, `data_by_symbol: dict[str, pd.DataFrame]`, `windows: list[tuple[pd.Timestamp, pd.Timestamp]]`
- Produces: `run_sandboxed_evaluation(...) -> SandboxRunResult`（`kind: "gate_evaluated" | "repairable" | "infra_error"`）、`prepare_sandbox_data(data_by_symbol, windows) -> tuple[str, str]`（共有データディレクトリパス, windowsファイルパス）

- [ ] **Step 1: 実装**

`python/src/backtest/sandbox_executor.py`:

```python
"""Claude生成ルールコードを Docker サンドボックスで隔離実行するオーケストレーター。

AST静的検査（高速フィルタ、主たる安全境界ではない）→ Dockerサンドボックス実行
（--network none・読み取り専用マウント、主たる安全境界）の順で実行し、結果を
既存の FactoryEvaluation 互換の形で返す。生成コード自体はこのモジュール内では
一切execしない（execするのはサンドボックスコンテナ内の scripts/sandbox_evaluate_rule.py）。
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from config.settings import (
    FACTORY_SANDBOX_CPU_LIMIT,
    FACTORY_SANDBOX_IMAGE,
    FACTORY_SANDBOX_MEMORY_LIMIT,
    FACTORY_SANDBOX_TIMEOUT_SECONDS,
)
from src.backtest.ast_safety_check import check_source_safety
from src.backtest.types import FactoryEvaluation, FactoryHypothesis
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SandboxRunResult:
    """サンドボックス実行1回分の結果。

    kind:
        "gate_evaluated" — 正常にバックテスト完了。evaluation を保持。
        "repairable"     — AST違反・実行時例外。repair_detail に修復用の情報。
                            ゲート判定結果はここに含まれない（別経路）。
        "infra_error"    — Dockerの起動失敗・タイムアウト等、コード起因でない。
    """

    kind: str
    evaluation: Optional[FactoryEvaluation] = None
    repair_detail: Optional[str] = None
    infra_detail: Optional[str] = None


def prepare_sandbox_data(
    data_by_symbol: dict[str, pd.DataFrame],
    windows: list[tuple[pd.Timestamp, pd.Timestamp]],
) -> tuple[str, str]:
    """バッチ全体で使い回す共有データディレクトリ/windowsファイルを1回だけ作る。

    呼び出し元が一晩のバッチにつき1回だけ呼び、返り値のパスを
    run_sandboxed_evaluation に渡す。呼び出し元は使用後にディレクトリを
    削除する責務を持つ（tempfile.TemporaryDirectory 等で管理）。
    """
    data_dir = tempfile.mkdtemp(prefix="factory_sandbox_data_")
    for symbol, df in data_by_symbol.items():
        df.to_parquet(os.path.join(data_dir, f"{symbol}.parquet"))

    windows_fd, windows_path = tempfile.mkstemp(
        prefix="factory_sandbox_windows_", suffix=".json"
    )
    raw = [[w[0].isoformat(), w[1].isoformat()] for w in windows]
    with os.fdopen(windows_fd, "w", encoding="utf-8") as f:
        json.dump(raw, f)

    return data_dir, windows_path


def _detect_self_image() -> str:
    """FACTORY_SANDBOX_IMAGE が未設定の場合、自コンテナのイメージ名を推測する。"""
    result = subprocess.run(
        ["docker", "inspect", "--format", "{{.Config.Image}}", os.environ.get("HOSTNAME", "")],
        capture_output=True,
        text=True,
        check=False,
    )
    image = result.stdout.strip()
    if not image:
        raise RuntimeError(
            "FACTORY_SANDBOX_IMAGE が未設定で、自コンテナのイメージ名も検出できませんでした"
        )
    return image


def run_sandboxed_evaluation(
    hypothesis: FactoryHypothesis,
    shared_data_dir: str,
    windows_file: str,
) -> SandboxRunResult:
    """1候補をAST検査 → Dockerサンドボックスで評価する。"""
    spec = hypothesis.rule_spec
    source_code = spec["source_code"]

    safety = check_source_safety(source_code)
    if not safety.passed:
        detail = "; ".join(f"{v.line}行目: {v.reason}" for v in safety.violations)
        return SandboxRunResult(kind="repairable", repair_detail=f"静的検査で拒否: {detail}")

    with tempfile.TemporaryDirectory(prefix="factory_sandbox_src_") as src_dir:
        source_path = os.path.join(src_dir, "candidate.py")
        with open(source_path, "w", encoding="utf-8") as f:
            f.write(source_code)

        image = FACTORY_SANDBOX_IMAGE or _detect_self_image()
        cmd = [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--read-only",
            "--tmpfs",
            "/tmp",
            "--memory",
            FACTORY_SANDBOX_MEMORY_LIMIT,
            "--cpus",
            FACTORY_SANDBOX_CPU_LIMIT,
            "-e",
            "STOCKFIXER_SANDBOX=1",
            "-v",
            f"{src_dir}:/sandbox/src:ro",
            "-v",
            f"{shared_data_dir}:/sandbox/data:ro",
            "-v",
            f"{windows_file}:/sandbox/windows.json:ro",
            image,
            "python",
            "scripts/sandbox_evaluate_rule.py",
            "--source-file",
            "/sandbox/src/candidate.py",
            "--class-name",
            spec["class_name"],
            "--rule-name",
            spec["rule_name"],
            "--description",
            spec["description"],
            "--market",
            hypothesis.market,
            "--lookback-years",
            str(hypothesis.lookback_years),
            "--data-dir",
            "/sandbox/data",
            "--windows-file",
            "/sandbox/windows.json",
        ]

        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=FACTORY_SANDBOX_TIMEOUT_SECONDS,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return SandboxRunResult(
                kind="infra_error", infra_detail="サンドボックス実行がタイムアウトしました"
            )

        if proc.returncode not in (0, 1):
            return SandboxRunResult(
                kind="infra_error",
                infra_detail=f"docker run 異常終了 (code={proc.returncode}): {proc.stderr[-2000:]}",
            )

        try:
            payload = json.loads(proc.stdout.strip().splitlines()[-1])
        except (json.JSONDecodeError, IndexError):
            return SandboxRunResult(
                kind="infra_error",
                infra_detail=f"サンドボックス出力の解析に失敗: {proc.stdout[-2000:]}",
            )

        if payload.get("status") == "error":
            return SandboxRunResult(
                kind="repairable",
                repair_detail=f"サンドボックス実行時エラー: {payload.get('traceback', '')[-2000:]}",
            )

        ev = payload["evaluation"]
        evaluation = FactoryEvaluation(
            hypothesis=hypothesis,
            sharpe_ratio=ev["sharpe_ratio"],
            sharpe_per_trade=ev["sharpe_per_trade"],
            win_rate=ev["win_rate"],
            num_trades=ev["num_trades"],
            max_drawdown=ev["max_drawdown"],
            total_return=ev["total_return"],
            window_returns=ev["window_returns"],
            n_symbols=ev["n_symbols"],
        )
        return SandboxRunResult(kind="gate_evaluated", evaluation=evaluation)
```

- [ ] **Step 2: 結合テストを書く**

`python/tests/integration/backtest/test_sandbox_executor.py`:

```python
import shutil

import pandas as pd
import pytest

from src.backtest.sandbox_executor import prepare_sandbox_data, run_sandboxed_evaluation
from src.backtest.types import FactoryHypothesis

pytestmark = pytest.mark.skipif(
    shutil.which("docker") is None, reason="Docker が利用できない環境ではスキップ"
)


def _sample_data() -> dict[str, pd.DataFrame]:
    dates = pd.date_range("2024-01-01", periods=60, freq="B")
    df = pd.DataFrame(
        {"Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.0, "Volume": 1_000_000},
        index=dates,
    )
    return {"TEST": df}


def _windows():
    dates = pd.date_range("2024-01-01", periods=60, freq="B")
    return [(dates[0], dates[29]), (dates[29], dates[-1])]


def test_safe_rule_evaluates_successfully():
    data_dir, windows_file = prepare_sandbox_data(_sample_data(), _windows())
    hypothesis = FactoryHypothesis(
        rule_spec={
            "type": "generated_code",
            "source_code": (
                "class SafeRule:\n"
                "    name = 'safe_rule'\n"
                "    description = 'test'\n"
                "    def generate_signal(self, df):\n"
                "        import pandas as pd\n"
                "        return pd.Series(0, index=df.index)\n"
            ),
            "class_name": "SafeRule",
            "rule_name": "safe_rule",
            "description": "test",
        },
        market="us",
    )
    result = run_sandboxed_evaluation(hypothesis, data_dir, windows_file)
    assert result.kind == "gate_evaluated"
    assert result.evaluation is not None
    assert result.evaluation.n_symbols == 1


def test_banned_import_rejected_without_docker_run():
    data_dir, windows_file = prepare_sandbox_data(_sample_data(), _windows())
    hypothesis = FactoryHypothesis(
        rule_spec={
            "type": "generated_code",
            "source_code": "import os\n\nclass BadRule:\n    pass\n",
            "class_name": "BadRule",
            "rule_name": "bad_rule",
            "description": "test",
        },
        market="us",
    )
    result = run_sandboxed_evaluation(hypothesis, data_dir, windows_file)
    assert result.kind == "repairable"
    assert "os" in result.repair_detail


def test_crashing_rule_returns_repairable_with_traceback():
    data_dir, windows_file = prepare_sandbox_data(_sample_data(), _windows())
    hypothesis = FactoryHypothesis(
        rule_spec={
            "type": "generated_code",
            "source_code": (
                "class CrashingRule:\n"
                "    name = 'crashing_rule'\n"
                "    description = 'test'\n"
                "    def generate_signal(self, df):\n"
                "        raise ValueError('boom')\n"
            ),
            "class_name": "CrashingRule",
            "rule_name": "crashing_rule",
            "description": "test",
        },
        market="us",
    )
    result = run_sandboxed_evaluation(hypothesis, data_dir, windows_file)
    assert result.kind == "repairable"
    assert "boom" in result.repair_detail
```

- [ ] **Step 3: テスト実行（Dockerが使える環境で）**

Run: `cd python && python -m pytest tests/integration/backtest/test_sandbox_executor.py -v`
Expected: 3 passed（`stockfixer:dev` イメージがビルド済みであること。未ビルドの場合は先に `docker build -t stockfixer:dev ./python` を実行）

- [ ] **Step 4: Commit**

```bash
git add python/src/backtest/sandbox_executor.py python/tests/integration/backtest/test_sandbox_executor.py
git commit -m "feat: サンドボックス起動オーケストレーターを追加"
```

---

### Task 8: Claude生成・修復ループ

**Files:**
- Create: `python/src/backtest/claude_rule_generator.py`
- Test: `python/tests/unit/backtest/test_claude_rule_generator.py`

**Interfaces:**
- Consumes: `src.infrastructure.llm.factory.get_text_review_port()`（`TextReviewPort.complete(*, system, user, model, max_tokens, schema) -> str`）、`src.backtest.sandbox_executor.run_sandboxed_evaluation`
- Produces: `generate_claude_hypotheses(market, existing_rule_catalog, champion_sharpe, shared_data_dir, windows_file, count) -> list[FactoryEvaluation]`（AST/実行時エラーは修復ループ後に諦めたら結果に含めない。ゲート不合格は含める＝呼び出し元の `apply_gate` に委ねる）

- [ ] **Step 1: 失敗するテストを書く**

`python/tests/unit/backtest/test_claude_rule_generator.py`:

```python
import json
from unittest.mock import MagicMock, patch

from src.backtest.claude_rule_generator import generate_claude_hypotheses
from src.backtest.sandbox_executor import SandboxRunResult
from src.backtest.types import FactoryEvaluation, FactoryHypothesis

_VALID_RESPONSE = json.dumps(
    {
        "rule_name": "novel_rule",
        "class_name": "NovelRule",
        "description": "何か新しいルール",
        "source_code": "class NovelRule:\n    name = 'novel_rule'\n"
        "    description = 'x'\n    def generate_signal(self, df):\n"
        "        import pandas as pd\n        return pd.Series(0, index=df.index)\n",
    }
)


def _make_hypothesis() -> FactoryHypothesis:
    return FactoryHypothesis(
        rule_spec={
            "type": "generated_code",
            "source_code": "class NovelRule:\n    pass\n",
            "class_name": "NovelRule",
            "rule_name": "novel_rule",
            "description": "x",
        },
        market="us",
    )


def test_disabled_flag_returns_empty(monkeypatch):
    monkeypatch.setattr("src.backtest.claude_rule_generator.FACTORY_CLAUDE_RULEGEN_ENABLED", False)
    result = generate_claude_hypotheses(
        market="us",
        champion_sharpe=1.0,
        shared_data_dir="dummy",
        windows_file="dummy.json",
    )
    assert result == []


@patch("src.backtest.claude_rule_generator.run_sandboxed_evaluation")
@patch("src.backtest.claude_rule_generator.get_text_review_port")
def test_gate_evaluated_candidate_returned(mock_port_factory, mock_sandbox, monkeypatch):
    monkeypatch.setattr("src.backtest.claude_rule_generator.FACTORY_CLAUDE_RULEGEN_ENABLED", True)
    monkeypatch.setattr("src.backtest.claude_rule_generator.FACTORY_CLAUDE_RULEGEN_COUNT", 1)

    mock_port = MagicMock()
    mock_port.complete.return_value = _VALID_RESPONSE
    mock_port_factory.return_value = mock_port

    evaluation = FactoryEvaluation(hypothesis=_make_hypothesis(), sharpe_ratio=1.5, num_trades=50)
    mock_sandbox.return_value = SandboxRunResult(kind="gate_evaluated", evaluation=evaluation)

    result = generate_claude_hypotheses(
        market="us",
        champion_sharpe=1.0,
        shared_data_dir="dummy",
        windows_file="dummy.json",
    )
    assert len(result) == 1
    assert result[0].sharpe_ratio == 1.5
    mock_port.complete.assert_called_once()
    mock_sandbox.assert_called_once()


@patch("src.backtest.claude_rule_generator.run_sandboxed_evaluation")
@patch("src.backtest.claude_rule_generator.get_text_review_port")
def test_repairable_failure_retries_then_gives_up(mock_port_factory, mock_sandbox, monkeypatch):
    monkeypatch.setattr("src.backtest.claude_rule_generator.FACTORY_CLAUDE_RULEGEN_ENABLED", True)
    monkeypatch.setattr("src.backtest.claude_rule_generator.FACTORY_CLAUDE_RULEGEN_COUNT", 1)
    monkeypatch.setattr(
        "src.backtest.claude_rule_generator.FACTORY_CLAUDE_RULEGEN_MAX_REPAIR_ATTEMPTS", 2
    )

    mock_port = MagicMock()
    mock_port.complete.return_value = _VALID_RESPONSE
    mock_port_factory.return_value = mock_port

    mock_sandbox.return_value = SandboxRunResult(
        kind="repairable", repair_detail="静的検査で拒否: import os"
    )

    result = generate_claude_hypotheses(
        market="us",
        champion_sharpe=1.0,
        shared_data_dir="dummy",
        windows_file="dummy.json",
    )
    assert result == []
    # 初回 + 修復2回 = 3回呼ばれる
    assert mock_port.complete.call_count == 3
    assert mock_sandbox.call_count == 3


@patch("src.backtest.claude_rule_generator.run_sandboxed_evaluation")
@patch("src.backtest.claude_rule_generator.get_text_review_port")
def test_infra_error_does_not_consume_repair_budget(mock_port_factory, mock_sandbox, monkeypatch):
    monkeypatch.setattr("src.backtest.claude_rule_generator.FACTORY_CLAUDE_RULEGEN_ENABLED", True)
    monkeypatch.setattr("src.backtest.claude_rule_generator.FACTORY_CLAUDE_RULEGEN_COUNT", 1)

    mock_port = MagicMock()
    mock_port.complete.return_value = _VALID_RESPONSE
    mock_port_factory.return_value = mock_port

    mock_sandbox.return_value = SandboxRunResult(kind="infra_error", infra_detail="timeout")

    result = generate_claude_hypotheses(
        market="us",
        champion_sharpe=1.0,
        shared_data_dir="dummy",
        windows_file="dummy.json",
    )
    assert result == []
    # インフラ起因は修復リトライしない（1回のみ呼ばれる）
    assert mock_port.complete.call_count == 1
    assert mock_sandbox.call_count == 1
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `cd python && python -m pytest tests/unit/backtest/test_claude_rule_generator.py -v`
Expected: `ModuleNotFoundError: No module named 'src.backtest.claude_rule_generator'`

- [ ] **Step 3: 実装**

`python/src/backtest/claude_rule_generator.py`:

```python
"""Claudeに新しい TradingRule 実装を発想させ、機械的な失敗のみ修復リトライする。

ゲート判定（DSR/PBO等）への修復リトライは意図的に実装しない。ゲート不合格は
「バグ」ではなく「良いルールではなかった」という正当な判定結果であり、ここに
修復ループを回すと過学習対策そのものを破壊するため（p-hacking化）。
"""

from __future__ import annotations

import json
from typing import Any, Optional

from config.settings import (
    FACTORY_CLAUDE_RULEGEN_COUNT,
    FACTORY_CLAUDE_RULEGEN_ENABLED,
    FACTORY_CLAUDE_RULEGEN_MAX_REPAIR_ATTEMPTS,
    FACTORY_CLAUDE_RULEGEN_MAX_TOKENS,
    FACTORY_CLAUDE_RULEGEN_MODEL,
)
from src.backtest.sandbox_executor import SandboxRunResult, run_sandboxed_evaluation
from src.backtest.types import FactoryEvaluation, FactoryHypothesis
from src.infrastructure.llm.factory import get_text_review_port
from src.utils.logger import get_logger

logger = get_logger(__name__)

_SYSTEM_PROMPT = (
    "あなたはクオンツトレーディングシステムのルール開発者です。"
    "OHLCV + テクニカル指標（macd, macd_signal, macd_diff, ema_fast, ema_slow, atr, rsi, "
    "bb_upper, bb_middle, bb_lower, bb_width, stoch_k, stoch_d, obv, volume_ratio, "
    "volume_price_trend, volume_ma_deviation, day_of_week, month, is_month_end, "
    "および w_/m_ 接頭辞の週足/月足マルチタイムフレーム特徴量）を受け取り、"
    "1=buy, -1=sell, 0=hold を返す generate_signal(self, df) を実装した新しい "
    "売買ルールクラスを1つ提案してください。"
    "既存の手書きルール（出来高ブレイクアウト・EMAモメンタム・RSI逆張り・"
    "ボリンジャーバンド・MACD+RSI・ボラティリティブレイクアウト）とは異なる着眼点を"
    "選んでください。単一銘柄OHLCVと上記指標列のみが利用可能です（他銘柄・マクロ指標・"
    "ネットワークアクセスは一切使えません）。"
    "importはpandas/numpy/taのみ許可されます。os/subprocess/socket等は一切使わないこと。"
    '{"rule_name": str, "class_name": str, "description": str, "source_code": str} '
    "のJSONのみを出力してください（説明文やMarkdownのコードフェンスは不要）。"
)

_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "rule_name": {"type": "string"},
        "class_name": {"type": "string"},
        "description": {"type": "string"},
        "source_code": {"type": "string"},
    },
    "required": ["rule_name", "class_name", "description", "source_code"],
    "additionalProperties": False,
}


def _parse_response(text: str) -> Optional[dict]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    required = {"rule_name", "class_name", "description", "source_code"}
    if not isinstance(data, dict) or not required.issubset(data):
        return None
    return data


def _generate_one_candidate(
    market: str, repair_context: Optional[str] = None
) -> Optional[FactoryHypothesis]:
    port = get_text_review_port()
    user_prompt = f"マーケット: {market}\n新しいルールを1つ提案してください。"
    if repair_context:
        user_prompt += (
            f"\n\n直前の提案には以下の問題がありました。修正して同じJSON形式で再提案して"
            f"ください:\n{repair_context}"
        )
    try:
        text = port.complete(
            system=_SYSTEM_PROMPT,
            user=user_prompt,
            model=FACTORY_CLAUDE_RULEGEN_MODEL,
            max_tokens=FACTORY_CLAUDE_RULEGEN_MAX_TOKENS,
            schema=_RESPONSE_SCHEMA,
        )
    except Exception:
        logger.error("[claude_rule_generator] Claude呼び出し失敗", exc_info=True)
        return None

    data = _parse_response(text)
    if data is None:
        logger.warning("[claude_rule_generator] 応答JSONのスキーマ不正: %s", text[:500])
        return None

    return FactoryHypothesis(
        rule_spec={
            "type": "generated_code",
            "source_code": data["source_code"],
            "class_name": data["class_name"],
            "rule_name": data["rule_name"],
            "description": data["description"],
        },
        market=market,
    )


def _generate_and_evaluate_with_repair(
    market: str, shared_data_dir: str, windows_file: str
) -> Optional[FactoryEvaluation]:
    """1候補につき初回生成＋最大 FACTORY_CLAUDE_RULEGEN_MAX_REPAIR_ATTEMPTS 回の修復。

    修復対象は「機械的な壊れ方」（静的検査違反・実行時例外・応答JSON不正）のみ。
    ゲート判定（DSR/PBO等）が絡む結果はここでは判定せず、gate_evaluated ならそのまま返す
    （合否は呼び出し元の apply_gate に委ねる）。
    """
    repair_context: Optional[str] = None
    attempts = FACTORY_CLAUDE_RULEGEN_MAX_REPAIR_ATTEMPTS + 1

    for attempt in range(attempts):
        hypothesis = _generate_one_candidate(market, repair_context=repair_context)
        if hypothesis is None:
            # 応答JSON不正 or Claude呼び出し失敗も「機械的な壊れ方」として修復対象にする
            repair_context = "前回の応答がJSONとして解析できませんでした。厳密なJSONで再提案してください。"
            continue

        result: SandboxRunResult = run_sandboxed_evaluation(
            hypothesis, shared_data_dir, windows_file
        )
        if result.kind == "gate_evaluated":
            return result.evaluation
        if result.kind == "infra_error":
            logger.warning(
                "[claude_rule_generator] インフラ起因の失敗のためこの候補をスキップ: %s",
                result.infra_detail,
            )
            return None
        # kind == "repairable"
        logger.info(
            "[claude_rule_generator] 修復リトライ %d/%d: %s",
            attempt + 1,
            attempts - 1,
            result.repair_detail,
        )
        repair_context = result.repair_detail

    logger.warning("[claude_rule_generator] 修復予算を使い切ったためこの候補を諦めます")
    return None


def generate_claude_hypotheses(
    market: str,
    champion_sharpe: float,
    shared_data_dir: str,
    windows_file: str,
) -> list[FactoryEvaluation]:
    """1晩分のClaude生成候補を生成・サンドボックス評価する。

    既定無効（FACTORY_CLAUDE_RULEGEN_ENABLED=False）。champion_sharpe は現状
    プロンプトへの直接利用はしていない（将来のプロンプト改善用に引数として残す）。
    """
    del champion_sharpe  # 将来のプロンプト改善用に予約（現状は未使用）
    if not FACTORY_CLAUDE_RULEGEN_ENABLED:
        return []

    evaluations: list[FactoryEvaluation] = []
    for i in range(FACTORY_CLAUDE_RULEGEN_COUNT):
        logger.info("[claude_rule_generator] 候補 %d/%d 生成中...", i + 1, FACTORY_CLAUDE_RULEGEN_COUNT)
        evaluation = _generate_and_evaluate_with_repair(market, shared_data_dir, windows_file)
        if evaluation is not None:
            evaluations.append(evaluation)
    return evaluations
```

- [ ] **Step 4: テストが通ることを確認**

Run: `cd python && python -m pytest tests/unit/backtest/test_claude_rule_generator.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add python/src/backtest/claude_rule_generator.py python/tests/unit/backtest/test_claude_rule_generator.py
git commit -m "feat: Claudeルール生成・修復ループを追加"
```

---

### Task 9: `run_factory_batch()` への統合

**Files:**
- Modify: `python/src/backtest/factory.py`
- Test: `python/tests/unit/backtest/test_factory_claude_integration.py`

**Interfaces:**
- Consumes: `src.backtest.claude_rule_generator.generate_claude_hypotheses`, `src.backtest.sandbox_executor.prepare_sandbox_data`
- Produces: `run_factory_batch()` の返り値 `FactoryBatchResult.evaluated` に、有効時はClaude生成候補の評価も含まれる（既存のDSR/PBO/ゲート/記録/レポート処理をそのまま通る）

- [ ] **Step 1: 失敗するテストを書く**

`python/tests/unit/backtest/test_factory_claude_integration.py`:

```python
from unittest.mock import MagicMock, patch

import pandas as pd

from src.backtest.factory import run_factory_batch
from src.backtest.types import FactoryEvaluation, FactoryHypothesis


def _sample_data():
    dates = pd.date_range("2024-01-01", periods=120, freq="B")
    return pd.DataFrame(
        {"Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.0, "Volume": 1_000_000},
        index=dates,
    )


@patch("src.backtest.factory.prepare_sandbox_data")
@patch("src.backtest.factory.generate_claude_hypotheses")
@patch("src.backtest.factory._load_symbol_data")
def test_claude_hypotheses_included_when_enabled(
    mock_load_data, mock_generate, mock_prepare, monkeypatch
):
    monkeypatch.setattr("src.backtest.factory.FACTORY_CLAUDE_RULEGEN_ENABLED", True)
    mock_load_data.return_value = {"TEST": _sample_data()}
    mock_prepare.return_value = ("/tmp/dummy_data", "/tmp/dummy_windows.json")

    claude_hypothesis = FactoryHypothesis(
        rule_spec={
            "type": "generated_code",
            "source_code": "class X:\n    pass\n",
            "class_name": "X",
            "rule_name": "claude_rule",
            "description": "x",
        },
        market="us",
    )
    mock_generate.return_value = [
        FactoryEvaluation(
            hypothesis=claude_hypothesis,
            sharpe_ratio=2.0,
            sharpe_per_trade=0.1,
            num_trades=40,
            max_drawdown=-0.05,
        )
    ]

    result = run_factory_batch(market="us", symbols=["TEST"], budget=1, n_windows=4)

    labels = [e.hypothesis.rule_spec.get("rule_name") for e in result.evaluated]
    assert "claude_rule" in labels
    mock_generate.assert_called_once()


@patch("src.backtest.factory.generate_claude_hypotheses")
@patch("src.backtest.factory._load_symbol_data")
def test_claude_hypotheses_skipped_when_disabled(mock_load_data, mock_generate, monkeypatch):
    monkeypatch.setattr("src.backtest.factory.FACTORY_CLAUDE_RULEGEN_ENABLED", False)
    mock_load_data.return_value = {"TEST": _sample_data()}

    run_factory_batch(market="us", symbols=["TEST"], budget=1, n_windows=4)

    mock_generate.assert_not_called()
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `cd python && python -m pytest tests/unit/backtest/test_factory_claude_integration.py -v`
Expected: `AttributeError: <module 'src.backtest.factory'> does not have the attribute 'generate_claude_hypotheses'`

- [ ] **Step 3: `run_factory_batch()` を拡張**

`python/src/backtest/factory.py` のimport群に追加:

```python
from config.settings import FACTORY_CLAUDE_RULEGEN_ENABLED
from src.backtest.claude_rule_generator import generate_claude_hypotheses
from src.backtest.sandbox_executor import prepare_sandbox_data
```

`run_factory_batch()` 内、`windows = _window_bounds(start, end, n_windows)` の直後（`evaluations = [evaluate_hypothesis(h, data, windows) for h in batch]` の前）に追加:

```python
    windows = _window_bounds(start, end, n_windows)
    evaluations = [evaluate_hypothesis(h, data, windows) for h in batch]

    if FACTORY_CLAUDE_RULEGEN_ENABLED:
        control_sharpes_pre = [
            e.sharpe_ratio for e in evaluations if e.hypothesis.is_control and e.num_trades > 0
        ]
        pre_champion_sharpe = max(control_sharpes_pre) if control_sharpes_pre else float("nan")
        shared_data_dir, windows_file = prepare_sandbox_data(data, windows)
        try:
            claude_evaluations = generate_claude_hypotheses(
                market, pre_champion_sharpe, shared_data_dir, windows_file
            )
            evaluations.extend(claude_evaluations)
        finally:
            shutil.rmtree(shared_data_dir, ignore_errors=True)
            Path(windows_file).unlink(missing_ok=True)
```

`import shutil` と `from pathlib import Path` をファイル先頭のimport群に追加する（`os` は Task 5 で既に追加済みのためここでは不要）。テストのモックパス（`/tmp/dummy_data` 等）は実在しないため、`ignore_errors=True` / `missing_ok=True` の両方が無いとテストで `FileNotFoundError` になる点に注意。

- [ ] **Step 4: テストが通ることを確認**

Run: `cd python && python -m pytest tests/unit/backtest/test_factory_claude_integration.py -v`
Expected: 2 passed

- [ ] **Step 5: 既存の factory テスト全体に回帰がないことを確認**

Run: `cd python && python -m pytest tests/unit/backtest/ -v`
Expected: 全件 PASS

- [ ] **Step 6: Commit**

```bash
git add python/src/backtest/factory.py python/tests/unit/backtest/test_factory_claude_integration.py
git commit -m "feat: run_factory_batch()にClaude生成候補を統合"
```

---

### Task 10: Issue本文でのソースコード表示

**Files:**
- Modify: `python/src/backtest/factory_report.py`
- Test: `python/tests/unit/backtest/test_factory_report_generated.py`

**Interfaces:**
- Produces: `_build_issue_body()` が `rule_spec["type"] == "generated_code"` のとき、スペックのJSON丸出しではなく ```python フェンスでソースコードを読みやすく表示する

- [ ] **Step 1: 失敗するテストを書く**

`python/tests/unit/backtest/test_factory_report_generated.py`:

```python
from src.backtest.factory_report import write_report
from src.backtest.types import FactoryEvaluation, FactoryHypothesis


def test_issue_body_renders_source_code_block(tmp_path, monkeypatch):
    monkeypatch.setattr("src.backtest.factory_report.get_results_dir", lambda: str(tmp_path))

    hypothesis = FactoryHypothesis(
        rule_spec={
            "type": "generated_code",
            "source_code": "class NovelRule:\n    name = 'novel_rule'\n",
            "class_name": "NovelRule",
            "rule_name": "novel_rule",
            "description": "新しい着眼点のルール",
        },
        market="us",
    )
    evaluation = FactoryEvaluation(
        hypothesis=hypothesis,
        sharpe_ratio=1.8,
        dsr=0.96,
        pbo=0.2,
        num_trades=45,
        max_drawdown=-0.1,
        win_rate=0.55,
        total_return=0.2,
        window_returns=[0.01, 0.02],
        n_symbols=3,
    )
    path = write_report(evaluation, champion_sharpe=1.0, period=("2024-01-01", "2025-01-01"))

    import json

    with open(path, encoding="utf-8") as f:
        report = json.load(f)
    body = report["issue_body"]
    assert "```python" in body
    assert "class NovelRule" in body
    assert "新しい着眼点のルール" in body
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `cd python && python -m pytest tests/unit/backtest/test_factory_report_generated.py -v`
Expected: FAIL（`"```python" in body` が False。現状は `rule_spec` をJSONとして丸ごとダンプするのみ）

- [ ] **Step 3: `_build_issue_body()` を拡張**

`python/src/backtest/factory_report.py` の `_build_issue_body()` 内、「### スペック」セクションを組み立てている箇所を以下に置き換える:

```python
def _build_spec_section(spec: dict) -> str:
    if spec.get("type") == "generated_code":
        return f"""### 生成ルール（Claude提案）

**{spec.get('rule_name', '?')}**: {spec.get('description', '')}

```python
{spec.get('source_code', '')}
```
"""
    return f"""### スペック

```json
{json.dumps(spec, ensure_ascii=False, indent=2)}
```
"""
```

`_build_issue_body()` 内の以下の行:

```python
    return f"""## 戦略仮説（自動生成）

夜間ファクトリーのゲートを通過した仮説です。`hypothesis_hash={h.hypothesis_hash}`
{pbo_warning}

### スペック

```json
{json.dumps(h.rule_spec, ensure_ascii=False, indent=2)}
```

- マーケット: {h.market}
```

を以下に置き換える:

```python
    spec_section = _build_spec_section(h.rule_spec)
    return f"""## 戦略仮説（自動生成）

夜間ファクトリーのゲートを通過した仮説です。`hypothesis_hash={h.hypothesis_hash}`
{pbo_warning}

{spec_section}
- マーケット: {h.market}
```

（残りの `評価期間` 以降は変更しない）

- [ ] **Step 4: テストが通ることを確認**

Run: `cd python && python -m pytest tests/unit/backtest/test_factory_report_generated.py -v`
Expected: 1 passed

- [ ] **Step 5: 既存の factory_report テストに回帰がないことを確認**

Run: `cd python && python -m pytest tests/unit/backtest/test_factory_report.py -v`
Expected: 全件 PASS（既存の atomic/and/or スペックは従来通りJSON表示のまま）

- [ ] **Step 6: Commit**

```bash
git add python/src/backtest/factory_report.py python/tests/unit/backtest/test_factory_report_generated.py
git commit -m "feat: 生成ルールのIssue本文をソースコードフェンスで読みやすく表示"
```

---

## 全体テスト・仕上げ

- [ ] **Step 1: CI相当の一括チェック**

Run: `cd python && .\check-ci.ps1`（Windows）
Expected: lint / mypy / pylint / import-linter / unit tests（cov≥80%）すべて green

- [ ] **Step 2: Docker統合確認（Docker利用可能な環境で）**

Run: `docker build -t stockfixer:dev ./python && cd python && python -m pytest tests/integration/backtest/test_sandbox_executor.py -v`
Expected: 3 passed

- [ ] **Step 3: 設計書との整合確認**

`docs/superpowers/specs/2026-07-22-claude-rule-generation-design.md` を再読し、実装が「新テーブルなし」「サンドボックス限定ガード」「ゲート不合格への修復リトライ禁止」「人間PRレビュー必須（auto-okなし）」の各方針から逸脱していないことを確認する。
