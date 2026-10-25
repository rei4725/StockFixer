# utils/ の業務の処理を domain/・BC へ移す Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `src/utils/` にある外とやり取りしない売買の判断・表・暦を `src/domain/` へ、reporting だけが使う `drift_thresholds` を `src/reporting/` へ移し、`utils/` に業務のモジュールが戻らないよう CI で止める。振る舞いは変えない。

**Architecture:** 移した本体は新しい場所に置き、元の `src/utils/<name>.py` には公開の名前だけを再輸出する転送を残す（`drift_thresholds` を除く）。呼び出し元とテストの import は同じ PR で新しい場所へ書き換える。転送は最後の PR で消す。

**Tech Stack:** Python 3 / pandas / pytest / import-linter（`python/.importlinter`）

**Spec:** `docs/superpowers/specs/2026-10-25-utils-business-logic-relocation-design.md`

## Global Constraints

- コマンドはすべて `python/` で実行する。Windows では `py` を使う。
- ベースは `develop`。ブランチは `refactor/<対象>`。コミットは Conventional Commits（`refactor:` など）。
- PR 本文には CLAUDE.md の必須セクション（`version_impact` など）を入れる。振る舞いを変えないため `version_impact` は `none` とする（`docs/VERSIONING_POLICY.md` で確かめる）。
- 各 PR の前に `.\check-ci.ps1` を通す（lint / mypy / pylint / import-linter / unit cov≥80% / bandit / pip-audit）。
- `.importlinter` の `ignore_imports` は0件のまま。
- 1ファイル600行未満（NF-704）。
- `src/domain/` は `src.domain` 以外の `src.*` と `config` を import しない。`domain/` へ移す処理はログを出さない。
- 移したモジュールの既存のテストは、import の書き換え以外は変えない。
- `src.utils.*` を指す既存の `mock.patch`（`sector_constraints.yf.Ticker`・`sector_constraints.get_symbol_sector`・`optimal_params_loader.OPTIMAL_PARAMS_PATH`・`signal_generator.get_optimal_params`）は残る側を指しているので変えない。

## Review Focus

1. 転送経由の import が、新しい場所と **同じ物** を返すこと（別のコピーだと、表を書き換える処理やテストの patch が片方にしか効かない）→ Task 1〜3 に `is` で比べるテスト。
2. `domain/` の新しいモジュールが `config` や `src.utils` を import しないこと（`config` は import-linter の root 外なので契約では捕まらない）→ Task 6 のテスト。
3. `SignalGenerator.generate_signal` が、`rolling_std` 有り無し・RSI 列無し・`Volume` 列無しのどの入力でも、移す前と同じ `pd.Series` を返すこと → Task 4 の比べるテスト。
4. `drift_thresholds` を移した後、`orchestration/jobs/drift_check.py` を patch しているテストの対象の文字列が古い場所のまま空振りしないこと → Task 5 の Step 1 で確かめる。
5. 転送が残っている間に `utils/` へ新しいファイルが足されたら落ちること（転送のファイルは一時的に許す）→ Task 6 のテスト。

---

### Task 0: 出す前の確かめ（PR 1・2・3 の直前に毎回行う）

- [ ] **Step 1:** `git fetch && git switch develop && git pull --ff-only` を実行し、`git switch -c refactor/<対象>` で新しいブランチを作る。
- [ ] **Step 2:** 開いた PR の中に、その PR で移すファイルを触っているものがないか確かめる。
  Run: `gh pr list --state open --json number,files --jq '.[] | select(any(.files[]; .path | test("src/utils/(regime_weights|jp_market_holidays|sector_constraints|signal_generator|drift_thresholds)[.]py"))) | .number'`
  Expected: 何も出ない。出たら、その PR の中身を見る。移すファイルの **中身** を書き換えていれば、その PR のマージを待つ。import だけなら進めてよい。

---

### Task 1: `regime_weights` を `domain/` へ（PR 1 `refactor/domain-pure-rules`）

**Files:**
- Create: `python/src/domain/regime_weights.py`（`src/utils/regime_weights.py` の中身をそのまま移す）
- Modify: `python/src/utils/regime_weights.py`（転送だけにする）
- Modify: `python/src/backtest/portfolio/simulation.py:19`、`python/src/trading/signal_generator.py:9-12`
- Test: `python/tests/unit/test_utils_shims.py`（新規。Task 2・3 も同じファイルに足す）

**Interfaces:**
- Produces: `src.domain.regime_weights.REGIME_SECTOR_WEIGHTS: dict[str, dict[str, float]]`、`src.domain.regime_weights.get_regime_sector_weight(regime: str, sector: str) -> float`

- [ ] **Step 1: 失敗するテストを書く**

```python
def test_regime_weights_shim_reexports_domain_objects():
    import src.domain.regime_weights as new
    import src.utils.regime_weights as old
    assert old.REGIME_SECTOR_WEIGHTS is new.REGIME_SECTOR_WEIGHTS
    assert old.get_regime_sector_weight is new.get_regime_sector_weight
```

- [ ] **Step 2:** `py -m pytest tests/unit/test_utils_shims.py -v` を実行し、`ModuleNotFoundError: No module named 'src.domain.regime_weights'` で FAIL することを確かめる。
- [ ] **Step 3:** 本体を `src/domain/regime_weights.py` へ移し、docstring を「domain の共有ルール表」に直す。`src/utils/regime_weights.py` は「転送: 2026-10-25 設計書 §5。PR 5 で消す」の docstring と、`from src.domain.regime_weights import REGIME_SECTOR_WEIGHTS as REGIME_SECTOR_WEIGHTS` などの明示の再輸出だけにする。`simulation.py` と `trading/signal_generator.py` の import を `src.domain.regime_weights` へ書き換える（`trading/signal_generator.py` の再輸出の形 `X as X  # noqa: F401` はそのまま残す）。
- [ ] **Step 4:** `py -m pytest tests/unit/test_utils_shims.py tests/unit/test_signal_generator.py -v` と `lint-imports` を実行し、PASS と3契約の KEPT を確かめる。
- [ ] **Step 5:** コミット `refactor: regime_weights を domain へ移す`

### Task 2: `jp_market_holidays` を `domain/` へ（PR 1 に続けて）

**Files:**
- Create: `python/src/domain/jp_market_holidays.py`（中身をそのまま移す）
- Modify: `python/src/utils/jp_market_holidays.py`（転送だけにする。`_HOLIDAYS_BY_YEAR` は載せない）
- Modify: `python/src/market_data/pipeline.py:42`、`python/src/prediction/predict_unified.py:24`、`python/src/trading/daily_loss_window.py:7`
- Modify: `python/tests/unit/test_jp_market_holidays.py:5`、`python/tests/unit/test_predict_single_stale_bars.py:14`、`python/tests/unit/test_predict_unified_stale_features.py:16`
- Test: `python/tests/unit/test_utils_shims.py`

**Interfaces:**
- Produces: `src.domain.jp_market_holidays.is_jp_market_holiday(day: date) -> bool`、`is_jp_business_day(day: date) -> bool`、`_HOLIDAYS_BY_YEAR: dict[int, frozenset[date]]`

- [ ] **Step 1: 失敗するテストを足す**

```python
def test_jp_market_holidays_shim_reexports_domain_functions():
    import src.domain.jp_market_holidays as new
    import src.utils.jp_market_holidays as old
    assert old.is_jp_market_holiday is new.is_jp_market_holiday
    assert old.is_jp_business_day is new.is_jp_business_day
```

- [ ] **Step 2:** `py -m pytest tests/unit/test_utils_shims.py -v` を実行し、新しいテストが `ModuleNotFoundError` で FAIL することを確かめる。
- [ ] **Step 3:** 本体を移し、転送を作り、上の6か所の import を `src.domain.jp_market_holidays` へ書き換える（`test_jp_market_holidays.py` の `_HOLIDAYS_BY_YEAR` も新しい場所から import する）。
- [ ] **Step 4:** `py -m pytest tests/unit/test_utils_shims.py tests/unit/test_jp_market_holidays.py tests/unit/test_predict_single_stale_bars.py tests/unit/test_predict_unified_stale_features.py -v` が PASS。
- [ ] **Step 5:** コミット `refactor: jp_market_holidays を domain へ移す`

### Task 3: `filter_by_sector_cap` を `domain/sector_cap.py` へ（PR 1 の最後）

**Files:**
- Create: `python/src/domain/sector_cap.py`（`filter_by_sector_cap` と、それが使う `TypeVar` `T` だけを移す）
- Modify: `python/src/utils/sector_constraints.py`（`filter_by_sector_cap` を消し、`from src.domain.sector_cap import filter_by_sector_cap as filter_by_sector_cap  # noqa: F401` で再輸出。`get_symbol_sector` などの yfinance の部分はそのまま）
- Modify: `python/src/backtest/portfolio/simulation.py:20`、`python/src/trading/capital_allocator.py:25`、`python/src/trading/execution/selection.py:14`（`filter_by_sector_cap` は domain から、`get_symbol_sector` は utils から import する形に分ける）
- Modify: `python/tests/unit/test_sector_constraints.py:4`（`filter_by_sector_cap` を domain から import する）
- Test: `python/tests/unit/test_utils_shims.py`

**Interfaces:**
- Produces: `src.domain.sector_cap.filter_by_sector_cap(items: Iterable[T], max_sector_positions: int, sector_getter: Callable[[T], str], initial_counts: Mapping[str, int] | None = None) -> list[T]`

- [ ] **Step 1: 失敗するテストを足す**

```python
def test_sector_constraints_shim_reexports_domain_filter():
    import src.domain.sector_cap as new
    import src.utils.sector_constraints as old
    assert old.filter_by_sector_cap is new.filter_by_sector_cap
```

- [ ] **Step 2:** `py -m pytest tests/unit/test_utils_shims.py -v` で FAIL を確かめる。
- [ ] **Step 3:** 移して、上の import を書き換える。`utils/sector_constraints.py` で使われなくなった `Counter`・`Iterable`・`Mapping`・`TypeVar` の import を消す（`Callable` も使われなければ消す）。
- [ ] **Step 4:** `py -m pytest tests/unit/test_utils_shims.py tests/unit/test_sector_constraints.py -v` が PASS。続けて `.\check-ci.ps1` が全部 PASS。
- [ ] **Step 5:** コミット `refactor: filter_by_sector_cap を domain へ移す`
- [ ] **Step 6:** push して `develop` 向けに PR 1 を作り、CI が緑なら人間の承認を待たずにマージしてよい（StockFixer の修正は AI の判断でマージしてよいという人間の方針。実口座の作業ではないため）。

### Task 4: `SignalGenerator` の判断を `domain/signal_rules.py` へ（PR 2 `refactor/signal-rules-domain`）

先に Task 0 を行う。

**Files:**
- Create: `python/src/domain/signal_rules.py`
- Modify: `python/src/utils/signal_generator.py`（`generate_signal` が新しい関数を呼ぶだけになる。コンストラクタと `config.settings` の import は残す）
- Test: `python/tests/unit/test_signal_rules.py`（新規）

**Interfaces:**
- Produces: `src.domain.signal_rules.generate_trade_signals(data: pd.DataFrame, prediction: pd.Series, base_threshold: float, volume_window_days: int, volume_multiplier: float, rolling_std: pd.Series | None = None) -> pd.Series`
- `SignalGenerator.generate_signal(data, prediction, rolling_std=None)` は `generate_trade_signals(data, prediction, self.base_threshold, VOLUME_FILTER_WINDOW_DAYS, VOLUME_FILTER_MULTIPLIER, rolling_std)` を返す。

- [ ] **Step 1: 失敗するテストを書く**（`test_signal_rules.py`。純粋関数とクラスが同じ出力を返すこと）

```python
import numpy as np
import pandas as pd
import pytest

from config.settings import VOLUME_FILTER_MULTIPLIER, VOLUME_FILTER_WINDOW_DAYS
from src.domain.signal_rules import generate_trade_signals
from src.utils.signal_generator import SignalGenerator


def _frame(with_rsi: bool, with_volume: bool) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    idx = pd.date_range("2026-01-01", periods=60, freq="B")
    cols = {}
    if with_rsi:
        cols["RSI"] = rng.uniform(10, 90, len(idx))
    if with_volume:
        cols["Volume"] = rng.integers(1_000, 10_000, len(idx)).astype(float)
    return pd.DataFrame(cols, index=idx)


@pytest.mark.parametrize("with_rsi", [True, False])
@pytest.mark.parametrize("with_volume", [True, False])
@pytest.mark.parametrize("with_std", [True, False])
def test_pure_function_matches_signal_generator(with_rsi, with_volume, with_std):
    data = _frame(with_rsi, with_volume)
    rng = np.random.default_rng(1)
    prediction = pd.Series(rng.normal(0, 0.01, len(data)), index=data.index)
    rolling_std = prediction.rolling(5).std() if with_std else None

    expected = SignalGenerator(base_threshold=0.005).generate_signal(data, prediction, rolling_std)
    actual = generate_trade_signals(
        data, prediction, 0.005, VOLUME_FILTER_WINDOW_DAYS, VOLUME_FILTER_MULTIPLIER, rolling_std
    )
    pd.testing.assert_series_equal(actual, expected)


def test_zero_average_std_falls_back_to_fixed_threshold():
    data = _frame(False, False)
    prediction = pd.Series(0.006, index=data.index)
    rolling_std = pd.Series(0.0, index=data.index)
    result = generate_trade_signals(data, prediction, 0.005, 20, 1.5, rolling_std)
    assert (result == "Buy").all()
```

- [ ] **Step 2:** `py -m pytest tests/unit/test_signal_rules.py -v` を実行し、`ModuleNotFoundError: No module named 'src.domain.signal_rules'` で FAIL を確かめる。
- [ ] **Step 3:** `generate_trade_signals` を実装する。今の `generate_signal` の本体をそのまま移し、`self.base_threshold`・`VOLUME_FILTER_WINDOW_DAYS`・`VOLUME_FILTER_MULTIPLIER` を引数に置き換える。docstring とコメントは移す。`SignalGenerator.generate_signal` は Interfaces のとおり新しい関数を呼ぶだけにする（docstring は残す）。`utils/signal_generator.py` の docstring に「判断の本体は `src.domain.signal_rules`。このクラスは optimal_params と設定の値を渡す入れ物」と書く。
- [ ] **Step 4:** `py -m pytest tests/unit/test_signal_rules.py tests/unit/test_signal_generator.py -v` が PASS。続けて `.\check-ci.ps1` が全部 PASS。
- [ ] **Step 5:** コミット `refactor: 売買シグナルの判断を domain の純粋関数へ移す`、push して PR 2 を作り、CI が緑ならマージする。

### Task 5: `drift_thresholds` を `reporting/` へ（PR 3 `refactor/drift-thresholds-reporting`）

先に Task 0 を行う。転送は置かない（設計書 §5）。

**Files:**
- Create: `python/src/reporting/drift_thresholds.py`（中身をそのまま移す）
- Delete: `python/src/utils/drift_thresholds.py`
- Modify: `python/src/orchestration/jobs/drift_check.py:9`、`python/src/reporting/kpi.py:16`、`python/src/reporting/monthly.py:24`

**Interfaces:**
- Produces: `src.reporting.drift_thresholds.DriftThresholds`、`DriftSelection`、`load_drift_thresholds() -> DriftThresholds`、`select_drifted_symbols(summary: pd.DataFrame, thresholds: DriftThresholds) -> DriftSelection`

- [ ] **Step 1:** 古い場所を指す文字列が残っていないか、移す前に一覧にする。
  Run: `git grep -n "utils.drift_thresholds\|utils import drift_thresholds" -- src tests`
  Expected: 上の3か所だけ。`patch("src.utils.drift_thresholds...` が出たら、その patch も Step 2 で書き換える対象に入れる。
- [ ] **Step 2:** `git mv src/utils/drift_thresholds.py src/reporting/drift_thresholds.py` で移し、Step 1 で出た箇所をすべて `src.reporting.drift_thresholds` へ書き換える。
- [ ] **Step 3:** `git grep -n "utils.drift_thresholds" -- src tests` が何も出さないこと、`lint-imports` で3契約が KEPT であることを確かめる。
- [ ] **Step 4:** `.\check-ci.ps1` が全部 PASS。
- [ ] **Step 5:** コミット `refactor: drift_thresholds を reporting へ移す`、push して PR 3 を作り、CI が緑ならマージする。

### Task 6: 戻らない仕組み（PR 4 `refactor/utils-allowlist`）

**Files:**
- Create: `python/tests/unit/test_utils_allowlist.py`
- Modify: `CLAUDE.md`（Architecture の `utils/` の行と Key Rules）
- Modify: `python/.importlinter`（Contract 1 の注釈の「型とポートのみ」の行）

**Interfaces:**
- Produces: `ALLOWED_UTILS_MODULES: dict[str, str]`（`src/utils/` からの相対パス → 置いてよい理由）と `TEMPORARY_SHIMS: dict[str, str]`（同、一時的に許す理由）。Task 7 がこの2つを書き換える。

- [ ] **Step 1: テストを書く**

```python
"""utils/ に業務のモジュールが戻らないよう止める（2026-10-25 設計書 §6）。"""

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"

ALLOWED_UTILS_MODULES: dict[str, str] = {
    # 今の src/utils/ の .py を全部、理由つきで並べる（db/ の下も含む。migrations/ の .sql は対象外）
}
TEMPORARY_SHIMS: dict[str, str] = {
    "regime_weights.py": "転送: PR 5 で消す",
    "jp_market_holidays.py": "転送: PR 5 で消す",
}


def _utils_modules() -> set[str]:
    root = SRC / "utils"
    return {p.relative_to(root).as_posix() for p in root.rglob("*.py") if "__pycache__" not in p.parts}


def test_no_unlisted_module_in_utils():
    unlisted = _utils_modules() - ALLOWED_UTILS_MODULES.keys() - TEMPORARY_SHIMS.keys()
    assert not unlisted, f"utils/ に許可リストに無いモジュールがある: {sorted(unlisted)}"


def test_allowlist_has_no_stale_entries():
    stale = (ALLOWED_UTILS_MODULES.keys() | TEMPORARY_SHIMS.keys()) - _utils_modules()
    assert not stale, f"許可リストに消えたモジュールが残っている: {sorted(stale)}"


def test_domain_imports_nothing_outside_domain():
    bad = []
    for path in (SRC / "domain").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else (
                [node.module or ""] if isinstance(node, ast.ImportFrom) and node.level == 0 else []
            )
            for name in names:
                top = name.split(".")[0]
                if top == "config" or (top == "src" and not name.startswith("src.domain")):
                    bad.append(f"{path.name}: {name}")
    assert not bad, f"domain が外を import している: {bad}"
```

  `ALLOWED_UTILS_MODULES` には、その時点の `src/utils/` の全 `.py`（`__init__.py` を含む `db/` 配下も）を、1行の理由つきで載せる。`ml_base.py`・`sector_constraints.py`・`signal_generator.py`・`optimal_params_loader.py` の理由には「外とのやり取りを持つため残す。後続: 設計書 §8」と書く。
- [ ] **Step 2:** わざと `src/utils/_probe.py` を空で作って `py -m pytest tests/unit/test_utils_allowlist.py -v` を実行し、`test_no_unlisted_module_in_utils` が FAIL することを確かめてから `_probe.py` を消す。もう一度実行して3件とも PASS。
- [ ] **Step 3:** CLAUDE.md を直す。Architecture の図の `utils/` の行を `utils/            道具だけ（DB, logging, retry, path, 外部 API の薄いクライアント）` に、`domain/` の行に「＋ルールの表・暦・純粋なルール関数」を足す。Key Rules に次の一文を足す:
  `- **utils/ に業務の処理を置かない**: 売買の判断・表・暦は domain/、1つの BC しか使わない処理はその BC へ。utils/ に新しいモジュールを足すときは tests/unit/test_utils_allowlist.py に理由を書く。`
  Shared Kernel の箇条に `domain/regime_weights.py`・`domain/jp_market_holidays.py`・`domain/sector_cap.py`・`domain/signal_rules.py` を1行ずつ足す。
- [ ] **Step 4:** `.importlinter` の Contract 1 の注釈の「src.domain は型とポートのみを持つ共有カーネルであり」を「src.domain は型・ポート・ルール定数・例外・純粋なルール関数を持つ共有カーネルであり」に直す。`lint-imports` が KEPT のまま。
- [ ] **Step 5:** `.\check-ci.ps1` が全部 PASS（CLAUDE.md を触るので pre-commit の Doc Drift も通ること）。コミット `test: utils の許可リストと domain の import の検査を足す`、push して PR 4 を作り、CI が緑ならマージする。

### Task 7: 転送を消す（PR 5 `refactor/remove-utils-shims`。PR 3 のマージから1週間後を目安）

**Files:**
- Delete: `python/src/utils/regime_weights.py`、`python/src/utils/jp_market_holidays.py`
- Modify: `python/src/utils/sector_constraints.py`（`filter_by_sector_cap` の再輸出の行を消す）
- Modify: `python/tests/unit/test_utils_allowlist.py`（`TEMPORARY_SHIMS` を空にする）
- Delete: `python/tests/unit/test_utils_shims.py`
- Modify: その時点で古い場所を import している `src/`・`tests/` のファイル

- [ ] **Step 1:** `git fetch && git switch develop && git pull --ff-only` の後、古い場所を使っている箇所を一覧にする。
  Run: `git grep -nE "utils(\.| import )(regime_weights|jp_market_holidays)|utils\.sector_constraints import .*filter_by_sector_cap" -- src tests`
  Expected: 一覧が出る（無くてもよい）。開いた PR も Task 0 の Step 2 の式で確かめ、該当があれば PR 番号を PR 5 の本文に書く。
- [ ] **Step 2:** Step 1 の箇所を新しい場所へ書き換え、上の Files を消す・直す。
- [ ] **Step 3:** Step 1 のコマンドが何も出さないこと、`py -m pytest tests/unit/test_utils_allowlist.py -v` が PASS（`stale` にならない）ことを確かめる。
- [ ] **Step 4:** `.\check-ci.ps1` が全部 PASS。コミット `refactor: utils の転送を消す`、push して PR 5 を作り、CI が緑ならマージする。
