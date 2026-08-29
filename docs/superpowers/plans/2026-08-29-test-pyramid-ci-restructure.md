# Test Pyramid / CI 再分類 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** PRごとに重量級のフルパイプラインテスト（実DB＋実モデル学習）がCIの必須チェックとして毎回・重複実行される構造を解消し、開発ループを高速化する。

**Architecture:** テストの物理配置（`tests/unit` / `tests/integration` / `tests/e2e`）は変更しない。代わりに (1) 既存の `slow` マーカーを実態に合わせて正しく付与し、(2) 重いフィクスチャの重複実行をなくし、(3) CIトリガーとconcurrency設定を見直し、(4) 開発者向けドキュメントから「フルテストを毎回回す」導線を除去する。

**Tech Stack:** pytest, pytest-timeout, GitHub Actions, PostgreSQL(CI service container)

**Spec:** 本セッションの調査結果（`python/tests/e2e/`, `python/tests/integration/`, `.github/workflows/{unit,integration}-tests.yml` の現状分析）。別ドキュメントなし、本プランが仕様を兼ねる。

## Global Constraints

- 既存テストの合格状態を絶対に壊さない（各ステップでローカル実行して確認する）
- `pytest.ini` の `addopts = --timeout=30` は変更しない。個別ファイルの `pytestmark` で上書きされている前提を壊さない
- 物理ファイル移動は行わない（`tests/integration/conftest.py` の autouse `_isolate_db`（関数スコープ）と `tests/e2e/conftest.py` の `e2e_db_env`（モジュール/セッションスコープの独自コネクション管理）が衝突するため、e2e内の3ファイルは e2e ディレクトリに留める）
- ローカル検証には `docker compose up -d postgres`（`check-ci.ps1` と同じ起動法）で実DBを使う

---

### Task 1: CI concurrency設定の追加

**Files:**
- Modify: `.github/workflows/unit-tests.yml`
- Modify: `.github/workflows/integration-tests.yml`

**Interfaces:** なし（ワークフロー設定のみ）

- [ ] **Step 1: 両ファイルの `on:` ブロック直後に concurrency 設定を追加**

`unit-tests.yml`:
```yaml
name: Unit Tests

on:
  pull_request:
  push:
    branches:
      - develop
    paths:
      - "python/**"
      - "!python/experiments/**"
      - ".github/workflows/unit-tests.yml"

concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

jobs:
```

`integration-tests.yml`:
```yaml
name: Integration / E2E Tests

on:
  pull_request:
  push:
    branches:
      - develop
    paths:
      - "python/**"
      - ".github/workflows/integration-tests.yml"
  workflow_dispatch:

concurrency:
  group: ${{ github.workflow }}-${{ github.ref }}
  cancel-in-progress: true

jobs:
```

`workflow_dispatch:` はTask 4のslowジョブを手動発火できるようにするため、このタイミングで追加する。

- [ ] **Step 2: YAML構文確認**

Run: `py -c "import yaml,glob;[yaml.safe_load(open(f, encoding='utf-8')) for f in glob.glob('.github/workflows/*.yml')]"`
Expected: 例外なく終了（構文エラーがあれば `yaml.scanner.ScannerError` 等が出る）

- [ ] **Step 3: コミット**

```bash
git add .github/workflows/unit-tests.yml .github/workflows/integration-tests.yml
git commit -m "ci: PRの重複pushで古いワークフロー実行をキャンセルするconcurrency設定を追加"
```

---

### Task 2: `e2e_db_env` フィクスチャの重複実行解消 + slowマーク付与

**Files:**
- Modify: `python/tests/e2e/conftest.py`
- Modify: `python/tests/e2e/test_full_pipeline.py`
- Modify: `python/tests/e2e/test_backtest_regression.py`
- Modify: `python/tests/e2e/test_data_quality.py`

**Interfaces:**
- Produces: `e2e_db_env` fixture は引き続き `models_dir` / `market` / `symbol` / `ohlcv` を含む dict を yield する（呼び出し側のシグネチャは不変）

**背景:** `e2e_db_env` は現在 `scope="module"` のため、これを使う3ファイル（`test_full_pipeline.py`, `test_backtest_regression.py`, `test_data_quality.py`）がそれぞれ独立に「合成データ投入→特徴量生成→実モデル学習」を重複実行している。同じ乱数シード(42)で生成した同一データに対する重複作業であり、`scope="session"` に上げることで1回にまとめられる。

**既知の副作用と対策（advisorレビュー指摘）:** `test_full_pipeline.py` は `prediction_results` に書き込み、`test_data_quality.py` はそれを読む。現状は別モジュールのため互いのデータは見えない設計だが、session化すると同一トランザクション上に両方のデータが乗り、**実行順序に依存する**ようになる。アルファベット順では `test_backtest_regression.py` → `test_data_quality.py` → `test_full_pipeline.py` の順で収集されるため、`test_data_quality.py` は `test_full_pipeline.py` の書き込み前に走る点は現状と変わらない（question: 元々`test_data_quality.py`は自分でデータを見るだけで`test_full_pipeline.py`の副作用に依存していないか要確認）。Step 4 で「全体実行」と「単体ファイル実行」の両方が通ることを確認する。

- [ ] **Step 1: `e2e_db_env` を `scope="module"` から `scope="session"` に変更**

`python/tests/e2e/conftest.py:94` を編集:
```python
@pytest.fixture(scope="session")
def e2e_db_env(e2e_ohlcv, tmp_path_factory):
```
（`e2e_ohlcv` 自体は `scope="module"` のままだと session スコープの fixture から参照できない — pytest はより狭いスコープの fixture を広いスコープの fixture が要求するとエラーになる。`e2e_ohlcv` も `scope="session"` に上げる。）

`python/tests/e2e/conftest.py:60` も併せて編集:
```python
@pytest.fixture(scope="session")
def e2e_ohlcv() -> pd.DataFrame:
```

docstring中の「モジュール全体で1つの環境を共有する」という記述も実態に合わせて更新:
```python
    """
    E2E テスト用の孤立環境を構築して yield する。

    tests/e2e/ 配下でこのフィクスチャを使う全モジュールが「セッションで1つ」
    の環境を共有する（各テストごと・各モジュールごとにロールバックしない）。
    合成データ投入・特徴量生成・実モデル学習は3ファイル分同一内容のため、
    重複実行を避けるためセッションスコープにしている。
    ...(以下既存の説明を継続)
    """
```

- [ ] **Step 2: 3ファイルの `pytestmark` をリスト化して `slow` を追加**

`test_full_pipeline.py:21`:
```python
pytestmark = [pytest.mark.timeout(300), pytest.mark.slow]
```

`test_backtest_regression.py:24`（既存が `pytest.mark.timeout(300)` の行）:
```python
pytestmark = [pytest.mark.timeout(300), pytest.mark.slow]
```

`test_data_quality.py:20`（既存が `pytest.mark.timeout(120)` の行）:
```python
pytestmark = [pytest.mark.timeout(120), pytest.mark.slow]
```

**注意:** 単純な再代入で `pytest.mark.slow` だけにしてしまうと `timeout` 指定が消え、`pytest.ini` の `addopts = --timeout=30` が効いてしまい重いテストがタイムアウトで落ちる。必ずリストにする。

- [ ] **Step 3: Postgresを起動してローカルでcollect確認**

Run:
```bash
cd python
docker compose up -d postgres
py -m pytest tests/e2e/ --collect-only -q -m "not slow"
py -m pytest tests/e2e/ --collect-only -q -m "slow"
```
Expected: 1つ目は `test_cli_smoke.py` の非slowテストのみが収集される（0件にならないこと＝exit code 5にならないことを確認）。2つ目は3ファイル分のテストが収集される。

- [ ] **Step 4: 実行順序依存の確認（session化の副作用チェック）**

Run:
```bash
py -m pytest tests/e2e/ -v --timeout=300 -m slow
py -m pytest tests/e2e/test_data_quality.py -v --timeout=300
```
Expected: 両方とも全テストPASS。2つ目（単体実行）が失敗する場合、`test_data_quality.py` が `test_full_pipeline.py` の副作用（`prediction_results`書き込み等）に暗黙依存している証拠なので、その依存を無くすか、そのテストだけ独自にデータを用意するよう修正する。

- [ ] **Step 5: コミット**

```bash
git add python/tests/e2e/conftest.py python/tests/e2e/test_full_pipeline.py python/tests/e2e/test_backtest_regression.py python/tests/e2e/test_data_quality.py
git commit -m "test: e2e_db_envをセッションスコープ化し重複実行を解消、重量テストにslowマークを付与"
```

---

### Task 3: `tests/integration/` 内の紛らわしい `_e2e` 命名を整理

**Files:**
- Rename: `python/tests/integration/test_backtest_e2e.py` → `python/tests/integration/test_backtest_pipeline.py`
- Rename: `python/tests/integration/test_backtest_optimize_e2e.py` → `python/tests/integration/test_backtest_optimize.py`
- Rename: `python/tests/integration/test_prediction_pipeline_optimal_params_e2e.py` → `python/tests/integration/test_prediction_optimal_params.py`
- Rename: `python/tests/integration/test_stress_test_e2e.py` → `python/tests/integration/test_stress_test.py`
- Modify: `python/tests/README.md`

**背景:** これらは全て `tests/integration/` 配下にあり、ディレクトリ自体が既に「integration」を表しているため、ファイル名の `_e2e` サフィックスは紛らわしいだけで意味を持たない（コード上の参照は無いことを確認済み。唯一の参照は `python/tests/README.md:126` のドキュメント内コード例）。中身（クラス名・テスト内容）は変更しない、リネームのみ。

- [ ] **Step 1: git mv でリネーム**

```bash
cd python
git mv tests/integration/test_backtest_e2e.py tests/integration/test_backtest_pipeline.py
git mv tests/integration/test_backtest_optimize_e2e.py tests/integration/test_backtest_optimize.py
git mv tests/integration/test_prediction_pipeline_optimal_params_e2e.py tests/integration/test_prediction_optimal_params.py
git mv tests/integration/test_stress_test_e2e.py tests/integration/test_stress_test.py
```

- [ ] **Step 2: README.md内の参照を更新**

`python/tests/README.md:126`:
```markdown
- `test_backtest_optimize.py::TestBacktestOptimizeE2E::test_optimization_metrics_dtype_fix`
```
（クラス名 `TestBacktestOptimizeE2E` はファイル内で変更しないためそのまま）

- [ ] **Step 3: collectで確認**

Run: `py -m pytest tests/integration/ --collect-only -q 2>&1 | tail -5`
Expected: エラーなく収集完了（import違反やファイル名衝突がないこと）

- [ ] **Step 4: コミット**

```bash
git add tests/integration/test_backtest_pipeline.py tests/integration/test_backtest_optimize.py tests/integration/test_prediction_optimal_params.py tests/integration/test_stress_test.py tests/README.md
git commit -m "test: tests/integration配下の紛らわしい_e2eサフィックスを整理"
```

---

### Task 4: `integration-tests.yml` にslow専用ジョブを追加

**Files:**
- Modify: `.github/workflows/integration-tests.yml`

**Interfaces:**
- Consumes: Task 1で追加済みの `workflow_dispatch:` トリガー、Task 2で付与した `slow` マーク

**方針:** 既存の `e2e-test` ジョブ（`-m "not slow"`）はPR・push双方で現状通り実行し続ける（advisorレビュー指摘の通り、こちらは変更不要）。新たに `e2e-test-slow` ジョブを追加し、`pull_request` イベント以外（`push: develop` および手動 `workflow_dispatch`）でのみ、slowマーク付きテストを実行する。

- [ ] **Step 1: `integration-tests.yml` の末尾に新ジョブを追加**

```yaml
  e2e-test-slow:
    if: github.event_name != 'pull_request'
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: python
    services:
      postgres:
        image: postgres:16-alpine
        env:
          POSTGRES_DB: stockfixer
          POSTGRES_USER: stockfixer
          POSTGRES_PASSWORD: stockfixer_ci
        ports:
          - 5432:5432
        options: >-
          --health-cmd "pg_isready -U stockfixer -d stockfixer"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
    env:
      DATABASE_URL: postgresql://stockfixer:stockfixer_ci@localhost:5432/stockfixer

    steps:
      - name: Checkout
        uses: actions/checkout@v7

      - name: Setup Python
        uses: actions/setup-python@v7
        with:
          python-version: "3.12"

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          # mlflow>=3.12.0 は pandas<3 を要求するため requirements.txt に競合がある。
          # mlflow を --no-deps で別途インストールすることで競合を回避する。
          grep -v '^mlflow' requirements.txt | pip install -r /dev/stdin
          pip install "mlflow>=3.12.0" --no-deps
          pip install -r requirements-dev.txt

      - name: Run E2E tests (slow)
        run: python -m pytest tests/e2e/ -v --timeout=300 -m "slow"

      - name: Upload logs on failure
        if: failure()
        uses: actions/upload-artifact@v7
        with:
          name: e2e-test-slow-logs
          path: Logs/
```

- [ ] **Step 2: YAML構文確認**

Run: `py -c "import yaml; yaml.safe_load(open('.github/workflows/integration-tests.yml', encoding='utf-8'))"`
Expected: 例外なし

- [ ] **Step 3: コミット**

```bash
git add .github/workflows/integration-tests.yml
git commit -m "ci: 重量e2eテストをPR必須チェックから外しdevelop push/手動発火専用ジョブへ分離"
```

---

### Task 5: 開発フロー推奨手順からフルテスト連打の記述を除去

**Files:**
- Modify: `python/tests/README.md`

**背景:** `python/tests/README.md:128-142` の「開発フロー推奨例」ステップ2が `python -m pytest tests/ -v`（unit+integration+e2e全部）を「ローカル検証完了」時に毎回回す手順として案内している。これが実質、開発者に手元でe2eを含むフルスイートを繰り返し実行させる唯一のドキュメント上の導線であり、今回の依頼の核心（開発フロー中に何度もe2eをやる仕組みをやめたい）に直接該当する。

- [ ] **Step 1: 該当セクションを実態（層ごとに使い分ける）に合わせて書き換え**

`python/tests/README.md:128-142` を以下に置き換え:
```markdown
## 開発フロー推奨例

```powershell
# 1. 機能開発中：Unit Test を実行（高速フィードバック、外部依存なし）
python -m pytest tests/unit/test_backtester_unit.py -v

# 2. PR 前：ローカルCI相当の一括チェック（unitのみ、check-ci.ps1と同等）
cd python; .\check-ci.ps1

# 3. integration/e2e は PR作成後にCIが自動実行する（PR: unit + integration + 軽量e2e）
#    ローカルで個別に確認したい場合のみ、対象を絞って実行する:
python -m pytest tests/integration/test_xxx.py -v

# 4. 重量級の e2e（実DB + 実モデル学習）は develop push 後 / 手動発火でのみ実行される。
#    ローカルで確認したい場合（Postgres起動が必要）:
docker compose up -d postgres
python -m pytest tests/e2e/ -v --timeout=300 -m "slow"
```
```

- [ ] **Step 2: コミット**

```bash
cd python
git add tests/README.md
git commit -m "docs: 開発フロー推奨例からフルテストスイート連打の導線を除去"
```

---

## 最終確認

- [ ] Task 1〜5 のコミットを積んだブランチで `check-ci.ps1` を実行し、既存のunitテスト・lint類が壊れていないことを確認する
- [ ] PR本文の `version_impact` は `none`（テスト構成・CI設定のみでプロダクトコード非変更のため）とし、`docs/VERSIONING_POLICY.md` の基準と矛盾しないか最終チェックする
- [ ] PR説明に「PRのCI待ち時間がどう変わる見込みか」（例: e2e-testジョブは変わらず速いまま、integration-testジョブは変わらず、developへのマージ後にe2e-test-slowが1回だけ走る）を明記する
