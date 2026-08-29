# Test Suite Hardening Plan B (DB接続再利用 + pytest-xdist opt-in) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** unit テストの実行時間の大半を占める「テストごとの Postgres 接続確立コスト」を排除し、pytest-xdist によるローカル並列実行をオプトインで使えるようにする。

**Architecture:** `tests/unit/conftest.py` の `_isolate_db` フィクスチャを、テストごとに `psycopg.connect()`/`close()` する方式から、セッションスコープで1本の接続を張って使い回す方式へ変更する。分離の仕組み自体（トランザクション開始→ロールバック）は変更しない。分離契約が壊れていないことを機械的に検証する回帰テストを新設する。並列実行は `pytest-xdist` を `requirements-dev.txt` に追加し README にオプトインの使い方を記載するのみに留め、CIワークフローは一切変更しない。

**Tech Stack:** Python, pytest, psycopg3, Postgres (docker-compose `postgres-test` サービス)

**Spec:** なし（本セッション内の調査に基づく直接発注。根拠となる実測値は本ドキュメントの Global Constraints と Task 1 に記載する）

## Global Constraints

- 実測根拠: raw `psycopg.connect()` + `SELECT 1` + `rollback()` + `close()` のサイクルは N=50 サンプルで中央値 約23.33ms、平均 約20.22ms。unit テストは2686件あり、テストごとの接続確立コストの合計は概算で **約54秒**（unit テスト全体の実行時間 約97〜100秒の半分近く）。
- `tests/e2e/conftest.py` には一切手を加えない（フィクスチャのライフサイクルが繊細で、Plan A から継続する制約）。
- `.github/workflows/*.yml` は一切変更しない。pytest-xdist はこのプランではローカルのオプトイン手段としてのみ導入し、CI へは配線しない（並列実行下でのテスト間コリジョン——例: 複数テストファイルが同じ固定シンボル名`"TESTBT"`等を使う——のリスクが未検証のため）。
- `_isolate_db` の分離セマンティクス（テストごとにトランザクションを開始してロールバックする）は変更しない。変更するのは接続の生成・破棄のタイミングのみ。
- このリポジトリは `.git` を複数 worktree で共有している。`git stash` / `git stash pop` を使う前は必ず `git stash list` で既存のスタッシュ（無関係な過去セッションのものを含む）を確認すること。
- サブエージェントを dispatch する際は `isolation: "worktree"` パラメータを使わないこと。作業ディレクトリはプロンプト本文で明示的に指定する。

---

### Task 1: `_isolate_db` の共有接続化 + 分離契約の回帰テスト + 実測

**Files:**
- Modify: `python/tests/unit/conftest.py`
- Create: `python/tests/unit/test_db_isolation_guard.py`

**Interfaces:**
- Consumes: `src.utils.db._connection.set_test_connection`, `close_connection`（既存、シグネチャ変更なし）。`src.utils.data_path_utils.get_database_url`（既存）。`src.utils.db.system_config.get_config_value`, `set_config_value`（既存、シグネチャ変更なし）。
- Produces: 新しいセッションスコープ fixture `_shared_test_connection`（他のフィクスチャや後続タスクから参照されない、このタスク内で閉じた変更）。

- [ ] **Step 1: 変更前のベースライン実行時間を記録する**

`python/` ディレクトリで以下を実行し、末尾のサマリ行（`=== N passed, ... in X.XXs ===` のような行）をそのままメモしておく。この後 Step 7 で変更後の数値と比較する。

Run:
```powershell
cd python
python -m pytest tests/unit/ -q --cov=src --cov-branch --cov-report=term-missing --cov-fail-under=80
```

Expected: 現行 develop 相当のテストの結果（`test_strategy_factory.py::TestRunFactoryBatch` の3件がタイムゾーン依存で日本時間のマシンではローカルで失敗することが既知。これは本タスクと無関係な既存の不具合であり、このタスクで修正しない）。ここで得られる **pass数・fail数・実行時間** を記録する。

- [ ] **Step 2: `tests/unit/conftest.py` の DB 隔離セクションを書き換える**

`python/tests/unit/conftest.py` 内、以下の既存コード（`_test_database_ready` の直後にある `_isolate_db` フィクスチャ全体）を:

```python
@pytest.fixture(autouse=True)
def _isolate_db(_test_database_ready):
    """全 unit テストを1トランザクションに包み、テスト終了時にロールバックする。

    _connection.py は呼び出し側が commit() しない設計のため、共有接続を
    そのままロールバックするだけで全ての書き込みを巻き戻せる
    （#548: 本番DB破損事故 / PR#556: filelock起因のCI一斉失敗、両方の
    事故クラスがこの方式では構造的に発生しなくなる）。
    """
    import psycopg

    from src.utils.data_path_utils import get_database_url
    from src.utils.db._connection import close_connection, set_test_connection

    con = psycopg.connect(get_database_url(), autocommit=False)
    # 接続を明示的にトランザクション開始状態にしてから注入する。こうしないと
    # _connection.py 側で `with con.transaction():` のようなネスト保護を
    # 使った場合、テスト最初の呼び出しが「ネストではなく最外殻」と誤認されて
    # 誤ってCOMMITしてしまう（テスト分離が壊れ、本物のPostgresへ書き込みが
    # 漏れる）。SELECT 1 で最初のトランザクションを確実に開始させておく。
    con.execute("SELECT 1")
    set_test_connection(con)
    try:
        yield
    finally:
        con.rollback()
        set_test_connection(None)
        con.close()
        close_connection()
```

次の内容へ**そっくり置き換える**（フィクスチャを1つ追加し、`_isolate_db` の中身を接続の使い回し版へ変更する）:

```python
@pytest.fixture(scope="session")
def _shared_test_connection(_test_database_ready):
    """テストセッション全体で使い回す共有DB接続。

    従来は各テストで psycopg.connect() → rollback() → close() を繰り返しており、
    接続確立コスト（実測: 1回あたり中央値約23ms、2686件で合計約54秒）が
    unitテスト全体の実行時間の半分近くを占めていた。このフィクスチャで接続
    そのものはセッション中1本だけ張り、テストごとの分離は `_isolate_db` 側の
    「テストごとにトランザクションを開始してロールバックする」という既存の
    仕組みでそのまま担保する（分離の強度は変えず、接続確立の回数だけを減らす）。
    """
    import psycopg

    from src.utils.data_path_utils import get_database_url

    con = psycopg.connect(get_database_url(), autocommit=False)
    yield con
    con.close()


@pytest.fixture(autouse=True)
def _isolate_db(_shared_test_connection):
    """全 unit テストを1トランザクションに包み、テスト終了時にロールバックする。

    _connection.py は呼び出し側が commit() しない設計のため、共有接続を
    そのままロールバックするだけで全ての書き込みを巻き戻せる
    （#548: 本番DB破損事故 / PR#556: filelock起因のCI一斉失敗、両方の
    事故クラスがこの方式では構造的に発生しなくなる）。

    接続自体は `_shared_test_connection`（セッションスコープ）で使い回し、
    ここではテストごとに「トランザクションの開始（SELECT 1）」と
    「ロールバック」だけを行う。
    """
    from src.utils.db._connection import close_connection, set_test_connection

    con = _shared_test_connection
    # 接続を明示的にトランザクション開始状態にしてから注入する。こうしないと
    # _connection.py 側で `with con.transaction():` のようなネスト保護を
    # 使った場合、テスト最初の呼び出しが「ネストではなく最外殻」と誤認されて
    # 誤ってCOMMITしてしまう（テスト分離が壊れ、本物のPostgresへ書き込みが
    # 漏れる）。SELECT 1 で毎テストの最初のトランザクションを確実に開始させておく。
    con.execute("SELECT 1")
    set_test_connection(con)
    try:
        yield
    finally:
        con.rollback()
        set_test_connection(None)
        close_connection()
```

`_isolate_db` の finally 節から `con.close()` を削除した点に注意すること（接続はセッション終了時に `_shared_test_connection` が1回だけ閉じる。`close_connection()` の呼び出しはそのまま残す——これは本番用の別プールを管理する関数で、`_test_connection` とは無関係なので変更不要）。

- [ ] **Step 3: 分離契約の回帰テストを新規作成する**

`python/tests/unit/test_db_isolation_guard.py` を新規作成し、以下の内容をそのまま書く:

```python
"""_isolate_db（tests/unit/conftest.py）の分離契約を検証する回帰テスト。

DB接続をテストセッション全体で使い回す方式（接続確立コストの削減が目的）に
変更した際、「あるテストで書き込んだ値が、次のテストへ漏れ出さないこと」が
壊れていないかを機械的に確認する。

このファイル内の2関数は必ず test_a → test_b の順で実行される想定。
pytest はデフォルトでファイル内の定義順にテストを収集するため、この2関数名は
アルファベット順・定義順のどちらで並んでも同じ順序になるよう
`test_a_`/`test_b_` というプレフィックスを付けている。リネームする場合は
この順序依存を壊さないよう注意すること。
"""

from src.utils.db.system_config import get_config_value, set_config_value

_MARKER_KEY = "_isolation_guard_marker"


def test_a_write_marker():
    set_config_value(_MARKER_KEY, "should_not_persist")


def test_b_marker_not_visible_in_next_test():
    value = get_config_value(_MARKER_KEY)
    assert value is None, (
        "前のテスト(test_a_write_marker)で書き込んだ値がロールバックされずに"
        "残っている。DB分離(_isolate_db)が壊れている可能性がある。"
    )
```

- [ ] **Step 4: 回帰テスト単体を実行し、分離が機能していることを確認する**

Run:
```powershell
python -m pytest tests/unit/test_db_isolation_guard.py -v
```

Expected: `2 passed`。`test_b_marker_not_visible_in_next_test` が失敗する場合、共有接続化によって分離契約が壊れているということなので、Step 2 の変更を見直すこと（自己解決を試みてよいが、原因を特定できない場合はブロッカーとして報告する）。

- [ ] **Step 5: DBに触れないテストで「未オープンのトランザクションに対する rollback」が問題を起こさないことを確認する**

`_shared_test_connection` は接続をセッション開始時に1回作るが、実際にDBへ触れないテスト（例: 純粋なロジックのみのテスト）でも `_isolate_db` の finally 節で `con.rollback()` が毎回呼ばれる。psycopg3 はトランザクション未開始の接続への `rollback()` を許容するはずだが、念のため実際のテストファイルで確認する。

Run:
```powershell
python -m pytest tests/unit/test_domain_types.py -v
```

Expected: 全件 `PASSED`。エラーや警告（`rollback()` に起因するもの）が出力に含まれないこと。

- [ ] **Step 6: unit テストスイート全体を実行し、既存の失敗件数から増えていないことを確認する**

Run:
```powershell
python -m pytest tests/unit/ -q --cov=src --cov-branch --cov-report=term-missing --cov-fail-under=80
```

Expected: Step 1 で記録した pass数・fail数と**一致する**こと（`test_strategy_factory.py::TestRunFactoryBatch` の3件の既知のタイムゾーン依存失敗を除き、新たな失敗が発生していないこと）。カバレッジゲート（80%以上）も通ること。

- [ ] **Step 7: 変更前後の実行時間を記録する**

Step 1 と Step 6 で得られたサマリ行（pass数・fail数・実行時間）を両方とも実装レポートに記載する。短縮幅が事前見積もりの約54秒を大きく下回っていても構わないが、その場合は正直にその実測値を報告すること（見積もりを結論として書き直さない）。

- [ ] **Step 8: コミット**

```bash
git add python/tests/unit/conftest.py python/tests/unit/test_db_isolation_guard.py
git commit -m "perf: unitテストのDB接続をセッション全体で使い回し接続確立コストを削減"
```

---

### Task 2: pytest-xdist のオプトイン導入（CI配線なし）

**Files:**
- Modify: `python/requirements-dev.txt`
- Modify: `python/tests/README.md`

**Interfaces:**
- Consumes: なし（Task 1 の変更と独立）。
- Produces: なし（ドキュメントと依存関係の追加のみ）。

- [ ] **Step 1: `pytest-xdist` を依存関係に追加する**

`python/requirements-dev.txt` の `# テスト` セクション（`pytest>=9.1.1` 等が並んでいる箇所）に1行追加する。変更前:

```
# テスト
pytest>=9.1.1
pytest-timeout==2.4.0
pytest-cov==7.1.0
hypothesis>=6.165.2
```

変更後:

```
# テスト
pytest>=9.1.1
pytest-timeout==2.4.0
pytest-cov==7.1.0
pytest-xdist>=3.6.1
hypothesis>=6.165.2
```

インストールする:

```powershell
pip install -r requirements-dev.txt
```

- [ ] **Step 2: `-n auto` で一度実行し、結果をそのまま記録する**

Run:
```powershell
python -m pytest tests/unit/ -n auto -q
```

Expected: 実行が完了すること。**この時点で赤（新規の失敗やエラー、ワーカークラッシュ）が出た場合は、このタスク内でデバッグしようとせず、失敗内容をそのまま実装レポートに記載して Step 3 に進まず停止すること**（原因は複数テストファイルが同じ固定シンボル名を使うこと等による並列実行下でのコリジョンが疑われるが、原因調査は本プランのスコープ外。レビューア／プランナー側の判断に委ねる）。

問題なく完走した場合は pass数・fail数・実行時間を記録し、Step 3 に進む。

- [ ] **Step 3: README にオプトインの使い方を記載する**

Step 2 が green だった場合のみ実施する。`python/tests/README.md` の `### ショートカットスクリプト` セクション（172行目付近、`./test.sh e2e-slow` の行の後、`## Unit Test 実装チェックリスト` の見出しの前）に、以下のセクションをそのまま追加する:

```markdown
### 並列実行（オプション・ローカル限定）

`pytest-xdist` を使うと unit テストをプロセス並列で実行できる。ローカルでの
高速化用途のオプトイン手段であり、**CIには配線していない**（複数テスト
ファイルが同じ固定シンボル名を使っている等、並列実行下でのテスト間干渉が
未検証のため）。

```powershell
python -m pytest tests/unit/ -n auto -q
```

失敗が出た場合は並列実行特有の干渉を疑い、まず `-n 0`（直列）で再現するか
確認すること。
```

- [ ] **Step 4: コミット**

```bash
git add python/requirements-dev.txt python/tests/README.md
git commit -m "chore: pytest-xdistをローカルオプトインの並列実行手段として追加"
```
