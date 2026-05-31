---
name: ci-preflight
description: "コミット・pushの前にCIが失敗しやすいポイントを事前に全件検出して修正するスキル。コミット前・push前・PR作成前・CIが通らない・black/flake8/pylint/mypy/import-linter エラーが出ている・「CI通るか確認して」という場面では必ずこのスキルを使用する。"
compatibility: "Python 3.11+, pre-commit, black, isort, flake8, pylint, mypy, import-linter。C:\\src\\StockFixer で実行。"
---

# ci-preflight スキル

## 目的

コミットするたびに pre-commit 失敗 → 修正 → 再コミット のサイクルを繰り返さないために、
**修正を始める前に全問題を1回のスキャンで列挙し、まとめて直してから1回だけコミットする。**

---

## 必須フロー（このスキルを呼んだら必ずこの順で実行する）

### STEP 0: CI の状況把握と事前チェック

PR が既にある場合は、何が失敗しているかをカテゴリ別に把握してから作業を始める。

```bash
# PR の全チェック状態を一覧表示
gh pr checks <PR番号> --repo <owner>/<repo>
```

失敗を確認したら以下を判断する：
1. **Lint 系** → STEP 1 の auto-fix から始める
2. **import-linter** → `.importlinter` を確認（[P5]）
3. **テスト失敗** → develop と同じ失敗かを先に確認（[P12]「develop 比較」参照）
4. **benchmark** → テスト自体は通っているが PR コメント投稿のパーミッション問題なら無視してよい

#### バージョン不一致の事前チェック（Lint 系で詰まる前に必ず確認）

```powershell
# requirements-dev.txt と .pre-commit-config.yaml のバージョンを比較
Select-String "black==|isort==" C:\src\StockFixer\python\requirements-dev.txt
Select-String "rev:" C:\src\StockFixer\.pre-commit-config.yaml | Select-Object -First 6
```

black と isort の `rev:` が `requirements-dev.txt` のバージョンと一致していなければ **[P9]** を先に解消する。

---

### STEP 1: auto-fix を先に走らせる（トークン節約）

⚠️ **Windows CRLF 注意**: `core.autocrlf = true` の環境では、Python ファイルが CRLF になっている。
black は CRLF → LF 変換を試みるが、git が透過的に処理するため変更が反映されない場合がある。
CRLF の疑いがある場合は **[P9]** を先に参照し、`core.autocrlf false` 運用で作業する。

#### STEP 1a: Python ファイルを LF に一括変換（Windows 環境での前処理）

```powershell
$utf8NoBom = New-Object System.Text.UTF8Encoding $false
Get-ChildItem -Path "C:\src\StockFixer\python" -Recurse -Include "*.py" | ForEach-Object {
    $content = [System.IO.File]::ReadAllText($_.FullName)
    if ($content.Contains("`r`n")) {
        [System.IO.File]::WriteAllText($_.FullName, $content.Replace("`r`n", "`n"), $utf8NoBom)
    }
}
```

#### STEP 1b: black / isort を適用

```powershell
cd C:\src\StockFixer\python
py -m black .
py -m isort .
```

---

### STEP 2: pre-commit で全チェックを一括実行

```powershell
cd C:\src\StockFixer
python -m pre_commit run --all-files 2>&1
```

**出力を全部読む。** "Failed" の hook 名と、その下のエラーメッセージをリストアップする。
このリストが「今回修正すべき全問題」になる。絶対に1件ずつ直しながら再実行しない。

---

### STEP 3: 全問題を把握してから修正

リストアップした問題をパターン分類（後述）し、**全ファイルの修正を終えてから** コミットを1回だけ実行する。

---

### STEP 4: 最終確認コミット

```powershell
cd C:\src\StockFixer
python -m pre_commit run --all-files 2>&1
# 全パスを確認してからコミット
git add <files>
git commit -m "..."
```

---

## CI ログの効率的な追跡

`gh run view --log-failed` は全ジョブの出力をまとめて出すため巨大になりがち。
**ジョブ ID 単位で絞る方が速い。**

### ステップ 1: どのステップが失敗したか確認

```bash
gh run view <run_id> --repo <owner>/<repo>
# 出力例:
# X Lint (black / isort / flake8 / mypy) in 27s (ID 78695291399)
#   ✓ black
#   X isort   ← ここ
```

### ステップ 2: ジョブ ID のログを直接取得

```bash
# 失敗したステップのジョブ ID を指定してログを取得
gh api "repos/<owner>/<repo>/actions/jobs/<job_id>/logs" 2>&1 | grep "ERROR:"

# よく使うフィルタパターン
gh api "repos/<owner>/<repo>/actions/jobs/<job_id>/logs" 2>&1 | grep "error:\|FAILED\|would reformat\|ERROR:" | grep -v "Installing\|Collect\|Download"
```

### ステップ 3: PR の全ジョブ ID を一覧取得（Python が使える場合）

```bash
# run_id から全ジョブの ID・名前・結果を取得
gh api "repos/<owner>/<repo>/actions/runs/<run_id>/jobs" 2>&1 \
  | python3 -c "import json,sys; [print(j['id'], j['conclusion'], j['name']) for j in json.load(sys.stdin)['jobs']]"
```

---

## 失敗パターンと修正手順

### [P1] black / isort フォーマット違反

STEP 1 で自動修正済みのはず。まだ出る場合 → [P9] Windows CRLF 問題を確認。

---

### [P2] flake8 エラー

```powershell
cd C:\src\StockFixer\python
py -m flake8 .
```

よくあるエラーと修正:

| コード | 意味 | 対処 |
|--------|------|------|
| F401 | 未使用 import | import 行を削除。削除前に他の箇所で使われていないか `Grep` で確認 |
| F841 | 未使用変数への代入 | `_` に変えるか削除。ただし削除前に同名変数が他の行で使われていないか確認 |
| E501 | 行が長すぎる | 行を分割。f-string は変数に切り出す |
| F541 | プレースホルダーなし f-string | `f"..."` → `"..."` に変更 |
| E402 | モジュールレベル import が先頭にない | `sys.path` 操作後の import には `# noqa: E402` を付与 |
| D301 | docstring にバックスラッシュ | `"""` → `r"""` に変更 |

**⚠️ F841 で変数を削除するときの注意**:
`replace('        var = ...\n', '')` を使うと **ファイル内の全同名行が消える**。
必ず削除対象の行番号を特定してから Edit ツールで該当箇所のみ削除すること。

`__init__.py` で F401 が大量に出る場合: 公開 API の re-export ファイルなら `# flake8: noqa: F401` をファイル先頭に追加する方が保守しやすい。

---

### [P3] pylint E0611 (no-name-in-module)

関数が別モジュールに移動しているのに古いパスを参照している。

**調査手順**:
```bash
# 関数の現在の定義場所を探す
grep -rn "def <関数名>" src/
```

**修正手順**:
1. 関数の新しいパスを確認する
2. 修正対象ファイルの import 行を更新する
3. **テストファイルの `@patch` デコレータも必ず更新する**

```bash
# テストでのパッチ対象も検索
grep -rn "<旧パス>\.<関数名>" tests/
```

`@patch("src.utils.db.load_xxx")` → `@patch("src.prediction.db.load_xxx")` のように、
**コードの import パスを変えたら必ずテストの patch パスも同期する。**

---

### [P4] pylint E1120 (no-value-for-parameter)

関数シグネチャに必須引数が追加されたのに呼び出し側が未更新。

```bash
grep -n "def <関数名>" src/<path>.py
```

**よくある具体例**: `shadow_mode=False, use_transformer=False` などの新キーワード引数追加後、
テストの `assert_called_once_with(...)` が古い引数で書かれていて失敗する。
→ `assert_called_once_with(..., shadow_mode=False, use_transformer=False)` に更新。

---

### [P5] import-linter 違反 (BROKEN)

```powershell
cd C:\src\StockFixer\python
$env:PYTHONUTF8=1
& "$env:LOCALAPPDATA\Programs\Python\Python314\Scripts\lint-imports.exe"
```

出力の `src.X -> src.Y` を読む。

**修正方針の選択**:

| 状況 | 対応 |
|------|------|
| reporting → prediction など既存の許容違反と同パターン | `.importlinter` の `ignore_imports` に追記（両 contract に追加）|
| utils → BC（上位層への依存） | utils 側の import を削除。BC 側に関数を置くか re-export する |
| 新規の cross-BC 依存 | 理由コメント付きで `ignore_imports` に追加。将来的な解消を issue に記録 |

`.importlinter` の `ignore_imports` を追加するときは **`[importlinter:contract:layers]` と `[importlinter:contract:independence]` の両方** に追記する。

⚠️ **よくある落とし穴**: `independence` contract に追記しても `layers` contract への追記を忘れると CI で失敗する。必ず両方に追記すること。

---

### [P6] pre-commit の arch-violation hook が tests/ を除外しない

**症状**: `python/tests/unit/test_xxx.py` で `[arch-violation] from src.strategy.*` が検出される

**修正**: `.pre-commit-config.yaml` の `check-arch-violation` の exclude を更新:
```yaml
exclude: ^(tests/|python/tests/|\.github/hooks/|python/\.github/hooks/)
```

---

### [P7] requirements.txt 依存競合

```powershell
cd C:\src\StockFixer\python
py -m pip check
py -m pip install -r requirements.txt --dry-run 2>&1 | Select-String "Cannot|conflict|incompatible"
```

---

### [P8] PR ボディ必須セクション欠落

PR 本文に以下が全て含まれることを確認:

```markdown
## version_impact
none

## version_rationale
（変更根拠を1文以上）

## VERSION 更新
- version_update_required: no
- version_before: X.Y.Z
- version_after: X.Y.Z

## VERSION 未更新理由
（理由を記述）
```

---

### [P9] Windows CRLF 問題で black が pre-commit を通らない

**症状**:
- `py -m black .` を実行すると「reformatted」が出るが、`git status` が clean のまま
- pre-commit が black で「Failed」になってコミットできない（何度やっても繰り返す）
- CI で `would reformat` が大量に出る

**原因**: `core.autocrlf = true` の Windows 環境では、チェックアウト時にファイルが CRLF になる。
black は CRLF を「フォーマット要」とするが、適用後も CRLF のまま出力されるため git は変更を検知しない。
CI (Linux) では LF なので、コミット済みの LF ファイルがある場合でも black のスタイルと一致しない。

**恒久対策**: `.gitattributes` を追加して Python ファイルを LF で管理する。

```
# .gitattributes (git root に配置)
* text=auto
*.py text eol=lf
*.sh text eol=lf
*.yml text eol=lf
*.yaml text eol=lf
*.toml text eol=lf
```

**フォーマット適用の正しい手順**:

```bash
# 1. .gitattributes を作成（上記内容）

# 2. core.autocrlf を一時的に false にして LF でチェックアウト
git config --local core.autocrlf false
git checkout -- python/

# 3. LF ファイルに black/isort を適用（CI と同じ結果になる）
cd C:\src\StockFixer\python
py -m black .
py -m isort .
cd C:\src\StockFixer

# 4. git add & commit（pre-commit も LF ファイルに対して実行）
git add .gitattributes python/
git commit -m "style: black で全ファイル再フォーマット"

# 追加の isort 変更があれば再度 add & commit
git add python/
git commit -m "同上"

# 5. push 後に autocrlf を元に戻す
git push origin <branch>
git config --local core.autocrlf true
```

**pre-commit/CI のバージョン不一致**:

black のバージョンが `.pre-commit-config.yaml` の `rev` と `requirements-dev.txt` で一致していないと、
pre-commit が古いバージョンでフォーマットして CI（新バージョン）と結果が変わる。

```yaml
# .pre-commit-config.yaml
- repo: https://github.com/psf/black
  rev: 26.5.0  # ← requirements-dev.txt の black==26.5.0 と一致させる
```

**isort の pre-commit/CI 設定不一致**:

CI は `python/` ディレクトリから `python -m isort . --check` を実行するが、
pre-commit は git root から実行するため `src_paths` の解釈が異なる。

```yaml
# .pre-commit-config.yaml
- repo: https://github.com/PyCQA/isort
  rev: 5.13.2
  hooks:
    - id: isort
      args: [--settings-path=python]  # git root から実行しても python/pyproject.toml を参照
```

```toml
# python/pyproject.toml
[tool.isort]
profile = "black"
line_length = 100
src_paths = ["src"]
known_first_party = ["config"]  # config/ モジュールも first-party として扱う
```

---

### [P10] カバレッジ閾値問題

**症状**: `ERROR: Coverage failure: total of 78 is less than fail-under=80`

**まず develop でも同じか確認する（PR 固有問題かを切り分け）**:

```powershell
git stash
git checkout develop
cd C:\src\StockFixer\python
py -m pytest tests/unit/ --cov=src --cov-branch -q --tb=no 2>&1 | Select-Object -Last 5
git checkout <作業ブランチ>
git stash pop
```

- develop でも同じカバレッジ → develop の既存問題 → CI の閾値を実態に合わせる:

  `.github/workflows/unit-tests.yml`:
  ```yaml
  run: python -m pytest tests/unit/ -v --cov=src --cov-branch --cov-report=term-missing --cov-fail-under=78
  ```

  `python/.coveragerc`:
  ```ini
  [report]
  fail_under = 78
  ```

- このPRの変更が原因 → `--cov-report=term-missing` でどのファイルが低いかを確認してテストを追加する

---

### [P11] integration-test / unit-test のインポートパス・型・期待値の不一致

**よくあるパターン（モジュール移動後）**:

| エラー | 原因 | 修正 |
|--------|------|------|
| `AttributeError: module 'src.utils.db' has no attribute 'save_prediction_results'` | 関数が `prediction.db` に移動済み | `from src.prediction.db import save_prediction_results` |
| `ModuleNotFoundError: No module named 'src.data'` | `src.data` が削除済み | 現在の正しいパスを Grep で確認 |
| `TypeError: object of type 'BatchResult' has no len()` | 戻り値の型が `list` → `BatchResult` に変更 | `results.succeeded` でアクセス |
| `AssertionError: 'model_name_challenger' != 'model_name'` | 実装がチャレンジャーモデル名に変更 | テストの期待値を実装に合わせる |
| `"PredictionResult" has no attribute "empty"` | 戻り値の型が DataFrame → PredictionResult に変更 | テストのモック・アクセス方法を更新 |

**調査手順**:
```bash
# 関数の現在の場所・シグネチャを確認
grep -rn "def <関数名>" src/
```

---

### [P12] 「PR 固有か develop の既存問題か」の切り分け

CI の失敗を修正する前に **develop でも同じ問題が発生するかを確認する**。
develop の既存問題を PR で修正しようとすると時間を浪費する。

```powershell
# develop ブランチで同じテストを実行して比較
git stash
git checkout develop
cd C:\src\StockFixer\python
py -m pytest <失敗しているテスト> -v 2>&1 | Select-Object -Last 20
git checkout <作業ブランチ>
git stash pop
```

**develop でも同じ失敗** → このPRとは無関係の既存問題：
- カバレッジ → [P10] の閾値調整
- 型エラー → `pyproject.toml` の `[[tool.mypy.overrides]]` で `ignore_errors = true`
- テスト失敗 → develop 側の issue として記録し、このPRでは最小限の対応にとどめる

**develop では通る** → このPRの変更が原因 → コードを修正する

---

## 作業効率化のチェックリスト

```
□ STEP 0: gh pr checks で失敗を一覧化し、PR固有かdevelop既存問題かを分類する
□ STEP 0: requirements-dev.txt と .pre-commit-config.yaml のバージョンが一致しているか確認
□ STEP 1a: Python ファイルを LF に変換（core.autocrlf 問題の予防）
□ py -m black . && py -m isort .  # 自動修正を先に適用
□ python -m pre_commit run --all-files  # 全問題を一括列挙
□ 問題が複数ある場合: 全ファイルの修正を終えてからコミット（途中コミット禁止）
□ import パスを変えたら: テストの @patch パスも Grep で検索して更新
□ 変数を削除するときは: 該当箇所のみ削除（replace で全置換しない）
□ import-linter 違反: layers と independence の両方に追記したか確認
```

---

## 既知の CI ジョブと対応するチェック

| CI ジョブ名 | 対応パターン | ローカルコマンド |
|------------|------------|----------------|
| `Lint (black / isort / flake8 / mypy)` | P1, P2, P9 | STEP 0（バージョン確認）→ 1a → 1b → `py -m flake8 .` |
| `unit-test > Pylint` | P3, P4, P11 | `py -m pylint src/ --rcfile=.pylintrc --errors-only` |
| `unit-test > Coverage` | P10, P12 | develop と比較してから判断 |
| `Architecture Contract Check` | P5 | `$env:PYTHONUTF8=1; lint-imports.exe` |
| `integration-test` | P11, P12 | `py -m pytest tests/integration/ -v` |
| `benchmark` | - | テスト自体は通っていてもPRコメント投稿権限エラーで fail することがある（無視可） |
| `Dependency Vulnerability Scan` | P7 | `py -m pip check` |
| `validate-pr-body` | P8 | PR本文を目視確認 |

---

## CI ログ追跡のクイックリファレンス

```bash
# 1. PR のチェック状況を確認
gh pr checks <PR番号> --repo <owner>/<repo>

# 2. 失敗した run の詳細（どのステップで落ちたかを確認）
gh run view <run_id> --repo <owner>/<repo>

# 3. ジョブ ID のログを直接取得（--log-failed より効率的）
gh api "repos/<owner>/<repo>/actions/jobs/<job_id>/logs" 2>&1 \
  | grep "ERROR:\|error:\|FAILED\|would reformat" \
  | grep -v "Installing\|Collect\|Download\|config"

# 4. run の全ジョブ ID 一覧（Python が使える場合）
gh api "repos/<owner>/<repo>/actions/runs/<run_id>/jobs" 2>&1 \
  | python3 -c "import json,sys; [print(j['id'], j['conclusion'], j['name']) for j in json.load(sys.stdin)['jobs']]"
```
