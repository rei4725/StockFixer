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

### STEP 1: auto-fix を先に走らせる（トークン節約）

black / isort は手動修正不要なツールなので、診断の前に適用してしまう。

```powershell
cd C:\src\StockFixer\python
py -m black .
py -m isort .
```

### STEP 2: pre-commit で全チェックを一括実行

```powershell
cd C:\src\StockFixer
python -m pre_commit run --all-files 2>&1
```

**出力を全部読む。** "Failed" の hook 名と、その下のエラーメッセージをリストアップする。
このリストが「今回修正すべき全問題」になる。絶対に1件ずつ直しながら再実行しない。

### STEP 3: 全問題を把握してから修正

リストアップした問題をパターン分類（後述）し、**全ファイルの修正を終えてから** コミットを1回だけ実行する。

### STEP 4: 最終確認コミット

```powershell
cd C:\src\StockFixer
python -m pre_commit run --all-files 2>&1
# 全パスを確認してからコミット
git add <files>
git commit -m "..."
```

---

## 失敗パターンと修正手順

### [P1] black / isort フォーマット違反

STEP 1 で自動修正済みのはず。まだ出る場合:

```powershell
py -m black <file>
py -m isort <file>
```

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
# 関数定義を確認して新しい引数を把握する
grep -n "def <関数名>" src/<path>.py
```

引数を Optional にしてフォールバック動作を実装するか、呼び出し側に引数を追加する。

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

---

### [P6] pre-commit の arch-violation hook が tests/ を除外しない

**症状**: `python/tests/unit/test_xxx.py` で `[arch-violation] from src.strategy.*` が検出される

**原因**: `.pre-commit-config.yaml` の `exclude: ^(tests/|...)` が git root 相対パスを考慮していない。
`python/tests/` で始まるパスは除外されない。

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
none   # major / minor / patch / none のいずれか

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

## 作業効率化のチェックリスト

コミット前に以下をまとめて確認してトークンを節約する:

```
□ py -m black . && py -m isort .  # 自動修正を先に適用
□ python -m pre_commit run --all-files  # 全問題を一括列挙
□ 問題が複数ある場合: 全ファイルの修正を終えてからコミット（途中コミット禁止）
□ import パスを変えたら: テストの @patch パスも Grep で検索して更新
□ 変数を削除するときは: 該当箇所のみ削除（replace で全置換しない）
```

---

## 既知の CI ジョブと対応するチェック

| CI ジョブ名 | 対応パターン | ローカルコマンド |
|------------|------------|----------------|
| `Lint (black / isort / flake8 / mypy)` | P1, P2 | `py -m black . && py -m isort . && py -m flake8 .` |
| `unit-test > Pylint` | P3, P4 | `py -m pylint src/ --rcfile=.pylintrc --errors-only` |
| `Architecture Contract Check` | P5 | `$env:PYTHONUTF8=1; lint-imports.exe` |
| `Dependency Vulnerability Scan` | P7 | `py -m pip check` |
| `validate-pr-body` | P8 | PR本文を目視確認 |
