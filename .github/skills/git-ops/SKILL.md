---
name: git-ops
description: "Git操作（コミット、ブランチ、push、diff、status等）を実行する。git・バージョン管理・コミット・ブランチ操作・変更確認の話題では必ずこのスキルを使用する。変更のpush・pull・stash・履歴確認が絡む場合も使用する。"
compatibility: "Git 2.0+。GitKraken MCPツール使用。"
---

## Goal
Git操作を効率的に実行し、バージョン管理とコード変更の追跡を行う。

## Available Operations

### 1. ステータス確認
現在の作業ツリーの状態を確認する。
```
mcp_gitkraken_git_status(directory="c:\\src\\StockFixer")
```
- 変更ファイル、ステージング状態、ブランチ情報を表示

### 2. 変更内容の確認
#### コミット履歴
```
mcp_gitkraken_git_log_or_diff(
    directory="c:\\src\\StockFixer",
    action="log",
    since="1 week ago",
    authors=["user@example.com"]
)
```
- `since`/`until`: 期間指定（例: "2 weeks ago", "yesterday", "2024-01-01"）
- `authors`: 特定のコミッター絞り込み

#### 差分確認
```
mcp_gitkraken_git_log_or_diff(
    directory="c:\\src\\StockFixer",
    action="diff",
    revision_range="main..develop"
)
```
- `revision_range`: 比較範囲（例: "HEAD", "main..feature", "abc123", "HEAD~5..HEAD"）

#### ファイル単位の変更履歴（blame）
```
mcp_gitkraken_git_blame(
    directory="c:\\src\\StockFixer",
    file="python/run_data_creation.py"
)
```
- 各行の最終変更者・コミット・日時を表示

### 3. ブランチ操作
#### ブランチ一覧
```
mcp_gitkraken_git_branch(
    directory="c:\\src\\StockFixer",
    action="list"
)
```

#### 新規ブランチ作成
```
mcp_gitkraken_git_branch(
    directory="c:\\src\\StockFixer",
    action="create",
    branch_name="feature/new-feature"
)
```

#### ブランチ切り替え
```
mcp_gitkraken_git_checkout(
    directory="c:\\src\\StockFixer",
    branch="feature/new-feature"
)
```

### 4. ステージング・コミット
#### ファイルをステージング
```
mcp_gitkraken_git_add_or_commit(
    directory="c:\\src\\StockFixer",
    action="add",
    files=["python/run_data_creation.py", "python/src/data/data_fetcher.py"]
)
```
- `files`を省略すると全ファイルをステージング

#### コミット（必ずチェック後に実施）
> **重要**: コミット前に必ず以下のチェックを実行すること。エラーが残った状態でのコミットは禁止。
```powershell
# コミット前チェック（python/ ディレクトリで実行）
cd C:\src\StockFixer
pre-commit run --all-files
```
チェックがすべてパスしたことを確認してからコミットする。
```
mcp_gitkraken_git_add_or_commit(
    directory="c:\\src\\StockFixer",
    action="commit",
    message="fix: データ取得エラーハンドリングを改善",
    files=["python/run_data_creation.py"]
)
```
- `files`を省略すると全ステージング済みファイルをコミット

### 5. リモート操作
#### Push
```
mcp_gitkraken_git_push(
    directory="c:\\src\\StockFixer"
)
```

### 6. 一時退避（stash）
```
mcp_gitkraken_git_stash(
    directory="c:\\src\\StockFixer",
    name="WIP: データパイプライン改善中"
)
```
- 作業中の変更を一時的に退避

### 7. ワークツリー操作
#### ワークツリー一覧
```
mcp_gitkraken_git_worktree(
    directory="c:\\src\\StockFixer",
    action="list"
)
```

#### ワークツリー追加
```
mcp_gitkraken_git_worktree(
    directory="c:\\src\\StockFixer",
    action="add",
    path="c:\\src\\StockFixer-feature",
    branch="feature/new-feature"
)
```

## Common Workflows

### 作業開始前の必須手順（スキップ禁止）
1. **fetch** でリモートの最新状態を取得する
```
mcp_gitkraken_git_fetch(
    directory="c:\\src\\StockFixer"
)
```
2. **status / branch -vv** で現在のブランチと追跡状況を確認する
```
mcp_gitkraken_git_status(directory="c:\\src\\StockFixer")
mcp_gitkraken_git_branch(directory="c:\\src\\StockFixer", action="list")
```
3. ベースブランチ（`develop`）が最新でない場合は pull して同期する
```
mcp_gitkraken_git_pull(directory="c:\\src\\StockFixer")
```
4. 作業ブランチが既に存在する場合は checkout → pull してから作業を再開する

> **なぜ必要か**: fetch なしで作業すると古い状態にコミットを積み、後から大量のコンフリクト解消が発生する。特に複数 PR が並行するときに顕著。

### 新機能開発開始
1. 上記「作業開始前の必須手順」を実行する
2. `git_status` で現在の状態確認
3. `git_branch` (action="create") で新規ブランチ作成
4. `git_checkout` でブランチ切り替え
5. コード変更後、`git_add_or_commit` でコミット

### コード変更前の確認
1. `git_status` で変更ファイル確認
2. `git_log_or_diff` (action="diff") で具体的な差分確認
3. `git_blame` で特定行の変更履歴確認

### コミット前の整理
1. `git_status` で変更ファイル確認
2. `git_add_or_commit` (action="add", files=[...]) で必要なファイルのみステージング
3. **（必須）** pre-commit チェックを実行し、全項目パスを確認
   ```powershell
   cd C:\src\StockFixer
   pre-commit run --all-files
   ```
   - エラーがある場合は修正してから再度 `add` → チェックを繰り返す
   - Black/isort による自動修正が発生した場合は修正後ファイルを再 `add` すること
4. `git_add_or_commit` (action="commit") でコミット
5. `git_push` でリモートに反映

### PR作成前のボディ検証チェック（必須）

PR を作成・更新する前に、以下のセクションが **すべて** 含まれていることを確認すること。
不足があると `PR Body Validation` CI が失敗する。

| セクション見出し | 必須条件 |
|---|---|
| `## version_impact` | `major` / `minor` / `patch` / `none` のいずれか1語 |
| `## version_rationale` | 空・プレースホルダー不可。変更根拠を1文以上記述 |
| `## VERSION 更新` | `version_update_required: yes` または `version_update_required: no` を含む |
| `## VERSION 未更新理由` | **常に見出しが必要**（`version_update_required: yes` の場合は「該当なし」等で可） |

**`version_update_required` と `version_impact` の対応ルール:**

| version_impact | version_update_required | VERSION 未更新理由 |
|---|---|---|
| major / minor / patch | `yes` 必須 | 見出しのみ（「該当なし」等でOK） |
| none | `no` 必須 | 未更新理由を必ず記述 |

**PR ボディテンプレート（コピー用）:**
```markdown
## version_impact

minor

## version_rationale

（変更根拠を記述）

## VERSION 更新

- version_update_required: yes
- version_before: X.Y.Z
- version_after: X.Y.Z

## VERSION 未更新理由

（該当なし。version_update_required: yes のため不要）
```

> **なぜ必要か**: CI の `validate-pr-body` ジョブは4つのセクション見出しを `require_section` で常時チェックする。`version_update_required: yes` でも `## VERSION 未更新理由` 見出しが存在しないと失敗する。

### PRマージ前の必須確認（手動運用）
1. GitHub の PR 画面で Actions タブを開く
2. `Unit Tests` ワークフローが最新コミットで `Success` になっていることを確認
3. 失敗時はログを確認して修正コミットを push し、再実行結果を確認
4. `Unit Tests` が成功するまでマージしない

### requirements*.txt 変更時の追加手順（PR前に必須）
`requirements.txt` または `requirements-dev.txt` を変更した場合は、PR 作成前にローカルで脆弱性スキャンを実行すること。
CI でのみ検出すると手戻りが発生するため、必ずローカルで事前確認する。

```powershell
cd C:\src\StockFixer\python
pip install pip-audit
pip-audit -r requirements.txt
pip-audit -r requirements-dev.txt
```

- 脆弱性が報告された場合: 修正バージョン以上に更新してから PR を作成する
- 問題なければそのまま PR を作成する

> **なぜ必要か**: `security.yml` は PR 時に pip-audit を実行して脆弱性検出でCI FAILする。ローカル確認をスキップすると、PR作成後に修正コミットが増えてレビューが汚れる。

### 一時的に別作業が必要な場合
1. `git_stash` で現在の変更を退避
2. 別作業実施
3. `git_stash` popで退避した変更を復元（ターミナルで `git stash pop` 実行）

## Best Practices

### コミットメッセージ規約
```
<type>: <subject>

<body>
```
- **type**: feat（新機能）, fix（バグ修正）, docs（ドキュメント）, refactor（リファクタリング）, test（テスト）, chore（雑務）
- **subject**: 50文字以内の簡潔な変更内容
- **body**: 必要に応じて詳細説明

例:
```
feat: 統合モデル予測機能を追加

- run_predict.py に--unified-modelオプション追加
- 複数モデルの予測値を平均して出力
- 予測結果をDuckDBに保存
```

### ブランチ命名規約
- `feature/機能名`: 新機能開発
- `fix/バグ内容`: バグ修正
- `refactor/対象`: リファクタリング
- `docs/対象`: ドキュメント更新

### コミット粒度
- 1コミット = 1つの論理的変更単位
- 関係ない変更は別コミットに分離
- 大規模変更は複数の小さなコミットに分割

### PR運用ルール（プラン制限時の代替）
- ブランチ保護の必須ステータスチェックが使えない場合、PRごとに手動で Actions 結果を確認してからマージする
- 最低条件: `Unit Tests` が Success
- 失敗中の PR はマージ禁止

## Troubleshooting

### コンフリクト発生時
1. `git_status` でコンフリクトファイル確認
2. ファイルを手動編集してコンフリクト解決
3. `git_add_or_commit` (action="add") で解決済みファイルをステージング
4. `git_add_or_commit` (action="commit") でマージコミット完了

### 誤ってコミットした場合
```powershell
# 最新コミットを取り消し（変更は保持）
git reset --soft HEAD~1

# 変更も含めて完全に取り消し
git reset --hard HEAD~1
```

### リモートと同期できない場合
```powershell
# リモートの最新を取得
git fetch origin

# 現在のブランチにマージ
git merge origin/develop

# または rebase
git rebase origin/develop
```

## References
- [Git公式ドキュメント](https://git-scm.com/doc)
- [GitHub Flow](https://docs.github.com/ja/get-started/quickstart/github-flow)
- [Conventional Commits](https://www.conventionalcommits.org/ja/)


## コミット前自動レビュー機構（Pre-commit Hooks）

### 概要
Git のコミット前に自動的にコード品質チェック・型検証・フォーマット検査を実行する仕組み。エラーがあるとコミットがブロックされます。

### チェック内容
1. **コードフォーマット（Black）**: PEP 8 スタイルに自動修正
2. **インポート整理（isort）**: Python import をアルファベット順に自動整理
3. **PEP 8 準拠性（Flake8）**: スタイル違反を検出
4. **型安全性（mypy）**: Python型ヒントを検証
5. **コード品質（Pylint）**: 軽量版で致命的エラーを検出
6. **ファイル整備（pre-commit-hooks）**: 末尾改行修正、大ファイルブロック等
7. **コミットメッセージ（カスタムスクリプト）**: Conventional Commits スタイル検証

### セットアップ
```powershell
cd C:\src\StockFixer\python

# 依存パッケージをインストール
pip install -r requirements.txt

# Git hooksを登録
cd ..
pre-commit install
pre-commit install --hook-type commit-msg
```

### 使用方法
コミット時に自動的にチェックが実行され、エラーがある場合は修正して再度コミットしてください。

通常のコミット
```
git add python/run_data_creation.py
git commit -m "fix(data-pipeline): エラーハンドリング改善"
```

手動でレビュー実行
```
# 全ファイルチェック
pre-commit run --all-files

# 特定のhookのみ実行
pre-commit run black --all-files
pre-commit run mypy --all-files
```

## Notes
- `directory` パラメータは常に `c:\\src\\StockFixer` を使用（Windowsパス形式）
- GitKraken MCPツールは基本的なGit操作をカバー
- 高度な操作（rebase -i、cherry-pick等）は `run_in_terminal` でGitコマンド直接実行
- `get_changed_files` ツールで現在の変更ファイルとdiffを取得可能
