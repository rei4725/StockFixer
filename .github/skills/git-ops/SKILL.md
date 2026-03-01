---
name: git-ops
description: "Git操作（コミット、ブランチ、push、diff、status等）を実行する。git、バージョン管理、コミット、ブランチ、push、pull、diff、status、checkout、stash、blame、履歴、変更確認の話題で使用する。"
metadata:
  author: StockFixer
  version: "1.0"
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
    revision_range="main..feature/training"
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

#### コミット
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

### 新機能開発開始
1. `git_status` で現在の状態確認
2. `git_branch` (action="create") で新規ブランチ作成
3. `git_checkout` でブランチ切り替え
4. コード変更後、`git_add_or_commit` でコミット

### コード変更前の確認
1. `git_status` で変更ファイル確認
2. `git_log_or_diff` (action="diff") で具体的な差分確認
3. `git_blame` で特定行の変更履歴確認

### コミット前の整理
1. `git_status` で変更ファイル確認
2. `git_add_or_commit` (action="add", files=[...]) で必要なファイルのみステージング
3. `git_add_or_commit` (action="commit") でコミット
4. `git_push` でリモートに反映

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
git merge origin/feature/training

# または rebase
git rebase origin/feature/training
```

## References
- [Git公式ドキュメント](https://git-scm.com/doc)
- [GitHub Flow](https://docs.github.com/ja/get-started/quickstart/github-flow)
- [Conventional Commits](https://www.conventionalcommits.org/ja/)

## Notes
- `directory` パラメータは常に `c:\\src\\StockFixer` を使用（Windowsパス形式）
- GitKraken MCPツールは基本的なGit操作をカバー
- 高度な操作（rebase -i、cherry-pick等）は `run_in_terminal` でGitコマンド直接実行
- `get_changed_files` ツールで現在の変更ファイルとdiffを取得可能
