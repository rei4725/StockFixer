Git 操作を実行する。Claude Code は git コマンドを Bash ツールで直接実行できる。

## 作業開始前の必須手順（スキップ禁止）
```bash
git fetch
git status        # または git branch -vv
git pull          # ベースブランチが古い場合
```
fetch なしで作業すると古い状態にコミットを積み、後から大量のコンフリクト解消が発生する。

## 通常ワークフロー

### ブランチ作成 & 切り替え
```bash
git checkout -b feature/new-feature   # 作成 & 切り替え
git checkout feature/existing         # 切り替えのみ
git branch -vv                        # ブランチ一覧
```

### ステージング & コミット前チェック（必須）
```bash
# コミット前チェック（エラーが残った状態でのコミットは禁止）
pre-commit run --all-files

# チェックパス後にステージング & コミット
git add python/run_data_creation.py python/src/data/data_fetcher.py
git commit -m "fix: データ取得エラーハンドリングを改善"
```

### コミットメッセージ規約
```
<type>: <subject>   # 50文字以内

<body>（必要に応じて）
```
type: `feat`（新機能）, `fix`（バグ修正）, `docs`, `refactor`, `test`, `chore`

### Push & PR
```bash
git push origin feature/new-feature

# PR 作成
gh pr create --title "feat: <概要>" --body "$(cat <<'EOF'
## version_impact
minor

## version_rationale
（変更根拠を記述）

## VERSION 更新
- version_update_required: yes
- version_before: X.Y.Z
- version_after: X.Y.Z

## VERSION 未更新理由
（該当なし）
EOF
)"
```

## PR ボディ必須セクション（CI の `validate-pr-body` がチェック）
| セクション | 必須条件 |
|---|---|
| `## version_impact` | `major` / `minor` / `patch` / `none` のいずれか1語 |
| `## version_rationale` | 空・プレースホルダー不可、1文以上 |
| `## VERSION 更新` | `version_update_required: yes` または `no` を含む |
| `## VERSION 未更新理由` | **常に見出しが必要**（yes の場合は「該当なし」等でも可） |

## PRマージ前の必須確認
1. Actions タブで `Unit Tests` が最新コミットで **Success** になっていることを確認
2. 失敗時はログ確認 → 修正コミット → 再確認
3. `Unit Tests` が成功するまでマージしない

## requirements*.txt 変更時（PR 前に必須）
```bash
pip install pip-audit
pip-audit -r requirements.txt
pip-audit -r requirements-dev.txt
```

## よく使うコマンド
```bash
git log --oneline --decorate -10         # 最近の履歴
git diff HEAD~1                          # 直前のコミットとの差分
git diff origin/develop..HEAD            # ブランチ間差分
git blame python/src/services/data_pipeline.py  # 行ごとの変更者
git stash push -m "WIP: データパイプライン改善中"   # 一時退避
git stash pop                            # 退避を復元
git worktree list                        # ワークツリー一覧
```

## コンフリクト発生時
```bash
git status                              # コンフリクトファイル確認
# → 手動でファイルを編集して解決
git add <解決済みファイル>
git commit                              # マージコミット完了
```

## ブランチ命名規約
- `feature/機能名`, `fix/内容`, `refactor/対象`, `docs/対象`
