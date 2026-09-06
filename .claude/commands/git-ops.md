Git 操作を Bash ツールで直接実行する（生 git コマンド版）。GitKraken MCP を使った運用フロー・Issue/PR操作・コミットメッセージ規約・ブランチ命名規約は [`.claude/skills/git-ops/SKILL.md`](../skills/git-ops/SKILL.md) と `CLAUDE.md` の Git Workflow を参照（ここには複製しない）。

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
git add <ファイル>
git commit -m "fix: データ取得エラーハンドリングを改善"
```

### Push & PR
```bash
git push origin feature/new-feature

# PR 本文テンプレートは docs/VERSIONING_POLICY.md セクション3を参照してコピー
gh pr create --title "feat: <概要>" --body "$(cat <<'EOF'
（テンプレートをここに貼る）
EOF
)"
```

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
git log --oneline --decorate -10      # 最近の履歴
git diff HEAD~1                       # 直前のコミットとの差分
git diff origin/develop..HEAD         # ブランチ間差分
git blame <ファイル>                   # 行ごとの変更者
git stash push -m "WIP: 説明"          # 一時退避
git stash pop                         # 退避を復元
git worktree list                     # ワークツリー一覧
```

## コンフリクト発生時
```bash
git status                              # コンフリクトファイル確認
# → 手動でファイルを編集して解決
git add <解決済みファイル>
git commit                              # マージコミット完了
```
