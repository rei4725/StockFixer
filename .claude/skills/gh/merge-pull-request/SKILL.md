# PRマージ

プルリクエストをマージする際は、GitHub CLI の `gh pr merge` コマンドを使用してください。

## 使用例

```bash
# インタラクティブモードでマージ（現在のブランチのPR）
gh pr merge

# PR番号を指定してマージ
gh pr merge 123

# マージ方法を指定
gh pr merge --merge      # マージコミット
gh pr merge --squash     # スカッシュマージ
gh pr merge --rebase     # リベースマージ

# マージ後にローカルブランチを削除
gh pr merge --delete-branch

# 自動マージを有効化（CI通過後に自動マージ）
gh pr merge --auto --squash
```
