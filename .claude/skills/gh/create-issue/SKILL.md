# Issue作成

Issueを作成する際は、GitHub CLI の `gh issue create` コマンドを使用してください。

## 使用例

```bash
# インタラクティブモードで作成
gh issue create

# タイトルと本文を指定して作成
gh issue create --title "バグ: ○○が動作しない" --body "## 再現手順\n1. ○○を実行\n2. ○○が発生"

# ラベルを付けて作成
gh issue create --title "機能要望: ○○" --label "enhancement"

# 担当者を指定して作成
gh issue create --title "タスク: ○○" --assignee "@me"
```
