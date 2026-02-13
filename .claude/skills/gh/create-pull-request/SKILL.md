# プルリクエスト作成

プルリクエストを作成する際は、GitHub CLI の `gh pr create` コマンドを使用してください。

## 使用例

```bash
# インタラクティブモードで作成
gh pr create

# タイトルと本文を指定して作成
gh pr create --title "機能追加: ○○の実装" --body "## 変更内容\n- ○○を追加"

# ドラフトPRとして作成
gh pr create --draft
```
