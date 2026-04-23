---
description: 指定されたイシューと実装に対するプルリクエストを作成します。
tools:
  [
    "execute",
    "read",
    "search",
    "todo",
    "web",
    "ms-vscode.vscode-websearchforcopilot/websearch",
  ]
---

与えられたイシューと実装に対する、プルリクエストを作成してください。

## 手順 (#tool:todo)

1. PR が作成できる状態にあるのか確認する
   - ドキュメント更新の忘れがないか
   - 未コミットの変更がないか
   - テスト (CI) が通過するか
   - PR ボディに必須セクションが含まれているか（`validate-pr-body` CI）
2. 作成にふさわしくない状況だと判断される場合、修正案を示して終了します。そうでなければ PR を作成します。
3. 作成された PR の内容とリンクをユーザーに通知します。

## Notes

- 関連する Issue がある場合、その Issue 番号を含めてください (e.g., `Closes #<number>`)
- GitHub Issue に追加のコメントが必要であれば、コメントを残しておいてください。

### PR ボディの必須セクション（`validate-pr-body` CI）

PR 作成・更新時に `validate-pr-body` ジョブが自動実行され、以下の4セクションが存在しないと CI が失敗する。
PR ボディを作成する際は必ず全セクションを含めること。

| セクション見出し | 必須の値 |
|---|---|
| `## version_impact` | `major` / `minor` / `patch` / `none` のいずれか1語 |
| `## version_rationale` | 空・プレースホルダー不可。変更根拠を1文以上記述 |
| `## VERSION 更新` | `version_update_required: yes` または `version_update_required: no` を含む |
| `## VERSION 未更新理由` | **常に見出しが必要**（`version_update_required: yes` の場合は「該当なし」等で可） |

**`version_impact` と `version_update_required` の対応ルール:**

| version_impact | version_update_required | VERSION 未更新理由 |
|---|---|---|
| major / minor / patch | `yes` 必須 / version_before・version_after も必須 | 「該当なし」等でOK |
| none | `no` 必須 | 未更新理由を1文以上記述 |

**PR ボディテンプレート:**
```markdown
## version_impact

none

## version_rationale

（変更根拠を1文以上記述）

## VERSION 更新

- version_update_required: no
- version_before:
- version_after:

## VERSION 未更新理由

（未更新理由を1文以上記述）
```

詳細は `.github/skills/git-ops/SKILL.md` の「PR作成前のボディ検証チェック」セクションを参照。

## ツール

- #tool:ms-vscode.vscode-websearchforcopilot/websearch: ウェブ検索
- `gh`: GitHub リポジトリの操作

## ドキュメント

- `docs/`
- `README.md`
- `CONTRIBUTING.md`
