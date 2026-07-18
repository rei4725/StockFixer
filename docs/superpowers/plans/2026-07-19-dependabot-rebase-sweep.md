# Dependabot PR 自動処理（コンフリクト自動リベース依頼）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Dependabot PR の `semver-minor` 更新も自動マージ対象に加え、他PRのマージで後発的に生じるコンフリクトを検知して自動で `@dependabot rebase` を依頼するワークフローを追加する。

**Architecture:** (A) 既存 `.github/workflows/dependabot-auto-merge.yml` の auto-merge 有効化条件を `semver-patch` のみから `semver-patch` または `semver-minor` に拡張する1行変更。(B) 新規 `.github/workflows/dependabot-rebase-sweep.yml` を追加し、`push`(develop)/`schedule`(30分毎cron)/`workflow_dispatch` をトリガーに、オープンな Dependabot PR のうち `mergeable == CONFLICTING` なものへ `@dependabot rebase` コメントを投げる。マージ自体は(A)で有効化された GitHub 純正 auto-merge の永続フラグに委ねるため、(B) はマージ処理を一切行わない。

**Tech Stack:** GitHub Actions (YAML), `gh` CLI, `jq`。Python側のコード変更は無し。

## Global Constraints

- `major` 更新（`version-update:semver-major`）は常に自動化対象外。既存の人手レビューフローを維持する。
- Discord通知などの追加通知は行わない（設計時にユーザーが明示的に不要と回答済み）。
- 新設ワークフローは update-type 判定を一切行わない（`dependabot/fetch-metadata` は `pull_request` イベント文脈でのみ正しく動作するため、既存ワークフロー側に判定を残す）。
- 新設ワークフローはステートレスに保つ（GitHub側の `mergeable`・コメント履歴のみを状態源とする）。
- rebase依頼の重複コメントを防ぐガードを必ず入れる（直近のrebase依頼コメント以降に新規コミットが無ければスキップ）。
- 既存の `dependabot-auto-merge.yml` の「Notify on security advisory patch update」ステップは変更しない（スコープ外）。

---

### Task 1: 既存ワークフローの auto-merge 対象を semver-minor にも拡張

**Files:**
- Modify: `.github/workflows/dependabot-auto-merge.yml:40-45`

**Interfaces:**
- Consumes: なし（既存ステップの `if` 条件のみを変更する自己完結タスク）
- Produces: なし（後続タスクはこのファイルに依存しない）

- [ ] **Step 1: 現在の該当ステップを確認する**

Read: `.github/workflows/dependabot-auto-merge.yml` の40〜45行目。現在の内容は以下のはず（異なっていたら作業前に差分を確認すること）:

```yaml
      - name: Enable auto-merge for patch updates
        if: steps.metadata.outputs.update-type == 'version-update:semver-patch'
        run: gh pr merge --auto --squash "$PR_URL"
        env:
          PR_URL: ${{ github.event.pull_request.html_url }}
          GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

- [ ] **Step 2: ステップ名と `if` 条件を semver-minor も含む形に書き換える**

`.github/workflows/dependabot-auto-merge.yml` の該当ブロックを以下に置き換える:

```yaml
      - name: Enable auto-merge for patch/minor updates
        if: |
          steps.metadata.outputs.update-type == 'version-update:semver-patch' ||
          steps.metadata.outputs.update-type == 'version-update:semver-minor'
        run: gh pr merge --auto --squash "$PR_URL"
        env:
          PR_URL: ${{ github.event.pull_request.html_url }}
          GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

- [ ] **Step 3: YAML構文を検証する**

Run (リポジトリルートから):
```bash
py -c "import yaml; yaml.safe_load(open('.github/workflows/dependabot-auto-merge.yml', encoding='utf-8')); print('OK')"
```
Expected: `OK` と表示され、例外が出ないこと。

- [ ] **Step 4: `if` 条件がYAML上「1つの複数行文字列」として解釈されることを確認する**

Run:
```bash
py -c "
import yaml
doc = yaml.safe_load(open('.github/workflows/dependabot-auto-merge.yml', encoding='utf-8'))
step = doc['jobs']['dependabot']['steps'][-1]
assert step['name'] == 'Enable auto-merge for patch/minor updates', step['name']
assert 'semver-patch' in step['if'] and 'semver-minor' in step['if'], step['if']
print('OK:', repr(step['if']))
"
```
Expected: `OK: ...` に続けて `semver-patch` と `semver-minor` の両方を含む文字列が表示されること。

- [ ] **Step 5: コミットする**

```bash
git add .github/workflows/dependabot-auto-merge.yml
git commit -m "feat: Dependabot自動マージ対象にsemver-minorを追加"
```

---

### Task 2: コンフリクト検知・自動リベース依頼ワークフローを新設

**Files:**
- Create: `.github/workflows/dependabot-rebase-sweep.yml`

**Interfaces:**
- Consumes: Task 1 の変更には依存しない（独立したワークフローファイル。ただしTask 1で有効化されたauto-mergeが、このタスクが依頼したリベース後のマージ完了を担う、という実行時の連携がある）
- Produces: なし（最終タスク）

- [ ] **Step 1: ローカルで列挙クエリの構文を検証する**

Run（このリポジトリで実際に `gh` CLI が使えることを利用した事前検証。オープンなDependabot PRが無くてもエラーにならず空文字を返せばよい）:
```bash
gh pr list --state open --author "app/dependabot" --json number,mergeable --jq '.[] | select(.mergeable == "CONFLICTING") | .number'
```
Expected: エラーなく終了する（該当PRが無ければ何も出力されない）。

- [ ] **Step 2: 重複防止ロジック（dedup判定）の jq フィルタをローカルで検証する**

Run（過去にマージ済みのPR番号566に対して、実際のコメント/コミット構造でフィルタが期待通り動くか確認する。`author.login` はGraphQL経由だと `github-actions[bot]` ではなく `github-actions`（角括弧なし）で返る点に注意——事前検証で確認済み）:
```bash
gh pr view 566 --json comments,commits --jq '
  [.comments[]
    | select(.author.login == "github-actions")
    | select(.body | test("@dependabot rebase"))
    | .createdAt
  ] | sort | last // "NONE"'
```
Expected: エラーなく実行され、`"NONE"` または日時文字列のいずれかが返ること（実際のコメント内容次第でどちらでも良い。ここではフィルタ構文自体にエラーが出ないことを確認するのが目的）。

- [ ] **Step 3: ワークフローファイルを新規作成する**

Create `.github/workflows/dependabot-rebase-sweep.yml`:

```yaml
name: Dependabot Rebase Sweep

on:
  push:
    branches: [develop]
  schedule:
    - cron: "*/30 * * * *"
  workflow_dispatch:

permissions:
  pull-requests: write

jobs:
  rebase-sweep:
    runs-on: ubuntu-latest
    steps:
      - name: Request rebase for conflicting Dependabot PRs
        run: |
          set -euo pipefail

          pr_numbers=$(gh pr list --state open --author "app/dependabot" --json number,mergeable \
            --jq '.[] | select(.mergeable == "CONFLICTING") | .number')

          if [ -z "$pr_numbers" ]; then
            echo "No conflicting Dependabot PRs found."
            exit 0
          fi

          for pr in $pr_numbers; do
            (
              set -e
              info=$(gh pr view "$pr" --json comments,commits)
              last_commit_date=$(echo "$info" | jq -r '.commits[-1].committedDate')
              last_rebase_comment_date=$(echo "$info" | jq -r '
                [.comments[]
                  | select(.author.login == "github-actions")
                  | select(.body | test("@dependabot rebase"))
                  | .createdAt
                ] | sort | last // empty')

              if [ -n "$last_rebase_comment_date" ] && [[ "$last_rebase_comment_date" > "$last_commit_date" ]]; then
                echo "PR #$pr: rebase already requested since last commit, skipping."
                exit 0
              fi

              echo "PR #$pr: requesting rebase."
              gh pr comment "$pr" --body "@dependabot rebase"
            ) || echo "PR #$pr: skipped due to an error, continuing."
          done
        env:
          GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

- [ ] **Step 4: YAML構文を検証する**

Run:
```bash
py -c "import yaml; yaml.safe_load(open('.github/workflows/dependabot-rebase-sweep.yml', encoding='utf-8')); print('OK')"
```
Expected: `OK` と表示され、例外が出ないこと。

- [ ] **Step 5: トリガー・パーミッション・ジョブ構造を検証する**

Run:
```bash
py -c "
import yaml
doc = yaml.safe_load(open('.github/workflows/dependabot-rebase-sweep.yml', encoding='utf-8'))
on = doc[True] if True in doc else doc['on']
assert on['push']['branches'] == ['develop'], on['push']
assert on['schedule'][0]['cron'] == '*/30 * * * *', on['schedule']
assert 'workflow_dispatch' in on, on
assert doc['permissions'] == {'pull-requests': 'write'}, doc['permissions']
step = doc['jobs']['rebase-sweep']['steps'][0]
assert '@dependabot rebase' in step['run'], 'rebase コメント文言が無い'
print('OK')
"
```
Expected: `OK` と表示されること。（YAMLパーサーは `on:` キーを真偽値 `True` として解釈することがあるため、`doc[True] if True in doc else doc['on']` で両対応している）

- [ ] **Step 6: 埋め込みシェルスクリプトの構文をローカルで検証する（shellcheck相当）**

Run（`run:` ブロックの中身を抽出してbash構文チェックのみ行う。ファイル書き出しはせず標準入力経由でチェックし、ネットワークアクセスは発生しない）:
```bash
py -c "
import yaml
doc = yaml.safe_load(open('.github/workflows/dependabot-rebase-sweep.yml', encoding='utf-8'))
print(doc['jobs']['rebase-sweep']['steps'][0]['run'])
" | bash -n /dev/stdin
echo "exit: $?"
```
Expected: `exit: 0`（bashの構文チェックのみでエラーが無いこと。実行はしない）。

- [ ] **Step 7: コミットする**

```bash
git add .github/workflows/dependabot-rebase-sweep.yml
git commit -m "feat: Dependabot PRのコンフリクトを検知して自動リベース依頼するワークフローを追加"
```

---

## 実地検証（両タスク共通・マージ後）

pytest等のユニットテスト対象外のため、develop ブランチにマージされた後の実地検証を行う:

1. 次回 Dependabot 週次実行（月曜）で複数PRが出た際、1件がマージされた後に残りのPRが `CONFLICTING` になるか観察する。
2. `dependabot-rebase-sweep.yml` の実行ログ（Actions タブ）で、該当PRに対して `@dependabot rebase` コメントが自動投稿されることを確認する。
3. Dependabotのリベース後、CIが green になった時点で `dependabot-auto-merge.yml` が有効化した auto-merge により自動でsquashマージされることを確認する（`semver-minor` のPRでも同様に動くことを含めて確認）。
4. 問題があれば、該当ワークフローファイルを無効化（削除 or `if: false`）して即座にロールバックする。
