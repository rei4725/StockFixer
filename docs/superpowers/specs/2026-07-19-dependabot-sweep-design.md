# Dependabot PR 自動処理（定期スイープ）設計

## 背景・課題

現状の `.github/workflows/dependabot-auto-merge.yml` は Dependabot PR が作成/更新された瞬間（`opened`/`synchronize`/`reopened`）にのみ反応し、`update-type: patch` のPRに対して `gh pr merge --auto --squash` を試みる。しかしこの方式には2つの穴がある。

1. **カスケードコンフリクトを検知できない**: 複数のDependabot PRが同時にオープンしている状態で1件をマージすると、`requirements.txt`/`requirements-dev.txt` を共有する他のPRにコンフリクトが生じる。しかしこのイベントは対象PR自身には発火しないため、誰かが気づいて手動で `@dependabot rebase` を依頼するまで放置される。2026-07-19の手動対応（PR #570, #568, #566, #565 の一括処理）で実際にこの現象が発生し、`#566`・`#568` が他PRのマージ後にコンフリクトした。
2. **`minor` 更新は自動化対象外**: 現行は `patch` のみが対象で、`minor` はCIが全green でも人が都度 `gh pr merge` する必要がある。

## 目的

Dependabot PR（`minor`/`patch`）に対する「CI確認 → コンフリクトならリベース依頼 → green化を待ってsquashマージ」という定型作業を、追加のワークフローで自動化する。`major` 更新は対象外とし、常に人間のレビューを必須とする。

## アーキテクチャ・トリガー

新規ワークフロー `.github/workflows/dependabot-sweep.yml` を追加する。既存の `dependabot-auto-merge.yml`（PRイベント駆動、初回マージ試行を担当）はそのまま残し、新設のsweepワークフローは **「他PRのマージによって後から生じたコンフリクトのフォローアップ」** を専任で担当する棲み分けにする。両ワークフローは役割が重複しないため、既存のロジックは変更しない。

トリガーは3種類:

- `push`（`develop` ブランチへのマージ検知。最速でカスケードコンフリクトの発生を捕捉する）
- `schedule`（cron、フォールバック用。push イベントを取りこぼした場合や、CI実行中で `mergeStateStatus` が確定していなかった場合の再チェックに使う。Dependabot自体のスケジュールは週次のため、高頻度cronでもコスト面は問題にならない）
- `workflow_dispatch`（手動即時実行の逃げ道）

## コンポーネント・判定ロジック

### 1. 対象PR列挙

```
gh pr list --author "app/dependabot" --state open --json number,title,mergeStateStatus,mergeable,labels
```

でオープンな Dependabot PR を全件取得する。

### 2. update-type 判定

`.github/dependabot.yml` の設定上、`python-runtime` / `python-dev` / `github-actions` の各グループは `update-types: [minor, patch]` に限定されている（`major` 更新はグループ化されず個別PRとして出てくる設計）。この既存の設定を根拠に、PRの Dependabot メタデータ（コミットトレーラー由来の `Dependabot-Update-Type`、または grouped PR かどうか）から `major` と判定されたPRは **何もせずスキップ**する（人手レビューに委ねる＝現状維持）。

### 3. 状態別アクション分岐（対象PRごとに）

- `mergeable == CONFLICTING` → `gh pr comment <n> --body "@dependabot rebase"` を実行
- `mergeStateStatus == CLEAN`（＝必須CIが全green）→ `gh pr merge <n> --squash --delete-branch`
- それ以外（CI実行中など）→ 何もせず次サイクルへ

### 通知について

Discord通知などの追加通知は行わない。マージ結果は通常のGitHub通知（PRクローズ）と `gh pr list` での確認に委ねる。

## データフロー

PRイベント/push/cron起動 → 対象PR一覧取得 → 各PRについて update-type 判定 → （rebase依頼 or マージ or 何もしない）。状態はGitHub側（PRの `mergeable`/`mergeStateStatus`）に一任し、ワークフロー側では状態を保持しない（ステートレスな設計）。

## エラーハンドリング・エッジケース

- `gh pr merge` が競合状態（`mergeable` が古いキャッシュ値だった場合など）で失敗しても、そのPRの処理だけ失敗として次のPRへ継続する。ワークフロー全体を失敗させない。
- rebase依頼をかけても解消しない（Dependabot側の制約等）PRは、サイクルを重ねるたびに「まだCONFLICTING→再度rebase依頼」を繰り返し得る。コメントスパムを避けるため、**直近のPRコメント履歴に自分（`github-actions[bot]`）による rebase依頼コメントが既にあり、かつそれ以降新しいコミットが積まれていない場合はスキップ**するガードを入れる。
- `major`（ungrouped）PRは対象外のため触れず、既存の人手レビューフローに委ねる。

## テスト・ロールアウト方針

GitHub Actionsのワークフロー自体はpytestのユニットテスト対象外（`python/`配下のCIカバレッジゲートとは別レイヤー）。検証は以下で行う。

- **実地検証**: 次にDependabotが週次でPRを出したタイミングで、実際に `minor`/`patch` PRが自動でrebase依頼→squash mergeまで流れることを確認する。
- **ロジックの切り出し**: 判定・分岐ロジックが複雑になった場合は、workflow内の埋め込みシェルではなく `python/scripts/` 等に切り出してユニットテスト可能にすることを検討する。ただし初期実装はYAGNIに従い、ワークフローYAML内に素直に書きシンプルに保つ。
- **ロールバック容易性**: 新規ワークフローファイル1本の追加であるため、問題が発生した場合は該当ファイルを無効化（削除 or `if: false`）するだけで、既存の `dependabot-auto-merge.yml` のみの状態に即座に戻せる。

## スコープ外（将来検討）

- スタックPR事故防止（親PRの `--delete-branch` マージで子PRがCLOSEDになる問題）は本設計の対象外。別途小規模な設計として扱う。
- `major` 更新の自動処理は対象外。常に手動レビュー。
