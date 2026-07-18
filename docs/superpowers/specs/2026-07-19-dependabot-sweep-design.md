# Dependabot PR 自動処理（コンフリクト自動リベース依頼）設計

## 背景・課題

現状の `.github/workflows/dependabot-auto-merge.yml` は Dependabot PR が作成/更新された瞬間（`opened`/`synchronize`/`reopened`）にのみ反応し、`update-type: semver-patch` のPRに対して `gh pr merge --auto --squash` （GitHub純正のauto-mergeを有効化）を試みる。しかしこの方式には2つの穴がある。

1. **カスケードコンフリクトを検知できない**: 複数のDependabot PRが同時にオープンしている状態で1件をマージすると、`requirements.txt`/`requirements-dev.txt` を共有する他のPRにコンフリクトが生じる。しかしこのイベントは対象PR自身には発火しないため、誰かが気づいて手動で `@dependabot rebase` を依頼するまで放置される。2026-07-19の手動対応（PR #570, #568, #566, #565 の一括処理）で実際にこの現象が発生し、`#566`・`#568` が他PRのマージ後にコンフリクトした。
2. **`semver-minor` 更新は自動化対象外**: 現行は `semver-patch` のみが対象で、`semver-minor` はCIが全green でも人が都度 `gh pr merge` する必要がある。

## 事前検証で判明した制約（重要）

計画作成前の技術検証で、当初想定had 2つの誤りが判明したため、設計を以下の前提に修正した。

- **`dependabot/fetch-metadata` アクションは `pull_request`/`pull_request_target` イベントの実行コンテキストに依存しており、`push`/`schedule` トリガーのジョブから任意のPR番号に対して後付けで呼び出すことはできない**。
- **Dependabotのコミットメッセージ本文（trailer）には `update-type` フィールドが常に含まれるとは限らない**（実例: streamlit更新PRの実コミットメッセージには `dependency-name`/`dependency-version`/`dependency-type` はあるが `update-type` 行は無かった）。またPRのブランチ名/タイトルが「grouped PRかどうか」を安定して示さない（`plotly`/`filelock`/`streamlit` はいずれも `python-runtime` グループのパターンに含まれる個別依存だが、実際は個別ブランチ名で作成され、それでいて `semver-minor`/`semver-patch` だった）。よって **grouped/ungroupedの見た目だけで major/minor/patch を判定するのは不可能**。

一方で、GitHub純正の **auto-merge フラグはPR単位で永続する**（新しいコミットが積まれても消えない）ことを実地で確認済み。2026-07-19、PR #568 は最初 `gh pr merge --auto --squash` が `Auto merge is not allowed for this repository` で失敗したが、Dependabotによるリベース後に同じワークフローが再実行された際は同じコマンドがそのまま成功し、CIがgreenになった時点で自動的にマージが完了した（＝1回目の失敗は一過性のもので、恒久的な設定不備ではなかった）。

この事実から、**update-type判定とsquash mergeの実行は既存の `dependabot-auto-merge.yml`（pull_requestイベント文脈）に任せ、新設ワークフローは「コンフリクトしているPRを見つけてリベースを依頼するだけ」に専念する**という、より単純で確実な設計に切り替える。

## 目的

1. 既存 `dependabot-auto-merge.yml` の自動マージ対象を `semver-patch` に加えて `semver-minor` にも広げる。
2. 「他PRのマージによって後から生じたコンフリクト」を検知し、`@dependabot rebase` を自動で依頼する新規ワークフローを追加する。リベース後のマージ完了は、(1)で有効化された永続的なauto-mergeフラグに委ねる（新設ワークフロー自身はマージを実行しない）。
3. `major` 更新（およびauto-mergeが有効化されていないPR全般）はこれまで通り対象外とし、常に人間のレビューを必須とする。

## アーキテクチャ・変更点

### 変更A: 既存ワークフローの条件拡張

`.github/workflows/dependabot-auto-merge.yml` の「Enable auto-merge for patch updates」ステップの `if` 条件を、`semver-patch` 単独から `semver-patch` または `semver-minor` にORで広げる。この部分は既存どおり `pull_request` イベント文脈で `dependabot/fetch-metadata@v3` を使うため、update-type判定の正確性は現状のまま維持される（今回の検証で発覚した問題はこの経路には影響しない）。

`major` 更新はこの条件に一致しないため、これまで通り自動マージ対象外（auto-mergeフラグが立たない）で維持される。

### 変更B: 新規ワークフロー（コンフリクト検知・リベース依頼）

新規ファイル `.github/workflows/dependabot-rebase-sweep.yml` を追加する。このワークフローは **update-type判定を一切行わない**。オープンなDependabot PRのうち `mergeable == CONFLICTING` なものを見つけ、`@dependabot rebase` を投げるだけの単機能ワークフローとする。

トリガーは3種類:

- `push`（`develop` ブランチへのマージ検知。最速でカスケードコンフリクトの発生を捕捉する）
- `schedule`（cron、フォールバック用。push イベントを取りこぼした場合の再チェックに使う。Dependabot自体のスケジュールは週次のため、高頻度cronでもコスト面は問題にならない）
- `workflow_dispatch`（手動即時実行の逃げ道）

`major` 更新PRやauto-mergeが有効化されていないPRであっても、コンフリクトしていればリベース依頼の対象に含める。リベースしてコンフリクトを解消しておくこと自体は、その後人間がレビューする際の助けになるだけで、勝手にマージされるわけではないため無害。

## コンポーネント

### 1. 対象PR列挙

```
gh pr list --author "app/dependabot" --state open --json number,mergeable,comments
```

でオープンな Dependabot PR を全件取得する。

### 2. コンフリクト検知とリベース依頼（対象PRごと）

- `mergeable == CONFLICTING` の場合のみ処理対象とする。
- 直近のPRコメントに、`github-actions[bot]` による `@dependabot rebase` 依頼コメントが**既に存在し、かつそのコメント以降に新しいコミットが積まれていない**場合はスキップする（同一コンフリクト状態に対する重複コメント防止）。
- 上記ガードに引っかからなければ `gh pr comment <n> --body "@dependabot rebase"` を実行する。

### 3. マージについて

このワークフローはマージを一切行わない。マージは変更Aで有効化された `gh pr merge --auto` の永続フラグが、リベース完了・CI green化を検知した時点でGitHub側が自動的に実行する。

### 通知について

Discord通知などの追加通知は行わない。

## データフロー

`push`/`schedule`/`workflow_dispatch` 起動 → オープンなDependabot PR一覧取得 → 各PRについて `mergeable == CONFLICTING` か判定 → （重複チェック後）rebase依頼コメント or 何もしない。状態はGitHub側（PRの `mergeable`、コメント履歴）に一任し、ワークフロー側では状態を保持しない（ステートレスな設計）。

## エラーハンドリング・エッジケース

- `gh pr comment` が失敗しても、そのPRの処理だけ失敗として次のPRへ継続する。ワークフロー全体を失敗させない。
- リベース依頼を送っても解消しない（Dependabot側の制約等）PRは、重複防止ガードにより一度依頼した後は新しいコミットが積まれるまで再依頼されない（スパム化しない）。
- `major` 更新やauto-merge非対象のPRであっても、コンフリクト解消（リベース依頼）自体は行う（無害なため）。実際にマージされるかどうかは変更Aの条件のみで決まる。

## テスト・ロールアウト方針

GitHub Actionsのワークフロー自体はpytestのユニットテスト対象外（`python/`配下のCIカバレッジゲートとは別レイヤー）。検証は以下で行う。

- **実地検証**: 次にDependabotが週次でPRを複数出すタイミングで、1件がマージされた後に残りのPRが自動的にリベース依頼→CI green化→auto-mergeによる自動マージまで流れることを確認する。
- **ロジックの切り出し**: 判定ロジックが複雑になった場合は、workflow内の埋め込みシェルではなく `python/scripts/` 等に切り出してユニットテスト可能にすることを検討する。ただし初期実装はYAGNIに従い、ワークフローYAML内に素直に書きシンプルに保つ。
- **ロールバック容易性**: 変更Aは1行の条件式変更、変更Bは新規ファイル1本の追加のみであるため、問題が発生した場合は差分を元に戻す/該当ファイルを無効化するだけで即座に既存状態に戻せる。

## スコープ外（将来検討）

- スタックPR事故防止（親PRの `--delete-branch` マージで子PRがCLOSEDになる問題）は本設計の対象外。別途小規模な設計として扱う。
- `major` 更新の自動マージは対象外。常に手動レビュー。
