## 概要

<!-- 変更内容を簡潔に記載 -->

## version_impact

none

<!--
↑ 上の行を `major` / `minor` / `patch` / `none` のいずれか1語に置き換える。
   - major: 後方互換性のない変更（公開API・設定の破壊的変更など）
   - minor: 後方互換のある機能追加
   - patch: 後方互換のあるバグ修正・内部リファクタ
   - none : ドキュメント / テスト / CI のみの変更で、リリース内容に影響しない
判定基準の正本: docs/VERSIONING_POLICY.md
-->

## version_rationale

ドキュメント / テスト / CI のみの変更でリリース内容に影響しないため。

<!-- ↑ 実際の変更に合わせて1文以上で必ず書き換える（空欄・TBD・N/A は不可）。 -->

## VERSION 更新

- version_update_required: no
- version_before:
- version_after:

<!--
version_impact が major / minor / patch の場合:
  - version_update_required: yes
  - version_before / version_after に SemVer 形式 (例: 1.2.3) を記載
  - VERSION ファイルも本PRで一緒に更新する
version_impact が none の場合:
  - version_update_required: no
  - version_before / version_after は空欄のままでよい
-->

## VERSION 未更新理由

ドキュメント / テスト / CI のみの変更で、リリース対象に含まれないため。

<!--
version_update_required: no のときは必須（空欄・TBD・N/A は不可）。
version_update_required: yes のときは「該当なし」と書いてよいが、見出しごと残すこと。
-->

## テスト

<!-- 実施したテストと結果（例: `python -m pytest tests/unit/ -v` 緑） -->
