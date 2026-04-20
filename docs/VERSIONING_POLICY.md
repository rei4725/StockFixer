# VERSIONING POLICY

StockFixer のバージョン管理は本書を正本とする。バージョン判定、PR 記載要件、VERSION 更新判断は本書に従う。

## 1. SemVer 基準

バージョンは SemVer を採用する。

- 形式: `MAJOR.MINOR.PATCH`
- `MAJOR`: 後方互換性のない変更
- `MINOR`: 後方互換性を維持した機能追加
- `PATCH`: 後方互換性を維持した修正・改善

## 2. version_impact の定義

PR では次のいずれかを必ず宣言する。

- `major`: 破壊的変更を含む
- `minor`: 機能追加・公開仕様拡張
- `patch`: バグ修正・非破壊の改善
- `none`: VERSION 更新不要

## 3. PR 必須要件

PR 本文には次の見出しを必須とする（テンプレート準拠）。

- `## version_impact`
- `## version_rationale`
- `## VERSION 更新`
	- `version_update_required: yes|no`
	- `version_before: x.y.z`
	- `version_after: x.y.z`
- `## VERSION 未更新理由`

追加ルール:

- `version_impact != none` の場合、`VERSION` 更新は必須
- `version_impact == none` の場合のみ `VERSION` 未更新を許可
- `major/minor/patch` の場合、`version_update_required: yes` かつ `version_before` / `version_after` の実値（SemVer）を必須とする
- `version_update_required: no` の場合、`## VERSION 未更新理由` の記載を必須とする

## 4. none の許容条件（例外条件）

`version_impact: none` を使えるのは、外部動作に影響しない変更に限る。

許容例:

- コメント修正、ドキュメント修正のみ
- CI や lint 設定のみの変更（ランタイム挙動不変）
- リファクタのみ（入出力・公開仕様・DB スキーマ不変）
- テストコードのみの変更

非許容例:

- API 入出力、CLI 引数、設定仕様の変更
- DB スキーマや保存フォーマットの変更
- 実行結果に影響するロジック変更

## 5. 判定フロー

1. 変更が外部利用者の挙動に影響するかを判定する
2. 破壊的変更なら `major`
3. 破壊的でない機能追加なら `minor`
4. 既存仕様内の修正・改善なら `patch`
5. 外部挙動に影響しない場合のみ `none`
6. `major/minor/patch` の場合は `VERSION` を更新し、PR に新旧バージョンを記載する
7. `none` の場合は未更新理由を PR に記載する

## 6. 運用参照

- 日常運用手順: `docs/OPERATIONS.md`
- バージョン運用スキル: `.github/skills/version-mgmt/SKILL.md`
