---
name: version-mgmt
description: "Gitタグによるバージョン管理とDockerイメージのバージョン付きビルドを行う。バージョン管理・タグ・リリース・バージョンアップ・セマンティックバージョニングの話題では必ずこのスキルを使用する。VERSIONファイルの更新やDockerイメージのバージョン連動が必要な場面でも使用する。"
compatibility: "Git 2.0+, Docker, docker-compose。プロジェクトルートで実行。"
---

## Goal
Git タグベースのセマンティックバージョニングでプロジェクトのバージョンを管理し、Dockerイメージと連動させる。

## Source Of Truth

バージョン判定の正本は `docs/VERSIONING_POLICY.md` とする。

- 本スキルで `version_impact` を判定する際は、必ず正本の判定基準に従う
- PR本文には `version_impact` / `version_rationale` / `version_update_required` を記載する
- `version_impact` が `major` / `minor` / `patch` の場合は `VERSION` 更新必須
- `version_impact` が `none` の場合のみ `VERSION` 未更新を許可し、未更新理由を記載する

## バージョン体系

### セマンティックバージョニング（SemVer）
```
v{MAJOR}.{MINOR}.{PATCH}
```

| 区分 | 変更契機 | 例 |
|------|---------|-----|
| MAJOR | 互換性のない大規模変更 | v1.0.0 → v2.0.0 |
| MINOR | 後方互換性のある機能追加 | v1.0.0 → v1.1.0 |
| PATCH | バグ修正・小改善 | v1.0.0 → v1.0.1 |

- タグには必ず `v` プレフィックスを付ける（例: `v1.2.3`）

## VERSION ファイル（必須）

`StockFixer/VERSION` がバージョンの **唯一の正として機能する**。

- バージョンアップ時は **必ず VERSION ファイルを先に更新**してからコミット・タグ作成を行う
- デプロイ時は **VERSION ファイルを参照**して `$env:VERSION` を設定する（Gitタグから取得しない）
- VERSION ファイルの内容は `v` プレフィックスなしの数値のみ（例: `1.2.3`）

```
StockFixer/
└── VERSION          # バージョン番号の正（例: 1.2.3）
```

## Procedure

### 1. 現在のバージョン確認

#### 最新タグの確認
```powershell
git describe --tags --abbrev=0
```

#### 全タグ一覧（バージョン降順）
```powershell
git tag -l "v*" --sort=-version:refname
```

#### タグとコミットの対応確認
```powershell
git log --oneline --decorate --tags
```

### 2. バージョンを上げる（VERSIONファイル更新 → コミット → タグ作成）

**手順はこの順序で行うこと：**
1. VERSION ファイルを編集して新バージョンを記載
2. VERSION ファイルをコミット
3. Gitタグを作成

#### PATCH バージョンアップ（バグ修正・小改善）
```powershell
# 現在のVERSIONファイルから取得して次のPATCHバージョンを計算
$current = (Get-Content C:\src\StockFixer\VERSION -Raw).Trim()
$parts = $current.Split(".")
$newVersionNum = "$($parts[0]).$($parts[1]).$([int]$parts[2] + 1)"
$newVersion = "v$newVersionNum"

# VERSIONファイルを更新
$newVersionNum | Set-Content C:\src\StockFixer\VERSION -NoNewline
git add VERSION
git commit -m "chore: bump version to $newVersionNum"
git tag -a $newVersion -m "Release $newVersion"
Write-Host "Created tag: $newVersion"
```

#### MINOR バージョンアップ（機能追加）
```powershell
$current = (Get-Content C:\src\StockFixer\VERSION -Raw).Trim()
$parts = $current.Split(".")
$newVersionNum = "$($parts[0]).$([int]$parts[1] + 1).0"
$newVersion = "v$newVersionNum"

$newVersionNum | Set-Content C:\src\StockFixer\VERSION -NoNewline
git add VERSION
git commit -m "chore: bump version to $newVersionNum"
git tag -a $newVersion -m "Release $newVersion"
Write-Host "Created tag: $newVersion"
```

#### MAJOR バージョンアップ（破壊的変更）
```powershell
$current = (Get-Content C:\src\StockFixer\VERSION -Raw).Trim()
$parts = $current.Split(".")
$newVersionNum = "$([int]$parts[0] + 1).0.0"
$newVersion = "v$newVersionNum"

$newVersionNum | Set-Content C:\src\StockFixer\VERSION -NoNewline
git add VERSION
git commit -m "chore: bump version to $newVersionNum"
git tag -a $newVersion -m "Release $newVersion"
Write-Host "Created tag: $newVersion"
```

### 3. タグをリモートにプッシュ
```powershell
# 特定のタグをプッシュ
git push origin v1.2.3

# 全タグをプッシュ
git push origin --tags
```

### 4. Docker イメージをバージョン付きでビルド

**デプロイ時のバージョンは `VERSION` ファイルを参照する。**

```powershell
cd C:\src\StockFixer

# VERSIONファイルからバージョンを取得（正）
$env:VERSION = (Get-Content .\VERSION -Raw).Trim()
if (-not $env:VERSION) { $env:VERSION = "dev" }

# ビルド情報を設定
$env:BUILD_DATE = (Get-Date -Format "yyyy-MM-ddTHH:mm:ssZ")
$env:GIT_COMMIT = git rev-parse --short HEAD

# ビルド＆起動
docker-compose up -d --build

Write-Host "Built image: stockfixer:$($env:VERSION)"
```

### 5. タグの削除（必要な場合のみ）
```powershell
# ローカルタグ削除
git tag -d v1.2.3

# リモートタグ削除
git push origin --delete v1.2.3
```

## 一括フロー（バージョンアップ → ビルド → プッシュ）

開発サイクルの典型的なフローをまとめて実行する。

```powershell
cd C:\src\StockFixer

# 1. バージョンアップの種類を決定（patch / minor / major）
$bumpType = "patch"  # ← 適宜変更

# 2. VERSIONファイルから現在のバージョン取得
$current = (Get-Content .\VERSION -Raw).Trim()
$parts = $current.Split(".")

# 3. 新バージョン計算
switch ($bumpType) {
    "patch" { $newVersionNum = "$($parts[0]).$($parts[1]).$([int]$parts[2] + 1)" }
    "minor" { $newVersionNum = "$($parts[0]).$([int]$parts[1] + 1).0" }
    "major" { $newVersionNum = "$([int]$parts[0] + 1).0.0" }
}
$newVersion = "v$newVersionNum"

# 4. VERSIONファイルを更新してコミット
$newVersionNum | Set-Content .\VERSION -NoNewline
git add VERSION
git commit -m "chore: bump version to $newVersionNum"
Write-Host "Version bumped: $newVersionNum"

# 5. タグ作成
git tag -a $newVersion -m "Release $newVersion"
Write-Host "Tagged: $newVersion"

# 6. Docker ビルド（VERSIONファイルを参照）
$env:VERSION = (Get-Content .\VERSION -Raw).Trim()
$env:BUILD_DATE = (Get-Date -Format "yyyy-MM-ddTHH:mm:ssZ")
$env:GIT_COMMIT = git rev-parse --short HEAD
docker-compose up -d --build
Write-Host "Built image: stockfixer:$($env:VERSION)"

# 7. コミット & タグをリモートにプッシュ
git push origin develop
git push origin $newVersion
Write-Host "Pushed tag: $newVersion"
```

## docker-compose.yml との連携

`docker-compose.yml` では `VERSION` 環境変数を参照してイメージ名を決定する。

```yaml
services:
  stockfixer:
    build:
      args:
        - VERSION=${VERSION:-dev}
    image: stockfixer:${VERSION:-dev}
```

- 環境変数 `VERSION` が未設定の場合は `dev` がデフォルト
- タグの `v` プレフィックスは Docker イメージ名では除去する（例: `v1.2.3` → `stockfixer:1.2.3`）

## Best Practices

### タグ作成タイミング
- テスト全件パス後にタグを作成する
- 未コミットの変更がない状態でタグを付ける
- コミットメッセージとタグメッセージの整合性を保つ

### タグメッセージの書き方
```
Release v1.2.3: 変更内容の要約

- 追加: 新機能の説明
- 修正: バグ修正の説明
- 改善: パフォーマンス改善等
```

### バージョンアップの判断基準
| 変更内容 | バージョン | 例 |
|---------|-----------|-----|
| バグ修正、ドキュメント更新 | PATCH | v1.0.0 → v1.0.1 |
| 新しい銘柄追加、新機能 | MINOR | v1.0.0 → v1.1.0 |
| DB スキーマ変更、API 仕様変更 | MAJOR | v1.0.0 → v2.0.0 |

### `none` の条件
- 以下をすべて満たす場合のみ `version_impact: none` を選択できる
  - 外部向けの挙動（API/CLI/設定/DBスキーマ）に変更がない
  - 実行結果に影響するロジック変更がない
  - 変更が docs / コメント / テスト / CI 設定調整などに限定される

## Troubleshooting

### タグが既に存在する場合
```
fatal: tag 'v1.0.0' already exists
```
→ 別のバージョン番号を使用するか、既存タグを削除してから再作成

### リモートにタグがプッシュされない
```powershell
# 明示的にタグをプッシュ（通常の git push ではタグは送信されない）
git push origin --tags
```

### Docker ビルドでバージョンが dev になる
→ 環境変数 `VERSION` が設定されていない。`VERSION` ファイルから取得して `$env:VERSION` に設定する。
```powershell
$env:VERSION = (Get-Content .\VERSION -Raw).Trim()
```

### VERSIONファイルとGitタグがずれた場合
→ VERSIONファイルを正として扱う。Gitタグを修正するか、VERSIONファイルを最新タグに合わせて更新してコミットする。

## References
- [Semantic Versioning 2.0.0](https://semver.org/lang/ja/)
- [docker-compose.yml](../../../docker-compose.yml)
- [Dockerfile](../../../python/Dockerfile)
