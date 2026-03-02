---
name: version-mgmt
description: "Gitタグによるバージョン管理とDockerイメージのバージョン付きビルドを行う。バージョン、version、タグ、tag、リリース、release、セマンティックバージョニング、semver、バージョンアップの話題で使用する。"
metadata:
  author: StockFixer
  version: "1.0"
compatibility: "Git 2.0+, Docker, docker-compose。プロジェクトルートで実行。"
---

## Goal
Git タグベースのセマンティックバージョニングでプロジェクトのバージョンを管理し、Dockerイメージと連動させる。

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

### 2. バージョンを上げる（タグ作成）

#### PATCH バージョンアップ（バグ修正・小改善）
```powershell
# 最新タグを取得して次のPATCHバージョンを計算
$latest = git describe --tags --abbrev=0 2>$null
if (-not $latest) { $latest = "v0.0.0" }
$parts = $latest.TrimStart("v").Split(".")
$newVersion = "v$($parts[0]).$($parts[1]).$([int]$parts[2] + 1)"
git tag -a $newVersion -m "Release $newVersion"
Write-Host "Created tag: $newVersion"
```

#### MINOR バージョンアップ（機能追加）
```powershell
$latest = git describe --tags --abbrev=0 2>$null
if (-not $latest) { $latest = "v0.0.0" }
$parts = $latest.TrimStart("v").Split(".")
$newVersion = "v$($parts[0]).$([int]$parts[1] + 1).0"
git tag -a $newVersion -m "Release $newVersion"
Write-Host "Created tag: $newVersion"
```

#### MAJOR バージョンアップ（破壊的変更）
```powershell
$latest = git describe --tags --abbrev=0 2>$null
if (-not $latest) { $latest = "v0.0.0" }
$parts = $latest.TrimStart("v").Split(".")
$newVersion = "v$([int]$parts[0] + 1).0.0"
git tag -a $newVersion -m "Release $newVersion"
Write-Host "Created tag: $newVersion"
```

#### 手動でバージョン指定
```powershell
git tag -a v1.2.3 -m "Release v1.2.3: 機能追加の説明"
```

### 3. タグをリモートにプッシュ
```powershell
# 特定のタグをプッシュ
git push origin v1.2.3

# 全タグをプッシュ
git push origin --tags
```

### 4. Docker イメージをバージョン付きでビルド

タグからバージョンを取得し、Docker イメージに適用する。

```powershell
cd C:\src\StockFixer

# Git タグからバージョンを取得
$VERSION = git describe --tags --abbrev=0 2>$null
if (-not $VERSION) { $VERSION = "dev" }
$VERSION = $VERSION.TrimStart("v")

# ビルド情報を設定
$env:VERSION = $VERSION
$env:BUILD_DATE = (Get-Date -Format "yyyy-MM-ddTHH:mm:ssZ")
$env:GIT_COMMIT = git rev-parse --short HEAD

# ビルド＆起動
docker-compose up -d --build

Write-Host "Built image: stockfixer:$VERSION"
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

# 2. 現在のバージョン取得
$latest = git describe --tags --abbrev=0 2>$null
if (-not $latest) { $latest = "v0.0.0" }
$parts = $latest.TrimStart("v").Split(".")

# 3. 新バージョン計算
switch ($bumpType) {
    "patch" { $newVersion = "v$($parts[0]).$($parts[1]).$([int]$parts[2] + 1)" }
    "minor" { $newVersion = "v$($parts[0]).$([int]$parts[1] + 1).0" }
    "major" { $newVersion = "v$([int]$parts[0] + 1).0.0" }
}

# 4. タグ作成
git tag -a $newVersion -m "Release $newVersion"
Write-Host "Tagged: $newVersion"

# 5. Docker ビルド
$env:VERSION = $newVersion.TrimStart("v")
$env:BUILD_DATE = (Get-Date -Format "yyyy-MM-ddTHH:mm:ssZ")
$env:GIT_COMMIT = git rev-parse --short HEAD
docker-compose up -d --build
Write-Host "Built image: stockfixer:$($env:VERSION)"

# 6. タグをリモートにプッシュ
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
→ 環境変数 `VERSION` が設定されていない。タグから取得して `$env:VERSION` に設定する。

## References
- [Semantic Versioning 2.0.0](https://semver.org/lang/ja/)
- [docker-compose.yml](../../../docker-compose.yml)
- [Dockerfile](../../../python/Dockerfile)
