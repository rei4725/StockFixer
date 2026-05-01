Git タグベースのセマンティックバージョニングでバージョンを管理し、Docker イメージと連動させる。

バージョン判定の正本: `docs/VERSIONING_POLICY.md`

## SOURCE OF TRUTH
`StockFixer/VERSION` がバージョンの唯一の正。
- バージョンアップ時は **必ず VERSION ファイルを先に更新**してからコミット・タグ作成
- Docker イメージ名は `v` プレフィックスなし（例: `stockfixer:1.2.3`）

## バージョン体系（SemVer）
| 区分 | 変更契機 |
|------|---------|
| MAJOR | 互換性のない大規模変更（DB スキーマ、API 仕様変更） |
| MINOR | 後方互換性のある機能追加（新銘柄、新機能） |
| PATCH | バグ修正・小改善、ドキュメント |
| none | docs/コメント/テスト/CI設定のみの変更 |

## バージョンアップ手順

### PATCH バージョンアップ
```powershell
$current = (Get-Content C:\src\StockFixer\VERSION -Raw).Trim()
$parts = $current.Split(".")
$newVersionNum = "$($parts[0]).$($parts[1]).$([int]$parts[2] + 1)"
$newVersionNum | Set-Content C:\src\StockFixer\VERSION -NoNewline
git add VERSION
git commit -m "chore: bump version to $newVersionNum"
git tag -a "v$newVersionNum" -m "Release v$newVersionNum"
```

### MINOR バージョンアップ
```powershell
$current = (Get-Content C:\src\StockFixer\VERSION -Raw).Trim()
$parts = $current.Split(".")
$newVersionNum = "$($parts[0]).$([int]$parts[1] + 1).0"
$newVersionNum | Set-Content C:\src\StockFixer\VERSION -NoNewline
git add VERSION
git commit -m "chore: bump version to $newVersionNum"
git tag -a "v$newVersionNum" -m "Release v$newVersionNum"
```

### MAJOR バージョンアップ
```powershell
$current = (Get-Content C:\src\StockFixer\VERSION -Raw).Trim()
$parts = $current.Split(".")
$newVersionNum = "$([int]$parts[0] + 1).0.0"
$newVersionNum | Set-Content C:\src\StockFixer\VERSION -NoNewline
git add VERSION
git commit -m "chore: bump version to $newVersionNum"
git tag -a "v$newVersionNum" -m "Release v$newVersionNum"
```

### タグをリモートにプッシュ
```powershell
git push origin v1.2.3    # 特定タグ
git push origin --tags    # 全タグ
```

## 現在のバージョン確認
```powershell
git describe --tags --abbrev=0
git tag -l "v*" --sort=-version:refname
```

## Docker ビルド（VERSION ファイルを参照）
```powershell
$env:VERSION = (Get-Content .\VERSION -Raw).Trim()
$env:BUILD_DATE = (Get-Date -Format "yyyy-MM-ddTHH:mm:ssZ")
$env:GIT_COMMIT = git rev-parse --short HEAD
docker-compose up -d --build
```

## Troubleshooting
- `fatal: tag 'v1.0.0' already exists` → 別バージョンを使用するか既存タグを削除して再作成
- Docker ビルドでバージョンが `dev` になる → `$env:VERSION` が未設定。VERSION ファイルから取得する
- VERSION ファイルと Git タグがずれた場合 → VERSION ファイルを正として扱い、Gitタグを修正する
