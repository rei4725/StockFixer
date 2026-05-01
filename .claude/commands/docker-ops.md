Docker コンテナのデプロイ・管理を行う。デプロイの中心は `weekly_redeploy.ps1`。

## 手動デプロイ手順

### Step 1: バージョンを上げる
```powershell
$current = (Get-Content VERSION).Trim()
$parts   = $current.Split(".")
$newVer  = "$($parts[0]).$($parts[1]).$([int]$parts[2] + 1)"
Set-Content VERSION $newVer
```

### Step 2: コミット & タグ
```powershell
git add VERSION
git commit -m "chore: bump version to $newVer"
git tag -a "v$newVer" -m "Release v$newVer"
```

### Step 3: デプロイ実行
```powershell
powershell -ExecutionPolicy Bypass -File C:\src\StockFixer\weekly_redeploy.ps1
```

`weekly_redeploy.ps1` の処理内容:
1. `git pull origin develop`
2. `python -m pytest tests/unit -v`（失敗時は即中断）
3. `VERSION` / `GIT_COMMIT` / `BUILD_DATE` を環境変数にセット
4. `docker-compose up -d --build`
5. コンテナ起動確認
6. 結果を `python/logs/redeploy.log` に記録

## 状態確認・ログ
```powershell
# コンテナ起動状態
docker ps --filter "name=stockfixer" --format "table {{.Names}}`t{{.Status}}`t{{.Image}}"

# アプリログ（リアルタイム）
docker-compose logs -f stockfixer

# デプロイログ
Get-Content C:\src\StockFixer\python\logs\redeploy.log -Tail 30

# 停止
docker-compose down
```

## 週次自動デプロイ設定
```powershell
# タスクスケジューラー登録（初回のみ・管理者権限）
C:\src\StockFixer\register_task.ps1

# 即時テスト実行
Start-ScheduledTask -TaskName "StockFixer Weekly Redeploy"

# 状態確認
Get-ScheduledTask -TaskName "StockFixer Weekly Redeploy" | Select-Object TaskName, State, LastRunTime, NextRunTime
```

## コンテナ構成
| 項目 | 値 |
|------|-----|
| コンテナ名 | `stockfixer` |
| イメージタグ | `stockfixer:<VERSION>` |
| 再起動ポリシー | `always` |

## ボリュームマウント
| ホスト側 | コンテナ側 | 用途 |
|---------|-----------|------|
| `./python/data` | `/app/data` | 株価データ・DuckDB永続化 |
| `./python/models` | `/app/models` | 学習済みモデル永続化 |
| `./python/results` | `/app/results` | 予測結果永続化 |
