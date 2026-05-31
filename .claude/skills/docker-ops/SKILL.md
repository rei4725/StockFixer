---
name: docker-ops
description: "Dockerコンテナのビルド・起動・ログ確認を行う。Docker・デプロイ・本番環境・weekly_redeploy・register_taskの話題では必ずこのスキルを使用する。コンテナ管理・docker-compose・スケジュールタスクの設定が絡む場合も使用する。"
compatibility: "Docker, docker-compose, PowerShell 5.1+。C:\\src\\StockFixer で実行。"
---

## Goal
デプロイは `weekly_redeploy.ps1` を中心に行う。
毎デプロイで `VERSION` ファイルのパッチ番号を +1 し、コミット・タグを打ってから実行する。

---

## 手動デプロイ手順（標準フロー）

### Step 1: パッチバージョンを上げる
```powershell
cd C:\src\StockFixer

# VERSIONファイルを読んでパッチ+1
$current = (Get-Content VERSION).Trim()
$parts   = $current.Split(".")
$newVer  = "$($parts[0]).$($parts[1]).$([int]$parts[2] + 1)"

Set-Content VERSION $newVer
Write-Host "バージョン: $current -> $newVer"
```

### Step 2: コミット & タグ
```powershell
git add VERSION
git commit -m "chore: bump version to $newVer"
git tag -a "v$newVer" -m "Release v$newVer"
```

### Step 3: デプロイ実行（weekly_redeploy.ps1）
```powershell
# weekly_redeploy.ps1 は VERSION ファイルを読んで docker-compose up --build する
powershell -ExecutionPolicy Bypass -File C:\src\StockFixer\weekly_redeploy.ps1
```

> `weekly_redeploy.ps1` の処理内容:
> 1. `git pull origin develop`
> 2. `python -m pytest tests/unit -v` を実行（失敗時は即中断）
> 3. `VERSION` / `GIT_COMMIT` / `BUILD_DATE` を環境変数にセット
> 4. `docker-compose up -d --build`
> 5. コンテナ起動確認
> 6. 結果を `Logs/redeploy.log` に記録

### デプロイゲート（必須）
- UnitTest が 1 件でも失敗した場合、`weekly_redeploy.ps1` は `exit 1` で終了する
- この場合、Docker イメージの再ビルド・再起動は実行されない
- ログ確認: `Get-Content C:\src\StockFixer\Logs\redeploy.log -Tail 100`

---

## 週次自動デプロイの設定

### タスクスケジューラーへの登録（初回のみ・管理者権限で実行）
```powershell
# 管理者 PowerShell で実行
C:\src\StockFixer\register_task.ps1
```
- 毎週土曜 04:00 に `weekly_redeploy.ps1` が自動実行される
- タスク名: `StockFixer Weekly Redeploy`

### タスクの即時テスト実行
```powershell
Start-ScheduledTask -TaskName "StockFixer Weekly Redeploy"
```

### タスクの状態確認
```powershell
Get-ScheduledTask -TaskName "StockFixer Weekly Redeploy" | Select-Object TaskName, State, LastRunTime, NextRunTime
```

---

## 状態確認・ログ

### コンテナ起動状態
```powershell
docker ps --filter "name=stockfixer" --format "table {{.Names}}\t{{.Status}}\t{{.Image}}"
```

### アプリログ（リアルタイム）
```powershell
docker-compose logs -f stockfixer
```

### デプロイログ
```powershell
Get-Content C:\src\StockFixer\Logs\redeploy.log -Tail 30
```

### 停止
```powershell
docker-compose down
```

---

## コンテナ構成

| 項目 | 値 |
|------|-----|
| コンテナ名 | `stockfixer` |
| イメージタグ | `stockfixer:<VERSION>` |
| 再起動ポリシー | `always` |
| ログドライバ | `json-file` (10MB × 3) |

### ボリュームマウント
| ホスト側 | コンテナ側 | 用途 |
|---------|-----------|------|
| `./python/data` | `/app/data` | 株価データ・DuckDB永続化 |
| `./python/models` | `/app/models` | 学習済みモデル永続化 |
| `./python/results` | `/app/results` | 予測結果永続化 |

### ビルド引数（自動セット）
| 変数 | 値 |
|------|-----|
| `VERSION` | `VERSION` ファイルの内容 |
| `GIT_COMMIT` | `git rev-parse --short HEAD` |
| `BUILD_DATE` | 実行時のISO日時 |

---

## References
- [weekly_redeploy.ps1](../../../weekly_redeploy.ps1)
- [register_task.ps1](../../../register_task.ps1)
- [docker-compose.yml](../../../docker-compose.yml)
- [Dockerfile](../../../python/Dockerfile)
