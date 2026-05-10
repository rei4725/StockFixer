# デプロイ Runbook

## 重要制約: Docker 単一プロセス起動

**DuckDB は1ファイルにつき読み書き接続を1つしか許可しない。**
本システムでは以下の制約を必ず守ること。

### NG パターン（絶対禁止）
- Docker コンテナ稼働中に ホストの Python から run_*.py を直接実行する
- 複数の docker-compose サービスが同一 stockfixer.duckdb を読み書きする
- `docker exec` で複数の run_*.py を同時並行起動する

### OK パターン
- コンテナ停止後にホストで run_*.py を実行する
- `docker exec stockfixer python run_*.py`（直列1本ずつ）
- 読み取り専用スクリプトはホストから並行実行可

### ロック状態の確認
`python/data/stockfixer.duckdb.lock` が存在する場合、プロセスが DB を使用中。

---

## 1. 正常デプロイ手順

週次自動再デプロイ（`weekly_redeploy.ps1`）が毎週月曜 02:00 に自動実行されるが、以下は手動で同等の操作を行う手順。

### 1-1. 前提確認

```powershell
# 作業ディレクトリ
cd C:\src\StockFixer

# コンテナ状態確認
docker ps --filter name=stockfixer

# 現在の VERSION 確認
Get-Content VERSION
```

### 1-2. git タグ付け & プッシュ

```powershell
# 最新コードを取得
git pull origin develop

# バージョンを更新（例: 1.14.0 → 1.15.0）
# VERSION ファイルを編集後
git add VERSION
git commit -m "chore: バージョンを 1.15.0 に更新"
git tag v1.15.0
git push origin develop --tags
```

### 1-3. ビルド用環境変数セット

```powershell
$env:VERSION    = (Get-Content VERSION).Trim()
$env:BUILD_DATE = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
$env:GIT_COMMIT = git rev-parse --short HEAD

Write-Host "VERSION=$($env:VERSION)  GIT_COMMIT=$($env:GIT_COMMIT)"
```

### 1-4. Docker イメージビルド & コンテナ再起動

```powershell
docker compose up -d --build
```

### 1-5. 起動確認

```powershell
# コンテナが Up になるまで待機（5秒後に確認）
Start-Sleep -Seconds 5
docker ps --filter name=stockfixer --format "{{.Status}}"

# 起動ログ確認
docker compose logs --tail 30
```

### 1-6. weekly_redeploy.ps1 を手動実行する場合

```powershell
# スクリプトはプロジェクトルートに配置されている
.\weekly_redeploy.ps1

# 実行ログ確認
Get-Content Logs\redeploy.log -Tail 50
```

`weekly_redeploy.ps1` は以下を自動で実行する:
1. `git pull origin develop`
2. pip パッケージ同期
3. Smoke Test / Unit Test / E2E Test（失敗時はデプロイ中断）
4. `VERSION` / `BUILD_DATE` / `GIT_COMMIT` を環境変数にセット
5. `docker compose up -d --build`
6. コンテナ起動確認 → `Logs/redeploy.log` に記録

---

## 2. ロールバック手順

デプロイ後に問題が発生した場合、前バージョンのイメージに戻す。

### 2-1. 問題の確認

```powershell
# エラーログ確認
docker compose logs --tail 100
Get-Content Logs\stockfixer_error.log -Tail 50
```

### 2-2. 現在のコンテナ停止

```powershell
docker compose stop
```

### 2-3. ロールバック先バージョンの確認

```powershell
# ローカルに保存されているイメージ一覧
docker images stockfixer

# git タグ一覧
git tag --sort=-version:refname | Select-Object -First 10
```

### 2-4. 前バージョンのコードに戻す

```powershell
# 前バージョンのタグ（例: v1.14.0）をチェックアウト
git checkout v1.14.0

# 前バージョンでイメージを再ビルド
$env:VERSION    = "1.14.0"
$env:BUILD_DATE = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
$env:GIT_COMMIT = git rev-parse --short HEAD

docker compose up -d --build
```

### 2-5. バージョン切り戻し後の確認

```powershell
# コンテナ稼働確認
docker ps --filter name=stockfixer

# イメージのバージョンラベル確認
docker inspect stockfixer --format '{{index .Config.Labels "org.opencontainers.image.version"}}'

# 日次パイプラインの疎通テスト
docker exec stockfixer python run_predict.py --mode single --market us --symbol AAPL
```

### 2-6. 正常バージョンに戻す

ロールバック後、develop ブランチの修正が完了したら以下で正常状態に戻す:

```powershell
git checkout develop
git pull origin develop
# → 「1. 正常デプロイ手順」を再実行
```

---

## 3. 手動デプロイ手順（Docker 直接操作）

スケジューラーや `weekly_redeploy.ps1` を使わず、手動で Docker を操作する場合。

### 3-1. コンテナ起動・停止

```powershell
# バックグラウンド起動
docker compose up -d

# 停止（データは volumes で保持）
docker compose stop

# 再開
docker compose start

# 再起動
docker compose restart

# 完全削除（データは volumes で残る）
docker compose down
```

> **注意**: Docker Desktop の GUI「Run」ボタンは使用しない。`env_file` 設定が適用されず `.env` の環境変数（`DISCORD_BOT_TOKEN` 等）がコンテナに渡らない。

### 3-2. イメージビルドのみ

```powershell
# 通常ビルド
docker compose build

# キャッシュなし完全再ビルド（依存ライブラリを丸ごと再インストールする場合）
docker compose build --no-cache
```

### 3-3. コンテナ内でのスクリプト手動実行

コンテナが起動中の場合は `docker exec` で直列に実行する（並列起動禁止）。

```powershell
# データ取得（全銘柄）
docker exec stockfixer python run_data_creation.py --batch

# 単一銘柄
docker exec stockfixer python run_data_creation.py --market jp --symbol 7203

# 統合モデル学習
docker exec stockfixer python run_unified_model_training.py

# 予測実行（Top10/Worst10）
docker exec stockfixer python run_predict.py --mode top10

# 日次パイプライン即時実行
docker exec stockfixer python run_scheduler.py --run-now daily

# 週次パイプライン即時実行
docker exec stockfixer python run_scheduler.py --run-now weekly
```

### 3-4. ログ確認

```powershell
# リアルタイムログ
docker compose logs -f

# 末尾100行
docker compose logs --tail 100

# エラーのみ
Get-Content Logs\stockfixer_error.log -Tail 100
```

---

## 4. バックアップのリストア手順（NF-602）

DuckDB データ（`python/data/stockfixer.duckdb`）および学習済みモデル（`python/models/`）のリストア手順。

### 4-1. リストア前の確認

```powershell
# コンテナを停止してから操作する
docker compose stop

# 現在の DB ファイルサイズ確認
Get-Item python\data\stockfixer.duckdb | Select-Object Name, Length, LastWriteTime
```

### 4-2. DuckDB ファイルのリストア

```powershell
# バックアップファイルの場所確認
Get-ChildItem python\data -Filter "*.duckdb.bak*"

# リストア（既存ファイルをバックアップしてから上書き）
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
Rename-Item python\data\stockfixer.duckdb "stockfixer.duckdb.broken_$timestamp"

# バックアップからリストア
Copy-Item python\data\stockfixer.duckdb.bak python\data\stockfixer.duckdb
```

### 4-3. モデルファイルのリストア

```powershell
# バックアップからモデルをリストア
Copy-Item -Recurse python\models.bak\* python\models\ -Force
```

### 4-4. リストア後の疎通確認

```powershell
# コンテナを再起動
docker compose up -d

# DB が読み取れるか確認
docker exec stockfixer python -c "
import duckdb
con = duckdb.connect('data/stockfixer.duckdb', read_only=True)
print(con.execute('SELECT COUNT(*) FROM stock_features').fetchone())
con.close()
"

# 予測が動作するか確認
docker exec stockfixer python run_predict.py --mode single --market us --symbol AAPL
```

### 4-5. ロールバック不可の場合（フルリビルド）

バックアップが存在しない場合や破損が激しい場合は、データを再取得して再構築する。

```powershell
# DB を空状態に初期化（注意: 全データが消える）
docker compose stop
Remove-Item python\data\stockfixer.duckdb -ErrorAction SilentlyContinue
Remove-Item python\data\stockfixer.duckdb.wal -ErrorAction SilentlyContinue

# コンテナ再起動後、全銘柄のデータを再取得
docker compose up -d
docker exec stockfixer python run_data_creation.py --batch
docker exec stockfixer python run_unified_model_training.py
```

---

## 5. DuckDB ロック競合時の復旧手順（NF-601）

### 5-1. 症状

```
IOException: Could not set lock on file 'python/data/stockfixer.duckdb'
RuntimeError: DuckDB書き込みロック取得タイムアウト
```

### 5-2. 原因の特定

```powershell
# ロックファイルの存在確認
Test-Path python\data\stockfixer.duckdb.lock

# DuckDB を使用中のプロセスを確認
Get-Process python -ErrorAction SilentlyContinue | Select-Object Id, ProcessName, StartTime

# スケジューラーのジョブ実行状況を確認
docker compose logs --tail 50 | Select-String "開始|完了|ERROR"
```

### 5-3. 復旧手順

1. **手動実行中のスクリプトを中断する**
   ```powershell
   # ホストで実行中の場合: Ctrl+C で中断
   # docker exec で実行中の場合: 別のターミナルで
   docker exec stockfixer pkill -f "run_"
   ```

2. **スケジューラーのジョブ完了を待つ**
   ```powershell
   # リアルタイムで完了ログを監視
   docker compose logs -f | Select-String "完了|failed|ERROR"
   ```

3. **ロックファイルが残存している場合は削除する**
   ```powershell
   # プロセスがすべて終了していることを確認してから実行
   Remove-Item python\data\stockfixer.duckdb.lock -ErrorAction SilentlyContinue
   Remove-Item python\data\stockfixer.duckdb.wal  -ErrorAction SilentlyContinue
   ```

4. **コンテナを再起動して正常状態に戻す**
   ```powershell
   docker compose restart

   # 起動確認
   Start-Sleep -Seconds 5
   docker ps --filter name=stockfixer --format "{{.Status}}"
   ```

5. **操作を再実行する**

### 5-4. 予防策

- スケジューラー稼働中の手動実行は `docker exec` 経由で**直列に1本ずつ**行う
- `docker exec` で複数コマンドを同時実行しない
- `docs/OPERATIONS.md` の「単一プロセス制約」を参照する
- 詳細な検出パターンは `docs/LOCK_DETECTION_GUIDE.md` を参照する

### 5-5. エラー復旧

上記手順で回復しない場合は `docs/INCIDENT_RESPONSE.md` の「DuckDB ロック競合」節を参照。

---

## 関連ドキュメント

| ドキュメント | 内容 |
|---|---|
| [OPERATIONS.md](OPERATIONS.md) | 日常運用手順・Docker 操作・スケジュールジョブ一覧 |
| [INCIDENT_RESPONSE.md](INCIDENT_RESPONSE.md) | 障害レベル定義・初動対応・ポストモーテムテンプレート |
| [LOCK_DETECTION_GUIDE.md](LOCK_DETECTION_GUIDE.md) | DuckDB ロック問題の自動検出ガイド |
| [VERSIONING_POLICY.md](VERSIONING_POLICY.md) | セマンティックバージョニング判定基準 |

---

*Last updated: 2026-05-10*
