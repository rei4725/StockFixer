# 運用手順書

StockFixer の Docker ビルド・デプロイ・日常運用に関する手順をまとめる。

## ロードマップ管理

- 収益改善に関する計画・優先度・進捗は `docs/ROADMAP_IDEAS.md` を正本として管理する
- 運用手順の変更が必要になった場合のみ、本書へ運用コマンドや手順を追記する
- 設計方針の変更が必要になった場合は `docs/ARCHITECTURE.md` を更新する

---

## 命名規約

### Docker 関連

| 項目 | 規約 | 例 |
|---|---|---|
| イメージ名 | `stockfixer` | 固定 |
| コンテナ名 | `stockfixer` | docker-compose.yml の `container_name` で固定 |
| サービス名 | `stockfixer` | docker-compose.yml の `services` キー |

### イメージタグ規約

| タグ | 用途 | 例 |
|---|---|---|
| `X.Y.Z` | リリースバージョン（セマンティックバージョニング） | `stockfixer:1.2.0` |
| `latest` | 最新安定版 | `stockfixer:latest` |
| `dev` | 開発版（デフォルト） | `stockfixer:dev` |
| `YYYYMMDD` | 日付ベース（日次運用記録用） | `stockfixer:20260301` |

### セマンティックバージョニング

| バージョン要素 | 変更タイミング | 例 |
|---|---|---|
| `MAJOR` (X) | 破壊的変更（DB スキーマ変更、API 非互換等） | `1.0.0` → `2.0.0` |
| `MINOR` (Y) | 機能追加（新モデル追加、新コマンド等） | `1.0.0` → `1.1.0` |
| `PATCH` (Z) | バグ修正、軽微な改善 | `1.0.0` → `1.0.1` |

### イメージに埋め込むメタデータ（OCI ラベル）

| ラベル | 内容 | 例 |
|---|---|---|
| `org.opencontainers.image.version` | バージョン | `1.2.0` |
| `org.opencontainers.image.created` | ビルド日時 | `2026-03-01T12:00:00Z` |
| `org.opencontainers.image.revision` | Git コミットハッシュ | `7ea6501` |
| `org.opencontainers.image.source` | リポジトリ URL | `https://github.com/rei4725/StockFixer` |

### ファイル・ディレクトリ

| 項目 | 規約 | 例 |
|---|---|---|
| 実行スクリプト | `run_*.py`（python/ 直下に集約） | `run_data_creation.py` |
| モデルファイル | `[モデル名].joblib` | `StockXGBoostModel.joblib` |
| 銘柄別モデルディレクトリ | `python/models/[market]_[symbol]/` | `python/models/jp_7203/` |
| 統合モデルディレクトリ | `python/models/unified/` | `python/models/unified/UnifiedStockXGBoost.joblib` |
| データディレクトリ | `python/data/[market]_[symbol]/` | `python/data/jp_7203/` |
| DB ファイル | `python/data/stockfixer.duckdb` | — |

### モデル命名

| モデル種別 | モデル名 | ファイル名 |
|---|---|---|
| 銘柄別 XGBoost | `StockXGBoostModel` | `StockXGBoostModel.joblib` |
| 銘柄別 LightGBM | `StockLightGBMModel` | `StockLightGBMModel.joblib` |
| 統合 XGBoost | `UnifiedStockXGBoost` | `UnifiedStockXGBoost.joblib` |
| 統合 LightGBM | `UnifiedStockLightGBM` | `UnifiedStockLightGBM.joblib` |

---

## 前提条件

- Docker Desktop がインストール済み
- `python/.env` に `DISCORD_BOT_TOKEN` 等の環境変数が設定済み
- 作業ディレクトリは `StockFixer/`（docker-compose.yml がある階層）

---

## Docker ビルド・起動

### 環境変数の設定

ビルド時にバージョン情報を渡すため、環境変数を設定する。

```powershell
# プロジェクトルートで実行
cd C:\src\StockFixer

# バージョン・ビルド情報を設定
$env:VERSION = "1.0.0"
$env:BUILD_DATE = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
$env:GIT_COMMIT = git rev-parse --short HEAD
```

> 環境変数を設定しなかった場合、タグは `dev`、メタデータは `unknown` になる。

### 初回ビルド＆起動

```powershell
# イメージビルド＆バックグラウンド起動
docker compose up -d --build
```

ビルド後のイメージ: `stockfixer:1.0.0`（VERSION 未指定時は `stockfixer:dev`）

### コード変更後の再ビルド

```powershell
# バージョンを上げて再ビルド
$env:VERSION = "1.1.0"
$env:BUILD_DATE = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ")
$env:GIT_COMMIT = git rev-parse --short HEAD
docker compose up -d --build
```

### ビルドのみ（起動しない）

```powershell
docker compose build
```

### キャッシュなしで完全再ビルド

```powershell
docker compose build --no-cache
```

### イメージ情報の確認

```powershell
# イメージ一覧
docker images stockfixer

# ラベル（メタデータ）の確認
docker inspect stockfixer:1.0.0 --format '{{json .Config.Labels}}' | ConvertFrom-Json
```

### コンテナ起動時の注意

**Docker Desktop の GUI から「Run」ボタンで起動しないこと。**  
GUI から直接 Run すると `docker-compose.yml` の `env_file` 設定が適用されず、`.env` の環境変数（`DISCORD_BOT_TOKEN` 等）がコンテナに渡らない。

必ず以下のコマンドで起動する:

```powershell
docker compose up -d
```

---

## コンテナ操作

| 目的 | コマンド |
|---|---|
| 起動（バックグラウンド） | `docker compose up -d` |
| 停止（データ保持） | `docker compose stop` |
| 再開 | `docker compose start` |
| 再起動 | `docker compose restart` |
| 完全削除（データは volumes で残る） | `docker compose down` |
| ログ確認（リアルタイム） | `docker compose logs -f` |
| ログ確認（末尾100行） | `docker compose logs --tail 100` |
| コンテナ状態確認 | `docker ps --filter name=stockfixer` |

---

## ログ確認

### ログファイルの場所

| ファイル | 内容 | ローテーション |
|---|---|---|
| `python/logs/stockfixer.log` | 全ログ（INFO以上） | 10MB×5世代 |
| `python/logs/stockfixer_error.log` | エラーのみ（ERROR以上） | 5MB×3世代 |

### ローカル実行時のログ確認

```powershell
# 最新ログをリアルタイム監視
Get-Content python\logs\stockfixer.log -Wait -Tail 50

# エラーログのみ確認
Get-Content python\logs\stockfixer_error.log -Tail 100
```

### Docker コンテナ内のログ確認

```powershell
# コンテナ内のログファイルを確認
docker exec stockfixer tail -f logs/stockfixer.log
docker exec stockfixer tail -100 logs/stockfixer_error.log
```

### ログレベルの変更

`LOG_LEVEL` 環境変数でログレベルを制御できる（デフォルト: INFO）。

```powershell
# ローカル実行時（DEBUG レベルに上げる）
$env:LOG_LEVEL = "DEBUG"
py run_data_creation.py --batch

# docker-compose.yml の environment セクションに追記
# LOG_LEVEL: DEBUG
```

---

## 手動スクリプト実行

コンテナ内でスクリプトを手動実行する場合は `docker exec` を使用する。

### データ取得

```powershell
# 全銘柄バッチ取得
docker exec stockfixer python run_data_creation.py --batch

# 単一銘柄
docker exec stockfixer python run_data_creation.py --market jp --symbol 7203
```

### 統合モデル学習

```powershell
# XGBoost + LightGBM 両方
docker exec stockfixer python run_unified_model_training.py

# 指定モデルのみ
docker exec stockfixer python run_unified_model_training.py --model-type XGBoostModel --no-both
```

### 銘柄別モデル作成

```powershell
# 全銘柄バッチ
docker exec stockfixer python run_model_creation.py --batch

# 単一銘柄
docker exec stockfixer python run_model_creation.py --market jp --symbol 7203
```

### 予測実行

```powershell
# Top10/Worst10（統合モデル）
docker exec stockfixer python run_predict.py --mode top10

# Top10/Worst10（銘柄別モデル）
docker exec stockfixer python run_predict.py --mode top10 --individual

# 単一銘柄
docker exec stockfixer python run_predict.py --mode single --market jp --symbol 7203

# ウォッチリスト全銘柄
docker exec stockfixer python run_predict.py --mode watchlist
```

### パイプライン即時実行

```powershell
# 日次パイプライン（データ取得 → 予測）を即時実行
docker exec stockfixer python run_scheduler.py --run-now daily

# 週次パイプライン（統合モデル再学習）を即時実行
docker exec stockfixer python run_scheduler.py --run-now weekly
```

---

## 自動スケジュール

コンテナ起動時に `run_scheduler.py --with-bot` が実行され、以下のジョブが自動で動作する。

| ジョブ | スケジュール | 内容 |
|---|---|---|
| `daily_pipeline` | 平日 19:00 | データ取得（バッチ） → 予測（Top10/Worst10） |
| `weekly_model_training` | 土曜 03:00 | 統合モデル再学習（XGBoost + LightGBM） |

---

## 週次自動再デプロイ（Windowsタスクスケジューラ）

毎週土曜 04:00 に `weekly_redeploy.ps1` が自動実行され、コードの最新化・Dockerイメージ再ビルド・コンテナ再起動を行う。

### 実行フロー

```
土曜 03:00  weekly_model_training（APScheduler・コンテナ内）
    ↓ 約30〜60分で完了
土曜 04:00  weekly_redeploy.ps1（Windowsタスクスケジューラ）
    1. git pull origin feature/training
    2. VERSION / BUILD_DATE / GIT_COMMIT を環境変数にセット
    3. docker-compose up -d --build
    4. コンテナ起動確認
    5. ログ出力 → python/logs/redeploy.log
```

### 関連ファイル

| ファイル | 説明 |
|---|---|
| `weekly_redeploy.ps1` | 再デプロイスクリプト本体（プロジェクトルート） |
| `register_task.ps1` | タスクスケジューラ再登録用スクリプト（管理者権限必要） |
| `python/logs/redeploy.log` | 実行ログ（UTF-8、追記形式） |

### タスクスケジューラ確認

```powershell
# タスク状態確認
schtasks /query /tn "StockFixer Weekly Redeploy" /fo LIST

# 即時テスト実行
schtasks /run /tn "StockFixer Weekly Redeploy"

# ログ確認
Get-Content C:\src\StockFixer\python\logs\redeploy.log -Tail 30
```

### タスク再登録（PCセットアップ時など）

```powershell
# 管理者PowerShellで実行
.\register_task.ps1
```

### 注意事項

- タスクは「対話型のみ」モードで登録されているため、**PCにログインしている状態**でのみ実行される
- 管理者権限で「バックグラウンドでも実行」にしたい場合は、`register_task.ps1` で再登録し、タスクスケジューラのGUIから「ユーザーのログオン状態にかかわらず実行する」に変更する
- `git pull` の認証は Git Credential Manager（Windows標準）に記憶済みの認証情報を使用する

---

## ローカル実行（Docker 不使用）

Docker を使わずローカルで直接実行する場合。

```powershell
cd C:\src\StockFixer\python

# 仮想環境の作成・有効化
py -m venv .venv
.\.venv\Scripts\Activate

# 依存パッケージのインストール
pip install -r requirements.txt

# 各スクリプトの実行
py run_data_creation.py --batch
py run_unified_model_training.py
py run_predict.py --mode top10
py run_scheduler.py --with-bot
```

---

## トラブルシューティング

### コンテナが起動しない

```powershell
# ログで原因確認
docker compose logs

# .env ファイルの存在確認
Test-Path python/.env
```

### DB ロックエラー

DuckDB は単一プロセスの排他書き込みロックを使用する。常駐コンテナを停止してから書き込み操作を行うこと。

```powershell
# 常駐コンテナを停止
docker compose stop

# 書き込みを伴う操作を実行
docker compose run --rm stockfixer python run_data_creation.py --batch

# 常駐コンテナを再開
docker compose start
```

### 差分更新処理での競合回避（2026-03 反映）

`run_data_creation.py --batch` / 日次パイプラインでは、DB競合回避のため処理を以下の2フェーズに分離している。

1. 並列フェーズ: データ取得・特徴量生成のみ（DB書き込みなし）
2. 逐次フェーズ: `market_data_raw` と `stock_features` のDB保存

この構成により、差分取得時の `market_data_raw` への `INSERT OR REPLACE` が並列で衝突するリスクを抑制する。

運用上の注意:

- DB書き込みを含むスクリプトは同時に複数起動しない
- 定期実行は `run_scheduler.py` の単一プロセス運用を維持する
- 手動実行を重ねる場合は先行ジョブ完了後に実行する

### イメージサイズが大きい場合

```powershell
# 不要なイメージ・キャッシュを削除
docker system prune -f
docker builder prune -f
```

---

## ボリュームマウント構成

| ホスト側パス | コンテナ内パス | 内容 |
|---|---|---|
| `python/data/` | `/app/data/` | DuckDB、銘柄別サブディレクトリ |
| `python/models/` | `/app/models/` | 学習済みモデル（.joblib） |
| `python/results/` | `/app/results/` | 予測結果 |
| `python/logs/` | `/app/logs/` | ログファイル（stockfixer.log 等） |

> `data/` `models/` `results/` `logs/` は `.dockerignore` でイメージから除外し、bind mount で永続化している。

---

*Last updated: 2026-03-16*
