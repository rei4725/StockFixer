---
name: docker-ops
description: >-
  Dockerコンテナのビルド・起動・ログ確認を行う。
  Docker、コンテナ、docker-compose、デプロイ、ビルド、
  本番環境、運用の話題で使用する。
metadata:
  author: StockFixer
  version: "1.0"
compatibility: >-
  Docker, docker-compose。プロジェクトルートで実行。
---

## Goal
Docker環境でのビルド・起動・ログ確認を正確に行う。

## Procedure

### ビルド＆起動
```bash
docker-compose up -d --build
```

### ログ確認
```bash
docker-compose logs -f stockfixer
```

### 停止
```bash
docker-compose down
```

### コンテナ構成
- 単一コンテナ構成（`stockfixer`）
- `restart: always` で常時稼働
- イメージ: `stockfixer:${VERSION:-dev}`

### ボリュームマウント
| ホスト側 | コンテナ側 | 用途 |
|---------|-----------|------|
| `./python/data` | `/app/data` | 株価データ・DuckDB永続化 |
| `./python/models` | `/app/models` | 学習済みモデル永続化 |
| `./python/results` | `/app/results` | 予測結果永続化 |

### 環境変数
- `python/.env` から `env_file` で読み込み
- ビルド引数: `BUILD_DATE`, `GIT_COMMIT`, `VERSION`

### ログ設定
- ドライバ: `json-file`
- 最大サイズ: 10MB × 3ファイル

## References
- [docker-compose.yml](../../../docker-compose.yml)
- [Dockerfile](../../../python/Dockerfile)
