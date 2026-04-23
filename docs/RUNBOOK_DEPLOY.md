# デプロイ・運用 Runbook

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

## エラー復旧
DuckDB ロック競合が発生した場合は `docs/OPERATIONS.md` の「DuckDB ロック競合エラー」を参照。
