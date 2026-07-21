# DuckDB → PostgreSQL 移行設計

## 背景・課題

StockFixer は現状、組み込み型の DuckDB（単一ファイル `python/data/stockfixer.duckdb`）をデータストアとして使っている。プロジェクトの成長に伴い、これが以下の形で運用上の負担になってきている。

1. **単一ライター制約による直列化**: DuckDB はプロセスあたり読み書き接続を1つしか持てないため、`src/utils/db/_connection.py` は `FileLock`（プロセス間mutex）で全DB操作を直列化している。スケジューラー・Bot・APIサーバーが同時に動く現行構成では、この直列化がロック競合・タイムアウト（`DbLockTimeoutError`）の温床になっている。
2. **テストの脆弱性**: 埋め込み型DBゆえ、テストが本物のファイルパス・ロック機構と直接やり取りせざるを得ない。実際に以下の事故が発生している。
   - unitテストが実際の `compact` 処理を本番DuckDBファイルに実行し破損させた事故（#548）。
   - `filelock` ライブラリのマイナーバージョンアップ（3.29.4→3.29.5）でロックファイルの削除挙動が変わり、integration testのteardownが `.duckdb.lock` 残置により一斉に `ENOTEMPTY` で失敗した事故（PR #556 で暫定対処）。
3. **スキーマ管理の二重化・乖離**: `_connection.py` の `_init_tables()`（inline DDL、起動のたび実行）と `src/utils/db/migrations/*.sql`（バージョン管理用の番号付きSQL）が併存しており、既に内容がズレている。`strategy_promotions`・`accuracy_weekly_snapshots`・`earnings_calendar`・`stock_fundamentals` 等、複数テーブルが `migrations/0001_initial.sql` に反映されておらず inline 側にしか存在しない。

これらは将来のマイクロサービス的な分割（例: 戦略ファクトリー夜間バッチや実発注ジョブの別プロセス化）を検討する上での前提条件でもある。埋め込み型DBのままではプロセスをまたいだ安全な並行書き込みが原理的に成立しないため、まずクライアント/サーバー型RDBへ移行し、堅牢な土台を作る。

**本設計のスコープはPostgreSQL移行のみ**であり、プロセス/サービス分割は別スペックとして後日扱う（本ドキュメントでは前提条件を整えるところまでとする）。

## 目的

1. DuckDBをPostgreSQL（自己ホスティング、docker-compose追加）に完全置換する。
2. `FileLock` による直列化を撤去し、Postgresのトランザクション分離による正しい並行書き込みに置き換える。
3. スキーマ管理を `migrations/` 配下の番号付きSQLファイルに一本化し、inline DDLとの乖離問題を解消する。
4. 既存の全テーブルデータ（`stock_features` の過去履歴、`prediction_results`、`paper_orders` 等の取引履歴を含む）を欠損なく引き継ぐ。
5. テストをトランザクションロールバック方式に切り替え、DB破損・ロック起因のテスト事故を構造的に起きなくする。

## 非対象（スコープ外）

- プロセス/サービス分割（戦略ファクトリー・実発注ジョブ等の別プロセス化）は別スペックとする。本設計はその前提条件を整えるところまで。
- DuckDBの分析性能を活かすハイブリッド構成（`postgres_scanner` でPostgresをDuckDBから読む等）は採用しない。完全置換とする。
- Kabu Station実発注のテーブル構造・ビジネスロジックの変更は対象外。既存スキーマのまま移行する。

## アーキテクチャ

- `docker-compose.yml` に `postgres` サービス（`postgres:16-alpine` を想定）を追加し、named volumeでデータを永続化する。
- scheduler / bot / api の全プロセスが同一Postgresインスタンスへ接続する。接続情報は `.env` の `DATABASE_URL` として保持し、`config/settings.py` 経由で読む（既存の `DISCORD_BOT_TOKEN` 等と同じ流儀）。
- `src/utils/db/_connection.py` の `FileLock` / `DbLockTimeoutError` 機構は撤去する。Postgresの通常のトランザクション分離レベルで安全な並行アクセスを処理できるため不要になる。

## スキーマ管理の一本化

- `_connection.py` の inline DDL（`_init_tables()`）は廃止し、スキーマ定義を `src/utils/db/migrations/` 配下の番号付きSQLファイルのみに一本化する。
- 現行の `_init_tables()` の最新状態（前述の乖離を解消した全テーブル）を正として、Postgres構文で `0001_baseline_postgres.sql` として書き起こす。
- `run_migrations()` はpsycopg版に置き換えるが、「`schema_migrations` テーブルで適用済みバージョンを管理する」という現行方式はそのまま踏襲する。

## データ移行手順（ビッグバング切り替え）

個人運用規模であり、数分〜数十分のダウンタイムは許容できる前提のもと、以下の手順で一括切り替えを行う。

1. **メンテナンス開始**: スケジューラー・Bot・APIを停止する（`docker-compose stop`）。
2. **DuckDBバックアップ**: `stockfixer.duckdb` を別名でコピー保存する（ロールバック用に検証期間中保持）。
3. **Postgres起動**: `docker-compose up -d postgres` → 上記マイグレーション一式を適用し、空スキーマを構築する。
4. **データ移行スクリプト実行**: 新規スクリプト `scripts/migrate_to_postgres.py` を作成し、DuckDBのPostgres Attach拡張を使って一括移行する。

   ```sql
   -- 接続文字列は例示。実スクリプトでは .env の DATABASE_URL から組み立てる
   ATTACH 'postgresql://user:pass@postgres:5432/stockfixer' AS pg (TYPE POSTGRES);
   TRUNCATE TABLE pg.stock_features;
   INSERT INTO pg.stock_features SELECT * FROM stock_features;
   -- 以下、_init_tables() のDDLから列挙した全テーブル分を機械的に繰り返す
   ```

   移行前に対象テーブルを `TRUNCATE` してからINSERTすることで、スクリプトの再実行を安全にする（べき等性の確保）。
5. **整合性検証**: 各テーブルの `COUNT(*)` をDuckDB側・Postgres側で突き合わせる。`paper_balance`・`paper_positions`・`paper_orders` など金額に関わるテーブルは合計値（`SUM(realized_pnl)` 等）も照合する。検証用スクリプトとして自動化し、不一致があれば切り替えを中止する。
6. **切り替え**: 検証OKなら `.env` の `DATABASE_URL` を有効化し、`_connection.py` の接続先をPostgresに向けて全プロセスを再起動する。
7. **保持期間**: DuckDBファイルは検証期間（目安2週間）バックアップとして残し、問題なければ削除する。

## アクセス層の変更

- `_connection.py` を全面書き換える: `duckdb.connect` → `psycopg`（psycopg3、`psycopg_pool.ConnectionPool` によりプロセスごとに接続プールを持つ）。
- `get_readonly_connection()` は「別プロセス専用」という現行の制約（DuckDBの排他仕様に起因）が消えるため、プールから接続を取得するだけの薄い関数に簡略化する。
- `src/utils/db/` 配下の各モジュール（`stock_features.py` 等、計14ファイル）はSQL文自体はほぼそのまま流用できる（標準SQLで書かれており、`INSERT ... ON CONFLICT DO UPDATE` のようなupsert構文もPostgres/DuckDB両対応）。プレースホルダ構文のみ機械的な置換が必要（DuckDBの `?` → psycopgの `%s`）。
- アクセス層はSQLAlchemyのようなORMを導入せず、現行同様「生SQL + ドライバ」の方針を維持する（CLAUDE.mdの「必要以上の抽象化をしない」方針との一貫性）。

## テスト戦略

- pytest fixtureで各テストを1トランザクションに包み、テスト終了時に必ず `ROLLBACK` する（本番相当のPostgresを使うがコミットしないため実データは汚れない）。
- ローカル: `docker-compose` の `postgres` サービスをテストでも共用する（テスト専用DB名かスキーマで本番と分離）。
- CI: GitHub Actions の `services: postgres:` を使い、ジョブ内で完結させる。
- この方式により、`filelock` バージョン起因のteardown事故（PR #556）や、テストが本番相当DBに実操作してしまう事故（#548）のクラスが構造的に起きなくなる。コミットしない限りテストが永続データへ影響を与えられないため。

## ロールバック・安全策

- DuckDBファイルは検証期間中バックアップとして保持する。
- 切り戻しが必要な場合は `.env` の `DATABASE_URL` を空に戻し、`_connection.py` を旧バージョンにgit revertすれば旧経路（DuckDB）に戻せる。
- データ移行スクリプトはテーブル単位で `TRUNCATE` してからINSERTするため、再実行しても安全（べき等）。

## リスク・検証ポイント

- Postgres移行後、バックテスト・スクリーニングのような大量スキャン系クエリの実行時間が現行のDuckDB比でどう変化するかは移行後に実測が必要。性能劣化が許容範囲を超える場合はインデックス追加・クエリ見直しで対応する（DuckDB分析性能の代替としてのハイブリッド構成は採用しない方針のため）。
- `_init_tables()` と `migrations/0001_initial.sql` の乖離箇所（`strategy_promotions` 等）を漏れなく `0001_baseline_postgres.sql` に反映すること。実装時に `_init_tables()` の最新版と突き合わせたチェックリストを作成し、テーブル数の一致を確認する。
