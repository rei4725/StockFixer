# PostgreSQL 切り替えランブック

対象: DuckDB → PostgreSQL のビッグバング切り替え（本番実施用）

## 前提

- 本プランの全タスク（Task 1〜17、および途中発見された Task 11.5・12.5 を含む）がdevelopにマージ済みであること
- `docker compose config` でpostgresサービスが定義されていることを確認済み
- `docker-compose.yml` は `stockfixer` サービスの `DATABASE_URL` を `postgres` サービス（`postgres:5432`、ユーザー`stockfixer`／DB`stockfixer`）向けに固定で設定済み（`environment:` に直書き）。この値は `python/.env` に同名の変数を書いても `environment:` 側が優先されて上書きされない点に注意する。
- Step 3〜5の `python scripts/...`（および `init_tables()`）実行は**ホスト側では行わず**、`docker compose run --rm --no-deps stockfixer ...` で `stockfixer` イメージのコンテナ内から実行する（各手順参照）。これにより `stockfixer` サービスに既に設定済みの `DATABASE_URL` をそのまま再利用でき、パスワードをホスト側シェルへ`export`する必要が一切ない。これは `pg_dump` 実行時にホスト入力・ホスト露出を避けている `backup_pipeline.py` の `_run_pg_dump`、および本ランブック「pg_dumpバックアップの復元手順」で `stockfixer` コンテナの `DATABASE_URL` を再利用しているのと同じ方針である。**`POSTGRES_PASSWORD`を既定値（`stockfixer_dev`）から変更する場合は、`docker-compose.yml`と同じ階層のプロジェクトルート`.env`（`python/.env`とは別物、`docker compose`の変数展開に使われるファイル）に`POSTGRES_PASSWORD`を設定しておくこと**。この値は`postgres`サービスの初期化パスワードと`stockfixer`サービスの`DATABASE_URL`の両方に展開されるため、Step 1を始める前（少なくともStep 3で`postgres`を起動する前）に設定しておく必要がある。忘れるとStep 3の`postgres`起動時点でパスワード不一致が生じる。

## 手順

1. メンテナンス開始（スケジューラー・Bot・APIを停止）

   ```bash
   docker compose stop stockfixer
   ```

2. DuckDBバックアップ

   ```bash
   mkdir -p python/data/backups
   cp python/data/stockfixer.duckdb python/data/backups/stockfixer_pre_postgres_$(date +%Y%m%d).duckdb
   ```

3. Postgres起動 + スキーマ初期化（必須）

   ```bash
   docker compose up -d postgres
   docker compose exec postgres pg_isready -U stockfixer -d stockfixer
   # start_period: 10s のため、起動直後は失敗することがある。失敗した場合は数秒待って再実行する
   # （例: until docker compose exec postgres pg_isready -U stockfixer -d stockfixer; do sleep 2; done）

   docker compose run --rm --no-deps stockfixer python -c "from src.utils.db import init_tables; init_tables()"
   ```

   `init_tables()` の実行は**必須**（省略不可）。マイグレーションは `_connection.py` がアプリ自身の初回接続時に `pg_advisory_lock` 経由で自動適用するが、それが発火するのは Step 6 でアプリコンテナが起動した時点であり、Step 4・5より後になる。一方 Step 4 の `migrate_to_postgres.py` は DuckDB からPostgresへ `psycopg` で直接接続するだけで `src.utils.db`／`_connection.py` を一切経由しないため、自動マイグレーションの経路に乗らない（スクリプト自身のdocstringにも「前提: 移行先PostgresにTask 2のマイグレーションが適用済みであること」と明記されている）。スキーマが無い状態で Step 4 を実行すると、最初に処理される `stock_features` テーブルへの `TRUNCATE TABLE pg."stock_features"` が「relation does not exist」で失敗する。Step 5 の `verify_postgres_migration.py` も同様にスキーマの事前作成が前提。そのため、このStepで明示的に `init_tables()` を実行してスキーマを作成しておくことが必須となる。

4. データ移行スクリプト実行

   ```bash
   docker compose run --rm --no-deps stockfixer python scripts/migrate_to_postgres.py --duckdb-path data/stockfixer.duckdb
   ```

5. 整合性検証

   ```bash
   docker compose run --rm --no-deps stockfixer python scripts/verify_postgres_migration.py --duckdb-path data/stockfixer.duckdb
   ```

   終了コード0を確認する（`docker compose run`はコンテナ内で実行したコマンドの終了コードをそのまま返す）。非ゼロの場合は手順を中止し、原因を調査する。

6. 切り替え

   `DATABASE_URL`は前述のとおり`docker-compose.yml`の`stockfixer`サービス定義に既に設定済みのため、`.env`側の追加編集は不要。`POSTGRES_PASSWORD`を既定値から変更する場合の対応は前提セクション参照（Step 1より前に設定済みであること）。

   ```bash
   docker compose up -d stockfixer
   docker compose logs -f stockfixer
   ```

   起動ログにマイグレーションエラーが無いこと、`/health`エンドポイントが200を返すことを確認する。

7. 保持期間

   `python/data/backups/stockfixer_pre_postgres_*.duckdb`は2週間保持し、問題なければ削除する。

## pg_dumpバックアップの復元手順（Task 12.5で判明した重要な注意点）

Task 12.5で `backup_pipeline.py` を `pg_dump`（カスタムフォーマット, `-Fc`）ベースに移行した際、`stockfixer` アプリコンテナに同梱される `pg_dump`（Debian trixieベースイメージのデフォルト、v17系）と、`postgres` サービスコンテナ（`postgres:16-alpine`、v16系）の間にバージョン差異があることが判明した。

**`pg_dump` のカスタムアーカイブフォーマットは「ダンプしたツールのバージョン」にアーカイブヘッダのバージョンが紐づき、より古い `pg_restore` は新しいアーカイブを読めない**（逆に新しい `pg_restore` は古いアーカイブを読める、という非対称な互換性）。実際に検証したところ、v17の `pg_dump` で作成したダンプを `docker compose exec postgres pg_restore`（v16）で読もうとすると `pg_restore: error: unsupported version (1.16) in file header` で失敗する。

**復元時は必ずダンプを作成したのと同じツール（`stockfixer` アプリコンテナ内の `pg_restore`）を使うこと**。またホスト側から直接`-h`/`-U`/パスワード等を指定する形にすると、このPostgresインスタンスはパスワード認証必須（`_run_pg_dump`が`PGPASSWORD`を明示的に渡しているのと同じ理由）のため非対話シェルではパスワード入力待ちで止まってしまう。`stockfixer` コンテナに既に設定済みの `DATABASE_URL` をそのまま使うのが最も確実:

```bash
# NG: postgresサービスコンテナのpg_restoreはバージョンが古く読めない
docker compose exec postgres pg_restore ...

# NG: ホストや他コンテナからパスワードなしで直接 -h/-U 指定すると認証待ちで止まる
docker compose exec stockfixer pg_restore -h postgres -U stockfixer -d stockfixer -c /app/data/backups/<timestamp>/stockfixer.dump

# OK: stockfixerアプリコンテナ内のpg_restoreを、同コンテナに設定済みのDATABASE_URLで実行する
docker compose exec stockfixer bash -c 'pg_restore -d "$DATABASE_URL" -c /app/data/backups/<timestamp>/stockfixer.dump'
```

なお、Task 12.5の検証は `pg_restore --list`（アーカイブのテーブル一覧読み取り）による互換性確認までで、実データを実際に空DBへ復元する完全なリストア手順の実地検証はまだ行っていない。本番復元が必要になった際は、上記コマンドの後に対象テーブルへのデータ反映を目視確認すること。

## ロールバック手順

問題が発生した場合:

```bash
docker compose stop stockfixer
git revert <該当コミット群>  # _connection.py 等をDuckDB版に戻す（docker-compose.yml のpostgresサービス定義・DATABASE_URL設定も同じコミット群に含まれる場合は併せて戻る）
docker compose up -d stockfixer
```

DuckDBファイルはStep 2でバックアップ済みのため、`python/data/stockfixer.duckdb`が
移行スクリプト実行時点のまま残っていることを確認する（移行スクリプトは読み取り専用接続のため元ファイルは変更されない）。
