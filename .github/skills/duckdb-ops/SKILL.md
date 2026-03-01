---
name: duckdb-ops
description: >-
  DuckDBのデータ確認・操作・移行を行う。
  DuckDB、DB、データベース、テーブル、stock_features、prediction_results、
  market_data_raw、データ確認、データ移行、check_data、migrateの話題で使用する。
metadata:
  author: StockFixer
  version: "1.0"
compatibility: >-
  Python 3.10+, DuckDB。python/ ディレクトリで実行。
---

## Goal
DuckDBのデータ確認・操作・移行を正確に行う。

## Procedure

### データ構造確認
```bash
cd python
py python/tools/check_data.py
```
- `stock_features` テーブルの列一覧・行数・銘柄数
- `market_data_raw` テーブルの件数・銘柄別カウント
- y列・Date列の有無チェック

### CSV → DuckDB 移行（初回セットアップ時）
```bash
py python/tools/migrate_csv_to_duckdb.py
```
- `stock_features` と `prediction_results` の両テーブルを対象
- 移行後に `verify_migration` で銘柄数・行数の一致を自動検証

### テーブル構成
| テーブル | 内容 | 主キー |
|---------|------|--------|
| `stock_features` | 特徴量データ（テクニカル指標＋ラグ特徴量＋y列） | market, symbol, row_num |
| `prediction_results` | 予測結果 | predicted_at, market, symbol |
| `market_data_raw` | 生OHLCV | market, symbol, date, timeframe |

### DB接続の使い方
```python
from src.utils.db import get_connection, get_readonly_connection

# 読み書き（シングルトン、アプリ内で共有）
con = get_connection()

# 読み取り専用（別プロセス向け、呼出側でclose()必要）
ro_con = get_readonly_connection()
try:
    df = ro_con.execute("SELECT * FROM stock_features LIMIT 10").fetchdf()
finally:
    ro_con.close()
```

### 主要DB関数
| 関数 | 説明 |
|------|------|
| `get_connection()` | シングルトン接続（threads=4, memory_limit=2GB） |
| `get_readonly_connection()` | 読み取り専用接続（別プロセス向け） |
| `upsert_stock_features(market, symbol, df)` | 特徴量をDELETE-INSERT |
| `load_stock_features(market, symbol)` | 1銘柄分の特徴量取得 |
| `load_all_stock_features()` | 全銘柄特徴量取得（統合モデル学習用） |
| `save_prediction_results(predicted_at, df)` | 予測結果保存 |
| `load_prediction_results(...)` | 予測結果取得（top_n/worst_nフィルタ対応） |
| `upsert_raw_ohlcv(rows)` | 生OHLCVをべき等INSERT |
| `get_all_symbols()` | 全銘柄(market, symbol)リスト |

### 重要な注意事項
- DB格納パス: `python/data/stockfixer.duckdb`
- 列はDataFrameに合わせて `_ensure_columns` で動的追加（ALTER TABLE ADD COLUMN）
- **並列書込禁止**: DuckDBはロック競合を起こすため、DB書込は必ず逐次実行

## References
- [DATABASE_SCHEMA.md](../../../docs/DATABASE_SCHEMA.md) — テーブル定義・列定義・データフローの詳細
- [db.py](../../../python/src/utils/db.py)
- [check_data.py](../../../python/tools/check_data.py)
- [migrate_csv_to_duckdb.py](../../../python/tools/migrate_csv_to_duckdb.py)
