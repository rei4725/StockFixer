DuckDB のデータ確認・操作・移行を正確に行う。

## データ確認
```bash
cd python
py python/tools/check_data.py
```
`stock_features` テーブルの列一覧・行数・銘柄数、`market_data_raw` の件数、y列・Date列の有無をチェックする。

## CSV → DuckDB 移行（初回セットアップ時）
```bash
py python/tools/migrate_csv_to_duckdb.py
```
`stock_features` と `prediction_results` の両テーブルを対象。移行後に `verify_migration` で自動検証。

## テーブル構成
| テーブル | 内容 | 主キー |
|---------|------|--------|
| `stock_features` | 特徴量データ（テクニカル指標＋ラグ特徴量＋y列） | market, symbol, row_num |
| `prediction_results` | 予測結果 | predicted_at, market, symbol |
| `market_data_raw` | 生OHLCV | market, symbol, date, timeframe |

## DB接続の使い方
```python
from src.utils.db import get_connection, get_readonly_connection

# 読み書き（シングルトン、アプリ内で共有）
con = get_connection()

# 読み取り専用（別プロセス向け、呼出側で close() 必要）
ro_con = get_readonly_connection()
try:
    df = ro_con.execute("SELECT * FROM stock_features LIMIT 10").fetchdf()
finally:
    ro_con.close()
```

## 主要DB関数
| 関数 | 説明 |
|------|------|
| `get_connection()` | シングルトン接続（threads=4, memory_limit=2GB） |
| `get_readonly_connection()` | 読み取り専用接続（別プロセス向け） |
| `upsert_stock_features(market, symbol, df)` | 特徴量を DELETE-INSERT |
| `load_stock_features(market, symbol)` | 1銘柄分の特徴量取得 |
| `load_all_stock_features()` | 全銘柄特徴量取得（統合モデル学習用） |
| `save_prediction_results(predicted_at, df)` | 予測結果保存 |
| `load_prediction_results(...)` | 予測結果取得（top_n/worst_n フィルタ対応） |
| `upsert_raw_ohlcv(rows)` | 生OHLCV をべき等INSERT |
| `get_all_symbols()` | 全銘柄 (market, symbol) リスト |

## 重要な注意事項
- DB格納パス: `python/data/stockfixer.duckdb`
- 列は DataFrame に合わせて `_ensure_columns` で動的追加（ALTER TABLE ADD COLUMN）
- **並列書込禁止**: DuckDB はロック競合を起こすため、DB書込は必ず逐次実行
- 詳細スキーマ: `docs/DATABASE_SCHEMA.md`
