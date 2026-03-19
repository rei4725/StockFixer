---
name: data-pipeline
description: "株価データの取得・特徴量生成・DuckDB保存のパイプラインを実行する。データ取得・更新・バッチ処理・特徴量生成の話題では必ずこのスキルを使用する。yfinance・OHLCV・stock_features・market_data_raw が絡む操作でも、明示的に言及されなくても使用する。"
compatibility: "Python 3.10+, yfinance, DuckDB。python/ ディレクトリで実行。"
---

## Goal
株価データの取得・テクニカル指標算出・特徴量生成・DuckDB保存を正確に行う。

## Procedure

### 単一銘柄のデータ取得
```bash
cd python
py run_data_creation.py --market jp --symbol 7203
```
- `--market`: 市場名（jp, us）
- `--symbol`: 銘柄コード（7203, AAPL）
- `--start_date` / `--end_date`: 任意の日付範囲指定（YYYY-MM-DD）

### ウォッチリスト全銘柄バッチ
```bash
cd python
py run_data_creation.py --batch
```
- 対象銘柄は `python/データ取得対象.csv`（market,symbol形式）から読み込む
- **フェーズ1**: データ取得＋特徴量生成（並列 max_workers=5）
- **フェーズ2**: DB書込（逐次実行、DuckDBロック制約回避）

### 内部処理フロー
1. `src/services/data_pipeline.py` → `fetch_stock_data_with_features()` でyfinanceからOHLCV取得
2. `src/features/technical_analysis.py` → テクニカル指標算出（SMA, RSI, MACD等）
3. `src/features/` → ラグ特徴量・ターゲット(y)列を自動付与
4. `src/utils/db.py` → `upsert_stock_features()` で `stock_features` テーブルにDELETE-INSERT
5. 生OHLCVは `market_data_raw` テーブルにもべき等INSERT

### 保存先
- 特徴量: DuckDB `stock_features` テーブル
- 生OHLCV: DuckDB `market_data_raw` テーブル
- DB格納パス: `python/data/stockfixer.duckdb`
- 期間: デフォルト過去5年分（end_date=現在、start_date=5年前）

## Key Functions
- `save_stock_data_with_features(market, symbol)` — フェッチ＋DB保存の一括実行
- `run_data_batch()` — ウォッチリスト全銘柄バッチ処理
- `load_target_symbols()` — CSVからウォッチリスト読み込み

## References
- [data_pipeline.py](../../../python/src/services/data_pipeline.py)
- [batch_runner.py](../../../python/src/services/batch_runner.py)
- [db.py](../../../python/src/utils/db.py)
