株価データの取得・テクニカル指標算出・特徴量生成・DuckDB保存を正確に行う。

## 実行コマンド

### 単一銘柄
```bash
cd python
py run_data_creation.py --market jp --symbol 7203
py run_data_creation.py --market us --symbol AAPL --start_date 2023-01-01 --end_date 2024-01-01
```

### ウォッチリスト全銘柄バッチ
```bash
py run_data_creation.py --batch
```
対象銘柄は `python/データ取得対象.csv` から読み込む。
- フェーズ1: データ取得＋特徴量生成（並列 max_workers=5）
- フェーズ2: DB書込（逐次実行 — DuckDBロック制約）

## 内部処理フロー
1. `src/services/data_pipeline.py` → `fetch_stock_data_with_features()` で yfinance から OHLCV 取得
2. `src/features/technical_analysis.py` → テクニカル指標算出（SMA, RSI, MACD 等）
3. ラグ特徴量・ターゲット(y)列を自動付与
4. `src/utils/db.py` → `upsert_stock_features()` で `stock_features` テーブルに DELETE-INSERT
5. 生OHLCV は `market_data_raw` テーブルにもべき等INSERT

## 保存先
- 特徴量: DuckDB `stock_features` テーブル
- 生OHLCV: DuckDB `market_data_raw` テーブル
- DB格納パス: `python/data/stockfixer.duckdb`

## Key Functions
- `save_stock_data_with_features(market, symbol)` — フェッチ＋DB保存の一括実行
- `run_data_batch()` — ウォッチリスト全銘柄バッチ処理
- `load_target_symbols()` — CSVからウォッチリスト読み込み
