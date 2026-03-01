# DuckDB テーブル定義

StockFixer が使用する DuckDB データベース（`python/data/stockfixer.duckdb`）のテーブル定義を記載する。

テーブルの初期化は `src/utils/db.py` の `_init_tables()` で行われる。

---

## テーブル一覧

| テーブル名 | 用途 | レコード単位 |
|-----------|------|-------------|
| `stock_features` | 特徴量データ（テクニカル指標＋ラグ特徴量＋ターゲット） | 銘柄×日付行 |
| `prediction_results` | 予測結果 | 銘柄×予測実行日時 |
| `market_data_raw` | 生OHLCVデータ | 銘柄×日付×時間軸 |

---

## stock_features

特徴量生成済みの株価データを格納するテーブル。銘柄ごとに DELETE → INSERT で全行入れ替えする。

### 固定列（CREATE TABLE で定義）

| 列名 | 型 | NULL | 説明 |
|------|-----|------|------|
| `market` | VARCHAR | NOT NULL | マーケット識別子（`jp`, `us`） |
| `symbol` | VARCHAR | NOT NULL | 銘柄コード（`7203`, `AAPL`） |
| `row_num` | INTEGER | NOT NULL | 行番号（時系列順序の保持用、0始まり） |

**主キー**: `(market, symbol, row_num)`

### 動的列（ALTER TABLE ADD COLUMN で自動追加）

DataFrame の列に応じて `_ensure_columns()` が自動的に列を追加する。型はpandas dtypeから推定される。

| pandas dtype | → SQL型 |
|-------------|---------|
| int系 | BIGINT |
| float系 | DOUBLE |
| bool | BOOLEAN |
| datetime64 | TIMESTAMP |
| その他 | VARCHAR |

#### 代表的な動的列

| 列名 | 型 | 説明 |
|------|-----|------|
| `Open` | DOUBLE | 始値 |
| `High` | DOUBLE | 高値 |
| `Low` | DOUBLE | 安値 |
| `Close` | DOUBLE | 終値 |
| `Volume` | BIGINT | 出来高 |
| `SMA_5` | DOUBLE | 5日単純移動平均 |
| `SMA_20` | DOUBLE | 20日単純移動平均 |
| `SMA_60` | DOUBLE | 60日単純移動平均 |
| `RSI_14` | DOUBLE | 14日RSI |
| `MACD` | DOUBLE | MACD |
| `MACD_signal` | DOUBLE | MACDシグナル |
| `BB_upper` | DOUBLE | ボリンジャーバンド上限 |
| `BB_lower` | DOUBLE | ボリンジャーバンド下限 |
| `Close_lag1` 〜 `Close_lag5` | DOUBLE | 終値のラグ特徴量 |
| `Volume_lag1` 〜 `Volume_lag5` | DOUBLE | 出来高のラグ特徴量 |
| `y` | DOUBLE | ターゲット変数（翌営業日終値） |
| `market_encoded` | BIGINT | マーケットの数値エンコード（us=0, jp=1） |

> **注意**: 実際の列は特徴量生成パイプラインの実装に依存し、上記は代表例。
> 列の完全な一覧は `py python/tools/check_data.py` で確認できる。

### CRUD操作

| 操作 | 関数 | 説明 |
|------|------|------|
| INSERT | `upsert_stock_features(market, symbol, df)` | DELETE → INSERT（銘柄単位で全行入替） |
| SELECT（1銘柄） | `load_stock_features(market, symbol)` | 管理列(market, symbol, row_num)を除外して返却 |
| SELECT（全銘柄） | `load_all_stock_features()` | market, symbol列付き（統合モデル学習用） |
| DELETE | `delete_stock_features(market, symbol)` | 銘柄単位で削除 |
| 銘柄一覧 | `get_all_symbols()` | `(market, symbol)` タプルのリスト |

---

## prediction_results

予測結果を格納するテーブル。予測実行のたびに対象銘柄の既存データを DELETE してから INSERT する。

### 列定義

| 列名 | 型 | NULL | 説明 |
|------|-----|------|------|
| `market` | VARCHAR | NOT NULL | マーケット識別子（`jp`, `us`） |
| `symbol` | VARCHAR | NOT NULL | 銘柄コード（`7203`, `AAPL`） |
| `predicted_at` | VARCHAR | NOT NULL | 予測実行日時（`YYYYMMDD_HHMMSS` 形式） |
| `current_price` | DOUBLE | NULL可 | 現在値（直近終値） |
| `avg_pred_price` | DOUBLE | NULL可 | 予想終値（複数モデルの平均） |
| `diff_ratio` | DOUBLE | NULL可 | 予想変化率 `(avg_pred_price - current_price) / current_price` |
| `model_count` | INTEGER | NULL可 | 予測に使用したモデル数 |

**主キー**: `(market, symbol, predicted_at)`

### CRUD操作

| 操作 | 関数 | 説明 |
|------|------|------|
| INSERT | `save_prediction_results(predicted_at, df)` | 対象銘柄の既存データを DELETE → INSERT |
| SELECT | `load_prediction_results(predicted_at, market, top_n, worst_n)` | predicted_at=None で最新、top_n/worst_n でフィルタ |
| 最新タイムスタンプ | `load_latest_prediction_timestamp()` | 最新の predicted_at を返す |
| マーケット一覧 | `load_prediction_markets(predicted_at)` | 指定タイムスタンプのマーケット一覧 |

---

## market_data_raw

yfinance から取得した生の OHLCV データを格納するテーブル。`INSERT OR REPLACE` でべき等保存される。

### 列定義

| 列名 | 型 | NULL | デフォルト | 説明 |
|------|-----|------|-----------|------|
| `market` | VARCHAR | NOT NULL | — | マーケット識別子（`jp`, `us`） |
| `symbol` | VARCHAR | NOT NULL | — | 銘柄コード（`7203`, `AAPL`） |
| `ticker` | VARCHAR | NOT NULL | — | yfinance向けティッカー（`7203.T`, `AAPL`） |
| `timeframe` | VARCHAR | NOT NULL | — | 時間軸（`1d` 等） |
| `ts` | TIMESTAMP | NOT NULL | — | タイムスタンプ（日足の場合は日付） |
| `open` | DOUBLE | NULL可 | — | 始値 |
| `high` | DOUBLE | NULL可 | — | 高値 |
| `low` | DOUBLE | NULL可 | — | 安値 |
| `close` | DOUBLE | NULL可 | — | 終値 |
| `volume` | BIGINT | NULL可 | — | 出来高 |
| `adj_close` | DOUBLE | NULL可 | — | 調整後終値 |
| `source` | VARCHAR | NOT NULL | `'yfinance'` | データソース |
| `ingested_at` | TIMESTAMP | NOT NULL | `CURRENT_TIMESTAMP` | データ取り込み日時 |

**主キー**: `(market, symbol, timeframe, ts)`

### CRUD操作

| 操作 | 関数 | 説明 |
|------|------|------|
| UPSERT | `upsert_raw_ohlcv(rows: list[dict])` | `INSERT OR REPLACE`（べき等）。保存行数を返却 |
| SELECT | `load_raw_ohlcv(market, symbol, start_date, end_date, timeframe)` | インデックス=Date、列名はyfinance形式（先頭大文字） |

### 読み出し時の列名変換

`load_raw_ohlcv()` は読み出し時にカラム名を yfinance 形式に変換する。

| DB列名 | → 返却時の列名 |
|--------|---------------|
| `open` | `Open` |
| `high` | `High` |
| `low` | `Low` |
| `close` | `Close` |
| `volume` | `Volume` |
| `adj_close` | `Adj Close` |

---

## テーブル間の関係

```
stock_features                    prediction_results
┌──────────────────────┐          ┌──────────────────────────┐
│ market   (PK)        │          │ market        (PK)       │
│ symbol   (PK)        │          │ symbol        (PK)       │
│ row_num  (PK)        │          │ predicted_at  (PK)       │
│ ...特徴量列(動的)...  │          │ current_price            │
│ y (ターゲット)        │          │ avg_pred_price           │
└──────────────────────┘          │ diff_ratio               │
        │                         │ model_count              │
        │ 学習データ               └──────────────────────────┘
        │                                    ▲
        ▼                                    │ 予測結果保存
  ┌──────────┐                               │
  │  モデル   │ ── 予測 ──────────────────────┘
  │ (.joblib) │
  └──────────┘
        ▲
        │ OHLCV → 特徴量生成
        │
market_data_raw
┌──────────────────────────┐
│ market     (PK)          │
│ symbol     (PK)          │
│ timeframe  (PK)          │
│ ts         (PK)          │
│ ticker                   │
│ open, high, low, close   │
│ volume, adj_close        │
│ source, ingested_at      │
└──────────────────────────┘
```

### データフロー

1. **市場データ取得**: yfinance → `market_data_raw`（生OHLCV保存）
2. **特徴量生成**: `market_data_raw` の OHLCV → テクニカル指標・ラグ特徴量算出 → `stock_features`
3. **モデル学習**: `stock_features` → XGBoost/LightGBM 学習 → `.joblib` 保存
4. **予測実行**: `stock_features`（最新行）+ `.joblib` → 予測 → `prediction_results`

---

## 接続管理

| 関数 | 用途 | 接続モード | ライフサイクル |
|------|------|-----------|---------------|
| `get_connection()` | 通常の読み書き | 読み書き可 | シングルトン（アプリ全体で1接続） |
| `get_readonly_connection()` | 別プロセスからの読み取り | 読み取り専用 | 呼出側で `close()` 必要 |
| `close_connection()` | 接続終了 | — | シングルトン接続を閉じる |

### 接続設定

```python
duckdb.connect(db_path, threads=4, memory_limit='2GB')
```

| パラメータ | 値 | 説明 |
|-----------|-----|------|
| `threads` | 4 | CPU並列処理スレッド数 |
| `memory_limit` | 2GB | メモリ使用上限 |

---

## ツール

| ツール | コマンド | 説明 |
|-------|---------|------|
| データ確認 | `py python/tools/check_data.py` | 列一覧・行数・銘柄数・y列有無を表示 |
| CSV→DB移行 | `py python/tools/migrate_csv_to_duckdb.py` | CSV → DuckDB 一括インポート＋検証 |

---

*Last updated: 2026-03-01*
