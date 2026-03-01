# テスト戦略とガイド

## ディレクトリ構成

```
tests/
├── unit/                              # Unit Test（全Mockingで隔離・11ファイル）
│   ├── __init__.py
│   ├── test_backtester.py            # Backtester基本動作
│   ├── test_backtester_unit.py       # Backtester.simulate_trading
│   ├── test_backtest_advanced.py     # ストップロス・テイクプロフィット・ポジションサイジング
│   ├── test_backtest_metrics.py      # メトリクス計算ロジック
│   ├── test_backtest_pipeline_unit.py # パイプラインロジック
│   ├── test_backtest_task.py         # ReturnRegressionTask
│   ├── test_data_path_utils.py       # パスユーティリティ
│   ├── test_df_to_string.py          # DataFrame文字列変換
│   ├── test_model_manager.py         # モデル管理ロジック
│   ├── test_signal_generator.py      # シグナル生成ロジック
│   └── test_technical_analysis.py    # テクニカル指標計算
├── integration/                       # Integration Test（実依存・11ファイル）
│   ├── __init__.py
│   ├── test_backtest_e2e.py          # バックテスト E2E
│   ├── test_backtest_optimize_e2e.py # 最適化 E2E
│   ├── test_batch_runner.py          # バッチ処理実行
│   ├── test_data_loader.py           # yfinance連携
│   ├── test_data_pipeline.py         # データパイプライン統合
│   ├── test_db.py                    # DuckDB操作（stock_features）
│   ├── test_db_raw.py                # DuckDB操作（market_data_raw）
│   ├── test_discord_bot.py           # Discord API連携
│   ├── test_model_training_pipeline.py # モデル学習パイプライン
│   ├── test_prediction_pipeline.py   # 予測パイプライン
│   └── test_scheduler_pipeline.py    # スケジューラー統合
├── conftest.py                       # pytest fixture 共有定義
└── README.md                         # このファイル
```

## テスト実行方法

### 全テスト実行
```powershell
cd python
python -m pytest tests/ -v
```

### Unit Test のみ（高速）
```powershell
python -m pytest tests/unit/ -v
```

### Integration Test のみ
```powershell
python -m pytest tests/integration/ -v
```

### 特定ファイルテストのみ
```powershell
python -m pytest tests/unit/test_backtester_unit.py -v
```

### 特定テストクラス/メソッドのみ
```powershell
python -m pytest tests/unit/test_backtester_unit.py::TestBacktesterSimulateTradingBasic::test_no_trades_on_all_hold_signal -v
```

### カバレッジ計測
```powershell
python -m pytest tests/ --cov=src --cov-report=html
```

### unittest ベース（pytest 非使用）
```powershell
# Unit Test
python -m unittest discover -s tests/unit -p "test_*.py"

# Integration Test
python -m unittest discover -s tests/integration -p "test_*.py"
```

## テスト戦略の説明

### Unit Test（`tests/unit/`）
**目的**: ロジック検証（外部依存なし）  
**特徴**:
- 全依存を Mock で隔離
- **実行時間**: <5秒
- **環境**: 開発マシンで常時実行可能
- **使用場面**: コミット前チェック、CI/CD パイプラインの初期段階

**対象ファイル（11ファイル）**:
- `test_backtester.py` — Backtester基本機能（買い・売り・手数料・スリッページ）
- `test_backtester_unit.py` — simulate_tradingメソッド詳細テスト
- `test_backtest_advanced.py` — ストップロス・テイクプロフィット・ポジションサイジング
- `test_backtest_metrics.py` — compute_metrics関数（勝率・profit_factor・シャープレシオ等）
- `test_backtest_pipeline_unit.py` — パイプラインヘルパー関数（ラベル生成・メトリクス集計）
- `test_backtest_task.py` — ReturnRegressionTask（make_labels・make_signal）
- `test_data_path_utils.py` — パスユーティリティ関数
- `test_df_to_string.py` — DataFrame文字列変換
- `test_model_manager.py` — ModelManager（モデル登録・作成・保存・ロード）
- `test_signal_generator.py` — SignalGenerator（シグナル生成ロジック）
- `test_technical_analysis.py` — テクニカル指標計算（RSI・MACD・ATR等）

### Integration Test（`tests/integration/`）
**目的**: 全フロー検証（実 DB・実モデル使用）  
**特徴**:
- DuckDB と実モデル（XGBoost）に依存
- **実行時間**: 数十秒～分単位
- **環境**: セットアップ済みマシンで実行
- **使用場面**: PR 時、ナイトリビルド

**対象ファイル（11ファイル）**:
- `test_backtest_e2e.py` — バックテストE2E（DB→特徴量→学習→バックテスト）
- `test_backtest_optimize_e2e.py` — グリッドサーチ実行〜メトリクス計算
- `test_batch_runner.py` — バッチ処理（並列実行・エラーハンドリング）
- `test_data_loader.py` — yfinance連携（株価取得・保存）
- `test_data_pipeline.py` — データパイプライン統合（取得→特徴量生成→DB保存）
- `test_db.py` — DuckDB操作（stock_featuresテーブルCRUD）
- `test_db_raw.py` — DuckDB操作（market_data_rawテーブルCRUD）
- `test_discord_bot.py` — Discord Bot（/forecastコマンド・メッセージ送信）
- `test_model_training_pipeline.py` — モデル学習パイプライン（学習→保存→ロード）
- `test_prediction_pipeline.py` — 予測パイプライン（モデルロード→予測→結果保存）
- `test_scheduler_pipeline.py` — スケジューラー統合（日次・週次ジョブ実行）

## conftest.py（共有 Fixture）

### 提供 Fixture

#### データ生成
- `sample_price_df` — 5行のサンプル株価 DataFrame
- `sample_features_df` — 10行の特徴量 DataFrame
- `sample_signal_series` — Buy/Hold/Sell シグナル

#### Mock オブジェクト
- `mock_model_manager` — ModelManager モック
- `mock_signal_generator` — SignalGenerator モック
- `mock_data_loader` — DataLoader モック

#### 複合 Fixture
- `backtester_with_mocks` — モック依存の Backtester インスタンス

#### 環境チェック
- `has_xgboost` — XGBoost 利用可能か
- `has_duckdb` — DuckDB 利用可能か

### 使用例

```python
def test_example(sample_price_df, sample_signal_series):
    """Fixture を利用したテスト"""
    bt = Backtester(...)
    result_df, metrics = bt.simulate_trading(sample_price_df, sample_signal_series)
    assert metrics["num_trades"] > 0
```

## メトリクス計算エラー修正（dtype エラー）

**問題**: `profit_factor` が `None` と数値の混在により、平均計算時に dtype エラー発生

**修正内容**:
- `src/services/backtest_optimize_pipeline.py` line 119-122
- `src/backtest/walk_forward.py` line 206-209

**検証テスト**:
- `test_backtest_optimize_unit.py::TestComputeMetrics`
- `test_backtest_optimize_e2e.py::TestBacktestOptimizeE2E::test_optimization_metrics_dtype_fix`

## 開発フロー推奨例

```powershell
# 1. 機能開発中：Unit Test を実行（高速フィードバック）
python -m pytest tests/unit/test_backtester_unit.py -v

# 2. ローカル検証完了：全テスト実行
python -m pytest tests/ -v

# 3. PR 前：カバレッジ確認
python -m pytest tests/ --cov=src --cov-report=term-missing

# 4. CI/CD で Integration Test も含めて検証
python -m pytest tests/integration/ -v
```

## トラブルシューティング

### `ModuleNotFoundError: No module named 'xgboost'`
XGBoost が未インストール。依存パッケージをインストール：
```powershell
pip install -r requirements.txt
```

### `FileNotFoundError: data/stockfixer.duckdb`
DuckDB ファイルが存在しない。データ取得を実行：
```powershell
python run_data_creation.py --batch
```

### `pytest: command not found`
pytest がインストールされていない（unittest は標準ライブラリ）。
unittest で実行：
```powershell
python -m unittest discover -s tests/unit
```

## 参考資料

- [pytest 公式ドキュメント](https://docs.pytest.org/)
- [unittest 公式ドキュメント](https://docs.python.org/3/library/unittest.html)
- [unittest.mock 公式ドキュメント](https://docs.python.org/3/library/unittest.mock.html)
