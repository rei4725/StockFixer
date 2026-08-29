# テスト戦略とガイド

## ディレクトリ構成

```
tests/
├── unit/                    # Unit Test（全Mockingで隔離）
├── integration/             # Integration Test（実依存）
├── conftest.py             # pytest fixture 共有定義
└── README.md               # このファイル
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
- **カバレッジ基準**: `src/` 全体で **80%以上**（`--cov-fail-under=80`）

### CI運用ルール（必須）
- Pull Request 作成/更新時に GitHub Actions で Unit Test を自動実行する
- CI が失敗している PR はマージしない
- Unit Test は外部API・実DBに依存させない（依存は Mock で隔離）

**対象テスト**:
Backtester（基本機能・詳細・ストップロス・テイクプロフィット・ポジションサイジング）、メトリクス計算、パイプラインロジック、パスユーティリティ、DataFrame変換、モデル管理、シグナル生成、テクニカル指標、最適パラメータ読込（Unit）

### Integration Test（`tests/integration/`）
**目的**: 全フロー検証（実 DB・実モデル使用）  
**特徴**:
- DuckDB と実モデル（XGBoost）に依存
- **実行時間**: 数十秒～分単位
- **環境**: セットアップ済みマシンで実行
- **使用場面**: PR 時、ナイトリビルド

**対象テスト**:
バックテストE2E、最適化E2E、バッチ処理、データ取得（yfinance）、データパイプライン統合、DuckDB操作、Discord Bot、モデル学習パイプライン、予測パイプライン、スケジューラー統合、最適パラメータ読込E2E

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
- `test_backtest_optimize.py::TestBacktestOptimizeE2E::test_optimization_metrics_dtype_fix`

## 開発フロー推奨例

```powershell
# 1. 機能開発中：Unit Test を実行（高速フィードバック、外部依存なし）
python -m pytest tests/unit/test_backtester_unit.py -v

# 2. PR 前：ローカルCI相当の一括チェック（unitのみ、check-ci.ps1と同等）
cd python; .\check-ci.ps1

# 3. integration/e2e は PR作成後にCIが自動実行する（PR: unit + integration + 軽量e2e）
#    ローカルで個別に確認したい場合のみ、対象を絞って実行する:
python -m pytest tests/integration/test_xxx.py -v

# 4. 重量級の e2e（実DB + 実モデル学習）は develop push 後 / 手動発火でのみ実行される。
#    ローカルで確認したい場合（Postgres起動が必要）:
docker compose up -d postgres
python -m pytest tests/e2e/ -v --timeout=300 -m "slow"
```

## Unit Test 実装チェックリスト

- テスト対象は1つの責務に絞る（1テスト1意図）
- テスト名は `test_期待する挙動_条件` 形式で命名する
- 外部依存（DB/API/ファイルI/O/時刻依存）は Mock または Fixture で隔離する
- 正常系と異常系（例外・境界値）を最低1ケースずつ含める
- `Arrange / Act / Assert` の3段構成を意識して可読性を保つ
- アサーションメッセージ不要の単純比較を優先し、過剰な分岐を避ける
- 失敗が再現しやすいようにランダム値は固定シードを使用する
- 変更したロジックには対応する Unit Test を同一PRで追加する

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
