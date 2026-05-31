---
name: backtest
description: "バックテストを実行して戦略の有効性を検証する。バックテスト・Walk-Forward・損益シミュレーション・パフォーマンス検証・シャープレシオの話題では必ずこのスキルを使用する。戦略の有効性確認や取引成績の検証が必要な場面でも使用する。"
compatibility: "Python 3.10+。python/ ディレクトリで実行。"
---

## Goal
モデル予測に基づく売買戦略のバックテストを実行し、パフォーマンスを検証する。

## Procedure

### 単一期間バックテスト
```bash
cd python
py run_backtest.py --market jp --symbol 7203
```

### Walk-Forward検証
```bash
py run_backtest.py --market jp --symbol 7203 --walk-forward --n-splits 5
```

### モデル・閾値・データソース指定
```bash
py run_backtest.py --market us --symbol AAPL \
    --model-type LightGBMModel --threshold 0.005 --source raw
```

### 引数一覧
| 引数 | デフォルト | 説明 |
|------|-----------|------|
| `--market` | `jp` | マーケット |
| `--symbol` | **必須** | 銘柄コード |
| `--model-type` | `XGBoostModel` | `XGBoostModel` or `LightGBMModel` |
| `--source` | `file` | `file`(DB特徴量), `api`(yfinance直接), `raw`(DB生OHLCVから再生成) |
| `--walk-forward` | False | Walk-Forward検証 |
| `--n-splits` | 5 | Walk-Forwardの分割数 |
| `--train-ratio` | 0.8 | 単一期間の学習データ比率 |
| `--threshold` | 0.0 | シグナル発生の変化率閾値 |
| `--initial-cash` | 1,000,000 | 初期資金 |
| `--fee-rate` | 0.001 | 取引手数料率 |
| `--slippage` | 0.0 | スリッページ |
| `--start-date` | なし | バックテスト開始日 |
| `--end-date` | なし | バックテスト終了日 |

### データソースの選択
| source | 説明 | 用途 |
|--------|------|------|
| `file` | DB `stock_features` テーブル | 通常使用（デフォルト） |
| `api` | yfinanceから直接取得 | DB未登録銘柄のテスト |
| `raw` | DB `market_data_raw` から特徴量再生成 | 生データからの再現検証 |

### 結果保存
- CSV形式で `python/results/` に保存
- メトリクス（総リターン、シャープレシオ、最大ドローダウン等）を標準出力

## Key Functions
- `run_backtest_single(...)` — 単一期間バックテスト
- `run_backtest_walk_forward(...)` — Walk-Forwardバックテスト
- `save_backtest_results(...)` — 結果をCSV保存
- `print_backtest_metrics(...)` — メトリクス出力

## References
- [backtest_pipeline.py](../../../python/src/services/backtest_pipeline.py)
- [run_backtest.py](../../../python/run_backtest.py)
