---
name: prediction
description: >-
  学習済みモデルを使用して株価予測を実行し、結果をDuckDBに保存する。
  予測、predict、forecast、Top10、Worst10、予想変化率、
  統合モデル予測、銘柄別モデル予測の話題で使用する。
metadata:
  author: StockFixer
  version: "1.0"
compatibility: >-
  Python 3.10+, 学習済みモデルが必要。python/ ディレクトリで実行。
---

## Goal
学習済みモデルで株価予測を実行し、マーケット別Top10/Worst10を出力・DB保存する。

## Procedure

### 全銘柄予測（統合モデル使用、デフォルト）
```bash
cd python
py run_predict.py
```

### 単一銘柄予測
```bash
py run_predict.py --mode single --market jp --symbol 7203
```

### ウォッチリスト予測
```bash
py run_predict.py --mode watchlist
```

### 銘柄別モデルを使用
```bash
py run_predict.py --individual
```

### 引数一覧
| 引数 | デフォルト | 説明 |
|------|-----------|------|
| `--mode` | なし（全銘柄） | `single` or `watchlist` |
| `--market` | なし | 市場名（singleモード時に必須） |
| `--symbol` | なし | 銘柄コード（singleモード時に必須） |
| `--individual` | False | 銘柄別モデルを使用 |

### 出力仕様
- 予測結果は DuckDB `prediction_results` テーブルに保存
- マーケット別 Top10 / Worst10 を標準出力に表示
- 予測値は直近データの翌営業日終値

### Discord連携
- 予測結果は `convert_df_for_discord()` でDiscord向けに整形
- 列名・順序: 「シンボル」「現在値」「予想終値」「予想変化率」
- 変化率: `(予想終値-現在値)/現在値`、有効数字2桁パーセント表示
- 値段は小数第3位で切り捨て

## Key Functions
- `predict_all_unified()` — 統合モデルで全銘柄予測
- `predict_all_individual()` — 銘柄別モデルで全銘柄予測
- `run_predict_single(market, symbol)` — 単一銘柄予測
- `output_top_worst_results(output_rows, mode)` — Top10/Worst10出力＋DB保存

## References
- [prediction_pipeline.py](../../../python/src/services/prediction_pipeline.py)
- [discord_bot.py](../../../python/src/api/discord_bot.py)
