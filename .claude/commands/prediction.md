学習済みモデルで株価予測を実行し、マーケット別 Top10/Worst10 を出力・DB保存する。

## 実行コマンド

```bash
cd python
py run_predict.py                                        # 全銘柄（統合モデル、デフォルト）
py run_predict.py --mode single --market jp --symbol 7203  # 単一銘柄
py run_predict.py --mode watchlist                        # ウォッチリスト予測
py run_predict.py --individual                            # 銘柄別モデルを使用
```

## 引数一覧
| 引数 | デフォルト | 説明 |
|------|-----------|------|
| `--mode` | なし（全銘柄） | `single` or `watchlist` |
| `--market` | なし | 市場名（single モード時に必須） |
| `--symbol` | なし | 銘柄コード（single モード時に必須） |
| `--individual` | False | 銘柄別モデルを使用 |

## 出力仕様
- 予測結果は DuckDB `prediction_results` テーブルに保存
- マーケット別 Top10 / Worst10 を標準出力に表示
- 予測値は直近データの翌営業日終値

## Discord 連携
- `convert_df_for_discord()` で整形
- 列名・順序: 「シンボル」「現在値」「予想終値」「予想変化率」
- 変化率: `(予想終値-現在値)/現在値`、有効数字2桁パーセント表示
- 値段は小数第3位で切り捨て

## Key Functions
- `predict_all_unified()` — 統合モデルで全銘柄予測
- `predict_all_individual()` — 銘柄別モデルで全銘柄予測
- `run_predict_single(market, symbol)` — 単一銘柄予測
- `output_top_worst_results(output_rows, mode)` — Top10/Worst10 出力＋DB保存
