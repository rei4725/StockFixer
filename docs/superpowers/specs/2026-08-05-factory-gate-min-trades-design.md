# 戦略ファクトリー: 銘柄あたり最低取引数ゲート

- Issue: #625
- 日付: 2026-08-05
- 関連: #598 / #564（本欠陥により不採用クローズ）、#369（ファクトリー Phase 1）

## 背景

戦略ファクトリーの合格ゲートが「1銘柄あたり1〜2取引 × 数十銘柄」という統計的アーティファクトを弾けない。#598 / #564 の再検証（2026-08-04）で判明した。夜間バッチは 05:00 で稼働しており、放置すると同性質の仮説が今後も合格し続ける。

### 根本原因

`_sharpe_per_trade`（`python/src/backtest/metrics/core.py:274`）は取引 PnL リストの mean/std である。取引がちょうど 2 回の銘柄では `std = |a-b|/√2` となり、2 回のリターンが近いと Sharpe が発散する。

`evaluate_hypothesis`（`python/src/backtest/factory.py:317-341`）はこの銘柄別 Sharpe を単純平均する。`apply_gate`（同 373 行）は `FACTORY_GATE_MIN_TRADES`（=30）を銘柄横断の**合計**取引数にしか適用しないため、この構成が素通りする。

### 実測（#598, IS 2024-07-25〜2026-07-25, jp 194銘柄）

- 買いシグナルが出たのは 69 銘柄、取引数合計 85 本
- 内訳: 2 取引の銘柄が 16、1 取引の銘柄が 53（1 取引は `len(pnl_list) < 2` で Sharpe 0.000）
- **報告 Sharpe 1.596 は、2 取引の 16 銘柄だけが作っている**
- 個別例: 5411 → 2 取引で Sharpe 24.465、6479 → 9.941

対照群も同じ汚染を受ける。OOS 2020-07-25〜2022-07-25 の `rsi_contrarian(30/70)` は 543 取引で Sharpe 77.156。ゲート閾値 30 の 18 倍の取引数がありながら、190 銘柄で約 2.9 取引/銘柄なので依然として少取引銘柄だらけである。**`champion_sharpe`（対チャンピオン改善ゲートの基準）自体が信用できない。**

## 目的

銘柄あたりの取引数が少なすぎる銘柄を集計から除外し、少取引アーティファクトが合格しないようにする。候補と対照群の双方に同じフィルタを適用することで `champion_sharpe` も浄化する。

## スコープ

含む:

1. 銘柄あたり最低取引数フィルタ + 有効銘柄数の下限ゲート
2. レポート / Issue 本文 / 批判的レビュー入力への新指標出力

含まない（別 Issue に切り出す）:

3. 評価値がデータ取得開始日に依存する問題（助走区間が評価期間に混入）
4. DSR の `n_obs` 単位不整合

## 設計

### 閾値

`python/config/settings.py` に追加し、env で上書き可能にする。

| 定数 | 既定値 | 意味 |
|---|---|---|
| `FACTORY_GATE_MIN_TRADES_PER_SYMBOL` | 3 | この本数未満の銘柄は集計から除外する |
| `FACTORY_GATE_MIN_EFFECTIVE_SYMBOLS` | 20 | 有効銘柄がこの数未満なら不合格 |

閾値 3 の根拠（IS 2024-07-25〜2026-07-25 実測）:

| ルール | 銘柄あたり取引数の分布 | 閾値3の有効銘柄 |
|---|---|---|
| #598 | 1取引:53, 2取引:16 | **0** |
| #564 | 1取引:39, 2取引:1 | **0** |
| ctl bollinger_band | 4取引以上のみ | 194（無傷） |
| ctl ema_momentum | 4取引以上のみ | 194（無傷） |
| ctl macd_rsi | 5取引以上のみ | 194（無傷） |
| ctl volatility_breakout | 7取引以上のみ | 194（無傷） |
| ctl rsi_contrarian | 1取引:4, 2取引:35, 3取引:46 … | 155（Sharpe 1.083→0.699） |
| ctl volume_breakout | 1取引:11, 2取引:36 … | 146 |

閾値 3 でアーティファクト 2 件は完全に排除され、健全な対照群 4 本は影響を受けない。

**閾値 2 は逆効果**である。Sharpe 0 の 1 取引銘柄が除外され、発散した 2 取引銘柄だけが残るため、#598 の平均 Sharpe は 1.596 → 6.885 に上昇する。閾値は 3 以上でなければならない。

有効銘柄数の下限 20 は、健全な対照群が閾値 3 で 146〜194 銘柄残ることから十分に緩い。合計取引数ゲートだけでは「2銘柄 × 20取引 = 40本」のような極端な集中を通してしまうため、独立した条件として必要である。

### モジュール構成

`factory.py` は現在 531 行で、CI の File Size Guard（上限 600 行）に対して余裕が小さい。また集計ロジックが DataFrame 越しにしかテストできない。そこで集計を純関数として切り出す。

```
src/backtest/factory_aggregation.py        （新規）
    SymbolMetrics            1銘柄の評価結果を保持する dataclass
    AggregatedMetrics        集計結果を保持する dataclass
    aggregate_symbol_metrics(rows, min_trades_per_symbol) -> AggregatedMetrics
```

`evaluate_hypothesis` は「銘柄ループで `SymbolMetrics` を溜める → `aggregate_symbol_metrics` を呼ぶ」構造になり、`factory.py` の行数はむしろ減る。

### データフロー

```
銘柄ごとに rule.generate_signal → backtester.simulate_trading      （現行どおり）
      ↓
SymbolMetrics(num_trades, sharpe, sharpe_per_trade, win_rate,
              total_return, max_drawdown) を溜める
      ↓
aggregate_symbol_metrics: num_trades >= min_trades_per_symbol の銘柄のみ採用
      ↓
sharpe_ratio / sharpe_per_trade / win_rate / total_return は有効銘柄の平均
max_drawdown は有効銘柄の最悪値（min）
num_trades は有効銘柄の合計
```

各指標の母数を明示する。

| 指標 | 母数 |
|---|---|
| `n_symbols`（既存） | データ取得に成功した銘柄数 |
| `n_symbols_with_signal` | 買いシグナルが 1 本以上出た銘柄数 |
| `n_effective_symbols` | `num_trades >= min_trades_per_symbol` を満たした銘柄数 |
| `avg_trades_per_symbol` | **シグナル発生銘柄**の取引数合計 ÷ `n_symbols_with_signal`（フィルタ前。フィルタの効き具合を診断するための値であるため） |
| `sharpe_ratio` ほか集計値 | `n_effective_symbols` |

`sharpe_per_trade` は DSR の入力であるため、DSR も有効銘柄のみに基づく値となる。DSR に渡す `n_obs`（銘柄横断の合計取引数）の単位不整合はスコープ外であり、本変更では従来どおり `evaluation.num_trades` を渡す。

### 変更するファイル

| ファイル | 変更内容 |
|---|---|
| `python/config/settings.py` | 定数 2 個を追加（`Field` 定義とモジュール直下の再公開の両方） |
| `python/src/backtest/factory_aggregation.py` | 新規。`SymbolMetrics` / `AggregatedMetrics` / `aggregate_symbol_metrics` |
| `python/src/backtest/types.py` | `FactoryEvaluation` に `n_symbols_with_signal` / `n_effective_symbols` / `avg_trades_per_symbol` を追加（既定値付き） |
| `python/src/backtest/factory.py` | `evaluate_hypothesis` を 2 段構えに変更、`apply_gate` に有効銘柄数条件を追加 |
| `python/src/backtest/factory_report.py` | Issue 本文のメトリクス表と `gate` ブロックに新指標を出力、既存ラベルの改称 |
| `python/src/backtest/hypothesis_review.py` | Claude へ渡すプロンプトに新指標を追加 |

### `apply_gate` の追加条件

```python
if evaluation.n_effective_symbols < FACTORY_GATE_MIN_EFFECTIVE_SYMBOLS:
    reasons.append(
        f"effective_symbols {evaluation.n_effective_symbols} "
        f"< {FACTORY_GATE_MIN_EFFECTIVE_SYMBOLS}"
    )
```

既存の `num_trades >= FACTORY_GATE_MIN_TRADES` は条件文を変更しない。フィルタ後の合計を見ることで自動的に厳しくなる。#598 では `num_trades 0 < 30` と `effective_symbols 0 < 20` の両方が不合格理由に並ぶ。

### レポート出力

Issue 本文のメトリクス表に 3 行追加し、母数が曖昧なラベルを改称する。

| 指標 | 変更 |
|---|---|
| Sharpe（銘柄平均） | → **Sharpe（有効銘柄平均）** に改称 |
| 取引数（合計） | → **取引数（有効銘柄合計）** に改称 |
| シグナル発生銘柄 | 新規 |
| 有効銘柄（N取引以上） | 新規。ゲート列に `>= FACTORY_GATE_MIN_EFFECTIVE_SYMBOLS` |
| 銘柄あたり平均取引数 | 新規 |

本文の `評価期間: … （N年、銘柄数 M）` は `データ取得銘柄数 M` に改める。従来の「銘柄数 194」は *データがあった* 銘柄数であり Sharpe の母数（実測 69）ではないため、誤読を招いていた。

`write_report` の `gate` ブロックにも同じ 3 指標を追加する。IssueAgent 側は `issue_title` / `issue_body` / `labels` のみを読むため、キー追加は後方互換である。

### 批判的レビューへの入力

`hypothesis_review.py:69-75` が Claude へ渡すプロンプトは現在「対象銘柄数 194 / 取引数 85」しか含まず、レビュアーが母数を誤認する構造になっている。実際 #598 のレビューは「194銘柄で約0.44件」と記述したが、正しくは「69銘柄で約1.23件」である。新指標 3 つを追加する。

### 窓別リターンは据え置く

`window_returns`（PBO の入力）は全銘柄のまま変更しない。PBO はバッチ単位の診断指標であり `apply_gate` の判定には使われていない。ここを変更すると PBO の意味自体が変わり本 Issue のスコープを超える。据え置く理由はコード上のコメントに残す。

### 後方互換

- `factory_runs` の過去行は `load_factory_hashes()`（重複排除）と `count_factory_runs()`（DSR の `n_trials`）にのみ使われ、`sharpe_ratio` は読まれない。過去行はそのまま残す。
- 出力済みの JSON レポートは「不変レポート」という設計思想のため削除しない。
- `sharpe_ratio` の算出方法が変わるため、本変更の前後で数値の直接比較はできない。`champion_sharpe` は毎バッチ対照群から再計算されるため自動的に新方式へ移行する。

## テスト

### `tests/unit/backtest/test_factory_aggregation.py`（新規）

1. 1 取引・2 取引の銘柄が Sharpe 平均から除外される（発散値が混入しない）
2. `num_trades` 合計がフィルタ後の値になる
3. `n_symbols_with_signal` / `n_effective_symbols` / `avg_trades_per_symbol` が正しい
4. 回帰: 全銘柄が 5 取引以上なら、フィルタの有無で結果が一致する
5. 有効銘柄 0 件で例外を出さず全値 0 を返す
6. `max_drawdown` は有効銘柄の最悪値（`min`）である

### `tests/unit/backtest/test_factory_gate.py`（新規）

7. `n_effective_symbols` が下限未満のとき不合格理由が出る
8. 閾値が env で上書きできる

### `tests/unit/backtest/test_factory_report_generated.py`（既存に追記）

9. 新指標が Issue 本文と `gate` ブロックに出力される

### 受け入れ検証（実データ）

ユニットテストとは別に、jp 194 銘柄の実データで以下を確認する。

- #598 / #564 のスペックが新ゲートで**不合格**となり、`gate_reasons` に有効銘柄数の理由が出る
- 健全な対照群 4 本（bollinger_band / ema_momentum / macd_rsi / volatility_breakout）が 194 銘柄のまま無傷である

## 受け入れ条件

- [ ] #598 / #564 のスペックを合格当時の期間で評価したとき、ゲートが不合格と判定する
- [ ] 合格レポートに「シグナル発生銘柄数」「有効銘柄数」「銘柄あたり平均取引数」が出力される
- [ ] 対照群 `champion_sharpe` が少取引銘柄の混入で発散しない
- [ ] 健全な対照群ルールの評価結果が本変更で悪化しない
- [ ] `check-ci.ps1` 相当のチェック（lint / mypy / pylint / import-linter / unit cov≥80%）が通る
