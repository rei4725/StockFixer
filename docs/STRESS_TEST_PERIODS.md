# ストレス期間定義（正本）

このドキュメントはストレステスト対象の歴史的暴落期間の定義を管理する正本です。

## ストレスシナリオ一覧

| シナリオID | 名称 | 期間 | 概要 |
|---|---|---|---|
| corona | コロナショック | 2020-02-01 〜 2020-03-31 | COVID-19パンデミックによる急落 |
| lehman | リーマンショック | 2008-09-01 〜 2008-10-31 | リーマン・ブラザーズ破綻に伴う金融危機 |

## 合格基準

| 指標 | 基準 |
|---|---|
| MDD（最大ドローダウン） | \|MDD\| ≤ 15% |

## シナリオ定義ファイル

シナリオは `python/src/services/stress_test_pipeline.py` の `STRESS_SCENARIOS` 定数で管理しています。
シナリオを追加・変更する際はこのドキュメントと定数を同期して更新してください。

```python
STRESS_SCENARIOS = {
    "corona": {
        "label": "コロナショック",
        "start_date": "2020-02-01",
        "end_date":   "2020-03-31",
    },
    "lehman": {
        "label": "リーマンショック",
        "start_date": "2008-09-01",
        "end_date":   "2008-10-31",
    },
}
MDD_THRESHOLD = 0.15
```

## 実行手順

```powershell
# 単一銘柄・全シナリオ（yfinanceから取得）
py run_stress_test.py --market jp --symbol 7203

# 単一銘柄・コロナショックのみ
py run_stress_test.py --market us --symbol AAPL --scenario corona

# 複数銘柄
py run_stress_test.py --market jp --symbols 7203 6758 9984

# MDD閾値をカスタム設定（20%）
py run_stress_test.py --market jp --symbol 7203 --mdd-threshold 0.20
```

## 結果出力

実行結果は `python/results/stress_test/stress_test_YYYYMMDD_HHMMSS.csv` に保存されます。

### 出力列

| 列名 | 説明 |
|---|---|
| market | マーケット識別子 |
| symbol | 銘柄シンボル |
| scenario_name | シナリオID (corona / lehman) |
| period_start | シナリオ開始日 |
| period_end | シナリオ終了日 |
| mdd | 最大ドローダウン（負の値） |
| sharpe_ratio | シャープレシオ |
| total_return | トータルリターン |
| win_rate | 勝率 |
| num_trades | 取引数 |
| max_consecutive_losses | 最大連敗数 |
| mdd_pass | MDD合格判定 (True/False) |

## 後続施策との依存関係

本ストレステスト（R-205）の検証結果は以下の施策の前提となります：

- **R-216**: バックテスト最適パラメータ自動ロード
- **R-217**: Kelly実績更新
- **R-307**: ドローダウン適応型資本配分

ストレスシナリオで MDD 15% 以下を確認した後に上記施策を実施することで、
リスク管理の統計的根拠を確立します。
