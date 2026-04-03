# 最適化バックテスト運用ガイド

## 概要

バックテスト最適化で特定された最適パラメータを、実際の取引に反映させるための手順です。

## ✅ 実装状況

### 1️⃣ **JSON保存格式**

最適化実行後、`python/config/optimal_params.json` に各銘柄の最適パラメータが自動保存されます。

```json
{
  "jp_1332": {
    "market": "jp",
    "symbol": "1332",
    "timestamp": "2026-03-01T18:32:50.818256",
    "sort_by": "sharpe_ratio",
    "threshold": 0.0,
    "stop_loss_pct": null,
    "take_profit_pct": null,
    "position_sizing": "atr",
    "position_fraction": 0.5,
    "atr_risk_pct": 0.02,
    "atr_multiplier": 1.0,
    "atr_min_fraction": 0.1,
    "atr_max_fraction": 1.0,
    "metrics": {
      "total_return": -0.1520988,
      "sharpe_ratio": 0.25460000000000005,
      "max_drawdown": -0.39801699999999995,
      "win_rate": 0.57524,
    "profit_factor": 1.06494,
    "avg_position_fraction": 0.78,
    "max_position_fraction": 0.99,
    "avg_position_value": 845000.0,
    "atr_fallback_trades": 0,
      "num_trades": 182
    }
  },
  "jp_1333": {
    "market": "jp",
    "symbol": "1333",
    "timestamp": "2026-03-01T18:33:09.091922",
    "sort_by": "sharpe_ratio",
    "threshold": 0.004,
    "metrics": { ... }
  }
}
```

### 2️⃣ **予測時にJSONから読み込み**

```python
from src.services.prediction_pipeline import get_optimal_params

# 銘柄ごとの最適パラメータを取得
params = get_optimal_params('jp', '1332')

if params:
    threshold = params['threshold']  # 0.0
    sharpe_ratio = params['metrics']['sharpe_ratio']  # 0.2546
    position_sizing = params.get('position_sizing', 'full')
    atr_risk_pct = params.get('atr_risk_pct', 0.02)
    # ...
```

## 🔄 利用シーン別の実装

### シーン1: Discord Bot での最適閾値表示

```python
# run_discord_bot.py の /forecast コマンド内で
optimal_params = get_optimal_params(market, symbol)
if optimal_params:
    embed.add_field(
        name="最適閾値",
        value=f"{optimal_params['threshold']:.4f} (SR: {optimal_params['metrics']['sharpe_ratio']:.2f})",
        inline=False
    )
```

### シーン2: SBI連携での動的閾値

```python
# src/sbi/order_executor.py で
optimal = get_optimal_params(market, symbol)
threshold = optimal.get('threshold', 0.0) if optimal else 0.0

# シグナル判定
if abs(pred_change_rate) >= threshold:
    # 注文実行
    place_order(...)
```

### シーン3: 定期的な最適化と反映

```bash
# 毎週日曜夜に実行スケジュール
# cron: run_backtest_optimize.py --market jp --symbol <symbol> --optimize-risk

# 結果は自動的にJSON更新
# → 翌営業日の予測・取引に反映
```

## 📊 複数銘柄の一括最適化

```bash
# 複数銘柄を順序実行
for symbol in 7203 7201 8306 9984; do
    py run_backtest_optimize.py --market jp --symbol $symbol --threshold-step 0.001
done

# → optimal_params.json に全銘柄が統合保存される
```

## 🛡️ 運用上の注意

### 最適パラメータ更新タイミング

| オプション | 推奨頻度 | 理由 |
|-----------|---------|------|
| 毎営業日 | ❌ | 過学習のリスク |
| **毎週** | ✅ | バランスの取れた更新 |
| 毎月 | ✅ | 安定性重視 |
| マーケット急変時 | ✅ | 臨機応変対応 |

### バックテスト閾値の設定

```bash
# 短期的な変動に強い（保守的）
--threshold-min 0.0 --threshold-max 0.01 --threshold-step 0.0005

# より積極的な売買
--threshold-min -0.005 --threshold-max 0.02 --threshold-step 0.001

# リスク調整を含める
--optimize-risk  # SL/TPも最適化

# ATR連動ポジションサイズも同時に最適化
--position-sizing atr --atr-risk-pcts 0.01,0.02,0.03 --atr-multipliers 0.5,1.0,1.5
```

### ATR連動ポジションサイズの運用メモ

- `--source file` では保存済み特徴量の `atr_lag1` を ATR 代替値として使用する
- `--source raw` / `--source api` では再生成した `atr` をそのまま使用する
- デフォルト値は `atr_risk_pct=0.02`, `atr_multiplier=1.0`, `atr_min_fraction=0.1`, `atr_max_fraction=1.0`
- 高ボラ局面で建玉比率が下がるかは `avg_position_fraction` と `max_position_fraction` で確認する

### 実測メモ（2026-04-03, jp/7203）

- `source=file`, `position_sizing=full`: total_return=-0.099952, sharpe_ratio=-0.0966, max_drawdown=-0.175299
- `source=file`, `position_sizing=atr`: total_return=-0.003798, sharpe_ratio=1.0794, max_drawdown=-0.116698, avg_position_fraction=0.802547
- `source=raw`, `position_sizing=atr`: total_return=0.128922, sharpe_ratio=2.5526, max_drawdown=-0.078342, avg_position_fraction=0.786479

少なくとも検証サンプルでは、ATR連動により建玉を抑えながらドローダウンとシャープレシオが改善した。

## 📈 パフォーマンス監視

### 最適パラメータの妥当性確認

```python
import json

with open('config/optimal_params.json') as f:
    params = json.load(f)

for key, p in params.items():
    metrics = p['metrics']
    # Win Rate が 50% 以上あるか確認
    if metrics['win_rate'] < 0.5:
        print(f"⚠️  {key}: 勝率が低い ({metrics['win_rate']:.1%})")

    # Sharpe Ratio が妥当か
    if metrics['sharpe_ratio'] < 0:
        print(f"⚠️  {key}: シャープレシオが負 ({metrics['sharpe_ratio']:.2f})")
```

## 🔧 トラブルシューティング

### JSONが見つからない

```bash
# パス確認
ls -la python/config/optimal_params.json

# なければ一度最適化を実行
py run_backtest_optimize.py --market jp --symbol 7203 --threshold-max 0.01
```

### パラメータが古い

```bash
# タイムスタンプを確認
python -c "
import json
with open('config/optimal_params.json') as f:
    params = json.load(f)
    for k, v in params.items():
        print(f'{k}: {v[\"timestamp\"]}')"
```

## 📝 まとめ

| 状況 | 対応 |
|-----|------|
| ✅ 最適化実行 → JSON自動保存 | `run_backtest_optimize.py` |
| ✅ 予측時に自動読込 | `prediction_pipeline.get_optimal_params()` |
| ✅ Discord Bot に表示 | `add_field()` で表示 |
| ✅ SBI連携で使用 | `order_executor.py` で動的閾値 |
| ❓ カスタム実装 | `config/optimal_params.json` を直接参照 |

## 関連ドキュメント

- [IMPLEMENTATION_BACKTEST_OPTIMIZE.md](IMPLEMENTATION_BACKTEST_OPTIMIZE.md) - 実装詳細
- [ARCHITECTURE.md](ARCHITECTURE.md) - バックテスト最適化のアーキテクチャ
- [OPERATIONS.md](OPERATIONS.md) - 運用手順書
