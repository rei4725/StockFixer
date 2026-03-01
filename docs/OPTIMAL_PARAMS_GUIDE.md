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
    "metrics": {
      "total_return": -0.1520988,
      "sharpe_ratio": 0.25460000000000005,
      "max_drawdown": -0.39801699999999995,
      "win_rate": 0.57524,
      "profit_factor": 1.06494,
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
```

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
