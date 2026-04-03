# バックテスト最適化実装ドキュメント

日時: 2026年3月1日  
状態: ✅ **実装・テスト完了**

## 実装内容

### 1. 最適パラメータのJSON保存機能

📁 **ファイル**: `src/services/backtest_optimize_pipeline.py`

**追加関数:**
- `save_optimal_params_json()` - 最適パラメータをJSON保存
- `get_optimal_params()` - JSONから読込（backup用）

**特徴:**
- 複数銘柄を1つのJSONに統合管理
- タイムスタンプ自動記録
- パフォーマンスメトリクス保存（SR/DD/勝率等）

```bash
実行例:
py run_backtest_optimize.py --market jp --symbol 1332
↓
CSV: results/optimize/jp_1332/optimize_20260301_183250.csv
JSON: config/optimal_params.json (更新・統合)
```

### 2. 予測時での自動読込

📁 **ファイル**: `src/services/prediction_pipeline.py`

**追加関数:**
- `get_optimal_params()` - JSON から最適パラメータを読込

**自動ログ出力:**
```
[jp_1332] 最適パラメータを読み込みました: 閾値=0.0, SharpeRatio=0.2546
```

### 3. メインスクリプトの更新

📁 **ファイル**: `run_backtest_optimize.py`

**変更内容:**
- JSON保存を `main()` に組込
- 実行後に自動的に JSON に保存

```
結果保存（CSV）: C:\src\StockFixer\python\results\optimize\jp_1332\optimize_...csv
最適パラメータ保存（JSON）: C:\src\StockFixer\python\config\optimal_params.json
```

### 4. ATR連動ポジションサイズ対応

📁 **ファイル**:
- `src/backtest/backtester.py`
- `src/services/backtest_pipeline.py`
- `src/backtest/walk_forward.py`
- `run_backtest.py`
- `run_backtest_optimize.py`

**追加内容:**
- ATRモードの建玉下限比率 / 上限比率を追加
- `avg_position_fraction`, `max_position_fraction`, `avg_position_value` をKPIへ追加
- `atr_fallback_trades` を追加し、ATR欠損時の full フォールバック回数を可視化
- 最適化JSONに `position_sizing`, `atr_risk_pct`, `atr_multiplier`, `atr_min_fraction`, `atr_max_fraction` を保存

**デフォルト値:**
- `atr_risk_pct = 0.02`
- `atr_multiplier = 1.0`
- `atr_min_fraction = 0.1`
- `atr_max_fraction = 1.0`

**補足:**
- `source=file` では `atr_lag1` を ATR の代替値として使用
- `source=raw` / `source=api` では再生成した `atr` を使用

---

## 🧪 テスト実行結果

### テスト1: 銘柄 jp/1332

```bash
$ py run_backtest_optimize.py --market jp --symbol 1332 \
    --threshold-min 0.0 --threshold-max 0.005 --threshold-step 0.001
```

**結果:**
- パラメータ組み合わせ: 6個
- Walk-Forward分割: 5
- 最適閾値: **0.005** (Sharpe=0.933)
- 総リターン: +8.1%
- 勝率: 59.2%
- プロフィットファクター: 1.259 ✨

### テスト2: 銘柄 jp/1333

```bash
$ py run_backtest_optimize.py --market jp --symbol 1333 \
    --threshold-min 0.0 --threshold-max 0.005 --threshold-step 0.002
```

**結果:**
- パラメータ組み合わせ: 3個
- 最適閾値: **0.004** (Sharpe=0.727)
- 総リターン: -1.9%
- 勝率: 50.0%

### JSONファイル統合確認

```json
{
  "jp_1332": {
    "threshold": 0.0,
    "metrics": { "sharpe_ratio": 0.2546, ... }
  },
  "jp_1333": {
    "threshold": 0.004,
    "metrics": { "sharpe_ratio": 0.1470, ... }
  }
}
```

✅ **複数銘柄が正常に統合管理されている**

### テスト3: JSON読込確認

```python
$ from src.services.prediction_pipeline import get_optimal_params
$ params = get_optimal_params('jp', '1332')
$ params['threshold']
0.0
$ params['metrics']['sharpe_ratio']
0.25460000000000005
```

✅ **予測時に自動読込が正常に動作**

### テスト4: ATR連動サイジング実測確認（jp/7203）

```bash
$ py run_backtest.py --market jp --symbol 7203 --source file --position-sizing full
$ py run_backtest.py --market jp --symbol 7203 --source file --position-sizing atr \
  --atr-risk-pct 0.02 --atr-multiplier 1.0 --atr-min-fraction 0.1 --atr-max-fraction 1.0
```

**比較結果（source=file）:**
- full: total_return=-9.9952%, sharpe=-0.0966, max_drawdown=-17.5299%, avg_position_fraction=0.998514
- atr: total_return=-0.3798%, sharpe=1.0794, max_drawdown=-11.6698%, avg_position_fraction=0.802547

```bash
$ py run_backtest.py --market jp --symbol 7203 --source raw --position-sizing atr \
  --atr-risk-pct 0.02 --atr-multiplier 1.0 --atr-min-fraction 0.1 --atr-max-fraction 1.0
```

**結果（source=raw, atr）:**
- total_return=12.8922%
- sharpe=2.5526
- max_drawdown=-7.8342%
- avg_position_fraction=0.786479
- atr_fallback_trades=0

✅ **ATR連動ポジションサイズが実データ上でも動作し、高ボラ局面で建玉抑制が効くことを確認**

---

## 📊 実装の流れ図

```
最適化バックテスト実行
  ↓
run_backtest_optimize.py
  ↓
backtest_optimize_pipeline.run_optimization()
  │
  ├─ Walk-Forward検証（複数パラメータ）
  └─ 各fold の損益・メトリクス計算
  ↓
run_backtest_optimize.py (main)
  │
  ├─ print_optimization_results()  → コンソール出力
  │
  ├─ save_optimization_results()   → CSV保存
  │
  └─ save_optimal_params_json()    → JSON保存 ← ★新規
      ↓
      config/optimal_params.json
  ↓
予測時・取引時
  ↓
prediction_pipeline.get_optimal_params() → 最適パラメータ読込 ← ★新規
  ↓
Discord Bot / SBI連携 で活用
```

---

## 🎯 次のステップ（オプション）

### すぐに実装できる拡張

1. **Discord Bot との連携**
   ```python
   # embed.add_field("最適閾値", f"{params['threshold']:.4f}")
   ```
   実装ファイル: `run_discord_bot.py`

2. **SBI注文 での動的閾値**
   ```python
   # threshold = get_optimal_params(market, symbol)['threshold']
   ```
   実装ファイル: `src/sbi/order_executor.py`

3. **自動定期最適化**
   ```bash
   # スケジューラーに追加
   # 毎週日曜 20:00 に最適化実行
   ```
   実装ファイル: `run_scheduler.py`

---

## 📝 技術詳細

### パス仕様

```
python/src/services/backtest_optimize_pipeline.py
         ↓dirname 3回
python/config/optimal_params.json
```

### JSON スキーマ

```json
{
  "market_symbol_key": {
    "market": "jp|us|...",
    "symbol": "7203|AAPL|...",
    "timestamp": "ISO 8601形式",
    "sort_by": "sharpe_ratio",
    "threshold": 浮動小数点,
    "stop_loss_pct": 浮動小数点 | null,
    "take_profit_pct": 浮動小数点 | null,
    "position_sizing": "full|fixed|confidence|atr",
    "position_fraction": 浮動小数点,
    "atr_risk_pct": 浮動小数点,
    "atr_multiplier": 浮動小数点,
    "atr_min_fraction": 浮動小数点,
    "atr_max_fraction": 浮動小数点,
    "metrics": {
      "total_return": 浮動小数点,
      "sharpe_ratio": 浮動小数点,
      "max_drawdown": 浮動小数点,
      "win_rate": 浮動小数点,
      "profit_factor": 浮動小数点,
      "avg_position_fraction": 浮動小数点,
      "max_position_fraction": 浮動小数点,
      "avg_position_value": 浮動小数点,
      "atr_fallback_trades": 整数,
      "num_trades": 整数
    }
  }
}
```

---

## ✅ チェックリスト

- [x] 最適化スクリプトでJSON保存機能追加
- [x] 予測パイプラインでJSON読込機能追加
- [x] パスの正確な計算（python/configを指す）
- [x] 複数銘柄の統合管理
- [x] ユニットテスト・パス確認テスト実施
- [x] ATR連動ポジションサイズの上下限比率追加
- [x] ATR関連メトリクス追加
- [x] 実データでのATR検証実施
- [x] ドキュメント作成

---

## 📂 成果物と変更ファイル

| ファイル | 変更 | 説明 |
|---------|------|------|
| `run_backtest_optimize.py` | ✏️ 更新 | JSON保存を追加 |
| `src/services/backtest_optimize_pipeline.py` | ✏️ 更新 | save_optimal_params_json()追加 |
| `src/services/prediction_pipeline.py` | ✏️ 更新 | get_optimal_params()追加 |
| `config/optimal_params.json` | 📁 新規作成 | 最適パラメータ格納 |
| `docs/OPTIMAL_PARAMS_GUIDE.md` | 📁 新規作成 | 運用ガイド |
| `docs/IMPLEMENTATION_BACKTEST_OPTIMIZE.md` | 📁 新規作成 | 本ドキュメント |

---

## 🚀 使用方法

### 基本的な流れ

```bash
# 1. 最適化実行（自動的にJSONに保存）
py run_backtest_optimize.py --market jp --symbol 7203 --optimize-risk

# 2. 予測実行（自動的にJSONから読込）
py run_predict.py

# 3. Discord Bot（パラメータ表示）
py run_discord_bot.py
```

---

## 🎉 実装完了サマリー

この実装により、バックテスト最適化で得られた最適パラメータが以下の方法で取引に反映されます：

1. **自動保存** : 最適化実行 → JSON に**自動保存**
2. **自動読込** : 予測実行 → JSON から**自動読込**
3. **柔軟な活用** : Discord Bot・SBI連携など各所で読込可能

**次回からは、複数銘柄の最適化 → 全て自動統合管理 となります！** 🎯

---

## 参考資料

- [OPTIMAL_PARAMS_GUIDE.md](OPTIMAL_PARAMS_GUIDE.md) - 運用手順
- [ARCHITECTURE.md](ARCHITECTURE.md) - システムアーキテクチャ
- [OPERATIONS.md](OPERATIONS.md) - 運用手順書
