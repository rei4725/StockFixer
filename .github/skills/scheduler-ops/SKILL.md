---
name: scheduler-ops
description: "スケジューラーの起動・管理・即時実行を行う。スケジューラー、scheduler、定期実行、日次パイプライン、週次学習、cron、daily、weeklyの話題で使用する。"
metadata:
  author: StockFixer
  version: "1.0"
compatibility: "Python 3.10+, APScheduler。python/ ディレクトリで実行。"
---

## Goal
APSchedulerベースのジョブスケジューラーを適切に運用する。

## Procedure

### スケジューラー起動
```bash
cd python
# 通常起動（BlockingScheduler）
py run_scheduler.py

# Discord Bot と同時起動（BackgroundScheduler）
py run_scheduler.py --with-bot
```

### パイプライン即時実行（テスト用）
```bash
py run_scheduler.py --run-now daily
py run_scheduler.py --run-now weekly
```

### スケジュール定義
| ジョブID | スケジュール | 処理内容 |
|----------|-------------|----------|
| `daily_pipeline` | 月〜金 19:00 JST | データバッチ取得 → 統合モデル予測 → Discord通知 |
| `weekly_model_training` | 毎週土曜 03:00 JST | XGBoost + LightGBM 統合モデル再学習 |

### 共通設定
- `misfire_grace_time=3600` — 1時間以内の遅延なら実行
- `coalesce=True` — 複数回分溜まっても1回だけ実行

### 日次パイプラインの処理フロー
1. `run_data_batch()` — ウォッチリスト全銘柄のデータ更新
2. `predict_all_unified()` — 統合モデルで全銘柄予測
3. Discord Webhook で Top10/Worst10 通知

### 週次学習の処理フロー
1. XGBoost 統合モデル再学習
2. LightGBM 統合モデル再学習
3. 保存先: `python/models/unified/`

## Key Functions
- `run_daily_pipeline()` — 日次パイプライン（データ→予測→通知）
- `run_weekly_training()` — 週次統合モデル再学習

## References
- [scheduler_pipeline.py](../../../python/src/services/scheduler_pipeline.py)
- [run_scheduler.py](../../../python/run_scheduler.py)
