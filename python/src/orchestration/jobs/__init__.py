"""スケジューラジョブパッケージ（#497 で scheduler.py から責務分割）。

cadence 別にジョブ定義を分割して保持する:
    - common   : 共通ヘルパー（エラーハンドリング・日付判定）
    - daily    : 日次・場中ジョブ
    - weekly   : 週次ジョブ
    - periodic : 月次・夜間ジョブ

外部からは後方互換のため従来どおり `src.orchestration.scheduler` 経由で
import できる（scheduler.py がファサードとして re-export する）。
"""
