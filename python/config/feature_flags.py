"""
機能フラグ（純粋なON/OFF真偽値のみ）の一元管理

しきい値・件数上限等の数値パラメータや秘密情報（APIキー・トークン）は
settings.py 側に残す。ここには「有効化しても安全性に直結する調整を伴わない」
真偽値フラグのみを置き、Claude Code 等の自動化エージェントが settings.py 全体
に触れることなくこのファイル単体の編集でフラグ切り替えを完結できるようにする。

env変数での上書きは settings.py の Settings クラスと同じ挙動
（pydantic-settings BaseSettings、.env読み込み対応、大文字小文字を区別）。
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class FeatureFlags(BaseSettings):
    """機能の有効/無効フラグ（真偽値のみ）。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
    )

    # ---------- ショートサイド（R-215） ----------
    ENABLE_SHORT_SIDE: bool = Field(default=False)

    # ---------- DB メンテナンス / retention（scheduler.run_weekly_db_maintenance） ----------
    # 月初週の週次メンテで VACUUM (ANALYZE) を実行し、肥大化・統計情報の陳腐化を防ぐか。
    DB_COMPACT_ENABLED: bool = Field(default=True)

    # ---------- Claude トレーダー（claude_agent.py） ----------
    CLAUDE_TRADER_ENABLED: bool = Field(default=False)

    # ---------- Claude 週次レビュー講評（reporting/llm_review.py） ----------
    # 週次レポートに Claude が生成した総評を添える。既定無効で安全にロールアウトする。
    LLM_REVIEW_ENABLED: bool = Field(default=False)

    # ---------- Claude バックテスト批判レビュー（backtest/critical_review.py） ----------
    # Claude にバックテスト方法論・設定を批判的レビューさせ、欠陥を Issue 草案 JSON 化する。
    # 既定無効。読み取り専用（コード変更・発注に非関与）。
    BACKTEST_REVIEW_ENABLED: bool = Field(default=False)

    # ---------- Claude 仮説単位レビュー（backtest/hypothesis_review.py） ----------
    # 戦略ファクトリーのゲート通過仮説1件ごとに Claude が過学習/偶然性リスクを評価する。
    # 既定無効。読み取り専用（ゲート判定・コード変更・発注には一切関与しない）。
    FACTORY_HYPOTHESIS_REVIEW_ENABLED: bool = Field(default=False)

    # ---------- 戦略ファクトリー自動昇格ループ: 昇格記録（orchestration/jobs/periodic.py） ----------
    # 既定無効。マージ検知ジョブはこのフラグが true になるまで一切のGitHub API呼び出しを行わない。
    STRATEGY_PROMOTION_CHECK_ENABLED: bool = Field(default=False)

    # ---------- Claude テスト穴埋めボット（quality/test_gap_review.py） ----------
    # coverage の未カバー行を Claude に渡し不足テストの追加案を Issue 草案 JSON 化する。
    # 既定無効。読み取り専用（テストコードは生成せず Issue 草案のみ）。
    TEST_GAP_ENABLED: bool = Field(default=False)

    # ---------- 昇格ゲート自動承認（#254） ----------
    AUTO_PROMOTE_MODEL: bool = Field(default=False)

    # ---------- 戦略ファクトリー（#369 Phase 1） ----------
    FACTORY_ENABLED: bool = Field(default=False)

    # ---------- Claudeルール生成（backtest/claude_rule_generator.py） ----------
    # 既定無効。夜間バッチでClaudeに新しいTradingRule実装を発想させ、Dockerサンドボックス
    # で隔離実行してから既存の戦略ファクトリーのゲートに通す。ゲート合格分のみIssue化され、
    # 人間のPRレビューを経て初めて本番反映される（Claudeは合否判定・本番反映に一切関与しない）。
    FACTORY_CLAUDE_RULEGEN_ENABLED: bool = Field(default=False)


flags = FeatureFlags()
