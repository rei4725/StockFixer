"""
アプリケーション設定値の一元管理

pydantic-settings の BaseSettings で env 変数を型安全に読み込み、
パース失敗・型不正は起動時に即時例外で fail させる。

既存コードとの互換のため、`from config.settings import MAX_DAILY_LOSS_RATE`
のような module-level 定数アクセスを維持する。
"""

import os
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from config.trading_policy import KELLY_CAP as _KELLY_CAP


class Settings(BaseSettings):
    """環境変数から読み込まれるアプリケーション設定。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
    )

    # ---------- リスク管理（risk_manager.py） ----------
    MAX_DAILY_LOSS_RATE: float = Field(default=0.02)
    MAX_POSITION_RATE: float = Field(default=0.10)
    MAX_CONSECUTIVE_LOSSES: int = Field(default=3)
    MAX_POSITIONS: int = Field(default=10)
    HALF_KELLY: Optional[float] = Field(default=None)

    # ---------- 自動発注（order_execution_pipeline.py） ----------
    MIN_CHANGE_RATIO: float = Field(default=0.003)
    BUY_THRESHOLD: float = Field(default=0.005)
    SELL_THRESHOLD: float = Field(default=-0.005)
    MAX_ORDERS_PER_RUN: int = Field(default=5)
    MAX_SECTOR_POSITIONS: int = Field(default=3)
    LIMIT_ORDER_AVG_VOLUME_THRESHOLD: int = Field(default=500_000)
    LIMIT_ORDER_SPREAD_PROXY_THRESHOLD: float = Field(default=0.01)
    LIMIT_ORDER_PRICE_BUFFER: float = Field(default=0.001)
    LIMIT_ORDER_LOOKBACK_DAYS: int = Field(default=5)
    EARNINGS_MASK_WINDOW_DAYS: int = Field(default=3)
    FEATURE_SELECTION_DROP_RATIO: float = Field(default=0.2)
    FEATURE_SELECTION_MIN_FEATURES: int = Field(default=10)
    PERMUTATION_IMPORTANCE_REPEATS: int = Field(default=5)
    FEATURE_SELECTION_PROTECT_TOP_SHAP: int = Field(default=10)

    # ---------- ペーパートレード（paper_broker.py） ----------
    PAPER_INITIAL_BALANCE: float = Field(default=1_000_000.0)

    # ---------- ショートサイド（R-215） ----------
    ENABLE_SHORT_SIDE: bool = Field(default=False)

    # ---------- 相関リスク管理（correlation_risk.py） ----------
    CORRELATION_ENC_THRESHOLD: float = Field(default=2.0)
    CORRELATION_WINDOW_DAYS: int = Field(default=20)
    CORRELATION_PAIRWISE_THRESHOLD: float = Field(default=0.7)
    CORRELATION_PAIRWISE_WINDOW_DAYS: int = Field(default=60)

    # ---------- DB メンテナンス / retention（scheduler.run_weekly_db_maintenance） ----------
    # 診断ログ（shap_values / feature_selection_log）の保持日数。
    # これより古い行は週次メンテで削除する（各グループの最新は常に保持）。
    DB_LOG_RETENTION_DAYS: int = Field(default=30)
    # 月初週の週次メンテで物理コンパクション（再構築で死領域を回収）を実行するか。
    # VACUUM はファイルを縮小しないため、定期的な再構築で肥大の再発を防ぐ。
    DB_COMPACT_ENABLED: bool = Field(default=True)
    # 週次メンテ後の DB ファイルサイズがこの GB を超えたら Discord に警告する。
    DB_SIZE_ALERT_GB: float = Field(default=20.0)

    # ---------- Claude トレーダー（claude_agent.py） ----------
    CLAUDE_TRADER_ENABLED: bool = Field(default=False)
    CLAUDE_TRADER_MODEL: str = Field(default="claude-opus-4-7")
    CLAUDE_TRADER_THINKING_BUDGET: int = Field(default=5000)
    CLAUDE_TRADER_MAX_TOKENS: int = Field(default=8192)

    # ---------- Claude 週次レビュー講評（reporting/llm_review.py） ----------
    # 週次レポートに Claude が生成した総評を添える。既定無効で安全にロールアウトする。
    LLM_REVIEW_ENABLED: bool = Field(default=False)
    LLM_REVIEW_MODEL: str = Field(default="claude-opus-4-8")
    LLM_REVIEW_MAX_TOKENS: int = Field(default=1024)

    # ---------- Claude バックテスト批判レビュー（backtest/critical_review.py） ----------
    # Claude にバックテスト方法論・設定を批判的レビューさせ、欠陥を Issue 草案 JSON 化する。
    # 既定無効。読み取り専用（コード変更・発注に非関与）。
    BACKTEST_REVIEW_ENABLED: bool = Field(default=False)
    BACKTEST_REVIEW_MODEL: str = Field(default="claude-opus-4-8")
    BACKTEST_REVIEW_MAX_TOKENS: int = Field(default=4096)

    # ---------- Claude テスト穴埋めボット（quality/test_gap_review.py） ----------
    # coverage の未カバー行を Claude に渡し不足テストの追加案を Issue 草案 JSON 化する。
    # 既定無効。読み取り専用（テストコードは生成せず Issue 草案のみ）。
    TEST_GAP_ENABLED: bool = Field(default=False)
    TEST_GAP_MODEL: str = Field(default="claude-opus-4-8")
    TEST_GAP_MAX_TOKENS: int = Field(default=4096)
    TEST_GAP_MAX_FILES: int = Field(default=5)
    TEST_GAP_MIN_COVERAGE: float = Field(default=80.0)
    # CLI argv 上限（Windows 32,767 文字）対策のプロンプト文字数バジェット
    TEST_GAP_CONTEXT_CHAR_BUDGET: int = Field(default=16000)

    # ---------- LLM バックエンド選択（infrastructure/llm/factory.py） ----------
    # "sdk": anthropic SDK（ANTHROPIC_API_KEY 従量課金・既定）
    # "cli": Claude Code CLI（Pro/Max サブスク認証・CLAUDE_CODE_OAUTH_TOKEN）
    LLM_BACKEND: str = Field(default="sdk")
    CLAUDE_CLI_BINARY: str = Field(default="claude")
    CLAUDE_CLI_TIMEOUT_SECONDS: int = Field(default=180)

    # ---------- モデルドリフト監視（R-274） ----------
    DRIFT_ALERT_WEEKS: int = Field(default=4)
    DRIFT_ALERT_THRESHOLD: float = Field(default=0.05)

    # ---------- 予測並列化（#266） ----------
    PREDICTION_MAX_WORKERS: int = Field(default=max(1, (os.cpu_count() or 2) // 2))

    # ---------- スケジューラリトライ設定 ----------
    SCHEDULER_MAX_RETRIES: int = Field(default=3)
    SCHEDULER_RETRY_BASE_WAIT_SECONDS: float = Field(default=30.0)

    # ---------- 出来高フィルター（signal_generator.py） ----------
    VOLUME_FILTER_WINDOW_DAYS: int = Field(default=20)
    VOLUME_FILTER_MULTIPLIER: float = Field(default=1.5)

    # ---------- 昇格ゲート自動承認（#254） ----------
    AUTO_PROMOTE_MODEL: bool = Field(default=False)

    # ---------- センチメント特徴量（R-404） ----------
    SENTIMENT_LOOKBACK_DAYS: int = Field(default=30)

    # ---------- バックテストのデフォルトスリッページ（#494・片道率） ----------
    # 流動性差を反映: 米大型株はタイト、日本株はやや広い。
    DEFAULT_SLIPPAGE_US: float = Field(default=0.0005)  # 5bps
    DEFAULT_SLIPPAGE_JP: float = Field(default=0.0010)  # 10bps

    # ---------- 戦略ファクトリー（#369 Phase 1） ----------
    FACTORY_ENABLED: bool = Field(default=False)
    FACTORY_NIGHTLY_BUDGET: int = Field(default=10)
    FACTORY_LOOKBACK_YEARS: int = Field(default=2)
    FACTORY_N_WINDOWS: int = Field(default=8)
    FACTORY_GATE_MIN_TRADES: int = Field(default=30)
    FACTORY_GATE_MIN_DSR: float = Field(default=0.95)
    FACTORY_GATE_MAX_PBO: float = Field(
        default=0.5
    )  # バッチ診断の閾値（per-hypothesis ゲートではない）
    FACTORY_GATE_MAX_DRAWDOWN: float = Field(default=-0.25)
    FACTORY_GATE_CHAMPION_MARGIN: float = Field(
        default=1.0
    )  # チャンピオン超え（×1.1 は到達不能だったため 1.0）


settings = Settings()

# ---------------------------------------------------------------------------
# 後方互換: 既存コードは `from config.settings import XXX` で import している
# ため、Settings インスタンスの値を module-level 定数として再エクスポートする。
# ---------------------------------------------------------------------------

MAX_DAILY_LOSS_RATE: float = settings.MAX_DAILY_LOSS_RATE
MAX_POSITION_RATE: float = settings.MAX_POSITION_RATE
MAX_CONSECUTIVE_LOSSES: int = settings.MAX_CONSECUTIVE_LOSSES
MAX_POSITIONS: int = settings.MAX_POSITIONS
HALF_KELLY: float = settings.HALF_KELLY if settings.HALF_KELLY is not None else _KELLY_CAP / 2

MIN_CHANGE_RATIO: float = settings.MIN_CHANGE_RATIO
BUY_THRESHOLD: float = settings.BUY_THRESHOLD
SELL_THRESHOLD: float = settings.SELL_THRESHOLD
MAX_ORDERS_PER_RUN: int = settings.MAX_ORDERS_PER_RUN
MAX_SECTOR_POSITIONS: int = settings.MAX_SECTOR_POSITIONS
LIMIT_ORDER_AVG_VOLUME_THRESHOLD: int = settings.LIMIT_ORDER_AVG_VOLUME_THRESHOLD
LIMIT_ORDER_SPREAD_PROXY_THRESHOLD: float = settings.LIMIT_ORDER_SPREAD_PROXY_THRESHOLD
LIMIT_ORDER_PRICE_BUFFER: float = settings.LIMIT_ORDER_PRICE_BUFFER
LIMIT_ORDER_LOOKBACK_DAYS: int = settings.LIMIT_ORDER_LOOKBACK_DAYS
EARNINGS_MASK_WINDOW_DAYS: int = settings.EARNINGS_MASK_WINDOW_DAYS
FEATURE_SELECTION_DROP_RATIO: float = settings.FEATURE_SELECTION_DROP_RATIO
FEATURE_SELECTION_MIN_FEATURES: int = settings.FEATURE_SELECTION_MIN_FEATURES
PERMUTATION_IMPORTANCE_REPEATS: int = settings.PERMUTATION_IMPORTANCE_REPEATS
FEATURE_SELECTION_PROTECT_TOP_SHAP: int = settings.FEATURE_SELECTION_PROTECT_TOP_SHAP

PAPER_INITIAL_BALANCE: float = settings.PAPER_INITIAL_BALANCE

ENABLE_SHORT_SIDE: bool = settings.ENABLE_SHORT_SIDE

CORRELATION_ENC_THRESHOLD: float = settings.CORRELATION_ENC_THRESHOLD
CORRELATION_WINDOW_DAYS: int = settings.CORRELATION_WINDOW_DAYS
CORRELATION_PAIRWISE_THRESHOLD: float = settings.CORRELATION_PAIRWISE_THRESHOLD
CORRELATION_PAIRWISE_WINDOW_DAYS: int = settings.CORRELATION_PAIRWISE_WINDOW_DAYS

CLAUDE_TRADER_ENABLED: bool = settings.CLAUDE_TRADER_ENABLED
CLAUDE_TRADER_MODEL: str = settings.CLAUDE_TRADER_MODEL
CLAUDE_TRADER_THINKING_BUDGET: int = settings.CLAUDE_TRADER_THINKING_BUDGET
CLAUDE_TRADER_MAX_TOKENS: int = settings.CLAUDE_TRADER_MAX_TOKENS

LLM_REVIEW_ENABLED: bool = settings.LLM_REVIEW_ENABLED
LLM_REVIEW_MODEL: str = settings.LLM_REVIEW_MODEL
LLM_REVIEW_MAX_TOKENS: int = settings.LLM_REVIEW_MAX_TOKENS

BACKTEST_REVIEW_ENABLED: bool = settings.BACKTEST_REVIEW_ENABLED
BACKTEST_REVIEW_MODEL: str = settings.BACKTEST_REVIEW_MODEL
BACKTEST_REVIEW_MAX_TOKENS: int = settings.BACKTEST_REVIEW_MAX_TOKENS

TEST_GAP_ENABLED: bool = settings.TEST_GAP_ENABLED
TEST_GAP_MODEL: str = settings.TEST_GAP_MODEL
TEST_GAP_MAX_TOKENS: int = settings.TEST_GAP_MAX_TOKENS
TEST_GAP_MAX_FILES: int = settings.TEST_GAP_MAX_FILES
TEST_GAP_MIN_COVERAGE: float = settings.TEST_GAP_MIN_COVERAGE
TEST_GAP_CONTEXT_CHAR_BUDGET: int = settings.TEST_GAP_CONTEXT_CHAR_BUDGET

LLM_BACKEND: str = settings.LLM_BACKEND
CLAUDE_CLI_BINARY: str = settings.CLAUDE_CLI_BINARY
CLAUDE_CLI_TIMEOUT_SECONDS: int = settings.CLAUDE_CLI_TIMEOUT_SECONDS

DRIFT_ALERT_WEEKS: int = settings.DRIFT_ALERT_WEEKS
DRIFT_ALERT_THRESHOLD: float = settings.DRIFT_ALERT_THRESHOLD

PREDICTION_MAX_WORKERS: int = settings.PREDICTION_MAX_WORKERS
SCHEDULER_MAX_RETRIES: int = settings.SCHEDULER_MAX_RETRIES
SCHEDULER_RETRY_BASE_WAIT_SECONDS: float = settings.SCHEDULER_RETRY_BASE_WAIT_SECONDS
VOLUME_FILTER_WINDOW_DAYS: int = settings.VOLUME_FILTER_WINDOW_DAYS
VOLUME_FILTER_MULTIPLIER: float = settings.VOLUME_FILTER_MULTIPLIER
AUTO_PROMOTE_MODEL: bool = settings.AUTO_PROMOTE_MODEL
SENTIMENT_LOOKBACK_DAYS: int = settings.SENTIMENT_LOOKBACK_DAYS
DEFAULT_SLIPPAGE_US: float = settings.DEFAULT_SLIPPAGE_US
DEFAULT_SLIPPAGE_JP: float = settings.DEFAULT_SLIPPAGE_JP
DB_LOG_RETENTION_DAYS: int = settings.DB_LOG_RETENTION_DAYS
DB_COMPACT_ENABLED: bool = settings.DB_COMPACT_ENABLED
DB_SIZE_ALERT_GB: float = settings.DB_SIZE_ALERT_GB

FACTORY_ENABLED: bool = settings.FACTORY_ENABLED
FACTORY_NIGHTLY_BUDGET: int = settings.FACTORY_NIGHTLY_BUDGET
FACTORY_LOOKBACK_YEARS: int = settings.FACTORY_LOOKBACK_YEARS
FACTORY_N_WINDOWS: int = settings.FACTORY_N_WINDOWS
FACTORY_GATE_MIN_TRADES: int = settings.FACTORY_GATE_MIN_TRADES
FACTORY_GATE_MIN_DSR: float = settings.FACTORY_GATE_MIN_DSR
FACTORY_GATE_MAX_PBO: float = settings.FACTORY_GATE_MAX_PBO
FACTORY_GATE_MAX_DRAWDOWN: float = settings.FACTORY_GATE_MAX_DRAWDOWN
FACTORY_GATE_CHAMPION_MARGIN: float = settings.FACTORY_GATE_CHAMPION_MARGIN
