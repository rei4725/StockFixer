"""
アプリケーション設定値の一元管理

pydantic-settings の BaseSettings で env 変数を型安全に読み込み、
パース失敗・型不正は起動時に即時例外で fail させる。

既存コードとの互換のため、`from config.settings import MAX_DAILY_LOSS_RATE`
のような module-level 定数アクセスを維持する。
"""

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
