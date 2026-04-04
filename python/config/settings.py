"""
アプリケーション設定値の一元管理

環境変数でオーバーライド可能。すべての数値設定はここから取得する。
"""

import os


def _float(env: str, default: float) -> float:
    val = os.getenv(env, "")
    try:
        return float(val) if val.strip() else default
    except ValueError:
        return default


def _int(env: str, default: int) -> int:
    val = os.getenv(env, "")
    try:
        return int(val) if val.strip() else default
    except ValueError:
        return default


# ---------------------------------------------------------------------------
# リスク管理（risk_manager.py）
# ---------------------------------------------------------------------------

#: 1日の最大損失率（口座残高に対する割合）。0 以下でガード無効
MAX_DAILY_LOSS_RATE: float = _float("MAX_DAILY_LOSS_RATE", 0.02)

#: 1銘柄の最大ポジション率（口座残高に対する割合）
MAX_POSITION_RATE: float = _float("MAX_POSITION_RATE", 0.10)

#: 当日の連続損失回数の上限（超過でその日の取引停止）
MAX_CONSECUTIVE_LOSSES: int = _int("MAX_CONSECUTIVE_LOSSES", 3)

#: 最大保有銘柄数
MAX_POSITIONS: int = _int("MAX_POSITIONS", 10)

#: Kelly 基準に掛ける安全係数（ハーフケリー = 0.5）
HALF_KELLY: float = _float("HALF_KELLY", 0.5)

# ---------------------------------------------------------------------------
# 自動発注（order_execution_pipeline.py）
# ---------------------------------------------------------------------------

#: 予測変化率がこれ以上なら買いシグナル
BUY_THRESHOLD: float = _float("BUY_THRESHOLD", 0.005)

#: 予測変化率がこれ以下なら売りシグナル
SELL_THRESHOLD: float = _float("SELL_THRESHOLD", -0.005)

#: 1回の実行で発注する最大銘柄数
MAX_ORDERS_PER_RUN: int = _int("MAX_ORDERS_PER_RUN", 5)

#: 同一セクターで同時に許容する最大銘柄数。0 以下で無効
MAX_SECTOR_POSITIONS: int = _int("MAX_SECTOR_POSITIONS", 3)

# ---------------------------------------------------------------------------
# ペーパートレード（paper_broker.py）
# ---------------------------------------------------------------------------

#: ペーパートレードの初期仮想残高（円）
PAPER_INITIAL_BALANCE: float = _float("PAPER_INITIAL_BALANCE", 1_000_000.0)
