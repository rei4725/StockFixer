"""
RiskManager — 自動売買リスク管理

OrderExecutionPipeline が注文を送信する前にゲートチェックを行い、
過大なリスクを伴う取引を自動的にブロックする。

ルール:
    1. 1日の最大損失額 = 口座残高の 2% まで
    2. 1銘柄の最大ポジション = 口座残高の 10% まで
    3. 当日の連続損失 3 回でその日の取引停止
    4. 最大保有銘柄数 = 10 銘柄
"""

from src.brokers.base import BrokerBase
from src.utils.logger import get_logger

logger = get_logger(__name__)

# --- リスクパラメータ（必要なら環境変数化） ---
MAX_DAILY_LOSS_RATE = 0.02  # 残高の 2%
MAX_POSITION_RATE = 0.10  # 残高の 10%
MAX_CONSECUTIVE_LOSSES = 3  # 連続損失でその日停止
MAX_POSITIONS = 10  # 最大保有銘柄数
HALF_KELLY = 0.5  # Kelly 係数に掛ける安全係数


class RiskError(Exception):
    """リスクチェックによる取引拒否"""


def _get_con():
    from src.utils.db import get_connection

    return get_connection()


class RiskManager:
    def __init__(self, broker: BrokerBase):
        self._broker = broker

    # ------------------------------------------------------------------
    # メインゲートチェック
    # ------------------------------------------------------------------

    def is_trading_allowed(self) -> bool:
        """
        全リスクルールを一括チェックする。
        一つでも違反があれば False を返す。
        """
        balance = self._broker.get_balance()

        # ルール 1: 当日損失上限
        daily_loss = self._get_daily_realized_loss()
        max_loss = balance * MAX_DAILY_LOSS_RATE
        if daily_loss >= max_loss:
            logger.warning(f"[risk] 当日損失上限超過: 損失={daily_loss:.0f}円 / 上限={max_loss:.0f}円 → 取引停止")
            return False

        # ルール 3: 連続損失
        consecutive = self._get_consecutive_losses()
        if consecutive >= MAX_CONSECUTIVE_LOSSES:
            logger.warning(f"[risk] 連続損失 {consecutive} 回 >= {MAX_CONSECUTIVE_LOSSES} 回 → 当日取引停止")
            return False

        # ルール 4: 最大保有銘柄数
        positions = self._broker.get_positions()
        if len(positions) >= MAX_POSITIONS:
            logger.warning(f"[risk] 保有銘柄数上限: {len(positions)} >= {MAX_POSITIONS} → 新規買い停止")
            return False

        return True

    # ------------------------------------------------------------------
    # ポジションサイジング
    # ------------------------------------------------------------------

    def calc_position_size(
        self,
        symbol: str,
        price: float,
        win_rate: float = 0.55,
        avg_win: float = 0.015,
        avg_loss: float = 0.008,
    ) -> int:
        """
        ハーフ Kelly 基準で発注株数を計算する。

        Args:
            symbol: 銘柄コード
            price: 発注価格（成行時は現在値を渡す）
            win_rate: 勝率（バックテスト実績 or デフォルト 55%）
            avg_win: 平均利益率（デフォルト 1.5%）
            avg_loss: 平均損失率（デフォルト 0.8%）

        Returns:
            発注株数（1株未満は切り捨て、0 になる場合は 0）
        """
        balance = self._broker.get_balance()

        # Kelly 比率
        if avg_win <= 0 or avg_loss <= 0:
            kelly = 0.0
        else:
            kelly = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win

        kelly = max(0.0, min(kelly, 1.0))  # クリッピング
        invest_amount = balance * kelly * HALF_KELLY

        # 1銘柄上限キャップ
        max_amount = balance * MAX_POSITION_RATE
        invest_amount = min(invest_amount, max_amount)

        if price <= 0:
            return 0

        qty = int(invest_amount / price)

        # 最低単元（日本株は原則 100 株単位）
        lot = 100
        qty = (qty // lot) * lot

        logger.debug(
            f"[risk] {symbol}: balance={balance:.0f} kelly={kelly:.3f} "
            f"invest={invest_amount:.0f} qty={qty}"
        )
        return qty

    # ------------------------------------------------------------------
    # 内部ヘルパー
    # ------------------------------------------------------------------

    def _get_daily_realized_loss(self) -> float:
        """当日の確定損失合計（プラスが損失）を返す"""
        con = _get_con()
        row = con.execute(
            """
            SELECT COALESCE(SUM(realized_pnl), 0.0)
            FROM trade_pnl
            WHERE DATE(closed_at) = CURRENT_DATE
              AND broker = ?
              AND realized_pnl < 0
            """,
            [self._broker.broker_name],
        ).fetchone()
        return abs(float(row[0])) if row else 0.0

    def _get_consecutive_losses(self) -> int:
        """直近の連続損失回数を返す"""
        con = _get_con()
        rows = con.execute(
            """
            SELECT realized_pnl
            FROM trade_pnl
            WHERE broker = ?
            ORDER BY closed_at DESC
            LIMIT ?
            """,
            [self._broker.broker_name, MAX_CONSECUTIVE_LOSSES],
        ).fetchall()
        count = 0
        for (pnl,) in rows:
            if pnl < 0:
                count += 1
            else:
                break
        return count
