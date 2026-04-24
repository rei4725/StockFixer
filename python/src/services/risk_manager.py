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

import os
from typing import Optional

from config.settings import (
    HALF_KELLY,
    MAX_CONSECUTIVE_LOSSES,
    MAX_DAILY_LOSS_RATE,
    MAX_POSITION_RATE,
    MAX_POSITIONS,
)
from config.trading_policy import (  # noqa: F401  # R-307/R-219 でポジションキャップ・DD制御に使用予定
    DEFAULT_AVG_LOSS,
    DEFAULT_AVG_WIN,
    DEFAULT_WIN_RATE,
    HIGH_CONFIDENCE_POSITION_CAP,
    MAX_ACCEPTABLE_DRAWDOWN,
)
from src.brokers.base import BrokerBase, OrderSide
from src.domain.types import TradingGateStatus
from src.utils.db._connection import _db_connection
from src.utils.logger import get_logger
from src.utils.optimal_params_loader import get_optimal_params

logger = get_logger(__name__)

DAILY_LOSS_RATE_ENV = "MAX_DAILY_LOSS_RATE"
DISABLE_DAILY_LOSS_GUARD_ENV = "DISABLE_DAILY_LOSS_GUARD"


class RiskError(Exception):
    """リスクチェックによる取引拒否"""


class RiskManager:
    def __init__(
        self,
        broker: BrokerBase,
        market: Optional[str] = None,
        symbol: Optional[str] = None,
    ):
        self._broker = broker
        # TODO: stop_loss_pct / take_profit_pct は現在 RiskManager 内で未使用（Dead Attribute）。
        #       将来の SL/TP チェックロジック（R-307 等）で参照予定。
        self.stop_loss_pct: Optional[float] = None
        self.take_profit_pct: Optional[float] = None

        # Kelly 入力パラメータ（BT実績値をロード、なければデフォルト値）
        self.kelly_win_rate: float = DEFAULT_WIN_RATE
        self.kelly_avg_win: float = DEFAULT_AVG_WIN
        self.kelly_avg_loss: float = DEFAULT_AVG_LOSS

        if market is not None and symbol is not None:
            params = get_optimal_params(market, symbol)
            if params is not None:
                v = params.get("stop_loss_pct")
                self.stop_loss_pct = float(v) if v is not None else None
                v = params.get("take_profit_pct")
                self.take_profit_pct = float(v) if v is not None else None
                logger.debug(
                    f"[{market}_{symbol}] optimal_params.json から SL/TP を自動ロード: "
                    f"stop_loss_pct={self.stop_loss_pct}, take_profit_pct={self.take_profit_pct}"
                )
                raw_metrics = params.get("metrics")
                metrics = raw_metrics if isinstance(raw_metrics, dict) else {}
                self.kelly_win_rate = float(metrics.get("win_rate", DEFAULT_WIN_RATE))
                self.kelly_avg_win = float(metrics.get("avg_win", DEFAULT_AVG_WIN))
                self.kelly_avg_loss = float(metrics.get("avg_loss", DEFAULT_AVG_LOSS))
                logger.debug(
                    f"[{market}_{symbol}] BT実績Kelly値をロード: "
                    f"win_rate={self.kelly_win_rate}, avg_win={self.kelly_avg_win}, "
                    f"avg_loss={self.kelly_avg_loss}"
                )
            else:
                logger.warning(
                    f"[{market}_{symbol}] optimal_params.json が見つからないか未登録のため "
                    f"stop_loss_pct/take_profit_pct はデフォルト値 (None) を使用します"
                )

    # ------------------------------------------------------------------
    # メインゲートチェック
    # ------------------------------------------------------------------

    def is_trading_allowed(self) -> bool:
        """
        全リスクルールを一括チェックする。
        一つでも違反があれば False を返す。
        """
        return self.evaluate_trading_gate().is_allowed

    def evaluate_trading_gate(self) -> TradingGateStatus:
        """発注可否と停止理由の詳細を返す。"""
        balance = self._broker.get_balance()
        positions = self._broker.get_positions()
        daily_loss_limit = self._resolve_daily_loss_limit(balance)
        daily_loss = 0.0

        if daily_loss_limit is not None:
            daily_loss = self._get_daily_realized_loss()

        # ルール 1: 当日損失上限
        if daily_loss_limit is not None and daily_loss >= daily_loss_limit:
            reason = f"日次損失上限に到達: 損失={daily_loss:.0f}円 / " f"上限={daily_loss_limit:.0f}円"
            logger.warning(f"[risk] {reason} → 新規発注停止")
            return TradingGateStatus(
                is_allowed=False,
                stop_active=True,
                reason_code="daily_loss_limit",
                reason=reason,
                daily_loss=daily_loss,
                daily_loss_limit=daily_loss_limit,
                consecutive_losses=0,
                consecutive_loss_limit=MAX_CONSECUTIVE_LOSSES,
                position_count=len(positions),
                max_positions=MAX_POSITIONS,
            )

        # ルール 3: 連続損失
        consecutive = self._get_consecutive_losses()
        if consecutive >= MAX_CONSECUTIVE_LOSSES:
            reason = f"連続損失 {consecutive} 回 >= {MAX_CONSECUTIVE_LOSSES} 回"
            logger.warning(f"[risk] {reason} → 当日取引停止")
            return TradingGateStatus(
                is_allowed=False,
                stop_active=True,
                reason_code="consecutive_losses",
                reason=reason,
                daily_loss=daily_loss,
                daily_loss_limit=daily_loss_limit,
                consecutive_losses=consecutive,
                consecutive_loss_limit=MAX_CONSECUTIVE_LOSSES,
                position_count=len(positions),
                max_positions=MAX_POSITIONS,
            )

        # ルール 4: 最大保有銘柄数
        if len(positions) >= MAX_POSITIONS:
            reason = f"保有銘柄数上限: {len(positions)} >= {MAX_POSITIONS}"
            logger.warning(f"[risk] {reason} → 新規買い停止")
            return TradingGateStatus(
                is_allowed=False,
                stop_active=False,
                reason_code="max_positions",
                reason=reason,
                daily_loss=daily_loss,
                daily_loss_limit=daily_loss_limit,
                consecutive_losses=consecutive,
                consecutive_loss_limit=MAX_CONSECUTIVE_LOSSES,
                position_count=len(positions),
                max_positions=MAX_POSITIONS,
            )

        return TradingGateStatus(
            is_allowed=True,
            stop_active=False,
            daily_loss=daily_loss,
            daily_loss_limit=daily_loss_limit,
            consecutive_losses=consecutive,
            consecutive_loss_limit=MAX_CONSECUTIVE_LOSSES,
            position_count=len(positions),
            max_positions=MAX_POSITIONS,
        )

    # ------------------------------------------------------------------
    # ポジションサイジング
    # ------------------------------------------------------------------

    def calc_position_size(
        self,
        symbol: str,
        price: float,
        win_rate: Optional[float] = None,
        avg_win: Optional[float] = None,
        avg_loss: Optional[float] = None,
        confidence_ratio: float = 1.0,
    ) -> int:
        """
        ハーフ Kelly 基準で発注株数を計算する。

        Args:
            symbol: 銘柄コード
            price: 発注価格（成行時は現在値を渡す）
            win_rate: 勝率（バックテスト実績 or デフォルト 55%）。None の場合は BT実績値を参照。
            avg_win: 平均利益率（デフォルト 1.5%）。None の場合は BT実績値を参照。
            avg_loss: 平均損失率（デフォルト 0.8%）。None の場合は BT実績値を参照。
            confidence_ratio: モデル信頼度係数（1/(1+model_std)）。1.0=最大信頼度。
                              モデル間分散が大きいほど小さくなり、ポジションを縮小する。

        Returns:
            発注株数（1株未満は切り捨て、0 になる場合は 0）
        """
        if win_rate is None:
            win_rate = self.kelly_win_rate
        if avg_win is None:
            avg_win = self.kelly_avg_win
        if avg_loss is None:
            avg_loss = self.kelly_avg_loss

        balance = self._broker.get_balance()

        if avg_win == 0.0 or avg_loss == 0.0:
            logger.warning(
                f"[{symbol}] Kelly計算: avg_win={avg_win}, avg_loss={avg_loss} "
                "のいずれかが0.0です。ポジションサイズが0になる可能性があります"
            )

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

        # モデル信頼度によるポジション縮小
        confidence_ratio = max(0.0, min(confidence_ratio, 1.0))
        qty = int(qty * confidence_ratio)

        # 最低単元（日本株は原則 100 株単位）
        lot = 100
        qty = (qty // lot) * lot

        logger.debug(
            f"[risk] {symbol}: balance={balance:.0f} kelly={kelly:.3f} "
            f"invest={invest_amount:.0f} confidence={confidence_ratio:.3f} qty={qty}"
        )
        return qty

    # ------------------------------------------------------------------
    # 内部ヘルパー
    # ------------------------------------------------------------------

    @staticmethod
    def _table_exists(con, table_name: str) -> bool:
        row = con.execute(
            """
            SELECT COUNT(*)
            FROM information_schema.tables
            WHERE table_name = ?
            """,
            [table_name],
        ).fetchone()
        return bool(row and row[0])

    @staticmethod
    def _resolve_daily_loss_rate() -> float | None:
        if os.getenv(DISABLE_DAILY_LOSS_GUARD_ENV, "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            logger.info(f"[risk] {DISABLE_DAILY_LOSS_GUARD_ENV}=1 のため日次損失ガードを無効化します")
            return None

        raw_value = os.getenv(DAILY_LOSS_RATE_ENV)
        if raw_value is None or raw_value.strip() == "":
            return MAX_DAILY_LOSS_RATE

        try:
            rate = float(raw_value)
        except ValueError:
            default_rate = f"{MAX_DAILY_LOSS_RATE:.2%}"
            logger.warning(
                f"[risk] {DAILY_LOSS_RATE_ENV}={raw_value!r} を解釈できないため" f"既定値 {default_rate} を使用します"
            )
            return MAX_DAILY_LOSS_RATE

        if rate <= 0:
            logger.info(f"[risk] {DAILY_LOSS_RATE_ENV}<=0 のため日次損失ガードを無効化します")
            return None
        return rate

    def _resolve_daily_loss_limit(self, balance: float) -> float | None:
        rate = self._resolve_daily_loss_rate()
        if rate is None:
            return None
        return balance * rate

    def _get_daily_realized_loss(self) -> float:
        """当日の確定損失合計（プラスが損失）を返す"""
        with _db_connection() as con:
            if self._broker.broker_name == "paper":
                row = con.execute(
                    """
                    SELECT COALESCE(SUM(realized_pnl), 0.0)
                    FROM paper_orders
                    WHERE status = 'filled'
                      AND side = ?
                      AND filled_at IS NOT NULL
                      AND DATE(filled_at) = CURRENT_DATE
                      AND realized_pnl < 0
                    """,
                    [int(OrderSide.SELL)],
                ).fetchone()
                return abs(float(row[0])) if row else 0.0

            if not self._table_exists(con, "trade_pnl"):
                logger.warning(
                    f"[risk] trade_pnl テーブル未作成のため日次損失を 0 として扱います: broker={self._broker.broker_name}"
                )
                return 0.0

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
        with _db_connection() as con:
            if self._broker.broker_name == "paper":
                rows = con.execute(
                    """
                    SELECT realized_pnl
                    FROM paper_orders
                    WHERE status = 'filled'
                      AND side = ?
                      AND realized_pnl IS NOT NULL
                    ORDER BY filled_at DESC
                    LIMIT ?
                    """,
                    [int(OrderSide.SELL), MAX_CONSECUTIVE_LOSSES],
                ).fetchall()
            else:
                if not self._table_exists(con, "trade_pnl"):
                    message = (
                        "[risk] trade_pnl テーブル未作成のため連続損失を 0 として扱います: "
                        f"broker={self._broker.broker_name}"
                    )
                    logger.warning(message)
                    return 0

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
