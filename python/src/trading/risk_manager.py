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
from datetime import date, datetime, timezone
from typing import Any, Optional

import numpy as np
import pandas as pd
import yfinance as yf

from config.settings import (
    DEFAULT_STOP_LOSS_PCT,
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
    KELLY_REGIME_MULTIPLIERS,
    MAX_ACCEPTABLE_DRAWDOWN,
    VIX_POSITION_SCALE,
    VIX_SPIKE_THRESHOLD,
)
from src.domain.exceptions import RiskError  # noqa: F401  # 後方互換のため re-export
from src.domain.trading_rules import get_lot_size
from src.trading.brokers.base import BrokerBase, OrderSide
from src.trading.daily_loss_window import daily_loss_window_start
from src.trading.position_valuation import broker_positions_value
from src.trading.types import TradingGateStatus
from src.trading.vix_run_cache import VixRunCache
from src.utils.db._connection import _db_connection
from src.utils.logger import get_logger
from src.utils.optimal_params_loader import get_optimal_params

logger = get_logger(__name__)

DAILY_LOSS_RATE_ENV = "MAX_DAILY_LOSS_RATE"
DISABLE_DAILY_LOSS_GUARD_ENV = "DISABLE_DAILY_LOSS_GUARD"

VIX_TICKER = "^VIX"
VIX_STALE_BUSINESS_DAYS = 5  # 返す VIX の日付がこれより古い営業日数なら WARNING（#823）


def _fetch_raw_vix_close() -> Optional[tuple[float, date]]:
    """yfinance から ^VIX の最新の終値とその日付を返す（#823。DB は VIX の生の値を持たない）。"""
    hist = yf.Ticker(VIX_TICKER).history(period="10d", auto_adjust=False, timeout=10)
    close = hist["Close"].dropna() if hist is not None and "Close" in hist.columns else None
    if close is None or close.empty:
        return None
    return float(close.iloc[-1]), pd.Timestamp(close.index[-1]).date()


def _fetch_lag1_vix() -> Optional[tuple[float, date]]:
    """stock_features の最新の行の vix_close_lag1（T-1 の値。#697 で生の列は無い）と行の日付。

    row_num は symbol ごとに採番され直すため date で降順ソートして銘柄横断の最新を取る。
    """
    with _db_connection() as con:
        row = con.execute("""
            SELECT vix_close_lag1, date
            FROM stock_features
            WHERE vix_close_lag1 IS NOT NULL
            ORDER BY date DESC
            LIMIT 1
            """).fetchone()
    if row is None:
        return None
    return float(row[0]), pd.Timestamp(row[1]).date()


def fetch_latest_vix() -> Optional[float]:
    """発注の時点で分かっている最新の VIX の終値（T）を返す（R-406・#823）。

    取れなければ vix_close_lag1（T-1）に WARNING つきで戻す。値の日付をログに出し、
    VIX_STALE_BUSINESS_DAYS 営業日より古ければ WARNING。どちらも取れなければ None。
    """
    result: Optional[tuple[float, date]] = None
    source = VIX_TICKER
    try:
        result = _fetch_raw_vix_close()
    except Exception:
        logger.warning("[risk] VIX の最新の終値の取得に失敗しました", exc_info=True)
    if result is None:
        source = "stock_features.vix_close_lag1"
        logger.warning("[risk] VIX の最新の終値が取れないため 1営業日前の lag1 で代わりにします")
        try:
            result = _fetch_lag1_vix()
        except Exception:
            logger.warning("[risk] VIX の取得に失敗しました", exc_info=True)
            return None
    if result is None:
        return None
    vix, as_of = result
    logger.info(f"[risk] VIX={vix:.2f} as_of={as_of} source={source}")
    age = int(np.busday_count(as_of, datetime.now(timezone.utc).date()))
    if age > VIX_STALE_BUSINESS_DAYS:
        logger.warning(f"[risk] VIX の値が古すぎます as_of={as_of} ({age}営業日前) source={source}")
    return vix


def compute_vix_position_scale(vix: Optional[float]) -> float:
    """VIX 水準に応じたポジションスケール係数を返す（R-406）。

    - vix が None または閾値未満: scale = 1.0（縮小なし）
    - vix >= VIX_SPIKE_THRESHOLD: scale = VIX_POSITION_SCALE（縮小）
    """
    if vix is None or vix < VIX_SPIKE_THRESHOLD:
        return 1.0
    return VIX_POSITION_SCALE


def compute_dd_capital_scale(dd_ratio: float, max_dd: float) -> float:
    """DD 進行率に応じたポジションスケール係数を返す（非線形縮小・R-307）。

    - DD ≤ 0          : scale = 1.0（縮小なし）
    - DD = max_dd     : scale = 0.25（最大縮小）
    - 回復期          : DD が減れば scale も自動回復（関数の対称性）

    Args:
        dd_ratio: 現在のドローダウン率 (0.0〜1.0)
        max_dd: 許容最大ドローダウン率（設定値）

    Returns:
        ポジションスケール係数 [0.25, 1.0]
    """
    if dd_ratio <= 0.0 or max_dd <= 0.0:
        return 1.0
    t = min(1.0, dd_ratio / max_dd)
    scale = 1.0 - 0.75 * (t**2)
    return max(0.25, float(scale))


class RiskManager:
    def __init__(
        self,
        broker: BrokerBase,
        market: Optional[str] = None,
        symbol: Optional[str] = None,
    ):
        self._broker = broker
        self._market = market
        self.stop_loss_pct: Optional[float] = None
        self.take_profit_pct: Optional[float] = None
        # 総資産と VIX は発注ループ中に銘柄ごとに参照されるため、インスタンス単位でキャッシュする
        self._total_equity_cache: Optional[float] = None
        self._vix_cache = VixRunCache(lambda: fetch_latest_vix())  # 日付が変われば読み直す

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
                    f"take_profit_pct は None を使用します"
                )
            if self.stop_loss_pct is None and DEFAULT_STOP_LOSS_PCT > 0:
                # 最適化は損切り無しで回しており、null のままだと損切りが一度も発動しない（#880）
                self.stop_loss_pct = float(DEFAULT_STOP_LOSS_PCT)
                logger.warning(
                    f"[{market}_{symbol}] stop_loss_pct が null のため、"
                    f"既定の損切りの線 DEFAULT_STOP_LOSS_PCT={self.stop_loss_pct} を使う"
                )

    # ------------------------------------------------------------------
    # SL/TP チェック
    # ------------------------------------------------------------------

    def check_sl_tp(
        self, avg_price: float, current_price: float, is_short: bool = False
    ) -> tuple[bool, bool, str]:
        """ポジションの含み損益率が SL/TP 閾値を超えているか判定する。

        is_short=True は空売りで、avg_price を売り建ての単価とし値上がりを損とする（#1120）。

        Returns:
            (sl_triggered, tp_triggered, reason)
            SL と TP が同時に成立することはないため、sl_triggered が True のとき tp_triggered は False。
        """
        if avg_price <= 0:
            return False, False, ""
        pnl_pct = (current_price - avg_price) / avg_price * (-1.0 if is_short else 1.0)
        sl_triggered = self.stop_loss_pct is not None and pnl_pct <= -self.stop_loss_pct
        tp_triggered = (
            not sl_triggered
            and self.take_profit_pct is not None
            and pnl_pct >= self.take_profit_pct
        )
        if sl_triggered and self.stop_loss_pct is not None:
            reason = f"SL発動: pnl={pnl_pct:.2%} <= -{self.stop_loss_pct:.2%}"
        elif tp_triggered and self.take_profit_pct is not None:
            reason = f"TP発動: pnl={pnl_pct:.2%} >= {self.take_profit_pct:.2%}"
        else:
            reason = ""
        return sl_triggered, tp_triggered, reason

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
            reason = (
                f"日次損失上限に到達: 損失={daily_loss:.0f}円 / " f"上限={daily_loss_limit:.0f}円"
            )
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
        market_regime: Optional[str] = None,
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
            発注株数（市場別の最低売買単位に切り捨て。日本株は 100 株単位。0 になる場合は 0）
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

        # レジーム別補正
        if market_regime is not None:
            regime_mult = KELLY_REGIME_MULTIPLIERS.get(market_regime, 1.0)
            invest_amount = invest_amount * regime_mult
            logger.debug(
                f"[risk] {symbol}: レジーム補正 regime={market_regime} mult={regime_mult:.2f}"
            )

        # 1銘柄上限キャップ
        max_amount = balance * MAX_POSITION_RATE
        invest_amount = min(invest_amount, max_amount)

        if price <= 0:
            return 0

        qty = int(invest_amount / price)

        # モデル信頼度によるポジション縮小
        confidence_ratio = max(0.0, min(confidence_ratio, 1.0))
        qty = int(qty * confidence_ratio)

        # R-307: DD適応型スケール（DD進行中は非線形縮小）
        dd_ratio = self.get_current_dd_ratio()
        dd_scale = compute_dd_capital_scale(dd_ratio, MAX_ACCEPTABLE_DRAWDOWN)
        if dd_scale < 1.0:
            logger.info(
                f"[risk] {symbol}: DD適応縮小 dd={dd_ratio:.1%} "
                f"max_dd={MAX_ACCEPTABLE_DRAWDOWN:.1%} scale={dd_scale:.2f}"
            )
        qty = int(qty * dd_scale)

        vix = self._vix_cache.get()  # R-406: VIX 急騰時のポジション縮小（1回の発注で1度だけ読む）
        vix_scale = compute_vix_position_scale(vix)
        if vix_scale < 1.0:
            logger.info(
                f"[risk] {symbol}: VIX急騰によるポジション縮小 "
                f"vix={vix:.1f} threshold={VIX_SPIKE_THRESHOLD:.1f} scale={vix_scale:.2f}"
            )
        qty = int(qty * vix_scale)

        # 市場別の最低売買単位に丸める（日本株=100株単位、米国株=1株。市場不明時は100株）
        lot = get_lot_size(self._market)
        qty = (qty // lot) * lot

        logger.debug(
            f"[risk] {symbol}: balance={balance:.0f} kelly={kelly:.3f} "
            f"invest={invest_amount:.0f} confidence={confidence_ratio:.3f} "
            f"dd_scale={dd_scale:.2f} qty={qty}"
        )
        return qty

    # ------------------------------------------------------------------
    # R-307: DD適応型資本配分
    # ------------------------------------------------------------------

    def get_total_equity(self) -> float:
        """総資産（現金残高 + 保有ポジション時価評価額）を返す（#696）。

        現金残高のみを DD 判定に使うと、資金を株に振り向けるほど「損失」と誤判定され
        DD適応縮小が不当に働き続けるため、総資産を基準にする。
        発注ループ中は銘柄ごとに参照されるためインスタンス単位でキャッシュする
        （買付は現金→株の振替であり総資産は変動しないため、ラン中の再計算は不要）。
        """
        if self._total_equity_cache is not None:
            return self._total_equity_cache

        balance = self._broker.get_balance()
        # 買い持ちの時価に、空売りの含み損益を足す（#1148）
        self._total_equity_cache = balance + broker_positions_value(self._broker, logger)
        return self._total_equity_cache

    def update_peak_balance(self) -> None:
        """総資産が過去最高値を超えた場合に peak_balance を更新する。
        run_daily_orders の先頭で呼び出すこと。
        """
        current = self.get_total_equity()
        with _db_connection() as con:
            row = con.execute("SELECT peak_balance FROM dd_state WHERE id = 1").fetchone()
            if row is None:
                con.execute("INSERT INTO dd_state (id, peak_balance) VALUES (1, %s)", [current])
            elif current > float(row[0]):
                con.execute("UPDATE dd_state SET peak_balance = %s WHERE id = 1", [current])

    def get_current_dd_ratio(self) -> float:
        """現在のドローダウン率（peak_balance と総資産の差分比）を返す。"""
        current = self.get_total_equity()
        with _db_connection() as con:
            row = con.execute("SELECT peak_balance FROM dd_state WHERE id = 1").fetchone()
        if row is None or float(row[0]) <= 0:
            return 0.0
        peak = float(row[0])
        return max(0.0, (peak - current) / peak)

    # ------------------------------------------------------------------
    # 内部ヘルパー
    # ------------------------------------------------------------------

    @staticmethod
    def _table_exists(con: Any, table_name: str) -> bool:
        row = con.execute(
            """
            SELECT COUNT(*)
            FROM information_schema.tables
            WHERE table_name = %s
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
            logger.info(
                f"[risk] {DISABLE_DAILY_LOSS_GUARD_ENV}=1 のため日次損失ガードを無効化します"
            )
            return None

        raw_value = os.getenv(DAILY_LOSS_RATE_ENV)
        if raw_value is None or raw_value.strip() == "":
            return MAX_DAILY_LOSS_RATE

        try:
            rate = float(raw_value)
        except ValueError:
            default_rate = f"{MAX_DAILY_LOSS_RATE:.2%}"
            logger.warning(
                f"[risk] {DAILY_LOSS_RATE_ENV}={raw_value!r} を解釈できないため"
                f"既定値 {default_rate} を使用します"
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
        """前の営業日（JST）の 0:00 から今までの確定損失合計（プラスが損失）を返す（#941）"""
        with _db_connection() as con:
            if self._broker.broker_name == "paper":
                row = con.execute(
                    """
                    SELECT COALESCE(SUM(realized_pnl), 0.0)
                    FROM paper_orders
                    WHERE status = 'filled'
                      AND side IN (%s, %s)
                      AND filled_at IS NOT NULL
                      AND filled_at >= %s
                      AND realized_pnl < 0
                    """,
                    [int(OrderSide.SELL), int(OrderSide.SHORT_COVER), daily_loss_window_start()],
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
                WHERE closed_at >= %s
                  AND broker = %s
                  AND realized_pnl < 0
                """,
                [daily_loss_window_start(), self._broker.broker_name],
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
                      AND side IN (%s, %s)
                      AND realized_pnl IS NOT NULL
                    ORDER BY filled_at DESC
                    LIMIT %s
                    """,
                    [int(OrderSide.SELL), int(OrderSide.SHORT_COVER), MAX_CONSECUTIVE_LOSSES],
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
                    WHERE broker = %s
                    ORDER BY closed_at DESC
                    LIMIT %s
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
