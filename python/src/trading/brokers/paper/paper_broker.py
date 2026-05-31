"""
PaperBroker — 仮想売買（ペーパートレード）Broker

kabu STATION® API が利用できない環境（APIキー未取得・テスト等）での
動作確認用 Broker 実装。

- 注文は yfinance の翌営業日始値で約定扱いにする
- ポジション・残高・注文履歴は DuckDB の paper_* テーブルで永続化
- BrokerBase と同一インターフェースのため、mode 変更だけで本番切り替え可能
"""

import uuid
from typing import Any, Callable

from config.settings import PAPER_INITIAL_BALANCE
from src.domain.ports import MarketDataPort
from src.trading.brokers.base import BrokerBase, OrderSide, OrderType
from src.utils.db._connection import _db_connection
from src.utils.logger import get_logger

logger = get_logger(__name__)

_INITIAL_BALANCE = PAPER_INITIAL_BALANCE


class PaperBroker(BrokerBase):
    """
    ペーパートレード用 Broker。APIキー不要で完全動作する。

    約定ルール:
        成行注文 → 翌営業日の始値で約定（yfinance から取得）
        指値注文 → 翌営業日の安値 <= 指値 <= 高値 なら約定、それ以外は失効
    """

    broker_name = "paper"

    def __init__(
        self,
        market_data_port: MarketDataPort,
        record_diff: Callable[..., None] | None = None,
    ) -> None:
        self._market_data = market_data_port
        self._record_diff = record_diff

    def get_token(self) -> str:
        """ペーパートレードはトークン不要。ダミー文字列を返す"""
        return "paper_mode"

    # ------------------------------------------------------------------
    # 注文
    # ------------------------------------------------------------------

    def send_order(
        self,
        symbol: str,
        side: OrderSide,
        qty: int,
        price: float = 0.0,
        order_type: OrderType = OrderType.MARKET,
    ) -> dict[str, Any]:
        """
        注文を DuckDB に記録し、翌営業日始値で仮約定させる。
        実際の約定は settle_pending_orders() が翌日の実行時に処理する。
        """
        order_id = str(uuid.uuid4())[:12]
        sym = symbol.replace(".T", "")
        with _db_connection() as con:
            con.execute(
                """
                INSERT INTO paper_orders
                    (order_id, symbol, side, qty, price, order_type, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, 'pending', CURRENT_TIMESTAMP)
                """,
                [order_id, sym, int(side), qty, price, int(order_type)],
            )
            if side == OrderSide.SHORT:
                # paper_short_positions に即時仮登録（加重平均単価更新）
                existing_short = con.execute(
                    "SELECT qty, avg_short_price FROM paper_short_positions WHERE symbol=?",
                    [sym],
                ).fetchone()
                if existing_short:
                    old_qty, old_avg = existing_short
                    new_qty = old_qty + qty
                    new_avg = (old_avg * old_qty + price * qty) / new_qty
                    con.execute(
                        "UPDATE paper_short_positions SET qty=?, avg_short_price=?, "
                        "updated_at=CURRENT_TIMESTAMP WHERE symbol=?",
                        [new_qty, new_avg, sym],
                    )
                else:
                    con.execute(
                        "INSERT INTO paper_short_positions "
                        "(symbol, qty, avg_short_price, opened_at, updated_at) "
                        "VALUES (?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)",
                        [sym, qty, price],
                    )
            elif side == OrderSide.SHORT_COVER:
                # SHORT_COVER は settle_pending_orders() で実現損益を確定する（pending 記録のみ）
                pass
        logger.info(
            f"[paper] 注文受付: order_id={order_id} symbol={symbol} "
            f"side={side.name} qty={qty} order_type={order_type.name}"
        )
        return {
            "order_id": order_id,
            "status": "pending",
            "message": "paper order queued",
        }

    def cancel_order(self, order_id: str) -> dict[str, Any]:
        """pending 状態の注文をキャンセルする"""
        with _db_connection() as con:
            con.execute(
                "UPDATE paper_orders SET status='cancelled' WHERE order_id=? AND status='pending'",
                [order_id],
            )
        logger.info(f"[paper] 注文キャンセル: order_id={order_id}")
        return {
            "order_id": order_id,
            "status": "cancelled",
            "message": "paper order cancelled",
        }

    def settle_pending_orders(self) -> list[dict[str, Any]]:
        """
        pending 状態の注文を yfinance の当日始値で約定処理する。
        スケジューラーから市場開始直後（9:05 頃）に呼び出す。

        Returns:
            約定した注文のリスト
        """
        # Phase 1: ペンディング注文を読み込む（接続をすぐに解放）
        with _db_connection() as con:
            rows = con.execute(
                "SELECT order_id, market, predicted_at, symbol, side, qty, price, "
                "signal_price, order_type "
                "FROM paper_orders WHERE status='pending'"
            ).fetchall()

        settled = []
        for (
            order_id,
            market,
            predicted_at,
            symbol,
            side,
            qty,
            limit_price,
            signal_price,
            order_type_val,
        ) in rows:
            ticker = f"{symbol}.T"
            try:
                hist = self._market_data.get_ohlcv(ticker, "2d")
                if hist.empty:
                    logger.warning(f"[paper] {symbol}: 株価取得失敗、スキップ")
                    continue

                today_row = hist.iloc[-1]
                open_price = float(today_row["Open"])
                high_price = float(today_row["High"])
                low_price = float(today_row["Low"])

                if order_type_val == int(OrderType.MARKET):
                    fill_price = open_price
                else:  # 指値
                    if side == int(OrderSide.BUY) and low_price <= limit_price:
                        fill_price = min(limit_price, open_price)
                    elif side == int(OrderSide.SELL) and high_price >= limit_price:
                        fill_price = max(limit_price, open_price)
                    elif side == int(OrderSide.SHORT) and high_price >= limit_price:
                        fill_price = max(limit_price, open_price)
                    elif side == int(OrderSide.SHORT_COVER) and low_price <= limit_price:
                        fill_price = min(limit_price, open_price)
                    else:
                        logger.info(f"[paper] {symbol}: 指値未達、失効")
                        with _db_connection() as con:
                            con.execute(
                                "UPDATE paper_orders SET status='expired' WHERE order_id=?",
                                [order_id],
                            )
                        continue

                # Phase 2: 約定処理（接続を再取得）
                with _db_connection() as con:
                    self._apply_fill(con, order_id, symbol, side, qty, fill_price)
                    con.execute(
                        "UPDATE paper_orders SET status='filled', "
                        "fill_price=?, filled_at=CURRENT_TIMESTAMP WHERE order_id=?",
                        [fill_price, order_id],
                    )
                if (
                    market
                    and predicted_at
                    and signal_price is not None
                    and self._record_diff is not None
                ):
                    self._record_diff(
                        market=str(market),
                        symbol=str(symbol),
                        predicted_at=str(predicted_at),
                        side=int(side),
                        signal_price=float(signal_price),
                        mode="paper",
                        order_id=str(order_id),
                        actual_price=fill_price,
                    )
                logger.info(
                    f"[paper] 約定: {symbol} {OrderSide(side).name} {qty}株 @ {fill_price:.1f}円"
                )
                settled.append(
                    {
                        "order_id": order_id,
                        "symbol": symbol,
                        "fill_price": fill_price,
                        "qty": qty,
                    }
                )

            except Exception as e:
                logger.error(f"[paper] 約定処理エラー ({symbol}): {e}", exc_info=True)

        return settled

    def _apply_fill(
        self, con: Any, order_id: str, symbol: str, side: int, qty: int, fill_price: float
    ) -> None:
        """約定に合わせてポジションと残高を更新する"""
        existing = con.execute(
            "SELECT qty, avg_price FROM paper_positions WHERE symbol=?", [symbol]
        ).fetchone()

        if side == int(OrderSide.BUY):
            cost = qty * fill_price
            # 残高を減少
            con.execute("UPDATE paper_balance SET balance = balance - ?", [cost])
            if existing:
                old_qty, old_avg = existing
                new_qty = old_qty + qty
                new_avg = (old_avg * old_qty + fill_price * qty) / new_qty
                con.execute(
                    "UPDATE paper_positions SET qty=?, avg_price=?, "
                    "updated_at=CURRENT_TIMESTAMP WHERE symbol=?",
                    [new_qty, new_avg, symbol],
                )
            else:
                con.execute(
                    "INSERT INTO paper_positions "
                    "(symbol, qty, avg_price, updated_at) VALUES (?, ?, ?, CURRENT_TIMESTAMP)",
                    [symbol, qty, fill_price],
                )
        elif side == int(OrderSide.SELL):
            proceeds = qty * fill_price
            con.execute("UPDATE paper_balance SET balance = balance + ?", [proceeds])
            if existing:
                old_qty, old_avg = existing
                new_qty = old_qty - qty
                if new_qty <= 0:
                    con.execute("DELETE FROM paper_positions WHERE symbol=?", [symbol])
                else:
                    con.execute(
                        "UPDATE paper_positions SET qty=?, "
                        "updated_at=CURRENT_TIMESTAMP WHERE symbol=?",
                        [new_qty, symbol],
                    )
            else:
                old_avg = fill_price  # ポジション不明時はPnL=0扱い
            realized_pnl = (fill_price - old_avg) * qty
            con.execute(
                "UPDATE paper_orders SET realized_pnl=? WHERE order_id=?",
                [realized_pnl, order_id],
            )
        elif side == int(OrderSide.SHORT):
            # 空売り新規: send_order で仮登録済みの avg_short_price を実際の約定値段で上書きする
            existing_short = con.execute(
                "SELECT qty, avg_short_price FROM paper_short_positions WHERE symbol=?",
                [symbol],
            ).fetchone()
            if existing_short:
                old_qty, old_avg = existing_short
                # 今回約定分のみで avg を再計算（send_order の仮登録を fill_price で補正）
                new_avg = ((old_avg * old_qty) - (old_avg - fill_price) * qty) / old_qty
                con.execute(
                    "UPDATE paper_short_positions SET avg_short_price=?, "
                    "updated_at=CURRENT_TIMESTAMP WHERE symbol=?",
                    [new_avg, symbol],
                )
            else:
                con.execute(
                    "INSERT INTO paper_short_positions "
                    "(symbol, qty, avg_short_price, opened_at, updated_at) "
                    "VALUES (?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)",
                    [symbol, qty, fill_price],
                )
        elif side == int(OrderSide.SHORT_COVER):
            # 空売り返済: paper_short_positions から qty を減算し実現損益を記録
            existing_short = con.execute(
                "SELECT qty, avg_short_price FROM paper_short_positions WHERE symbol=?",
                [symbol],
            ).fetchone()
            if existing_short:
                old_qty, old_avg = existing_short
                new_qty = old_qty - qty
                if new_qty <= 0:
                    con.execute("DELETE FROM paper_short_positions WHERE symbol=?", [symbol])
                else:
                    con.execute(
                        "UPDATE paper_short_positions SET qty=?, "
                        "updated_at=CURRENT_TIMESTAMP WHERE symbol=?",
                        [new_qty, symbol],
                    )
            else:
                old_avg = fill_price  # ポジション不明時はPnL=0扱い
            realized_pnl = (old_avg - fill_price) * qty
            con.execute(
                "UPDATE paper_orders SET realized_pnl=? WHERE order_id=?",
                [realized_pnl, order_id],
            )

    # ------------------------------------------------------------------
    # 照会
    # ------------------------------------------------------------------

    def get_short_positions(self) -> list[dict[str, Any]]:
        """保有中の空売りポジション一覧を返す。

        Returns:
            list[dict]: [{"symbol": str, "qty": int, "avg_short_price": float,
                          "current_price": float, "unrealized_pnl": float | None}]
        """
        with _db_connection() as con:
            rows = con.execute(
                "SELECT symbol, qty, avg_short_price FROM paper_short_positions WHERE qty > 0"
            ).fetchall()
        if not rows:
            return []

        results = []
        for sym, qty, avg_short in rows:
            ticker = f"{sym}.T"
            current_price = avg_short  # フォールバック
            try:
                hist = self._market_data.get_ohlcv(ticker, "2d")
                if not hist.empty:
                    current_price = float(hist["Close"].iloc[-1])
            except Exception:
                logger.warning(
                    "[paper] %s: 株価取得失敗（フォールバック値を使用）", sym, exc_info=True
                )
            # 空売りの未実現損益: 売り単価 - 現在価格が正なら利益
            unrealized_pnl = (avg_short - current_price) * qty
            results.append(
                {
                    "symbol": sym,
                    "qty": qty,
                    "avg_short_price": avg_short,
                    "current_price": current_price,
                    "unrealized_pnl": unrealized_pnl,
                }
            )
        return results

    def get_positions(self) -> list[dict[str, Any]]:
        """保有ポジション一覧を返す（現在価格・含み損益つき）"""
        with _db_connection() as con:
            rows = con.execute(
                "SELECT symbol, qty, avg_price FROM paper_positions WHERE qty > 0"
            ).fetchall()
        if not rows:
            return []

        results = []
        for sym, qty, avg in rows:
            ticker = f"{sym}.T"
            current_price = avg  # フォールバック
            try:
                hist = self._market_data.get_ohlcv(ticker, "2d")
                if not hist.empty:
                    current_price = float(hist["Close"].iloc[-1])
            except Exception:
                logger.warning(
                    "[paper] %s: 株価取得失敗（フォールバック値を使用）", sym, exc_info=True
                )
            unrealized_pnl = (current_price - avg) * qty
            results.append(
                {
                    "symbol": sym,
                    "qty": qty,
                    "avg_price": avg,
                    "current_price": current_price,
                    "unrealized_pnl": unrealized_pnl,
                }
            )
        return results

    def get_pnl_summary(self) -> dict[str, Any]:
        """
        損益サマリーを返す。

        Returns:
            {
                "realized_pnl": float,      # 通算実現損益
                "unrealized_pnl": float,    # 含み損益合計
                "total_pnl": float,         # 通算損益
                "balance": float,           # 現在残高
                "initial_balance": float,   # 初期残高
                "trade_count": int,         # 約定済み売り取引数
                "started_at": str | None,   # 最初の約定日時
            }
        """
        with _db_connection() as con:
            row = con.execute(
                "SELECT SUM(realized_pnl), COUNT(*), MIN(filled_at) "
                "FROM paper_orders WHERE status='filled' AND side IN (?, ?) "
                "AND realized_pnl IS NOT NULL",
                [int(OrderSide.SELL), int(OrderSide.SHORT_COVER)],
            ).fetchone()
        # SUM/COUNT クエリは必ず 1 行を返すが fetchone() の型は Optional なので明示ガード
        if row is None:
            realized_pnl = 0.0
            trade_count = 0
            started_at = None
        else:
            realized_pnl = float(row[0] or 0.0)
            trade_count = int(row[1] or 0)
            started_at = str(row[2]) if row[2] else None

        # 含み損益（現在ポジションから計算）
        positions = self.get_positions()
        unrealized_pnl = sum(p["unrealized_pnl"] for p in positions)

        balance = self.get_balance()
        return {
            "realized_pnl": realized_pnl,
            "unrealized_pnl": unrealized_pnl,
            "total_pnl": realized_pnl + unrealized_pnl,
            "balance": balance,
            "initial_balance": _INITIAL_BALANCE,
            "trade_count": trade_count,
            "started_at": started_at,
        }

    def get_balance(self) -> float:
        """現金余力（円）を返す"""
        with _db_connection() as con:
            row = con.execute("SELECT balance FROM paper_balance LIMIT 1").fetchone()
        return float(row[0]) if row else _INITIAL_BALANCE

    def get_orders(self) -> list[dict[str, Any]]:
        """当日の注文一覧を返す"""
        with _db_connection() as con:
            rows = con.execute("""
                SELECT order_id, symbol, side, qty, price, status
                FROM paper_orders
                WHERE DATE(created_at) = CURRENT_DATE
                ORDER BY created_at DESC
                """).fetchall()
        return [
            {
                "order_id": oid,
                "symbol": sym,
                "side": side,
                "qty": qty,
                "price": prc,
                "status": st,
            }
            for oid, sym, side, qty, prc, st in rows
        ]
