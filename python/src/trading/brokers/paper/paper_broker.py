"""
PaperBroker — 仮想売買（ペーパートレード）Broker

kabu STATION® API が利用できない環境（APIキー未取得・テスト等）での
動作確認用 Broker 実装。

- 注文は yfinance の翌営業日始値で約定扱いにする
- ポジション・残高・注文履歴は DuckDB の paper_* テーブルで永続化
- BrokerBase と同一インターフェースのため、mode 変更だけで本番切り替え可能
"""

import math
import uuid
from datetime import date
from typing import Any

from config.settings import PAPER_INITIAL_BALANCE
from src.domain.ports import MarketDataPort, TradeDiffSink
from src.domain.types import TradeDiffRecord
from src.trading.brokers.base import BrokerBase, OrderSide, OrderType
from src.trading.brokers.paper import split_adjust as sa
from src.trading.brokers.paper.bar_timing import bar_date_jst as _bar_date_jst
from src.trading.brokers.paper.bar_timing import created_after_bar_close as _created_after_bar_close
from src.trading.brokers.paper.bar_timing import parse_predicted_date as _parse_predicted_date
from src.trading.brokers.paper.fill import apply_fill
from src.trading.brokers.paper.pending_short import (
    pending_in_row_units,
    read_pending_shorts,
    split_filled_short,
)
from src.trading.position_valuation import (
    long_positions_unrealized_pnl,
    short_positions_unrealized_pnl,
)
from src.utils.db._connection import _db_connection
from src.utils.japan_time import now_jst
from src.utils.logger import get_logger

logger = get_logger(__name__)

_INITIAL_BALANCE = PAPER_INITIAL_BALANCE
# 保有の評価に使う最後の足の古さの上限（暦日）。連休の前の営業日の足は使い、売買停止の足は使わない
_PRICE_MAX_AGE_DAYS = 7


def _today_jst() -> date:
    """約定のジョブを回した日（JST）。テストで差し替える"""
    return now_jst().date()


def _undo_short_registration(
    con: Any, symbol: str, qty: int, price: Any, splits: Any = None, created_at: Any = None
) -> None:
    """取り消した空売りの分を paper_short_positions から戻す（加重平均単価も戻す, #1104）

    splits と created_at を渡すと、仮登録で分割を当てて足した株数を引く（#1339）。
    """
    row = con.execute(
        "SELECT qty, avg_short_price, updated_at FROM paper_short_positions WHERE symbol=%s",
        [symbol],
    ).fetchone()
    if not row:
        return
    cur_qty, cur_avg = int(row[0]), float(row[1] or 0.0)
    factor = 1.0 if created_at is None else sa.split_factor_between(splits, created_at, row[2])
    rest = cur_qty - int(round(qty * factor))
    if rest <= 0:
        con.execute("DELETE FROM paper_short_positions WHERE symbol=%s", [symbol])
        return
    rest_avg = (cur_avg * cur_qty - float(price or 0.0) * qty) / rest
    # 権利落ちの日の寄りの前に直した行の updated_at を、前の日と読まれる今の時刻に戻さない（#1339）
    con.execute(
        "UPDATE paper_short_positions SET qty=%s, avg_short_price=%s, "
        "updated_at=GREATEST(updated_at, CURRENT_TIMESTAMP) WHERE symbol=%s",
        [rest, rest_avg, symbol],
    )


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
        trade_diff_sink: TradeDiffSink,
    ) -> None:
        self._market_data = market_data_port
        self._trade_diff_sink = trade_diff_sink
        # 直前の settle_pending_orders で、例外のため pending に残した注文の件数（#1088）
        self.last_settle_errors = 0

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
        # PaperBroker は東証銘柄専用（settle で f"{symbol}.T" を組み立てる）ため market は "jp"。
        # ここで確定させることで、メタデータ補完（_link_paper_order_metadata）を経由しない
        # 発注経路でも market が NULL にならず、エクイティ再構成の時価評価が機能する。
        splits: Any = None
        if side == OrderSide.SHORT:
            # 分割前の空売りの保有に分割後の値段を混ぜて平均しないよう、先に直す（#1086）
            splits = sa.adjust_stored_short_for_splits(self._market_data, sym)
        with _db_connection() as con:
            if side == OrderSide.BUY:
                # 約定待ちの買いは保有にも残高にも出ないため、次の回が同じ銘柄を重ねて買う（#802）
                waiting = con.execute(
                    "SELECT order_id FROM paper_orders "
                    "WHERE symbol=%s AND side=%s AND status='pending' LIMIT 1",
                    [sym, int(OrderSide.BUY)],
                ).fetchone()
                if waiting:
                    logger.warning(
                        f"[paper] {sym}: 買いが約定待ち（order_id={waiting[0]}）のため、"
                        "次の買いを受けない"
                    )
                    return {
                        "order_id": "",
                        "status": "rejected",
                        "message": "pending buy exists",
                    }
            con.execute(
                """
                INSERT INTO paper_orders
                    (order_id, market, symbol, side, qty, price, order_type, status, created_at)
                VALUES (%s, 'jp', %s, %s, %s, %s, %s, 'pending', CURRENT_TIMESTAMP)
                """,
                [order_id, sym, int(side), qty, price, int(order_type)],
            )
            if side == OrderSide.SHORT:
                # paper_short_positions に即時仮登録（加重平均単価更新）
                existing_short = con.execute(
                    "SELECT qty, avg_short_price, updated_at FROM paper_short_positions "
                    "WHERE symbol=%s",
                    [sym],
                ).fetchone()
                if existing_short:
                    old_qty, old_avg, old_updated_at = existing_short
                    # 権利落ちの日の寄りの前に直した行は updated_at を戻さず、分割前の注文の株数は
                    # 分割を当てて足す（金額は分割で変わらないのでそのまま、#1339）
                    created_at = con.execute(
                        "SELECT created_at FROM paper_orders WHERE order_id=%s", [order_id]
                    ).fetchone()[0]
                    factor = sa.split_factor_between(splits, created_at, old_updated_at)
                    new_qty = old_qty + int(round(qty * factor))
                    new_avg = (old_avg * old_qty + price * qty) / new_qty
                    con.execute(
                        "UPDATE paper_short_positions SET qty=%s, avg_short_price=%s, "
                        "updated_at=GREATEST(updated_at, CURRENT_TIMESTAMP) WHERE symbol=%s",
                        [new_qty, new_avg, sym],
                    )
                else:
                    con.execute(
                        "INSERT INTO paper_short_positions "
                        "(symbol, qty, avg_short_price, opened_at, updated_at) "
                        "VALUES (%s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)",
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
        """pending 状態の注文をキャンセルする（空売りは send_order の即時の仮登録も戻す）"""
        with _db_connection() as con:
            pending = con.execute(
                "SELECT symbol, side, qty, price, created_at FROM paper_orders "
                "WHERE order_id=%s AND status='pending'",
                [order_id],
            ).fetchone()
            con.execute(
                "UPDATE paper_orders SET status='cancelled' WHERE order_id=%s AND status='pending'",
                [order_id],
            )
            if pending and int(pending[1]) == int(OrderSide.SHORT):
                sym = str(pending[0])
                splits = sa.fetch_splits(self._market_data, sym)
                _undo_short_registration(con, sym, int(pending[2]), pending[3], splits, pending[4])
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
        # Phase 1: ペンディング注文を読み込む（接続をすぐに解放）。出た順に約定させ、
        # 同じ銘柄の返済が先の空売りより先に約定しないようにする（#1225）
        with _db_connection() as con:
            rows = con.execute(
                "SELECT order_id, market, predicted_at, symbol, side, qty, price, "
                "signal_price, order_type, created_at "
                "FROM paper_orders WHERE status='pending' ORDER BY created_at, order_id"
            ).fetchall()

        settled = []
        errors = 0  # 例外で約定できず pending に残した件数
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
            created_at,
        ) in rows:
            ticker = f"{symbol}.T"
            try:
                hist = self._market_data.get_ohlcv(ticker, "2d")
                if hist.empty:
                    logger.warning(f"[paper] {symbol}: 株価取得失敗、スキップ")
                    continue

                # 9:05 時点で当日の行がまだ無いと、最後の行はシグナルを作った日以前の行になる。
                # その行で約定させると前営業日の値段で約定してしまうため、pending のまま
                # 据え置いて次回リトライさせる（当日の値を見ていないので指値の失効も判断しない）。
                signal_date = _parse_predicted_date(predicted_at)
                bar_date = _bar_date_jst(hist.index[-1])
                if signal_date is not None and bar_date <= signal_date:
                    logger.warning(
                        f"[paper] {symbol}: 当日の行が未取得（最後の行={bar_date} "
                        f"シグナル日={signal_date}）のため約定スキップ（pending 据え置き）"
                    )
                    continue
                # predicted_at を持たない注文（SL/TP・ルール・期限の手じまい）と、何日か据え置かれた
                # 注文も、今日（JST）の行でだけ約定させる。東証の休みの日は最後の行が前の営業日になる（#805）
                today = _today_jst()
                if bar_date < today:
                    logger.warning(
                        f"[paper] {symbol}: 今日の行が未取得（最後の行={bar_date} "
                        f"今日={today}）のため約定スキップ（pending 据え置き）"
                    )
                    continue
                # 引けの後に出た注文（遅れた 9:05 の約定が 16:30 のルールの発注の後に走ったときなど）を、
                # 注文より前の始値で約定させない。翌営業日の足で約定させる。us は東証の引けで区切らない
                if market != "us" and _created_after_bar_close(created_at, bar_date):
                    logger.warning(
                        f"[paper] {symbol}: 注文が足の日の引けの後（created_at={created_at}、"
                        f"足={bar_date}）のため約定スキップ（pending 据え置き）"
                    )
                    continue

                today_row = hist.iloc[-1]
                open_price = float(today_row["Open"])
                high_price = float(today_row["High"])
                low_price = float(today_row["Low"])

                # yfinance が NaN の OHLC を返すことがある（hist.empty は通過する）。
                # NaN のまま約定させると fill_price / 残高 / avg_price が NaN 汚染するため、
                # pending のまま据え置いて次回リトライさせる。
                if math.isnan(open_price) or math.isnan(high_price) or math.isnan(low_price):
                    logger.warning(
                        f"[paper] {symbol}: OHLC が NaN のため約定スキップ（pending 据え置き）"
                    )
                    continue

                order = (side, qty, limit_price, signal_price, created_at)
                adjusted = sa.adjust_order_for_splits(self._market_data, symbol, order, open_price)
                if adjusted is None:
                    continue
                qty, limit_price, signal_price = adjusted

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
                        with _db_connection() as con, con.transaction():
                            con.execute(
                                "UPDATE paper_orders SET status='expired' WHERE order_id=%s",
                                [order_id],
                            )
                            if side == int(OrderSide.SHORT):
                                # cancel_order と同じく仮登録を戻す。行も分割で直してあるので、
                                # 分割の直しの後の株数と指値で引く（#1219）
                                _undo_short_registration(con, symbol, qty, limit_price)
                        continue

                # Phase 2: 約定処理（接続を再取得）。接続は自動確定のため、1件分を囲み、
                # 途中で落ちたら残高・保有・注文を全部戻して pending に残す（#1088）
                with _db_connection() as con, con.transaction():
                    if side == int(OrderSide.BUY):
                        cash = float(
                            con.execute("SELECT balance FROM paper_balance LIMIT 1").fetchone()[0]
                        )
                        if qty * fill_price > cash:
                            # 発注は約定待ちを引かない残高で株数を決めるので、ここで検算する（#802）
                            logger.warning(
                                f"[paper] {symbol}: 現金不足（要 {qty * fill_price:.0f} / "
                                f"残 {cash:.0f}）のため買いを約定させない（rejected）"
                            )
                            con.execute(
                                "UPDATE paper_orders SET status='rejected' WHERE order_id=%s",
                                [order_id],
                            )
                            continue
                    qty = apply_fill(con, order_id, symbol, side, qty, fill_price)
                    if qty <= 0:
                        # 保有の無い決済は約定させない（評価額に幻の現金と建玉を入れない、#1190）
                        con.execute(
                            "UPDATE paper_orders SET status='rejected' WHERE order_id=%s",
                            [order_id],
                        )
                        continue
                    con.execute(
                        "UPDATE paper_orders SET status='filled', qty=%s, "
                        "fill_price=%s, filled_at=CURRENT_TIMESTAMP WHERE order_id=%s",
                        [qty, fill_price, order_id],
                    )
                if market and predicted_at and signal_price is not None:
                    self._trade_diff_sink.record(
                        TradeDiffRecord(
                            market=str(market),
                            symbol=str(symbol),
                            predicted_at=str(predicted_at),
                            side=int(side),
                            signal_price=float(signal_price),
                            mode="paper",
                            order_id=str(order_id),
                            actual_price=fill_price,
                        )
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
                errors += 1
                logger.error(f"[paper] 約定処理エラー ({symbol}): {e}", exc_info=True)

        self.last_settle_errors = errors
        if errors:
            logger.error(
                f"[paper] 約定処理の例外で pending に据え置いた注文: {errors} 件"
                "（当日の行が無い据え置きとは別）"
            )
        return settled

    # ------------------------------------------------------------------
    # 照会
    # ------------------------------------------------------------------

    def _latest_close(self, sym: str) -> float:
        """保有の評価に使う最新の終値。取れない・NaN・最後の足が古いときは NaN（#844）

        取得元の値（買値・売値）で埋めると損益0に見え、SL/TP や DD の判定が黙って外れる。
        """
        try:
            hist = self._market_data.get_ohlcv(f"{sym}.T", "2d")
        except Exception:
            logger.warning("[paper] %s: 株価取得失敗（評価できない）", sym, exc_info=True)
            return math.nan
        if hist.empty:
            logger.warning(f"[paper] {sym}: 株価が空（評価できない）")
            return math.nan
        close = float(hist["Close"].iloc[-1])
        bar_date = _bar_date_jst(hist.index[-1])
        if math.isnan(close) or (_today_jst() - bar_date).days > _PRICE_MAX_AGE_DAYS:
            logger.warning(f"[paper] {sym}: 終値が NaN か古い（最後の足={bar_date}、評価できない）")
            return math.nan
        return close

    def get_short_positions(self) -> list[dict[str, Any]]:
        """保有中の空売りポジション一覧を返す。

        Returns:
            list[dict]: [{"symbol": str, "qty": int, "pending_qty": int,
                          "avg_short_price": float, "current_price": float,
                          "unrealized_pnl": float | None}]
            qty は約定待ちの SHORT（仮登録）を含む（保有の上限に数える, #1146）。pending_qty は
            その約定待ちの株数で、avg_short_price と unrealized_pnl は約定した分だけで出す（#1225）。
            現在値が評価できない銘柄は current_price と unrealized_pnl が NaN になる。
        """
        with _db_connection() as con:
            rows = con.execute(
                "SELECT symbol, qty, avg_short_price, updated_at "
                "FROM paper_short_positions WHERE qty > 0"
            ).fetchall()
            pending = read_pending_shorts(con) if rows else {}
        if not rows:
            return []

        results = []
        for sym, raw_qty, raw_avg, updated_at in rows:
            qty, avg_short, split_known = sa.adjust_short_position_for_splits(
                self._market_data, sym, raw_qty, raw_avg, updated_at
            )
            orders = pending.get(str(sym))
            splits = sa.fetch_splits(self._market_data, sym) if orders else None
            pending_qty, filled_qty, avg_short = split_filled_short(
                pending_in_row_units(orders, splits, updated_at), int(raw_qty), qty, avg_short
            )
            current_price = self._latest_close(sym)
            if (
                filled_qty > 0
                and not split_known
                and sa.looks_like_split_gap(avg_short, current_price)
            ):
                logger.warning(
                    f"[paper] {sym}: 分割の記録が取れず、売り建ての単価と現在値の比が分割の比ほど違う"
                )
                current_price = math.nan
            # 空売りの未実現損益: 売り単価 - 現在価格が正なら利益。約定待ちの分は評価しない
            unrealized_pnl = (avg_short - current_price) * filled_qty if filled_qty > 0 else 0.0
            results.append(
                {
                    "symbol": sym,
                    "qty": qty,
                    "pending_qty": pending_qty,
                    "avg_short_price": avg_short,
                    "current_price": current_price,
                    "unrealized_pnl": unrealized_pnl,
                }
            )
        return results

    def get_positions(self) -> list[dict[str, Any]]:
        """保有ポジション一覧を返す（現在価格・含み損益つき。評価できない銘柄は NaN）"""
        with _db_connection() as con:
            rows = con.execute(
                "SELECT symbol, qty, avg_price, updated_at FROM paper_positions WHERE qty > 0"
            ).fetchall()
        if not rows:
            return []

        results = []
        for sym, qty, avg, updated_at in rows:
            qty, avg, split_known = sa.adjust_position_for_splits(
                self._market_data, sym, qty, avg, updated_at
            )
            current_price = self._latest_close(sym)
            if not split_known and sa.looks_like_split_gap(avg, current_price):
                logger.warning(
                    f"[paper] {sym}: 分割の記録が取れず、買値と現在値の比が分割の比ほど違う"
                )
                current_price = math.nan
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

    def get_pending_buy_symbols(self) -> set[str]:
        """約定待ちの BUY がある銘柄コードを返す（get_positions は約定前の買いを含まない、#1137）"""
        with _db_connection() as con:
            rows = con.execute(
                "SELECT DISTINCT symbol FROM paper_orders WHERE status='pending' AND side=%s",
                [int(OrderSide.BUY)],
            ).fetchall()
        return {str(sym).replace(".T", "") for (sym,) in rows}

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
                "FROM paper_orders WHERE status='filled' AND side IN (%s, %s) "
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
        # 評価できない（NaN の）保有は WARNING を出して飛ばす（#1162）
        unrealized_pnl = long_positions_unrealized_pnl(self.get_positions(), logger)
        # 空売りの含み損益も足す（#1148）
        unrealized_pnl += short_positions_unrealized_pnl(self.get_short_positions(), logger)

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
