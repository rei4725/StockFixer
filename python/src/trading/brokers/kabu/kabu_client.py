"""
KabuBroker — au カブコム証券 kabu STATION® API クライアント

kabu STATION® アプリ（常駐）が起動している前提で localhost:18080 を叩く。
APIパスワードは環境変数 KABU_API_PASSWORD から取得する。

公式仕様書: https://kabucom.github.io/kabusapi/reference/
"""

import math
import os
from datetime import datetime, timedelta
from typing import Any

import httpx

from src.trading.brokers.base import BrokerBase, BrokerError, OrderSide, OrderType
from src.trading.brokers.kabu.rate_limit import request_with_rate_limit_retry
from src.utils.logger import get_logger

logger = get_logger(__name__)

_BASE_URL = "http://localhost:18080/kabusapi"
_TOKEN_TTL_SECONDS = 3600  # トークン有効期間（1時間）
_TOKEN_REFRESH_BUFFER_SECONDS = 300  # 期限5分前に事前更新
_TOKEN_MAX_RETRIES = 2  # 1回リトライ（初回 + 1回）
_REC_TYPE_EXECUTION = 8  # /orders の Details の RecType: 8 = 約定
_STATE_DONE = 5  # /orders の State: 5 = 終了（全部約定・取消・失効）
# /orders の State の 1〜4 は終わっていない注文（1 = 待機、2 = 処理中、3 = 処理済で
# 約定待ち・一部約定、4 = 訂正取消送信中）。値の意味は仕様書の記憶による推測（#1521）
# /orders・/positions の product: 0 = すべて、1 = 現物、2 = 信用、3 = 先物、4 = OP
_PRODUCT_CASH = 1
# sendorder・cancelorder の応答の Result: 0 = 成功、それ以外は断り（#951）。
# 0 以外の個々の値の意味は仕様書で確かめていない（推測）。
_RESULT_OK = 0

# RequestSendOrder の値（kabu_STATION_API.yaml）
# Side: "1" = 売、"2" = 買。OrderSide の数とは逆なので、明示の表で写す。
_CASH_SIDE: dict[OrderSide, str] = {
    OrderSide.BUY: "2",
    OrderSide.SELL: "1",
}
# DelivType: 現物買は 2 = お預り金、現物売は 0 = 指定なし
_CASH_DELIV_TYPE: dict[OrderSide, int] = {
    OrderSide.BUY: 2,
    OrderSide.SELL: 0,
}
# FundType: 現物買は 02 = 保護、現物売は半角スペース2つが指定必須
_CASH_FUND_TYPE: dict[OrderSide, str] = {
    OrderSide.BUY: "02",
    OrderSide.SELL: "  ",
}
_ACCOUNT_TYPE_SPECIFIC = 4  # AccountType: 2 = 一般、4 = 特定、12 = 法人
_POSITION_SIDE_BUY = "2"  # /positions の Side: "1" = 売、"2" = 買


def _number(value: Any) -> float | None:
    """数として読める値を float で返す。null・数でない値・NaN は None を返す。"""
    if value is None or isinstance(value, bool):
        return None
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(num) else num


def _is_cash_long(item: Any) -> bool:
    """/positions の1行が現物の買い持ちなら True を返す（#1000）。

    システムは現物の買いしか建てないため、信用の建玉（MarginTradeType のある行）と
    売りの行（Side "1"）は SL/TP・売りシグナル・保有の上限に入れない。
    Side が無い行は、product で現物に絞った返りとして買い持ちとみなす。
    """
    if not isinstance(item, dict):
        return False
    if item.get("MarginTradeType") not in (None, 0, ""):
        logger.info(f"[kabu] {item.get('Symbol', '')}: 信用の建玉は保有に数えない")
        return False
    side = item.get("Side")
    if side is not None and str(side) != _POSITION_SIDE_BUY:
        logger.info(f"[kabu] {item.get('Symbol', '')}: 売りの建玉は保有に数えない")
        return False
    return True


def _check_result(data: Any, action: str) -> dict[str, Any]:
    """sendorder・cancelorder の応答の Result が 0 でなければ BrokerError を出す（#951）。

    Result が無い・数でない応答も、受け付けられたとは言えないため断りとして扱う。
    """
    if not isinstance(data, dict) or _number(data.get("Result")) != _RESULT_OK:
        raise BrokerError(f"[kabu] {action} が断られました: {data!r}")
    return data


def _position_row(item: dict[str, Any]) -> dict[str, Any]:
    """/positions の1行を保有の辞書に写す（#1010）。

    欄が null のときは .get の既定値が効かず None になるため、ここで数に直す。
    現在値が null なら評価額 Valuation ÷ 株数で埋め、埋められなければ NaN にする
    （紙の道と同じく position_valuation が時価0として数え、SL/TP は飛ばす）。
    """
    symbol = item.get("Symbol", "")
    qty_num = _number(item.get("LeavesQty", 0))
    if qty_num is None:
        logger.warning(f"[kabu] {symbol}: 建玉の株数 LeavesQty が読めず 0 として扱う")
        qty_num = 0.0
    qty = int(qty_num)

    avg_price = _number(item.get("Price", 0.0))
    if avg_price is None:
        logger.warning(f"[kabu] {symbol}: 建玉の取得単価 Price が読めず NaN として扱う")
        avg_price = math.nan

    current_price = _number(item.get("CurrentPrice", 0.0))
    if current_price is None:
        valuation = _number(item.get("Valuation"))
        if valuation is not None and qty > 0:
            current_price = valuation / qty
        else:
            logger.warning(f"[kabu] {symbol}: 建玉の現在値 CurrentPrice が読めず NaN として扱う")
            current_price = math.nan

    return {
        "symbol": symbol,
        "qty": qty,
        "avg_price": avg_price,
        "current_price": current_price,
    }


class KabuBroker(BrokerBase):
    """kabu STATION® API を使った証券操作クライアント"""

    broker_name = "kabu"

    def __init__(self, api_password: str | None = None, timeout: float = 10.0):
        """
        Args:
            api_password: kabu STATION® API パスワード。
                          省略時は環境変数 KABU_API_PASSWORD を使用。
            timeout: HTTP タイムアウト（秒）
        """
        self._api_password = api_password or os.environ.get("KABU_API_PASSWORD", "")
        self._timeout = timeout
        self._token: str = ""
        self._token_expires_at: datetime = datetime.min

    # ------------------------------------------------------------------
    # 認証
    # ------------------------------------------------------------------

    def get_token(self) -> str:
        """
        APIトークンを取得する。
        キャッシュ済みかつ期限まで5分超あれば再利用する。
        取得失敗時は最大1回リトライし、それでも失敗すれば BrokerError を raise する。
        """
        buffer = timedelta(seconds=_TOKEN_REFRESH_BUFFER_SECONDS)
        if self._token and datetime.now() < self._token_expires_at - buffer:
            return self._token

        if not self._api_password:
            raise EnvironmentError(
                "KABU_API_PASSWORD が設定されていません。"
                "環境変数を設定するか KabuBroker(api_password=...) で渡してください。"
            )

        last_exc: Exception | None = None
        for attempt in range(_TOKEN_MAX_RETRIES):
            try:
                resp = httpx.post(
                    f"{_BASE_URL}/token",
                    json={"APIPassword": self._api_password},
                    timeout=self._timeout,
                )
                resp.raise_for_status()
                self._token = resp.json()["Token"]
                self._token_expires_at = datetime.now() + timedelta(seconds=_TOKEN_TTL_SECONDS)
                logger.info("kabu STATION® API トークンを取得しました (attempt=%d)", attempt + 1)
                return self._token
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "kabu STATION® API トークン取得失敗 (attempt=%d/%d): %s",
                    attempt + 1,
                    _TOKEN_MAX_RETRIES,
                    exc,
                )

        raise BrokerError(
            f"kabu STATION® API トークン取得に失敗しました ({_TOKEN_MAX_RETRIES}回試行): {last_exc}"
        ) from last_exc

    # ------------------------------------------------------------------
    # 注文
    # ------------------------------------------------------------------

    @staticmethod
    def _require_trade_password() -> str:
        """取引パスワードを返す。未設定・空なら送る前に BrokerError で止める。"""
        password = os.environ.get("KABU_TRADE_PASSWORD", "")
        if not password:
            raise BrokerError(
                "KABU_TRADE_PASSWORD が設定されていません。空のまま注文・取消は送りません。"
            )
        return password

    def send_order(
        self,
        symbol: str,
        side: OrderSide,
        qty: int,
        price: float = 0.0,
        order_type: OrderType = OrderType.MARKET,
    ) -> dict[str, Any]:
        """
        現物株式の注文を発注する（東証・現物・特定口座）。

        Args:
            symbol: 銘柄コード（".T" なし、例: "7203"）
            side: OrderSide.BUY / SELL
            qty: 株数
            price: 指値価格（成行は 0.0）
            order_type: OrderType.MARKET / LIMIT

        Returns:
            {"order_id": str, "status": str, "message": str}

        Raises:
            BrokerError: SHORT / SHORT_COVER など現物の口で送れない side のとき（送らない）。
                応答の Result が 0 でないとき、OrderId が空のとき（#951）
        """
        if side not in _CASH_SIDE:
            raise BrokerError(
                f"[kabu] 現物の口では {side.name} の注文を送れません: symbol={symbol}"
            )
        trade_password = self._require_trade_password()
        token = self.get_token()
        body = {
            "Password": trade_password,
            "Symbol": symbol.replace(".T", ""),
            "Exchange": 1,  # 1 = 東証
            "SecurityType": 1,  # 1 = 株式
            "Side": _CASH_SIDE[side],
            "CashMargin": 1,  # 1 = 現物
            "DelivType": _CASH_DELIV_TYPE[side],
            "FundType": _CASH_FUND_TYPE[side],
            "AccountType": _ACCOUNT_TYPE_SPECIFIC,
            "Qty": qty,
            "FrontOrderType": int(order_type),
            "Price": price if order_type == OrderType.LIMIT else 0,
            "ExpireDay": 0,  # 0 = 当日有効
        }
        logger.info(f"[kabu] 注文送信: symbol={symbol} side={side.name} qty={qty} price={price}")
        resp = request_with_rate_limit_retry(
            lambda: httpx.post(
                f"{_BASE_URL}/sendorder",
                headers={"X-API-KEY": token},
                json=body,
                timeout=self._timeout,
            ),
            f"sendorder symbol={symbol}",
        )
        resp.raise_for_status()
        data = _check_result(resp.json(), f"注文 symbol={symbol}")
        order_id = str(data.get("OrderId") or "")
        if not order_id:
            raise BrokerError(
                f"[kabu] 注文 symbol={symbol} の応答に OrderId がありません: {data!r}"
            )
        logger.info(f"[kabu] 注文受付: order_id={order_id}")
        return {"order_id": order_id, "status": "accepted", "message": str(data)}

    def cancel_order(self, order_id: str) -> dict[str, Any]:
        """注文をキャンセルする"""
        trade_password = self._require_trade_password()
        token = self.get_token()
        body = {
            "OrderId": order_id,
            "Password": trade_password,
        }
        resp = httpx.put(
            f"{_BASE_URL}/cancelorder",
            headers={"X-API-KEY": token},
            json=body,
            timeout=self._timeout,
        )
        resp.raise_for_status()
        data = _check_result(resp.json(), f"取消 order_id={order_id}")
        logger.info(f"[kabu] 注文キャンセル: order_id={order_id}")
        return {"order_id": order_id, "status": "cancelled", "message": str(data)}

    # ------------------------------------------------------------------
    # 照会
    # ------------------------------------------------------------------

    def get_positions(self) -> list[dict[str, Any]]:
        """保有ポジション一覧を返す"""
        token = self.get_token()
        resp = request_with_rate_limit_retry(
            lambda: httpx.get(
                f"{_BASE_URL}/positions",
                headers={"X-API-KEY": token},
                params={"product": _PRODUCT_CASH},
                timeout=self._timeout,
            ),
            "positions",
        )
        resp.raise_for_status()
        raw = resp.json() or []
        return [_position_row(item) for item in raw if _is_cash_long(item)]

    def get_balance(self) -> float:
        """現金余力（円）を返す"""
        token = self.get_token()
        resp = request_with_rate_limit_retry(
            lambda: httpx.get(
                f"{_BASE_URL}/wallet/cash",
                headers={"X-API-KEY": token},
                timeout=self._timeout,
            ),
            "wallet/cash",
        )
        resp.raise_for_status()
        return float(resp.json().get("StockAccountWallet", 0.0))

    def get_orders(self) -> list[dict[str, Any]]:
        """当日の注文一覧を返す。

        price は注文の値段（成行は 0）のまま。fill_price は Details の約定の行
        （RecType 8）の値段を数量で加重した平均で、約定の行が無ければ None。
        filled は注文が終了（State 5）し、約定の数量が 1 以上のとき True (#952)。
        """
        token = self.get_token()
        resp = httpx.get(
            f"{_BASE_URL}/orders",
            headers={"X-API-KEY": token},
            params={"product": 0, "details": True},
            timeout=self._timeout,
        )
        resp.raise_for_status()
        raw = resp.json() or []
        orders: list[dict[str, Any]] = []
        for item in raw:
            fill_price, filled_qty = _execution_fill(item.get("Details"))
            state = item.get("State", "")
            orders.append(
                {
                    "order_id": item.get("ID", ""),
                    "symbol": item.get("Symbol", ""),
                    "side": item.get("Side", ""),
                    "qty": item.get("OrderQty", 0),
                    "price": item.get("Price", 0.0),
                    "status": state,
                    "fill_price": fill_price,
                    "filled_qty": filled_qty,
                    "filled": state == _STATE_DONE and filled_qty > 0,
                }
            )
        return orders

    def get_pending_buy_symbols(self) -> set[str]:
        """当日の終わっていない現物の買いの銘柄コードを返す（#1521）。

        読めないときは例外を上へ出す（0件と取り違えると重ね買いになる。#1465）。
        """
        return self._working_cash_symbols(OrderSide.BUY)

    def get_pending_sell_symbols(self) -> set[str]:
        """当日の終わっていない現物の売りの銘柄コードを返す（#1521）。読めないときは例外を出す。"""
        return self._working_cash_symbols(OrderSide.SELL)

    def _working_cash_symbols(self, side: OrderSide) -> set[str]:
        """/orders の現物の注文のうち、side が合い終了（State 5）でないものの銘柄を返す。"""
        token = self.get_token()
        resp = request_with_rate_limit_retry(
            lambda: httpx.get(
                f"{_BASE_URL}/orders",
                headers={"X-API-KEY": token},
                params={"product": _PRODUCT_CASH},
                timeout=self._timeout,
            ),
            "orders",
        )
        resp.raise_for_status()
        want_side = _CASH_SIDE[side]
        symbols: set[str] = set()
        for item in resp.json() or []:
            if str(item.get("Side", "")) != want_side or not _is_working(item):
                continue
            symbol = str(item.get("Symbol") or "").replace(".T", "")
            if symbol:
                symbols.add(symbol)
        return symbols


def _is_working(item: dict[str, Any]) -> bool:
    """注文が終わっていなければ True。State が読めない注文は終わっていないとして数える。"""
    state = _number(item.get("State"))
    if state is None:
        logger.warning(
            "[kabu] 注文 %s の State が読めず、約定待ちとして数える: %r",
            item.get("ID", ""),
            item.get("State"),
        )
        return True
    return int(state) != _STATE_DONE


def _execution_fill(details: Any) -> tuple[float | None, int]:
    """Details の約定の行から、数量で加重した約定値と約定の数量を返す。"""
    total_qty = 0
    total_amount = 0.0
    for row in details or []:
        if not isinstance(row, dict) or row.get("RecType") != _REC_TYPE_EXECUTION:
            continue
        try:
            price = float(row.get("Price") or 0.0)
            qty = int(float(row.get("Qty") or 0))
        except (TypeError, ValueError):
            continue
        if price <= 0 or qty <= 0:
            continue
        total_qty += qty
        total_amount += price * qty
    if total_qty <= 0:
        return None, 0
    return total_amount / total_qty, total_qty
