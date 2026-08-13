"""
yfinance 財務取得層。

``yf.Ticker(ticker)`` の ``.info`` / ``.financials`` / ``.income_stmt`` /
``.balance_sheet`` から財務指標を取得し、``FundamentalRecord`` に正規化する。
取得・パースに失敗した場合は例外を伝播させず ``None`` を返す。

リトライ・例外処理の方針は ``src.market_data.yf_client`` の既存ラッパに合わせる
（``with_retry`` で指数バックオフ、失敗は warning + exc_info=True でログ）。

⚠️ 本層が返すデータは yfinance の **最新スナップショットのみ** であり、
**ポイントインタイム非対応** である。**将来向きライブスクリーン（#434）のフィルタ
専用** とし、過去バックテスト（#431）には使用しないこと（PIT データは Stage 3）。
"""

from datetime import datetime, timezone
from typing import Optional

import pandas as pd
import yfinance as yf

from src.market_data.types import FundamentalRecord
from src.utils.data_path_utils import get_ticker
from src.utils.logger import get_logger
from src.utils.retry_helper import with_retry

logger = get_logger(__name__)

# .financials / .income_stmt の行ラベル候補（yfinance のバージョン差異を吸収）
_REVENUE_KEYS = ("Total Revenue", "TotalRevenue", "Revenue")
_OPERATING_INCOME_KEYS = ("Operating Income", "OperatingIncome")
_NET_INCOME_KEYS = ("Net Income", "NetIncome", "Net Income Common Stockholders")
_CASH_KEYS = (
    "Cash And Cash Equivalents",
    "CashAndCashEquivalents",
    "Cash",
    "Cash Cash Equivalents And Short Term Investments",
)


def _safe_float(value: object) -> Optional[float]:
    """値を float に変換する。None・NaN・変換不能は None。"""
    if value is None:
        return None
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if pd.isna(result):
        return None
    return result


def _row_series(df: Optional[pd.DataFrame], keys: tuple[str, ...]) -> Optional[pd.Series]:
    """財務 DataFrame から候補ラベルに一致する行（Series）を返す。見つからなければ None。"""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return None
    for key in keys:
        if key in df.index:
            # pre-commitフックのmypy(pandas-stubs同梱、requirements-dev.txt非連動)では
            # df.loc[key] の戻り値が Series | DataFrame | Scalar と推論され、
            # 宣言した Optional[pd.Series] に対し return-value エラーとなる。
            # CI側の `mypy src/`（リポジトリのpandas-stubsピン）では発生しない差異。
            return df.loc[key]  # type: ignore[return-value]
    return None


def _latest_value(df: Optional[pd.DataFrame], keys: tuple[str, ...]) -> Optional[float]:
    """財務 DataFrame の該当行から直近期（最新列）の値を返す。

    yfinance の財務 DataFrame は列が期間（新しい順）なので先頭列が直近。
    """
    series = _row_series(df, keys)
    if series is None or len(series) == 0:
        return None
    return _safe_float(series.iloc[0])


def _revenue_cagr_3y(financials: Optional[pd.DataFrame]) -> Optional[float]:
    """売上高の CAGR を算出する（最大4期＝3年分のスパンを使用）。

    年率 = (最新 / 最古) ** (1 / 期間数) - 1。
    値が2期未満、または開始/終了が非正の場合は None。
    """
    series = _row_series(financials, _REVENUE_KEYS)
    if series is None:
        return None
    # 列（期間）の新しい順 → 時系列昇順に並べ替え、欠損を除去
    values = [_safe_float(v) for v in series.tolist()]
    values = [v for v in values if v is not None]
    if len(values) < 2:
        return None
    # series は新しい順なので reverse して古い→新しい
    ordered = list(reversed(values))[-4:]
    start = ordered[0]
    end = ordered[-1]
    span = len(ordered) - 1
    if start is None or end is None or start <= 0 or end <= 0 or span < 1:
        return None
    return (end / start) ** (1.0 / span) - 1.0


def fetch_fundamentals(market: str, symbol: str) -> Optional[FundamentalRecord]:
    """
    yfinance から1銘柄の財務スナップショットを取得する。

    取得失敗（例外・空データ）時は ``None`` を返し、warning ログを残す
    （例外は伝播させない）。

    Args:
        market: マーケット識別子（例: "us", "jp"）
        symbol: 銘柄シンボル（例: "AAPL", "7203"）

    Returns:
        ``FundamentalRecord``。取得・パース不能時は ``None``。
    """
    ticker = get_ticker(market, symbol)
    try:
        ticker_obj = yf.Ticker(ticker)

        def _fetch() -> tuple:
            info = ticker_obj.info or {}
            financials = ticker_obj.financials
            income_stmt = ticker_obj.income_stmt
            balance_sheet = ticker_obj.balance_sheet
            return info, financials, income_stmt, balance_sheet

        info, financials, income_stmt, balance_sheet = with_retry(_fetch)
    except Exception:
        logger.warning("[fundamentals] 取得失敗 market=%s symbol=%s", market, symbol, exc_info=True)
        return None

    if not isinstance(info, dict):
        info = {}

    revenue = _safe_float(info.get("totalRevenue"))
    if revenue is None:
        revenue = _latest_value(financials, _REVENUE_KEYS)

    operating_income = _latest_value(income_stmt, _OPERATING_INCOME_KEYS)
    if operating_income is None:
        operating_income = _latest_value(financials, _OPERATING_INCOME_KEYS)

    net_income = _safe_float(info.get("netIncomeToCommon"))
    if net_income is None:
        net_income = _latest_value(income_stmt, _NET_INCOME_KEYS)
        if net_income is None:
            net_income = _latest_value(financials, _NET_INCOME_KEYS)

    eps = _safe_float(info.get("trailingEps"))
    roe = _safe_float(info.get("returnOnEquity"))

    op_margin = _safe_float(info.get("operatingMargins"))
    if op_margin is None and operating_income is not None and revenue:
        op_margin = operating_income / revenue

    net_margin = _safe_float(info.get("profitMargins"))
    if net_margin is None and net_income is not None and revenue:
        net_margin = net_income / revenue

    debt_to_equity = _safe_float(info.get("debtToEquity"))

    cash = _safe_float(info.get("totalCash"))
    if cash is None:
        cash = _latest_value(balance_sheet, _CASH_KEYS)

    market_cap = _safe_float(info.get("marketCap"))
    shares_outstanding = _safe_float(info.get("sharesOutstanding"))
    revenue_cagr_3y = _revenue_cagr_3y(financials)
    trailing_pe = _safe_float(info.get("trailingPE"))
    payout_ratio = _safe_float(info.get("payoutRatio"))

    record = FundamentalRecord(
        market=market,
        symbol=symbol,
        as_of=datetime.now(timezone.utc),
        revenue=revenue,
        operating_income=operating_income,
        net_income=net_income,
        eps=eps,
        roe=roe,
        op_margin=op_margin,
        net_margin=net_margin,
        debt_to_equity=debt_to_equity,
        cash=cash,
        market_cap=market_cap,
        shares_outstanding=shares_outstanding,
        revenue_cagr_3y=revenue_cagr_3y,
        trailing_pe=trailing_pe,
        payout_ratio=payout_ratio,
    )

    # 全フィールドが空（実質データ無し）なら取得失敗扱い
    metric_fields = (
        revenue,
        operating_income,
        net_income,
        eps,
        roe,
        market_cap,
        shares_outstanding,
    )
    if all(v is None for v in metric_fields):
        logger.warning("[fundamentals] 有効な財務指標なし market=%s symbol=%s", market, symbol)
        return None

    return record
