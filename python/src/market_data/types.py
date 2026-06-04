"""
market_data BC の共有データ型。
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class FundamentalRecord:
    """yfinance から取得した1銘柄の財務スナップショット。

    ⚠️ 本データは yfinance の **最新スナップショットのみ** で構成され、
    **ポイントインタイム（PIT）非対応** である。各フィールドは取得時点の
    最新値であり、過去日付に遡って当時の値を再現することはできない。

    そのため本データは **将来向きライブスクリーン（#434）のフィルタ専用** と
    し、過去バックテスト（#431）には使用しないこと。PIT 対応データは Stage 3
    で別途整備する。

    Attributes:
        market: マーケット識別子（例: "us", "jp"）
        symbol: 銘柄シンボル（例: "AAPL", "7203"）
        as_of: スナップショット基準時刻（UTC）
        revenue: 直近売上高
        operating_income: 直近営業利益
        net_income: 直近純利益
        eps: 1株当たり利益（trailing EPS）
        roe: 自己資本利益率
        op_margin: 営業利益率
        net_margin: 純利益率
        debt_to_equity: 負債資本倍率（D/E）
        cash: 現金及び現金同等物
        market_cap: 時価総額
        shares_outstanding: 発行済株式数
        revenue_cagr_3y: 売上高 CAGR（3年、算出不能時は None）
    """

    market: str
    symbol: str
    as_of: datetime
    revenue: Optional[float] = None
    operating_income: Optional[float] = None
    net_income: Optional[float] = None
    eps: Optional[float] = None
    roe: Optional[float] = None
    op_margin: Optional[float] = None
    net_margin: Optional[float] = None
    debt_to_equity: Optional[float] = None
    cash: Optional[float] = None
    market_cap: Optional[float] = None
    shares_outstanding: Optional[float] = None
    revenue_cagr_3y: Optional[float] = None
