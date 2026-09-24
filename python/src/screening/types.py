"""screening BC の型定義"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Position:
    """月次ライブ・スクリーンが追跡する保有ポジション（半自動運用）。

    売買は人間が最終判断するため、本テーブルは「人間が建てたと宣言した」
    ポジションの状態（保有/段階利確/撤退）を月次で更新するための台帳。
    """

    market: str
    symbol: str
    entry_date: str  # YYYY-MM-DD
    entry_price: float
    status: str = "open"  # "open" | "closed"
    held_fraction: float = 1.0  # 直近評価時点の保有比率（1.0=フル, 0.0=撤退完了）
    last_action: str = "entry"  # "entry" | "hold" | "scale_out" | "exit"
    last_reason: str = "entry"  # 直近イベントの理由（ma_break / trail_stop / scale_2x ...）
    last_multiple: float = 1.0  # 直近評価時点の倍率（price / entry_price）
    last_evaluated: str = ""  # 最終評価日（YYYY-MM-DD）


@dataclass
class MultibaggerCandidate:
    """質ゲート通過後の最終候補（TrendCandidate ＋ ファンダ指標 ＋ 合成スコア）。

    財務欠損銘柄を ``on_missing="penalize"`` で残す場合、ファンダ指標は None・
    ``fundamentals_missing=True`` となり、合成スコアは減点される。
    """

    market: str
    symbol: str
    multibagger_score: float  # 合成スコア（高いほど上位）
    trend_score: float  # トレンド・スクリーナーの score（#429）
    growth_score: float  # 成長スコア（順位正規化, 0〜1）
    quality_score: float  # 質スコア（順位正規化, 0〜1）
    close: float  # 直近終値
    revenue_cagr_3y: Optional[float]  # 売上CAGR(3年)
    roe: Optional[float]  # 自己資本利益率
    op_margin: Optional[float]  # 営業利益率
    net_margin: Optional[float]  # 純利益率
    debt_to_equity: Optional[float]  # 負債資本倍率(D/E)
    market_cap: Optional[float]  # 時価総額
    fundamentals_missing: bool  # 財務欠損で penalize 扱いか


@dataclass
class ValueCandidate:
    """バリュー・スクリーナー（低PER・低配当性向・財務安定）が返す候補銘柄。"""

    market: str
    symbol: str
    trailing_pe: float
    payout_ratio: float
    debt_to_equity: float  # パーセントポイント単位（yfinance実測準拠、例: 78.4 = D/E比0.78）
    net_income: float
    market_cap: Optional[float]  # 表示用。ゲート対象外のため欠損を許容する
