"""screening BC の型定義"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class HoldRules:
    """保有/撤退エンジンのルール（差し替え可能なパラメータ）。"""

    trail_ma_weeks: int = 40  # 40週(=200日)線割れで撤退
    trail_stop_pct: float = 0.35  # 高値から -35% で撤退（MA割れと OR）
    # 2倍/5倍で部分利確
    scale_out_multiples: list[float] = field(default_factory=lambda: [2.0, 5.0])
    scale_out_fraction: float = 0.20  # 各利確で保有の20%を売却（[] や 0 で「放置」=利確しない）


@dataclass
class PositionEvent:
    """保有シミュレーション中に発生した約定イベント。"""

    date: str  # YYYY-MM-DD
    action: str  # "entry" | "hold" | "scale_out" | "exit"
    price: float
    held_fraction: float  # 約定後の保有比率（1.0=フル, 0.0=撤退完了）
    reason: str  # "entry" | "ma_break" | "trail_stop" | "scale_2x" | "scale_5x" | "thesis_break"
    multiple: float  # エントリー価格に対する現在倍率（price / entry_price）


@dataclass
class TrendCandidate:
    """長期トレンド・スクリーナーが返す候補銘柄。"""

    market: str
    symbol: str
    score: float  # 合成スコア（高いほど上位）
    close: float  # 直近終値
    dist_from_52w_high: float  # 52週高値からの下落率（0=高値更新, 負値=下にある）
    above_200dma: bool  # 終値 > 200日SMA
    sma200_rising: bool  # 200日SMAが上向き（直近20日で上昇）
    return_6m: float  # 6ヶ月リターン
    return_12m: float  # 12ヶ月リターン
    avg_volume: float  # 平均出来高（流動性）


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
