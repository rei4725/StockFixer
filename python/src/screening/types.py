"""screening BC の型定義"""

from dataclasses import dataclass


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
