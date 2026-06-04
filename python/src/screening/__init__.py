"""screening BC: 長期トレンド・スクリーナー

既存の OHLCV 特徴量（DuckDB `stock_features`）のみを使い、長期上昇トレンドにある
multibagger 候補を抽出・ランキングする。予測はせず「現に強い株」を選ぶ。

依存は `utils` のみ（他 BC を import しない純粋 BC）。
"""

from src.screening.trend_screener import save_candidates, screen_trend_candidates
from src.screening.types import TrendCandidate

__all__ = ["TrendCandidate", "screen_trend_candidates", "save_candidates"]
