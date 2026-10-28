"""予想の回ごとに、ATR のクリップと特徴量の0埋めを数えてログに出す（#1318）。

predict_with_unified_model は銘柄ごとに呼ばれ、呼び出し元のテストの差し替えも
多いので、引数を増やさず、回を囲む collect_prediction_run_stats() の間だけ
record_clip() / record_zero_fill() で数える。予想はスレッドで並べて走るため、
集める先はスレッドをまたいで共有し、ロックで守る。
"""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
from threading import Lock
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

from src.utils.logger import get_logger

logger = get_logger(__name__)

# 0埋めの特徴量の名前をログに出す件数の上限
MAX_LOGGED_FEATURE_NAMES = 10


@dataclass
class ClipFillCounts:
    """1つの市場・ホライズンの数"""

    total: int = 0
    clipped_upper: int = 0
    clipped_lower: int = 0
    no_atr: int = 0
    fill_total: int = 0
    filled_symbols: int = 0
    filled_features: Counter = field(default_factory=Counter)


class PredictionRunStats:
    """1回の予想のクリップと0埋めの数（スレッドをまたいで安全）"""

    def __init__(self, label: str) -> None:
        self.label = label
        self._lock = Lock()
        self._counts: Dict[Tuple[str, int], ClipFillCounts] = {}

    def _get(self, market: str, horizon: int) -> ClipFillCounts:
        return self._counts.setdefault((market, horizon), ClipFillCounts())

    def record_clip(
        self, market: str, horizon: int, raw_ratio: float, max_ratio: Optional[float]
    ) -> None:
        """クリップの前の変化率と上限の幅（判定できないときは None）を数える"""
        with self._lock:
            counts = self._get(market, horizon)
            counts.total += 1
            if max_ratio is None:
                counts.no_atr += 1
            elif raw_ratio > max_ratio:
                counts.clipped_upper += 1
            elif raw_ratio < -max_ratio:
                counts.clipped_lower += 1

    def record_zero_fill(self, market: str, horizon: int, features: Sequence[str]) -> None:
        """1銘柄で0で埋めた特徴量（無ければ空）を数える"""
        with self._lock:
            counts = self._get(market, horizon)
            counts.fill_total += 1
            if features:
                counts.filled_symbols += 1
                counts.filled_features.update(set(features))

    def snapshot(self) -> Dict[Tuple[str, int], ClipFillCounts]:
        with self._lock:
            return dict(self._counts)

    def log_summary(self) -> None:
        """市場・ホライズンごとに、クリップと0埋めを1行ずつ INFO で出す"""
        for (market, horizon), counts in sorted(self.snapshot().items()):
            total = counts.total
            logger.info(
                "予想のATRクリップ: run=%s market=%s horizon=%d 銘柄数=%d "
                "上限=%d（%.1f%%） 下限=%d（%.1f%%） ATRなし=%d",
                self.label,
                market,
                horizon,
                total,
                counts.clipped_upper,
                _percent(counts.clipped_upper, total),
                counts.clipped_lower,
                _percent(counts.clipped_lower, total),
                counts.no_atr,
            )
            # 多くの銘柄で埋めた順、同じ数なら名前の順
            names = sorted(counts.filled_features, key=lambda n: (-counts.filled_features[n], n))
            shown = names[:MAX_LOGGED_FEATURE_NAMES]
            logger.info(
                "予想の特徴量の0埋め: run=%s market=%s horizon=%d 0埋めの銘柄数=%d/%d "
                "特徴量の数=%d 名前=%s（ほか%d件）",
                self.label,
                market,
                horizon,
                counts.filled_symbols,
                counts.fill_total,
                len(names),
                ",".join(shown) if shown else "なし",
                len(names) - len(shown),
            )


def _percent(part: int, total: int) -> float:
    return 100.0 * part / total if total else 0.0


_active: List[PredictionRunStats] = []
_active_lock = Lock()


@contextmanager
def collect_prediction_run_stats(label: str) -> Iterator[PredictionRunStats]:
    """囲んだ間の予想のクリップと0埋めを数え、終わりにログに出す"""
    stats = PredictionRunStats(label)
    with _active_lock:
        _active.append(stats)
    try:
        yield stats
    finally:
        with _active_lock:
            _active.remove(stats)
        try:
            stats.log_summary()
        except Exception:
            logger.error("予想のクリップと0埋めのログに失敗", exc_info=True)


def _current() -> List[PredictionRunStats]:
    with _active_lock:
        return list(_active)


def record_clip(market: str, horizon: int, raw_ratio: float, max_ratio: Optional[float]) -> None:
    """集めている回があれば、クリップの前の変化率と上限の幅を数える"""
    for stats in _current():
        stats.record_clip(market, horizon, raw_ratio, max_ratio)


def record_zero_fill(market: str, horizon: int, features: Sequence[str]) -> None:
    """集めている回があれば、1銘柄で0で埋めた特徴量を数える"""
    for stats in _current():
        stats.record_zero_fill(market, horizon, features)
