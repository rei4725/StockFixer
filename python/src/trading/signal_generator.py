from __future__ import annotations

import numpy as np
import pandas as pd

from src.domain.trading_rules import CONFLUENCE_BOOST_PER_HORIZON as _CONFLUENCE_BOOST_PER_HORIZON
from src.domain.trading_rules import DEFAULT_HORIZON_WEIGHTS as _DEFAULT_HORIZON_WEIGHTS
from src.utils.logger import get_logger
from src.utils.regime_weights import REGIME_SECTOR_WEIGHTS as REGIME_SECTOR_WEIGHTS  # noqa: F401
from src.utils.regime_weights import (  # noqa: F401
    get_regime_sector_weight as get_regime_sector_weight,
)
from src.utils.signal_generator import SignalGenerator as SignalGenerator  # noqa: F401

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# R-212: マルチホライズン統合シグナルスコア
# ---------------------------------------------------------------------------


def compute_multi_horizon_score(
    diff_ratio_1d: float,
    diff_ratio_3d: float | None = None,
    diff_ratio_5d: float | None = None,
    diff_ratio_10d: float | None = None,
    confluence_score: int | None = None,
    weights: tuple[float, float, float, float] = _DEFAULT_HORIZON_WEIGHTS,
) -> float:
    """
    1d/3d/5d/10d の予測変化率を重み付けで統合したシグナルスコアを返す。

    利用できないホライズンは除外して残りの重みを正規化する。
    confluence_score が高い（複数ホライズンで方向が一致）ほどスコアを増幅する。

    Args:
        diff_ratio_1d: 翌日予測変化率（必須）
        diff_ratio_3d: 3営業日後予測変化率（省略可）
        diff_ratio_5d: 5営業日後予測変化率（省略可）
        diff_ratio_10d: 10営業日後予測変化率（省略可）
        confluence_score: 1d 予測と同方向のホライズン数（0〜3）
        weights: (w_1d, w_3d, w_5d, w_10d) の重みタプル

    Returns:
        統合シグナルスコア（diff_ratio と同じ値域スケール）
    """
    candidate_pairs = [
        (diff_ratio_1d, weights[0]),
        (diff_ratio_3d, weights[1]),
        (diff_ratio_5d, weights[2]),
        (diff_ratio_10d, weights[3]),
    ]
    # None のホライズンを除外
    valid = [(r, w) for r, w in candidate_pairs if r is not None]
    if not valid:
        return diff_ratio_1d

    total_w = sum(w for _, w in valid)
    score = sum(r * w / total_w for r, w in valid)

    # confluence ボーナス: 方向一致ホライズンが多いほどスコアを強調
    if confluence_score is not None and confluence_score > 0:
        boost = 1.0 + confluence_score * _CONFLUENCE_BOOST_PER_HORIZON
        score *= boost

    return float(score)


def apply_multi_horizon_score_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    予測結果 DataFrame に ``multi_horizon_score`` 列を付与して返す。

    ``diff_ratio_3d/5d/10d`` や ``confluence_score`` が存在しない列でも動作する。

    Args:
        df: prediction_results から取得した DataFrame

    Returns:
        multi_horizon_score 列が追加されたコピー
    """
    result = df.copy()

    def _row_score(row: pd.Series) -> float:
        return compute_multi_horizon_score(
            diff_ratio_1d=float(row["diff_ratio"]),
            diff_ratio_3d=_opt_float(row, "diff_ratio_3d"),
            diff_ratio_5d=_opt_float(row, "diff_ratio_5d"),
            diff_ratio_10d=_opt_float(row, "diff_ratio_10d"),
            confluence_score=_opt_int(row, "confluence_score"),
        )

    result["multi_horizon_score"] = result.apply(_row_score, axis=1)
    return result


def _opt_float(row: pd.Series, col: str) -> float | None:
    if col not in row.index:
        return None
    v = row[col]
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _opt_int(row: pd.Series, col: str) -> int | None:
    if col not in row.index:
        return None
    v = row[col]
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# I-256: 決算・イベントカレンダーフィルター
# ---------------------------------------------------------------------------


def filter_event_signals(
    signals: pd.Series,
    event_dates: "pd.DatetimeIndex",
    days_before: int = 2,
    days_after: int = 1,
) -> pd.Series:
    """
    イベント日（決算発表等）前後の Buy シグナルを Hold に変換する。

    event_dates は呼び出し元が market_data.event_calendar.get_event_dates() などで
    取得して渡すこと（クロスBC依存を避けるため直接インポートしない）。

    Args:
        signals: generate_signal() が返す 'Buy'/'Sell'/'Hold' の Series
        event_dates: イベント日一覧（tz-naive DatetimeIndex）
        days_before: イベント日の何日前からエントリー抑制するか（デフォルト 2）
        days_after: イベント日の何日後までエントリー抑制するか（デフォルト 1）

    Returns:
        イベント近傍の Buy を Hold に変換したシグナル Series のコピー
    """
    if len(event_dates) == 0:
        return signals.copy()

    result = signals.copy()
    event_ts = pd.DatetimeIndex([pd.Timestamp(e).normalize() for e in event_dates])

    for idx in result.index[result == "Buy"]:
        check_ts = pd.Timestamp(idx).normalize()
        for ev_ts in event_ts:
            delta = (check_ts - ev_ts).days
            if -days_before <= delta <= days_after:
                result.loc[idx] = "Hold"
                break

    return result


if __name__ == "__main__":
    # テスト用のダミーデータ
    dates = pd.to_datetime(pd.date_range(start="2023-01-01", periods=100, freq="D"))
    dummy_data = pd.DataFrame(
        {
            "Open": 100 + (np.random.rand(100) - 0.5).cumsum(),
            "High": 101 + (np.random.rand(100) - 0.5).cumsum(),
            "Low": 99 + (np.random.rand(100) - 0.5).cumsum(),
            "Close": 100 + (np.random.rand(100) - 0.5).cumsum(),
            "Volume": np.random.randint(1000, 5000, 100),
            "RSI": np.random.uniform(20, 80, 100),
        },
        index=dates,
    )

    # ダミーの予測結果 (株価変化率を想定)
    dummy_prediction = pd.Series(np.random.uniform(-0.01, 0.01, 100), index=dates)
    # ローリングSTD（直近20日）
    dummy_rolling_std = dummy_prediction.rolling(20).std()
    generator = SignalGenerator()
    signals_fixed = generator.generate_signal(dummy_data, dummy_prediction)
    signals_dynamic = generator.generate_signal(
        dummy_data, dummy_prediction, rolling_std=dummy_rolling_std
    )

    print("固定閾値シグナル:")
    print(signals_fixed.value_counts())
    print("\n動的閾値シグナル:")
    print(signals_dynamic.value_counts())
