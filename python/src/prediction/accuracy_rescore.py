"""
足が直った後の、直近の採点の行の付け直し（#2155）

日次の採点は予想ごとに1回だけで、途中の足の終値で採点した行は、足が後で直っても古い実績のまま
残る。直近の行を読み、実績の足の今の終値が保存した実績と違う行だけ、実績・実績の変化率・
方向の的中を付け直す。基準の日・終値の欠けた足・記録待ちの分割の扱いは run_accuracy_check と
同じ部品を使い、予想の値は変えない。
"""

import math
from datetime import datetime

import numpy as np
import pandas as pd

from src.domain.split_steps import pending_split_steps
from src.prediction.base_bar_date import _actual_ratio_and_match, _base_bar_date
from src.prediction.db import load_prediction_accuracy, save_prediction_accuracy
from src.utils.db.market_data import load_raw_closes
from src.utils.logger import get_logger

logger = get_logger(__name__)

# 付け直しを見る期間（predicted_at の日付の新しい方からの日数）
RESCORE_DAYS = 10
# 足は DOUBLE で保存した値のまま比べるので、浮動小数の誤差（float32 の丸めの約 1e-7）だけを許す
_SAME_CLOSE_TOLERANCE = 1e-6
# 基準の終値も同じ足から予想に渡した値で、正しい行は 1e-6 の内で合う（本番で確かめた）。広げると
# 途中の足を基準にした予想のうち、確定の終値に近い行が混ざる。分割・併合の直しもここで外れる
_BASE_CLOSE_TOLERANCE = 1e-6
# 一番古い基準の日から足を読み始めるまでの余白（連休で基準の足が前にずれても届く日数）
_BAR_MARGIN = pd.Timedelta(days=20)


def _is_close(value: float, reference: float, tolerance: float) -> bool:
    return math.isclose(value, reference, rel_tol=tolerance, abs_tol=0.0)


def _load_close_bars(market: str, oldest_base: datetime) -> dict[str, pd.DataFrame]:
    """market の銘柄ごとの、終値のある足（index=日付、Close）を1回の読み込みで返す。"""
    df = load_raw_closes(market, start_date=oldest_base - _BAR_MARGIN, raise_on_error=True)
    df["close"] = df["close"].astype(float)  # 0件の表は object の列になるので型をそろえる
    df = df[np.isfinite(df["close"])].rename(columns={"close": "Close"})
    return {str(symbol): bars.set_index("ts")[["Close"]] for symbol, bars in df.groupby("symbol")}


def _rescore_market(
    market: str, rows: pd.DataFrame, horizon: int, counts: dict[str, int]
) -> list[dict]:
    """1つの市場の採点の行のうち、付け直した行を返す。付け直さない理由は counts に数える。"""
    bars_by_symbol = _load_close_bars(market, rows["base_day"].min())
    fixed: list[dict] = []
    for row in rows.to_dict("records"):
        bars = bars_by_symbol.get(str(row["symbol"]))
        pos = 0 if bars is None else int(bars.index.searchsorted(row["base_day"], side="right"))
        if bars is None or pos == 0 or pos + horizon > len(bars):
            counts["足なし"] += 1
            continue

        predicted_price, predicted_ratio = row["predicted_price"], row["predicted_ratio"]
        # 予想の時の値段（保存していない）を、保存した予想の値段と変化率から戻す
        current_price = predicted_price / (1 + predicted_ratio)
        if not _is_close(current_price, float(bars["Close"].iloc[pos - 1]), _BASE_CLOSE_TOLERANCE):
            counts["基準の足が合わない"] += 1
            continue

        actual_pos = pos + horizon - 1
        actual_price = float(bars["Close"].iloc[actual_pos])
        if _is_close(actual_price, row["actual_price"], _SAME_CLOSE_TOLERANCE):
            continue
        if pending_split_steps(bars.iloc[pos - 1 : actual_pos + 1], bars.index[-1]):
            counts["記録待ちの分割の段差"] += 1
            continue

        actual_ratio, direction_match = _actual_ratio_and_match(
            current_price, predicted_ratio, actual_price
        )
        fixed.append(
            {
                "market": market,
                "symbol": row["symbol"],
                "model_name": row["model_name"],
                "predicted_at": row["predicted_at"],
                "horizon": horizon,
                "predicted_price": predicted_price,
                "actual_price": actual_price,
                "predicted_ratio": predicted_ratio,
                "actual_ratio": actual_ratio,
                "direction_match": direction_match,
            }
        )
    return fixed


def rescore_fixed_bars(model_name: str, horizon: int = 1, days: int = RESCORE_DAYS) -> int:
    """
    直近の採点の行のうち、実績の足の終値が変わった行を付け直して保存する。

    Args:
        model_name: prediction_accuracy のモデル名ラベル（production / challenger）
        horizon: 対象のホライズン
        days: predicted_at の日付の新しい方から何日分を見るか

    Returns:
        付け直した行の数

    Raises:
        Exception: 採点の行・足の読み込みか、保存に落ちたとき（日次の [2/6] の知らせに乗せる）
    """
    rows = load_prediction_accuracy(
        horizon=horizon, limit=0, model_name=model_name, days=days, raise_on_error=True
    )
    n_read = len(rows)
    # 実績が空の行（記録待ちの分割 #2035）と、予想の時の値段を戻せない行は見ない
    values = rows[["predicted_price", "predicted_ratio", "actual_price"]].astype(float)
    rows = rows[np.isfinite(values).all(axis=1) & (values["predicted_ratio"] != -1)].copy()

    counts = {"足なし": 0, "基準の足が合わない": 0, "記録待ちの分割の段差": 0}
    fixed: list[dict] = []
    if not rows.empty:
        rows["base_day"] = [
            _base_bar_date(str(predicted_at), str(market))
            for predicted_at, market in zip(rows["predicted_at"], rows["market"])
        ]
        for market, market_rows in rows.groupby("market"):
            fixed.extend(_rescore_market(str(market), market_rows, horizon, counts))

    if fixed:
        save_prediction_accuracy(fixed)
    logger.info(
        "採点の付け直し (%s): 付け直し=%d件 (読み込み=%d件, 見た行=%d件, %s)",
        model_name,
        len(fixed),
        n_read,
        len(rows),
        ", ".join(f"{reason}={count}件" for reason, count in counts.items()),
    )
    return len(fixed)
