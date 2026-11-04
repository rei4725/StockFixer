"""シミュレーションの各足で使う約定の入力（シグナル・確信度・平均出来高・ATR）を作る。

Backtester から抽出（File Size Guard 対応・#1547）。ロジック自体の変更はなし。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd

# 動的スリッページの平均出来高算出ウィンドウ（バー数）— #494
_AVG_VOL_WINDOW = 20


@dataclass(frozen=True)
class ExecutionInputs:
    """約定の足で使う系列。"""

    exec_col: str
    avg_vol: pd.Series
    exec_avg_vol: pd.Series
    exec_signal: pd.Series
    exec_pred: Optional[pd.Series]
    exec_atr: Optional[pd.Series]


def build_execution_inputs(
    df: pd.DataFrame,
    signal: pd.Series,
    pred: Optional[pd.Series],
    close_col: str,
    execution_lag: int,
) -> ExecutionInputs:
    """実行ラグに合わせて、約定の時点で分かっている値だけの系列を作る。"""
    # #494: 動的スリッページ用の平均出来高系列（Volume 列が無ければ全て0でフラットのみ適用）
    if "Volume" in df.columns:
        avg_vol = df["Volume"].rolling(_AVG_VOL_WINDOW, min_periods=1).mean()
    else:
        avg_vol = pd.Series(0.0, index=df.index)
    # #493: 実行ラグ。シグナル/確信度を execution_lag バー遅延させ、翌バーで約定する。
    # 約定価格は翌バー始値(Open)。Open 列が無ければ Close にフォールバック。
    # SL/TP の判定と mark-to-market は当バー Close 基準。SL/TP の約定は翌バー始値（#964）。
    # 最終バーまでに約定先バーが無いシグナルは shift により自然に破棄される。
    exec_col = "Open" if (execution_lag > 0 and "Open" in df.columns) else close_col
    if execution_lag > 0:
        exec_signal = signal.reindex(df.index).shift(execution_lag)
        exec_pred = pred.reindex(df.index).shift(execution_lag) if pred is not None else None
    else:
        exec_signal = signal
        exec_pred = pred
    # シグナル約定（execution_lag > 0 なら翌バー始値）のコスト・サイジングには、
    # 約定時点で既知の前バーまでの平均出来高を使う（当バー出来高の先読みを防ぐ）。
    # 先頭バーは前バーが無いため 0（フラットのみ適用）。SL/TP の翌バー決済も同じ exec_avg_vol を使う。
    if execution_lag > 0:
        exec_avg_vol = avg_vol.shift(1).fillna(0.0)
    else:
        exec_avg_vol = avg_vol
    # #800: ATR サイジングも約定時点で既知の前バーまでの ATR を使う。
    # atr_lag1 があれば各行の前バー ATR としてそのまま使い（file/api/raw で同じ意味）、
    # 無ければ当バー高安引けを含む atr を 1 本ずらす。同バー約定は当バー atr のまま。
    exec_atr: Optional[pd.Series] = None
    if execution_lag > 0:
        if "atr_lag1" in df.columns:
            exec_atr = df["atr_lag1"]
        elif "atr" in df.columns:
            exec_atr = df["atr"].shift(1)
    elif "atr" in df.columns:
        exec_atr = df["atr"]
    return ExecutionInputs(exec_col, avg_vol, exec_avg_vol, exec_signal, exec_pred, exec_atr)
