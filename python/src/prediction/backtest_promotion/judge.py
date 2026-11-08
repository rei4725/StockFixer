"""
バックテストの勝ちと、仮の入れ替えのあとの戻しの判定（純粋関数）。

候補・production の物差しは baseline_metrics() の dict。
向きだけの的中率は判定に使わない。値が None の物差しはその条件の負けとして扱う。
"""

from __future__ import annotations

from typing import Optional

from src.prediction.constant_prediction import CONSTANT_PREDICTION_STD, is_constant_prediction
from src.prediction.types import RollbackDecision, WinDecision

HOLDOUT_TRADING_DAYS = 20  # 取っておく期間（営業日）
MIN_COVERAGE = 0.8  # 各モデルの n_baseline が expected_rows に対して要る割合
WATCH_TRADING_DAYS = 5  # 入れ替え後に見張る採点済みの営業日数
ROLLBACK_HIT_MARGIN = 0.03  # 実測がバックテストの候補よりこれ以上低いと見張りに当たる
UP_RATE_LOW = 0.1
UP_RATE_HIGH = 0.9
PATH_CHECK_SAMPLE = 50  # 道の一致の検査に使う最大銘柄数
PATH_CHECK_TOL = 1e-4  # 道の一致の許容差（diff_ratio の絶対値）

_EPS = 1e-12


def _get(metrics: dict, key: str) -> Optional[float]:
    value = metrics.get(key)
    return None if value is None else float(value)


def judge_win(
    candidate: dict,
    production: dict,
    daily_diff: list[float],
    expected_rows: int,
) -> WinDecision:
    """候補が production に勝ったかを判定する。

    Args:
        candidate: 候補の baseline_metrics()
        production: production の baseline_metrics()
        daily_diff: 日ごとの（候補 - production）の hit_rate_minus_majority
        expected_rows: 採点の行の見込み（市場ごとの y のある日の数 × 銘柄の数の合計）

    Returns:
        WinDecision（落ちた条件ごとに reasons へ1つ）
    """
    reasons: list[str] = []

    need = expected_rows * MIN_COVERAGE
    for label, m in (("候補", candidate), ("production", production)):
        n = _get(m, "n_baseline")
        if n is None or n < need:
            reasons.append(f"{label}の採点行が足りない（n_baseline={n}、必要 {need:g} 以上）")

    c_hit = _get(candidate, "hit_rate_minus_majority")
    p_hit = _get(production, "hit_rate_minus_majority")
    if c_hit is None or p_hit is None or not c_hit > p_hit:
        reasons.append(
            f"基準を引いた的中率が production を上回らない（候補 {c_hit}、production {p_hit}）"
        )

    c_mae = _get(candidate, "mae")
    p_mae = _get(production, "mae")
    if c_mae is None or p_mae is None or not c_mae < p_mae:
        reasons.append(f"MAE が production より小さくない（候補 {c_mae}、production {p_mae}）")

    c_rc = _get(candidate, "rank_corr")
    p_rc = _get(production, "rank_corr")
    if c_rc is None or p_rc is None or not (c_rc > p_rc and c_rc > 0):
        reasons.append(
            f"順位相関が production を上回り0より大きい、を満たさない（候補 {c_rc}、production {p_rc}）"
        )

    wins = sum(1 for d in daily_diff if d > 0)
    if not daily_diff or not wins * 2 > len(daily_diff):
        reasons.append(f"日ごとの勝ち越しが過半数に届かない（{wins}/{len(daily_diff)} 日）")

    # 順位相関は定数の予想では微小な揺れの順位を測るだけなので、ばらつきそのものを見る（#1602）
    c_std = _get(candidate, "prediction_std")
    if is_constant_prediction(c_std):
        reasons.append(
            f"候補の予想が全銘柄でほぼ同じ値（標準偏差 {c_std}、線 {CONSTANT_PREDICTION_STD}）"
        )

    return WinDecision(won=not reasons, reasons=reasons)


def judge_rollback(
    live: dict,
    backtest_candidate: dict,
    old_same_window: Optional[dict],
    new_same_window: Optional[dict] = None,
) -> RollbackDecision:
    """入れ替え後の実測（採点済み5営業日）から、戻すかを判定する。

    見張りの条件のどれかに当たり、かつ同じ5日・同じ行の旧 production の
    hit_rate_minus_majority と rank_corr がどちらも、同じ行の新 production を上回るときだけ戻す（#1729）。
    取っておいた20日の旧 production の数字とは比べない（窓が違うと相場の向きで上下が決まる）。
    実測の rank_corr が None のときは 0.0 として扱う（予想が一定で順位が付かない場合を通さない）。
    それ以外の実測の値が None の条件は当たりにしない（判断材料が無いまま戻さない）。
    旧 production を測れない（old_same_window が None）か、どちらかの値が None なら
    old_is_better は False（戻さない）。

    Args:
        live: 新 production の実測（5日の全行）の baseline_metrics()。見張りの条件に使う
        backtest_candidate: バックテストで測った候補（今の production）の物差し
        old_same_window: 同じ5日・同じ行の旧 production の baseline_metrics()。測れなければ None
        new_same_window: old_same_window と同じ行の新 production の baseline_metrics()。
            None なら live を使う
    """
    triggered: list[str] = []

    live_hit = _get(live, "hit_rate_minus_majority")
    bt_hit = _get(backtest_candidate, "hit_rate_minus_majority")
    if (
        live_hit is not None
        and bt_hit is not None
        and bt_hit - live_hit >= ROLLBACK_HIT_MARGIN - _EPS
    ):
        triggered.append(
            f"実測の的中率がバックテストより {ROLLBACK_HIT_MARGIN} 以上低い（実測 {live_hit}、バックテスト {bt_hit}）"
        )

    live_rc = _get(live, "rank_corr")
    if live_rc is None:
        live_rc = 0.0  # 順位相関が出ない（予想が一定など）のは見張りを通さない
    if live_rc <= 0:
        triggered.append(f"実測の順位相関が0以下（{live_rc}）")

    up_rate = _get(live, "predicted_up_rate")
    if up_rate is not None and (up_rate > UP_RATE_HIGH or up_rate < UP_RATE_LOW):
        triggered.append(f"上がる予想の割合が偏っている（{up_rate}）")

    new = live if new_same_window is None else new_same_window
    new_hit = _get(new, "hit_rate_minus_majority")
    new_rc = _get(new, "rank_corr")
    if new_rc is None:
        new_rc = 0.0  # 実測と同じく、順位が付かないのは 0 とする
    old = old_same_window or {}
    old_hit = _get(old, "hit_rate_minus_majority")
    old_rc = _get(old, "rank_corr")
    old_is_better = (
        old_hit is not None
        and old_rc is not None
        and new_hit is not None
        and old_hit > new_hit
        and old_rc > new_rc
    )

    return RollbackDecision(
        rollback=bool(triggered) and old_is_better,
        triggered=triggered,
        old_is_better=old_is_better,
    )
