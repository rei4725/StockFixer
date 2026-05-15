"""
運用監視ダッシュボード CLI（R-303）

損益・ドリフト・paper/real 乖離・実験結果を横断表示する read-only CLI。
将来の C#/Web UI にも流用しやすいよう各セクションを独立した集計関数に分離している。

使い方:
    from src.reporting.dashboard import run_dashboard
    run_dashboard(recent_days=30, drift_n=20)
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Optional

import pandas as pd
from tabulate import tabulate  # type: ignore[import-untyped]

from src.reporting.monthly import run_monthly_report
from src.utils.db import load_drift_summary, load_experiment_runs, load_paper_real_diff_summary
from src.utils.db._connection import _db_connection
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ドリフト閾値
_DRIFT_ERROR_THRESHOLD = 0.02
_DRIFT_ACCURACY_THRESHOLD = 0.45


def _now_jst_str() -> str:
    """現在時刻を JST 表示の文字列で返す。"""
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S") + " JST"


# ---------------------------------------------------------------------------
# セクション別集計関数（将来の Web API にも流用可能）
# ---------------------------------------------------------------------------


def _section_monthly_kpi() -> Optional[list[list]]:
    """月次 KPI を表形式の行リストで返す。なければ None。"""
    summary = run_monthly_report()

    def _pct(val: Optional[float]) -> str:
        return f"{val * 100:.2f}%" if val is not None else "N/A"

    def _f2(val: Optional[float]) -> str:
        return f"{val:.4f}" if val is not None else "N/A"

    return [
        ["Net Return", _pct(summary.net_return)],
        ["Max Drawdown", _pct(summary.max_drawdown)],
        ["Sharpe Ratio", _f2(summary.sharpe_ratio)],
        ["Hit Rate", _pct(summary.hit_rate)],
        ["Avg Slippage", _pct(summary.avg_slippage)],
        ["集計銘柄数", str(summary.symbol_count) if summary.symbol_count is not None else "N/A"],
        ["対象年月", summary.target_month],
        ["WF スナップショット", summary.wf_snapshot_file or "N/A"],
    ]


def _section_drift(drift_n: int) -> tuple[list[list], int]:
    """
    ドリフト超過銘柄の行リストと超過件数を返す。

    Returns:
        (rows, exceeded_count)
        rows: tabulate 用の行リスト
        exceeded_count: 閾値超過銘柄数
    """
    df = load_drift_summary(horizon=1, recent_n=drift_n)
    if df is None or df.empty:
        return [], 0

    error_mask = (
        df["mean_abs_error"] >= _DRIFT_ERROR_THRESHOLD
        if "mean_abs_error" in df.columns
        else pd.Series(False, index=df.index)
    )
    acc_mask = (
        df["direction_accuracy"] <= _DRIFT_ACCURACY_THRESHOLD
        if "direction_accuracy" in df.columns
        else pd.Series(False, index=df.index)
    )
    exceeded = df[error_mask | acc_mask]
    rows = []
    for _, row in exceeded.iterrows():
        rows.append(
            [
                row.get("market", ""),
                row.get("symbol", ""),
                f"{row.get('mean_abs_error', float('nan')):.4f}",
                f"{row.get('direction_accuracy', float('nan')):.4f}",
                int(row.get("n_samples", 0)),
            ]
        )
    return rows, len(exceeded)


def _section_paper_real_diff(recent_days: int) -> list[list]:
    """paper/real 乖離サマリーを表形式の行リストで返す。"""
    d = load_paper_real_diff_summary(recent_days=recent_days)
    return [
        ["追跡件数", str(d["tracked_count"])],
        ["比較可能件数", str(d["comparable_count"])],
        ["Paper Slippage 平均", f"{d['avg_paper_slippage'] * 100:.4f}%"],
        ["Real Slippage 平均", f"{d['avg_real_slippage'] * 100:.4f}%"],
        ["価格乖離率 平均 (abs)", f"{d['avg_abs_diff_ratio'] * 100:.4f}%"],
        ["価格乖離 最大 (abs)", f"¥{d['max_abs_price_diff']:.2f}"],
    ]


def _section_paper_balance() -> list[list]:
    """
    paper トレードの残高・損益サマリーを行リストで返す。

    実現損益: paper_orders.realized_pnl の合計（filled 注文のみ）
    含み損益（空売り）: paper_short_positions.unrealized_pnl の合計
    ロングポジションの含み損益はリアルタイム価格なしでは算出不可のため表示しない。
    """
    with _db_connection() as con:
        bal_row = con.execute("SELECT balance FROM paper_balance LIMIT 1").fetchone()
        balance = float(bal_row[0]) if bal_row else 0.0

        realized_row = con.execute(
            "SELECT SUM(realized_pnl) FROM paper_orders "
            "WHERE status = 'filled' AND realized_pnl IS NOT NULL"
        ).fetchone()
        realized_pnl = (
            float(realized_row[0]) if realized_row and realized_row[0] is not None else 0.0
        )

        short_row = con.execute(
            "SELECT SUM(unrealized_pnl) FROM paper_short_positions "
            "WHERE unrealized_pnl IS NOT NULL"
        ).fetchone()
        short_unrealized = float(short_row[0]) if short_row and short_row[0] is not None else 0.0

        open_positions_row = con.execute("SELECT COUNT(*) FROM paper_positions").fetchone()
        open_count = int(open_positions_row[0]) if open_positions_row else 0

    total_pnl = realized_pnl + short_unrealized

    return [
        ["残高 (現金)", f"¥{balance:,.0f}"],
        ["実現損益", f"¥{realized_pnl:,.0f}"],
        ["含み損益（空売り）", f"¥{short_unrealized:,.0f}"],
        ["合計損益", f"¥{total_pnl:,.0f}"],
        ["ロングポジション数", str(open_count)],
    ]


def _section_model_accuracy(limit: int) -> list[list]:
    """
    experiment_runs から直近の実験結果をモデル別に集計し行リストで返す。

    モデル名でグループ化し、平均 directional_accuracy / rmse を表示する。
    """
    df = load_experiment_runs(limit=limit)
    if df is None or df.empty:
        return []

    req_cols = {"model_name", "directional_accuracy", "rmse"}
    if not req_cols.issubset(df.columns):
        return []

    grp = (
        df.groupby("model_name")
        .agg(
            avg_da=pd.NamedAgg(column="directional_accuracy", aggfunc="mean"),
            avg_rmse=pd.NamedAgg(column="rmse", aggfunc="mean"),
            count=pd.NamedAgg(column="model_name", aggfunc="count"),
        )
        .reset_index()
        .sort_values("avg_da", ascending=False)
    )

    rows = []
    for _, row in grp.iterrows():
        da = row["avg_da"]
        rmse = row["avg_rmse"]
        da_str = f"{da:.4f}" if da is not None and not math.isnan(float(da)) else "N/A"
        rmse_str = f"{rmse:.6f}" if rmse is not None and not math.isnan(float(rmse)) else "N/A"
        rows.append([row["model_name"], da_str, rmse_str, int(row["count"])])
    return rows


# ---------------------------------------------------------------------------
# メインエントリ
# ---------------------------------------------------------------------------


def run_dashboard(recent_days: int = 30, drift_n: int = 20) -> None:
    """
    運用監視ダッシュボードを標準出力に表示する（read-only）。

    各セクションは独立した try/except でラップされているため、
    1 セクション失敗しても残りのセクションを表示し続ける。

    Args:
        recent_days: paper/real 乖離・月次 KPI の集計期間（日）
        drift_n: ドリフト判定に使う直近サンプル数（銘柄ごと）
    """
    print("=" * 60)
    print("  StockFixer 運用監視ダッシュボード")
    print("=" * 60)
    print(f"生成日時: {_now_jst_str()}")
    print(f"集計期間: 直近 {recent_days} 日 / ドリフト直近 {drift_n} 件")
    print()

    # --- [1] 月次 KPI ---
    print("--- [1] 月次 KPI ---")
    try:
        rows = _section_monthly_kpi()
        if rows:
            print(tabulate(rows, headers=["指標", "値"], tablefmt="simple"))
        else:
            print("  データなし")
    except Exception as e:
        logger.error("月次 KPI 取得失敗: %s", e, exc_info=True)
        print(f"  取得失敗: {e}")
    print()

    # --- [2] ドリフト状況 ---
    print(f"--- [2] ドリフト状況（直近 {drift_n} 件）---")
    print(
        f"  閾値: mean_abs_error >= {_DRIFT_ERROR_THRESHOLD}"
        f" または direction_accuracy <= {_DRIFT_ACCURACY_THRESHOLD}"
    )
    try:
        rows, count = _section_drift(drift_n)
        if count == 0:
            print("  閾値超過なし")
        else:
            print(f"  閾値超過銘柄数: {count}")
            print(
                tabulate(
                    rows,
                    headers=["market", "symbol", "mean_abs_error", "direction_accuracy", "n"],
                    tablefmt="simple",
                )
            )
    except Exception as e:
        logger.error("ドリフト取得失敗: %s", e, exc_info=True)
        print(f"  取得失敗: {e}")
    print()

    # --- [3] paper/real 乖離 ---
    print(f"--- [3] paper/real 乖離（直近 {recent_days} 日）---")
    try:
        rows = _section_paper_real_diff(recent_days)
        print(tabulate(rows, headers=["指標", "値"], tablefmt="simple"))
    except Exception as e:
        logger.error("paper/real 乖離取得失敗: %s", e, exc_info=True)
        print(f"  取得失敗: {e}")
    print()

    # --- [4] ペーパートレード残高・損益 ---
    print("--- [4] ペーパートレード残高・損益 ---")
    try:
        rows = _section_paper_balance()
        print(tabulate(rows, headers=["項目", "値"], tablefmt="simple"))
    except Exception as e:
        logger.error("ペーパー残高取得失敗: %s", e, exc_info=True)
        print(f"  取得失敗: {e}")
    print()

    # --- [5] 最近のモデル精度 ---
    _exp_limit = max(recent_days * 5, 100)
    print(f"--- [5] 最近のモデル精度（直近 {_exp_limit} 件の experiment_runs から集計）---")
    try:
        rows = _section_model_accuracy(limit=_exp_limit)
        if rows:
            print(
                tabulate(
                    rows,
                    headers=["model_name", "avg_direction_acc", "avg_rmse", "runs"],
                    tablefmt="simple",
                )
            )
        else:
            print("  データなし")
    except Exception as e:
        logger.error("モデル精度取得失敗: %s", e, exc_info=True)
        print(f"  取得失敗: {e}")
    print()

    print("=" * 60)
