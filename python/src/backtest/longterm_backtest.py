"""長期コホート・バイ&ホールド・バックテスト本体（Stage 1）

過去の各時点でトレンド・スクリーン(#429)を実行→上位候補にエントリー→
保有/撤退エンジン(#430 `simulate_position`)で数年ホールド、をシミュレートし、
「この規律なら過去に何倍株を何件捕まえ、ポートフォリオとして何倍・最大DD何%
だったか」を実データで測る。

ルックアヘッド禁止:
    各リスクリーン日 t のスクリーンは t 以前のデータのみを使う
    （`screen_trend_candidates(as_of=t)`）。ポジションのその後の評価は
    `simulate_position` がトレーリングで行うため未来データを先読みしない。

⚠️ 生存者バイアスについて:
    対象ユニバースは「現在 DuckDB に存在する銘柄」であり、過去に上場廃止・倒産
    した銘柄は含まれない。したがって本バックテストの倍率分布・リターンは
    生存者バイアスにより**楽観方向に歪む**。結果は上限寄りの目安として解釈すること。

import について:
    backtest BC から screening BC の純粋関数（`screen_trend_candidates` /
    `simulate_position`）を参照する。screening は .importlinter の独立性契約の
    modules に含まれないため契約違反にはならない（lint-imports contracts kept）。
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd

from src.backtest.metrics import _max_drawdown, fetch_benchmark_returns
from src.screening.hold_engine import simulate_position
from src.screening.trend_screener import screen_trend_candidates
from src.screening.types import HoldRules
from src.utils.data_path_utils import ensure_dir, get_results_dir
from src.utils.db.stock_features import get_all_symbols, load_stock_features
from src.utils.logger import get_logger

logger = get_logger(__name__)

# rescreen_freq -> pandas DateOffset（リスクリーン間隔）
_FREQ_OFFSETS: dict[str, pd.DateOffset] = {
    "weekly": pd.DateOffset(weeks=1),
    "monthly": pd.DateOffset(months=1),
    "quarterly": pd.DateOffset(months=3),
    "yearly": pd.DateOffset(years=1),
    "annual": pd.DateOffset(years=1),
}


def _load_price_map(market: str, end: str) -> dict[str, pd.DataFrame]:
    """market の全銘柄について date / Close を持つ価格系列を読み込む。

    end までに切り詰める（バックテスト窓外を評価しないため）。
    """
    price_map: dict[str, pd.DataFrame] = {}
    for m, symbol in get_all_symbols():
        if m != market:
            continue
        df = load_stock_features(m, symbol)
        if df is None or df.empty or "date" not in df.columns or "Close" not in df.columns:
            continue
        df = df[["date", "Close"]].copy()
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        df = df[df["date"] <= end].sort_values("date").reset_index(drop=True)
        if not df.empty:
            price_map[symbol] = df
    return price_map


def _build_calendar(price_map: dict[str, pd.DataFrame], start: str, end: str) -> list[str]:
    """全銘柄の取引日の和集合を [start, end] で昇順に並べたカレンダーを作る。"""
    dates: set[str] = set()
    for df in price_map.values():
        dates.update(df["date"].tolist())
    cal = sorted(d for d in dates if start <= d <= end)
    return cal


def _make_rescreen_dates(calendar: list[str], start: str, freq: str) -> list[str]:
    """リスクリーン日リストを作る。

    start から freq 間隔でターゲット日を生成し、各ターゲット日以降で最初に存在する
    取引日に丸める。カレンダー外（末尾超過）のターゲットは無視する。
    """
    if not calendar:
        return []
    offset = _FREQ_OFFSETS.get(freq, _FREQ_OFFSETS["quarterly"])
    last = pd.Timestamp(calendar[-1])
    target = pd.Timestamp(max(start, calendar[0]))

    out: list[str] = []
    seen: set[str] = set()
    while target <= last:
        target_str = target.strftime("%Y-%m-%d")
        # target 以降で最初の取引日
        nxt = next((d for d in calendar if d >= target_str), None)
        if nxt is not None and nxt not in seen:
            out.append(nxt)
            seen.add(nxt)
        target = target + offset
    return out


def _close_lookups(
    price_map: dict[str, pd.DataFrame], calendar: list[str]
) -> dict[str, dict[str, float]]:
    """銘柄ごとに calendar 上で前方補完した date->Close を作る（保有評価用）。"""
    lookups: dict[str, dict[str, float]] = {}
    for symbol, df in price_map.items():
        s = df.set_index("date")["Close"].astype(float)
        s = s.reindex(calendar).ffill()
        lookups[symbol] = s.to_dict()
    return lookups


def _events_by_date(events: list) -> dict[str, list]:
    """entry を除く PositionEvent を date ごとにまとめる。"""
    out: dict[str, list] = {}
    for ev in events:
        if ev.action == "entry":
            continue
        out.setdefault(ev.date, []).append(ev)
    return out


def _make_trade(pos: dict[str, Any], price_map: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """確定したポジションから trades_df の1行を組み立てる。

    max_multiple は entry〜exit 区間の最高終値 / entry_price（到達倍率）。
    multiple は exit_price / entry_price（最終実現倍率）。
    """
    symbol = pos["symbol"]
    entry_date = pos["entry_date"]
    exit_date = pos["exit_date"]
    entry_price = pos["entry_price"]

    df = price_map[symbol]
    window = df[(df["date"] >= entry_date) & (df["date"] <= exit_date)]
    peak = float(window["Close"].max()) if not window.empty else pos["exit_price"]
    max_multiple = peak / entry_price if entry_price > 0 else 0.0

    held_days = (pd.Timestamp(exit_date) - pd.Timestamp(entry_date)).days
    return {
        "symbol": symbol,
        "entry_date": entry_date,
        "exit_date": exit_date,
        "entry_price": round(entry_price, 4),
        "exit_price": round(pos["exit_price"], 4),
        "multiple": round(pos["exit_price"] / entry_price if entry_price > 0 else 0.0, 4),
        "max_multiple": round(max_multiple, 4),
        "exit_reason": pos["exit_reason"],
        "held_days": held_days,
    }


def _try_enter(
    date: str,
    market: str,
    top_n: int,
    max_positions: int,
    fee_rate: float,
    rules: HoldRules,
    end: str,
    cash: float,
    open_positions: dict[str, dict[str, Any]],
    price_map: dict[str, pd.DataFrame],
) -> float:
    """リスクリーン日 date で空き枠に新規エントリーする。更新後の cash を返す。"""
    empty_slots = max_positions - len(open_positions)
    if empty_slots <= 0 or cash <= 0:
        return cash

    candidates = screen_trend_candidates(market=market, top_n=top_n, as_of=date)
    new_syms = [c for c in candidates if c.symbol not in open_positions and c.symbol in price_map][
        :empty_slots
    ]
    if not new_syms:
        return cash

    # 空き枠で均等配分（候補が枠より少なければ余剰現金は温存）
    per_position = cash / empty_slots
    for cand in new_syms:
        symbol = cand.symbol
        series = price_map[symbol]
        series = series[series["date"] <= end]
        entry_row = series[series["date"] == date]
        if entry_row.empty:
            continue
        entry_price = float(entry_row["Close"].iloc[0])
        if entry_price <= 0 or per_position <= 0:
            continue

        shares = per_position * (1.0 - fee_rate) / entry_price
        cash -= per_position

        events = simulate_position(series, entry_date=date, rules=rules)
        if not events:
            cash += per_position
            continue

        open_positions[symbol] = {
            "symbol": symbol,
            "entry_date": date,
            "entry_price": entry_price,
            "original_shares": shares,
            "current_hf": 1.0,
            "events_by_date": _events_by_date(events),
        }
    return cash


def run_longterm_backtest(
    market: str = "us",
    start: str = "2021-01-01",
    end: str = "2026-01-01",
    rescreen_freq: str = "quarterly",
    top_n: int = 30,
    initial_cash: float = 1_000_000.0,
    max_positions: int = 10,
    fee_rate: float = 0.001,
    benchmark_ticker: str = "^GSPC",
    rules: Optional[HoldRules] = None,
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """長期コホート・バックテストを実行する。

    手順（ルックアヘッド厳禁）:
        1. start〜end を rescreen_freq で区切りリスクリーン日リストを作る。
        2. 各リスクリーン日 t で t 以前のデータのみで screen_trend_candidates を実行。
        3. 空き枠（max_positions 未満）に上位候補から新規エントリー。
        4. 各ポジションを simulate_position で評価し撤退/利確イベントを得る。
        5. 空き現金を均等配分し fee_rate を売買に課金。撤退で現金回収→再投資。
        6. 日次でポートフォリオ評価額を集計し equity_df を作る。

    ⚠️ 生存者バイアス: ユニバースは現存銘柄のみのため結果は楽観方向に歪む。

    Returns:
        (equity_df, metrics, trades_df)
        equity_df: columns = [date, portfolio_value]（日付昇順）
        trades_df: 1行=1ポジション
            [symbol, entry_date, exit_date, entry_price, exit_price,
             multiple, max_multiple, exit_reason, held_days]
    """
    rules = rules or HoldRules()

    price_map = _load_price_map(market, end)
    calendar = _build_calendar(price_map, start, end)
    if not calendar:
        logger.warning("バックテスト対象データがありません (market=%s)", market)
        empty_equity = pd.DataFrame(columns=["date", "portfolio_value"])
        empty_trades = pd.DataFrame(
            columns=[
                "symbol",
                "entry_date",
                "exit_date",
                "entry_price",
                "exit_price",
                "multiple",
                "max_multiple",
                "exit_reason",
                "held_days",
            ]
        )
        benchmark = fetch_benchmark_returns(benchmark_ticker, start, end)
        return (
            empty_equity,
            _compute_metrics({}, empty_trades, initial_cash, start, end, benchmark),
            empty_trades,
        )  # noqa: E501

    rescreen_dates = set(_make_rescreen_dates(calendar, start, rescreen_freq))
    close_lookups = _close_lookups(price_map, calendar)

    cash = initial_cash
    open_positions: dict[str, dict[str, Any]] = {}
    closed_trades: list[dict[str, Any]] = []
    equity_rows: list[dict[str, Any]] = []

    for date in calendar:
        # 1) 当日発生のイベント（利確・撤退）を処理して現金回収
        for symbol in list(open_positions.keys()):
            pos = open_positions[symbol]
            for ev in pos["events_by_date"].get(date, []):
                if ev.action == "scale_out":
                    delta = pos["current_hf"] - ev.held_fraction
                    if delta > 0:
                        sold = pos["original_shares"] * delta
                        cash += sold * ev.price * (1.0 - fee_rate)
                        pos["current_hf"] = ev.held_fraction
                elif ev.action == "exit":
                    sold = pos["original_shares"] * pos["current_hf"]
                    cash += sold * ev.price * (1.0 - fee_rate)
                    pos["current_hf"] = 0.0
                    pos["exit_date"] = ev.date
                    pos["exit_price"] = ev.price
                    pos["exit_reason"] = ev.reason
                    closed_trades.append(_make_trade(pos, price_map))
                    del open_positions[symbol]

        # 2) リスクリーン日なら空き枠にエントリー
        if date in rescreen_dates:
            cash = _try_enter(
                date,
                market,
                top_n,
                max_positions,
                fee_rate,
                rules,
                end,
                cash,
                open_positions,
                price_map,
            )

        # 3) 日次ポートフォリオ評価額
        held_value = sum(
            p["original_shares"] * p["current_hf"] * close_lookups[p["symbol"]].get(date, 0.0)
            for p in open_positions.values()
        )
        equity_rows.append({"date": date, "portfolio_value": round(cash + held_value, 2)})

    equity_df = pd.DataFrame(equity_rows)
    trades_df = pd.DataFrame(
        closed_trades,
        columns=[
            "symbol",
            "entry_date",
            "exit_date",
            "entry_price",
            "exit_price",
            "multiple",
            "max_multiple",
            "exit_reason",
            "held_days",
        ],
    )

    benchmark = fetch_benchmark_returns(benchmark_ticker, start, end)
    metrics = _compute_metrics(equity_df, trades_df, initial_cash, start, end, benchmark)
    return equity_df, metrics, trades_df


def _compute_metrics(
    equity_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    initial_cash: float,
    start: str,
    end: str,
    benchmark: dict,
) -> dict:
    """equity_df / trades_df から計測指標を組み立てる。"""
    if isinstance(equity_df, pd.DataFrame) and not equity_df.empty:
        final_value = float(equity_df["portfolio_value"].iloc[-1])
        equity_series = equity_df.set_index("date")["portfolio_value"]
        max_dd = _max_drawdown(equity_series)
    else:
        final_value = initial_cash
        max_dd = 0.0

    total_return = (final_value - initial_cash) / initial_cash if initial_cash > 0 else 0.0

    years = max((pd.Timestamp(end) - pd.Timestamp(start)).days / 365.25, 1e-9)
    cagr = (final_value / initial_cash) ** (1.0 / years) - 1.0 if final_value > 0 else -1.0

    n_trades = len(trades_df)
    if n_trades > 0:
        mm = trades_df["max_multiple"]
        n_2x = int((mm >= 2.0).sum())
        n_3x = int((mm >= 3.0).sum())
        n_5x = int((mm >= 5.0).sum())
        n_10x = int((mm >= 10.0).sum())

        realized = trades_df["multiple"]
        wins = trades_df[realized >= 1.0]
        losses = trades_df[realized < 1.0]
        win_rate = len(wins) / n_trades
        avg_win_multiple = float(wins["multiple"].mean()) if not wins.empty else 0.0
        avg_loss = float(losses["multiple"].mean() - 1.0) if not losses.empty else 0.0
        avg_held_days = float(trades_df["held_days"].mean())
    else:
        n_2x = n_3x = n_5x = n_10x = 0
        win_rate = avg_win_multiple = avg_loss = avg_held_days = 0.0

    bench_return = benchmark.get("total_return") if isinstance(benchmark, dict) else None
    alpha = (total_return - bench_return) if bench_return is not None else None

    return {
        "initial_cash": round(initial_cash, 2),
        "final_cash": round(final_value, 2),
        "total_return": round(total_return, 6),
        "cagr": round(cagr, 6),
        "max_drawdown": round(max_dd, 6),
        "n_trades": n_trades,
        "n_2x": n_2x,
        "n_3x": n_3x,
        "n_5x": n_5x,
        "n_10x": n_10x,
        "pct_2x": round(n_2x / n_trades, 4) if n_trades else 0.0,
        "pct_3x": round(n_3x / n_trades, 4) if n_trades else 0.0,
        "pct_5x": round(n_5x / n_trades, 4) if n_trades else 0.0,
        "pct_10x": round(n_10x / n_trades, 4) if n_trades else 0.0,
        "win_rate": round(win_rate, 4),
        "avg_win_multiple": round(avg_win_multiple, 4),
        "avg_loss": round(avg_loss, 6),
        "avg_held_days": round(avg_held_days, 1),
        "benchmark_ticker": benchmark.get("ticker") if isinstance(benchmark, dict) else None,
        "benchmark_return": bench_return,
        "alpha": round(alpha, 6) if alpha is not None else None,
    }


def save_results(equity_df: pd.DataFrame, trades_df: pd.DataFrame, market: str) -> tuple[str, str]:
    """equity / trades を results/backtest/longterm/ に CSV 保存しパスを返す。"""
    out_dir = ensure_dir(f"{get_results_dir()}/backtest/longterm")
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    equity_path = f"{out_dir}/equity_{market}_{ts}.csv"
    trades_path = f"{out_dir}/trades_{market}_{ts}.csv"
    equity_df.to_csv(equity_path, index=False)
    trades_df.to_csv(trades_path, index=False)
    logger.info("結果を保存: %s / %s", equity_path, trades_path)
    return equity_path, trades_path


def build_conclusion(metrics: dict, market: str, start: str, end: str, max_positions: int) -> str:
    """metrics から結論文を組み立てる（サマリ出力用）。"""
    initial = metrics.get("initial_cash", 0.0)
    final = metrics.get("final_cash", 0.0)
    mult = final / initial if initial > 0 else 0.0
    total_ret = metrics.get("total_return", 0.0)
    cagr = metrics.get("cagr", 0.0)
    max_dd = metrics.get("max_drawdown", 0.0)
    bench = metrics.get("benchmark_return")
    alpha = metrics.get("alpha")

    bench_part = (
        f"ベンチマーク（{metrics.get('benchmark_ticker')}: {bench:+.1%}）に対し "
        f"{alpha:+.1%}pt。"
        if bench is not None and alpha is not None
        else "ベンチマーク比較はデータ取得失敗のため省略。"
    )
    return (
        f"{market} を {start}〜{end} に四半期スクリーン・最大{max_positions}銘柄保有・"
        f"40週線割れまでホールドの規律で運用した場合、"
        f"資産は {initial:,.0f}円→{final:,.0f}円（{mult:.2f}倍, 総リターン {total_ret:+.1%}, "
        f"CAGR {cagr:+.1%}）、最大DD {max_dd:.1%}。"
        f"期間中に 2倍到達 {metrics.get('n_2x', 0)}件 / 3倍 {metrics.get('n_3x', 0)}件 / "
        f"5倍 {metrics.get('n_5x', 0)}件 / 10倍 {metrics.get('n_10x', 0)}件。"
        f"{bench_part}"
        " ※ユニバースは現存銘柄のみのため生存者バイアスで楽観方向に歪む点に注意。"
    )
