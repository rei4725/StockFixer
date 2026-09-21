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
    対象ユニバースは「現在 DB に存在する銘柄」であり、過去に上場廃止・倒産
    した銘柄は含まれない。したがって本バックテストの倍率分布・リターンは
    生存者バイアスにより**楽観方向に歪む**。結果は上限寄りの目安として解釈すること。

import について:
    backtest BC から screening BC の純粋関数（`screen_trend_candidates` /
    `simulate_position`）を参照する。既存のBC間依存であり、.importlinter の
    layers/independence 両契約に ignore_imports として明示登録済み（#638）。
    将来はBT専用ポート経由に切り出す想定（ロードマップ分類 [D]）。
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.backtest.execution import ExecutionModel
from src.backtest.longterm.config import LongtermBacktestConfig
from src.backtest.longterm.metrics import compute_longterm_metrics
from src.backtest.longterm.portfolio import OpenPosition, Portfolio
from src.backtest.longterm.prices import (
    build_calendar,
    close_lookups,
    load_price_map,
    make_rescreen_dates,
)
from src.backtest.metrics import fetch_benchmark_returns
from src.screening.hold_engine import simulate_position
from src.screening.trend_screener import screen_trend_candidates
from src.utils.logger import get_logger

logger = get_logger(__name__)

TRADE_COLUMNS = [
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


def group_events_by_date(events: list) -> dict[str, list]:
    """entry を除く PositionEvent を date ごとにまとめる。"""
    out: dict[str, list] = {}
    for ev in events:
        if ev.action == "entry":
            continue
        out.setdefault(ev.date, []).append(ev)
    return out


def _peak_multiple(
    price_map: dict[str, pd.DataFrame],
    symbol: str,
    entry_date: str,
    entry_price: float,
    exit_date: str,
    exit_price: float,
) -> float:
    """entry〜exit 区間の最高終値 / entry_price（到達倍率）を返す。

    区間に価格が 1 本も無い場合は exit_price を最高値とみなす（従来挙動）。
    """
    df = price_map[symbol]
    window = df[(df["date"] >= entry_date) & (df["date"] <= exit_date)]
    peak = float(window["Close"].max()) if not window.empty else exit_price
    return peak / entry_price if entry_price > 0 else 0.0


def enter_candidates(
    config: LongtermBacktestConfig,
    portfolio: Portfolio,
    date: str,
    price_map: dict[str, pd.DataFrame],
    execution: ExecutionModel,
) -> None:
    """リスクリーン日 date で空き枠に新規エントリーする。

    現金は `portfolio.cash` を読み、`portfolio.enter` が減算する。
    """
    empty_slots = config.max_positions - len(portfolio.positions)
    cash = portfolio.cash
    if empty_slots <= 0 or cash <= 0:
        return

    candidates = screen_trend_candidates(market=config.market, top_n=config.top_n, as_of=date)
    new_syms = [
        c for c in candidates if c.symbol not in portfolio.positions and c.symbol in price_map
    ][:empty_slots]
    if not new_syms:
        return

    # 空き枠で均等配分（候補が枠より少なければ余剰現金は温存）。
    # per_position はループ前の現金から一度だけ決め、ループ中は更新しない。
    per_position = cash / empty_slots
    for cand in new_syms:
        symbol = cand.symbol
        # price_map は load_price_map が SQL 側で end まで切り詰め済み。
        series = price_map[symbol]
        entry_row = series[series["date"] == date]
        if entry_row.empty:
            continue
        entry_price = float(entry_row["Close"].iloc[0])
        if entry_price <= 0 or per_position <= 0:
            continue

        # 予算 per_position で買える整数株数。端株は買わず、余剰は現金として温存する。
        shares = execution.max_affordable_qty(per_position, entry_price)
        if shares <= 0:
            continue
        cost = execution.buy_cost(shares, entry_price)

        # イベント生成を enter の前に済ませることで、空イベント時の
        # 現金巻き戻しが不要になる（simulate_position は現金に依存しない）。
        events = simulate_position(series, entry_date=date, rules=config.rules)
        if not events:
            continue

        portfolio.enter(
            OpenPosition(
                symbol=symbol,
                entry_date=date,
                entry_price=entry_price,
                shares=shares,
                current_hf=1.0,
                cost_basis=cost,
                events_by_date=group_events_by_date(events),
            )
        )


def _empty_result(
    config: LongtermBacktestConfig,
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """対象データが無いときの空の戻り値を組み立てる。"""
    empty_equity = pd.DataFrame(columns=["date", "portfolio_value"])
    empty_trades = pd.DataFrame(columns=TRADE_COLUMNS)
    benchmark = fetch_benchmark_returns(config.benchmark_ticker, config.start, config.end)
    return (
        empty_equity,
        compute_longterm_metrics(empty_equity, empty_trades, config, benchmark),
        empty_trades,
    )


def _trades_frame(portfolio: Portfolio) -> pd.DataFrame:
    """確定済み ClosedTrade を従来の 9 列の trades_df に整形する。"""
    rows: list[dict[str, Any]] = [
        {
            "symbol": t.symbol,
            "entry_date": t.entry_date,
            "exit_date": t.exit_date,
            "entry_price": round(t.entry_price, 4),
            "exit_price": round(t.exit_price, 4),
            "multiple": round(t.multiple, 4),
            "max_multiple": round(t.max_multiple, 4),
            "exit_reason": t.exit_reason,
            "held_days": t.held_days,
        }
        for t in portfolio.closed
    ]
    return pd.DataFrame(rows, columns=TRADE_COLUMNS)


def run_longterm_backtest(
    config: LongtermBacktestConfig,
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """長期コホート・バックテストを実行する。

    手順（ルックアヘッド厳禁）:
        1. start〜end を rescreen_freq で区切りリスクリーン日リストを作る。
        2. 各リスクリーン日 t で t 以前のデータのみで screen_trend_candidates を実行。
        3. 空き枠（max_positions 未満）に上位候補から新規エントリー。
        4. 各ポジションを simulate_position で評価し撤退/利確イベントを得る。
        5. 空き現金を均等配分し、手数料とスリッページを売買に課金（ExecutionModel）。
           撤退で現金回収→再投資。
        6. 日次でポートフォリオ評価額を集計し equity_df を作る。

    ⚠️ 生存者バイアス: ユニバースは現存銘柄のみのため結果は楽観方向に歪む。

    Returns:
        (equity_df, metrics, trades_df)
        equity_df: columns = [date, portfolio_value]（日付昇順）
        trades_df: 1行=1ポジション
            [symbol, entry_date, exit_date, entry_price, exit_price,
             multiple, max_multiple, exit_reason, held_days]
    """
    execution = ExecutionModel(config.costs)

    price_map = load_price_map(config.market, config.end)
    calendar = build_calendar(price_map, config.start, config.end)
    if not calendar:
        logger.warning("バックテスト対象データがありません (market=%s)", config.market)
        return _empty_result(config)

    rescreen_dates = set(make_rescreen_dates(calendar, config.start, config.rescreen_freq))
    lookups = close_lookups(price_map, calendar)

    portfolio = Portfolio(config.initial_cash)
    equity_rows: list[dict[str, Any]] = []

    for date in calendar:
        # 1) 当日発生のイベント（利確・撤退）を処理して現金回収
        for symbol in list(portfolio.positions.keys()):
            pos = portfolio.positions[symbol]
            for ev in pos.events_by_date.get(date, []):
                if ev.action == "scale_out":
                    portfolio.scale_out(symbol, ev, execution)
                elif ev.action == "exit":
                    max_multiple = _peak_multiple(
                        price_map,
                        symbol,
                        pos.entry_date,
                        pos.entry_price,
                        ev.date,
                        ev.price,
                    )
                    portfolio.close(symbol, ev, execution, max_multiple)
                    # 撤退後の同日イベントは保有が無いので処理しない。
                    break

        # 2) リスクリーン日なら空き枠にエントリー
        if date in rescreen_dates:
            enter_candidates(config, portfolio, date, price_map, execution)

        # 3) 日次ポートフォリオ評価額
        equity_rows.append(
            {"date": date, "portfolio_value": round(portfolio.equity(lookups, date), 2)}
        )

    equity_df = pd.DataFrame(equity_rows)
    trades_df = _trades_frame(portfolio)

    benchmark = fetch_benchmark_returns(config.benchmark_ticker, config.start, config.end)
    metrics = compute_longterm_metrics(equity_df, trades_df, config, benchmark)
    return equity_df, metrics, trades_df
