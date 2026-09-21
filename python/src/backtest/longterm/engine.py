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

from bisect import bisect_left
from typing import Any, Optional

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
from src.screening.types import TrendCandidate
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


def resolve_entry_date(calendar: list[str], screen_date: str, lag: int) -> Optional[str]:
    """スクリーン日から lag 営業日後の約定日を返す。

    screen_date がカレンダーに存在しない場合、またはカレンダー末尾を越える
    場合は None（そのリスクリーン日はエントリーを見送る）。

    lag が負数の場合は ValueError を送出する（`calendar[i + lag]` が
    Python の負インデックスとして末尾から silently 解決されてしまい、
    「未来の約定日」を意図せず返してしまう事故を防ぐため）。
    """
    if lag < 0:
        raise ValueError(f"execution_lag は 0 以上である必要があります: {lag}")
    i = bisect_left(calendar, screen_date)
    if i >= len(calendar) or calendar[i] != screen_date:
        return None
    j = i + lag
    return calendar[j] if j < len(calendar) else None


def enter_candidates(
    config: LongtermBacktestConfig,
    portfolio: Portfolio,
    entry_date: str,
    candidates: list[TrendCandidate],
    price_map: dict[str, pd.DataFrame],
    execution: ExecutionModel,
) -> None:
    """entry_date に、事前にスクリーン済みの候補リストから空き枠へ約定する。

    候補選定（`screen_trend_candidates(as_of=screen_date)`）は呼び出し側が
    リスクリーン日の時点で済ませ、ここでは受け取った候補（スコア降順）を
    entry_date 時点の状態（空き枠・現金）で埋めるだけを行う。
    リスクリーン日と約定日を分離した上で「空き枠に入る/入らない」「幾ら
    ずつ配分するか」を entry_date 時点の最新状態で再計算することで、
    途中の撤退・別エントリーで枠や現金が変化していてもポートフォリオが
    max_positions を超えて埋まることはない（呼び出しごとに毎回計算し直す
    ため、スクリーン時点の空き枠を固定して使い回すことはしない）。

    現金は `portfolio.cash` を読み、`portfolio.enter` が減算する。
    """
    empty_slots = config.max_positions - len(portfolio.positions)
    cash = portfolio.cash
    if empty_slots <= 0 or cash <= 0:
        return

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
        entry_row = series[series["date"] == entry_date]
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
        events = simulate_position(series, entry_date=entry_date, rules=config.rules)
        if not events:
            continue

        portfolio.enter(
            OpenPosition(
                symbol=symbol,
                entry_date=entry_date,
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
           結果は当日エントリーせず、t + execution_lag 営業日（`resolve_entry_date`）を
           キーにキュー（`pending_entries`）へ積む。
        3. ループが約定日に到達したら、その日の状態（空き枠・現金）で
           空き枠（max_positions 未満）に上位候補から新規エントリーする
           （`enter_candidates`）。screen_date 時点の枠数を固定して使い回すと
           約定日までの撤退・別エントリーで状況が変わっていても過去の空き枠数
           のまま埋めてしまい max_positions を超過しうるため、埋める直前に
           必ず再計算する。
        4. 各ポジションを simulate_position で評価し撤退/利確イベントを得る。
        5. 空き現金を均等配分し、手数料とスリッページを売買に課金（ExecutionModel）。
           撤退で現金回収→再投資。
        6. 日次でポートフォリオ評価額を集計し equity_df を作る（現金の減算・
           ポジションの計上は約定日にのみ起きるため、リスクリーン日当日の
           equity に未来の約定価格が先読みで混ざることはない）。

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
    # screen_date -> entry_date は resolve_entry_date が
    # 「カレンダー上のインデックス + lag」で決めるため screen_date ごとに
    # 一意な entry_date になり、異なる screen_date が同じキーに衝突すること
    # はない。
    pending_entries: dict[str, list[TrendCandidate]] = {}

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

        # 2) リスクリーン日ならスクリーンし、約定日（execution_lag 営業日後）
        #    をキーに候補をキューへ積む。ここではまだ現金もポジションも
        #    動かさない（当日の equity に未来の約定価格を混ぜないため）。
        if date in rescreen_dates:
            entry_date = resolve_entry_date(calendar, date, config.execution_lag)
            if entry_date is not None:
                pending_entries[entry_date] = screen_trend_candidates(
                    market=config.market, top_n=config.top_n, as_of=date
                )

        # 3) 本日が約定日として積まれていれば、その時点の空き枠・現金で約定する。
        queued = pending_entries.pop(date, None)
        if queued:
            enter_candidates(config, portfolio, date, queued, price_map, execution)

        # 4) 日次ポートフォリオ評価額
        equity_rows.append(
            {"date": date, "portfolio_value": round(portfolio.equity(lookups, date), 2)}
        )

    equity_df = pd.DataFrame(equity_rows)
    trades_df = _trades_frame(portfolio)

    benchmark = fetch_benchmark_returns(config.benchmark_ticker, config.start, config.end)
    metrics = compute_longterm_metrics(equity_df, trades_df, config, benchmark)
    return equity_df, metrics, trades_df
