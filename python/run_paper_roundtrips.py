"""紙の売買を往復（買いから売り）で組み、入った道別・月別にまとめて表示する（#1817）

読むだけ。paper_orders に書き込まない。
"""

import sys

from src.reporting.paper_roundtrips import (
    build_roundtrips,
    count_phantom_sells,
    load_paper_roundtrip_orders,
    summarize_roundtrips,
)
from src.utils.db import get_readonly_connection
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    con = get_readonly_connection()
    try:
        orders = load_paper_roundtrip_orders(con)
    finally:
        con.close()
    trips, open_lots, unmatched = build_roundtrips(orders)
    summary = summarize_roundtrips(trips)

    print("\n=== 紙の往復 道別 ===")
    for g in summary.by_route:
        print(
            f"  {g.route:<10} 往復 {g.trips:>4}  勝ち {g.wins:>4}  勝率 {g.win_rate * 100:5.1f}%"
            f"  損益 {g.pnl:>14,.0f}  平均保有 {g.avg_holding_days:5.1f}日"
        )
    print("\n=== 紙の往復 道と月 ===")
    for g in summary.by_month:
        print(f"  {g.route:<10} {g.key}  往復 {g.trips:>4}  勝ち {g.wins:>4}  損益 {g.pnl:>14,.0f}")
    print(f"\n  往復の損益の合計: {summary.total_pnl:,.0f}")
    print(f"  開いたままの買い: {len(open_lots)} 件 {sum(lot.qty for lot in open_lots)} 株")
    print(f"  株数を動かさなかった売り（realized_pnl なし）: {count_phantom_sells(orders)} 件")
    print(
        f"  買いの残りを超えた売り: {sum(unmatched.values())} 株 {dict(sorted(unmatched.items()))}"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"紙の往復の報告に失敗しました: {e}", exc_info=True)
        sys.exit(1)
