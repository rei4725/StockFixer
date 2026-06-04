"""
月次KPIレポート実行スクリプト（R-203 CLIラッパー）

月次 KPI サマリー（Net Return / Sharpe / MDD / Hit Rate / Avg Slippage）を
Walk-Forward レポートおよび DuckDB から集計して標準出力へ表示する。
"""

import argparse
import sys

from src.reporting.monthly import run_monthly_report
from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="月次KPIレポートを生成する")
    parser.add_argument(
        "--month",
        type=str,
        default=None,
        help="対象年月 YYYY-MM（省略時は当月）",
    )
    return parser.parse_args()


def main():
    from src.orchestration.port_wiring import wire_ports

    wire_ports()
    args = parse_args()
    summary = run_monthly_report(target_month=args.month)

    def _fmt_pct(val):
        return f"{val * 100:.2f}%" if val is not None else "データなし"

    def _fmt_f2(val):
        return f"{val:.4f}" if val is not None else "データなし"

    print(f"\n=== 月次KPIレポート {summary.target_month} ===\n")
    print("[最重要KPI]")
    print(f"  Net Return   : {_fmt_pct(summary.net_return)}")
    print(f"  Max Drawdown : {_fmt_pct(summary.max_drawdown)}")
    print(f"  Sharpe Ratio : {_fmt_f2(summary.sharpe_ratio)}")
    print("")
    print("[補助KPI]")
    print(f"  Hit Rate     : {_fmt_pct(summary.hit_rate)}  (直近30日)")
    print(f"  Avg Slippage : {_fmt_pct(summary.avg_slippage)}  (直近30日)")
    print("")
    print("[メタ情報]")
    print(f"  対象銘柄数   : {summary.symbol_count or 'N/A'}")
    print(f"  WFスナップ   : {summary.wf_snapshot_file or 'N/A'}")
    print(f"  生成日時     : {summary.generated_at[:19]}")
    print("")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"月次レポート生成に失敗しました: {e}", exc_info=True)
        sys.exit(1)
