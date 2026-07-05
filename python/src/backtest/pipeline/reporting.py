"""バックテスト結果の保存・表示・ベンチマーク比較。

CSV 保存、メトリクス出力、グラフ描画、ベンチマーク取得など
実行結果のレポーティング系ユーティリティ。
"""

import os
from datetime import datetime
from typing import Any, Optional

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def save_backtest_results(
    result_df: Optional[pd.DataFrame],
    metrics: Optional[dict[str, Any]],
    wf_df: Optional[pd.DataFrame],
    market: str,
    symbol: str,
    task_name: str,
) -> None:
    """
    バックテスト結果を CSV に保存する。

    Args:
        result_df: 取引ログ DataFrame（単一期間モード）
        metrics: メトリクス辞書（単一期間モード）
        wf_df: Walk-Forward 結果 DataFrame
        market: マーケット識別子
        symbol: 銘柄シンボル
        task_name: タスク名
    """
    from src.utils.data_path_utils import ensure_dir, get_results_dir

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(get_results_dir(), "backtest", f"{market}_{symbol}")
    ensure_dir(out_dir)

    if wf_df is not None and not wf_df.empty:
        path = os.path.join(out_dir, f"wf_{task_name}_{ts}.csv")
        wf_df.to_csv(path, index=False)
        print(f"\n結果保存: {path}")
    elif result_df is not None and not result_df.empty:
        path = os.path.join(out_dir, f"trades_{task_name}_{ts}.csv")
        result_df.to_csv(path, index=False)
        print(f"\n取引ログ保存: {path}")


def print_backtest_metrics(
    metrics: dict[str, Any], label: str = "", benchmark: Optional[dict[str, Any]] = None
) -> None:
    """
    バックテストメトリクスを標準出力に表示する。

    Args:
        metrics: compute_metrics が返す辞書
        label: ヘッダーラベル
        benchmark: fetch_benchmark_returns が返す辞書（オプション）
    """
    if not metrics:
        return
    print(f"\n{'='*50}")
    if label:
        print(f" {label}")
    print(f"{'='*50}")

    # net（手数料・スリッページ控除後）
    print("  [NET] 手数料・スリッページ控除後")
    print(f"  {'final_cash':20s}: {metrics.get('final_cash')}")
    print(f"  {'total_return':20s}: {metrics.get('total_return')}")
    print(f"  {'sharpe_ratio':20s}: {metrics.get('sharpe_ratio')}")
    print(f"  {'max_drawdown':20s}: {metrics.get('max_drawdown')}")
    print(f"  {'num_trades':20s}: {metrics.get('num_trades')}")
    print(f"  {'win_rate':20s}: {metrics.get('win_rate')}")
    print(f"  {'profit_factor':20s}: {metrics.get('profit_factor')}")
    if "avg_position_fraction" in metrics:
        print(f"  {'avg_position_fraction':20s}: {metrics.get('avg_position_fraction')}")
        print(f"  {'max_position_fraction':20s}: {metrics.get('max_position_fraction')}")
        print(f"  {'avg_position_value':20s}: {metrics.get('avg_position_value')}")
        print(f"  {'atr_fallback_trades':20s}: {metrics.get('atr_fallback_trades')}")

    # gross（比較用: コスト控除前）
    if "gross_total_return" in metrics:
        print(f"{'─'*50}")
        print("  [GROSS] コスト控除前（同一約定数量ベース）")
        print(f"  {'gross_final_cash':20s}: {metrics.get('gross_final_cash')}")
        print(f"  {'gross_total_return':20s}: {metrics.get('gross_total_return')}")
        print(f"  {'gross_sharpe_ratio':20s}: {metrics.get('gross_sharpe_ratio')}")
        print(f"  {'gross_max_drawdown':20s}: {metrics.get('gross_max_drawdown')}")
        print(f"  {'cost_impact_cash':20s}: {metrics.get('cost_impact_cash')}")
        print(f"  {'cost_impact_return':20s}: {metrics.get('cost_impact_return')}")

    if "mc_final_cash_p50" in metrics:
        print(f"{'─'*50}")
        print("  [Monte Carlo] 取引損益ブートストラップ (n=1000)")
        print(f"  {'max_drawdown_mean':20s}: {metrics.get('mc_max_drawdown_mean')}")
        print(f"  {'max_drawdown_p95':20s}: {metrics.get('mc_max_drawdown_p95')}")
        print(f"  {'final_cash_p05':20s}: {metrics.get('mc_final_cash_p05')}")
        print(f"  {'final_cash_p50':20s}: {metrics.get('mc_final_cash_p50')}")
        print(f"  {'final_cash_p95':20s}: {metrics.get('mc_final_cash_p95')}")

    if benchmark and benchmark.get("total_return") is not None:
        bm_ret = benchmark["total_return"]
        strategy_ret = metrics.get("total_return", 0.0)
        alpha = strategy_ret - bm_ret
        print(f"{'─'*50}")
        print(
            f"  {'ベンチマーク':20s}: {benchmark['ticker']} ({benchmark['start']} ～ {benchmark['end']})"
        )
        print(f"  {'BM リターン':20s}: {bm_ret:.4%}")
        print(f"  {'アルファ':20s}: {alpha:+.4%}")
    print(f"{'='*50}")


def plot_backtest_chart(
    result_df: pd.DataFrame,
    metrics: Optional[dict[str, Any]],
    market: str,
    symbol: str,
    initial_cash: float,
    file_notifier=None,
) -> None:
    """
    バックテスト結果グラフを保存し、オプションで通知する。

    Args:
        result_df: バックテスト結果 DataFrame
        metrics: メトリクス辞書
        market: マーケット識別子
        symbol: 銘柄シンボル
        initial_cash: 初期資金
        file_notifier: ``(path: str, title: str) -> None`` の呼び出し可能オブジェクト。
            None の場合は通知しない。呼び出し元（orchestration 等）で注入する。
    """
    from src.backtest.metrics import plot_backtest
    from src.utils.data_path_utils import get_results_dir

    out_dir = os.path.join(get_results_dir(), "backtest", f"{market}_{symbol}")
    chart_path = plot_backtest(
        result_df,
        metrics or {},
        out_dir,
        market=market,
        symbol=symbol,
        initial_cash=initial_cash,
    )
    if chart_path:
        logger.info(f"グラフ保存: {chart_path}")
        if file_notifier is not None:
            file_notifier(chart_path, f"{market.upper()}/{symbol} バックテスト結果")


def fetch_benchmark_for_result(
    result_df: pd.DataFrame,
    benchmark_name: str,
    fallback_start: Optional[str] = None,
    fallback_end: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    """
    バックテスト結果 DataFrame からベンチマーク比較データを取得する。

    Args:
        result_df: バックテスト結果 DataFrame（date 列推奨）
        benchmark_name: ベンチマーク識別子 ("n225", "sp500" など)
        fallback_start: result_df に date 列がない場合の開始日
        fallback_end: result_df に date 列がない場合の終了日

    Returns:
        fetch_benchmark_returns が返す辞書、または None
    """
    from src.backtest.metrics import BENCHMARK_TICKERS, fetch_benchmark_returns

    bm_ticker = BENCHMARK_TICKERS.get(benchmark_name, benchmark_name)
    bm_start = str(result_df["date"].min())[:10] if "date" in result_df.columns else fallback_start
    bm_end = str(result_df["date"].max())[:10] if "date" in result_df.columns else fallback_end
    if bm_start and bm_end:
        return fetch_benchmark_returns(bm_ticker, bm_start, bm_end)
    return None
