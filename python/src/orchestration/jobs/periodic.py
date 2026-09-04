"""月次・夜間スケジューラジョブ。

月次 KPI レポートと戦略ファクトリー夜間バッチ。
"""

from typing import Optional

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


_MIN_EQUITY_POINTS = 5


def _label_with_start_date(label: str, series: pd.Series) -> str:
    """系列自身の開始日をラベルに付与する（運用開始日が系列ごとに異なるため）。"""
    start = series.dropna().index[0].strftime("%Y-%m-%d")
    return f"{label} (since {start})"


def _build_monthly_equity_series(
    paper_equity: pd.Series,
    allocation_equity: pd.Series,
    benchmark: Optional[pd.Series],
) -> dict[str, pd.Series]:
    """月次損益曲線チャート用の系列 dict を組み立てる。

    各系列は運用開始日が異なるため、正規化は開始来推移として扱い、ラベルに
    開始日を明記して読者が誤って絶対水準を比較しないようにする。
    データ点数が少ない系列（運用開始直後）はノイズとして除外する。

    前提: paper_equity は呼び出し元（run_monthly_report_job）で
    len(equity.dropna()) >= _MIN_EQUITY_POINTS を確認済みであること。
    """
    series: dict[str, pd.Series] = {
        _label_with_start_date("Paper Trading", paper_equity): paper_equity
    }
    if allocation_equity is not None and len(allocation_equity.dropna()) >= _MIN_EQUITY_POINTS:
        series[_label_with_start_date("Allocation Bot", allocation_equity)] = allocation_equity
    if benchmark is not None and len(benchmark.dropna()) >= _MIN_EQUITY_POINTS:
        series[_label_with_start_date("S&P 500", benchmark)] = benchmark
    return series


def run_monthly_report_job() -> None:
    """
    月次実行（毎月1日 03:00）: 月次KPIレポートを生成・保存・Discord通知する。

    手順:
        1. Walk-Forward KPI を集計（Net Return / MDD / Sharpe）
        2. Hit Rate / Avg Slippage を集計
        3. paper/real 乖離・ドリフト状況を含む Markdown を results/monthly/ に保存
        4. Discord に通知
    """
    from config.settings import DRIFT_ALERT_THRESHOLD, DRIFT_ALERT_WEEKS
    from src.prediction.drift_monitor import check_weekly_hit_rate_drift
    from src.reporting.discord.discord_utils import send_monthly_report_notification
    from src.reporting.monthly import run_monthly_report, save_monthly_report_to_file

    def _drift_checker():
        return check_weekly_hit_rate_drift(weeks=DRIFT_ALERT_WEEKS, threshold=DRIFT_ALERT_THRESHOLD)

    logger.info("=== 月次レポート生成開始 ===")
    try:
        summary = run_monthly_report()
        report_path = save_monthly_report_to_file(summary, drift_checker=_drift_checker)
        logger.info("=== 月次レポート保存完了: %s ===", report_path)
    except Exception as e:
        logger.error("月次レポート生成失敗: %s", e, exc_info=True)
        raise

    try:
        send_monthly_report_notification(
            target_month=summary.target_month,
            net_return=summary.net_return,
            max_drawdown=summary.max_drawdown,
            sharpe_ratio=summary.sharpe_ratio,
            hit_rate=summary.hit_rate,
            avg_slippage=summary.avg_slippage,
            symbol_count=summary.symbol_count,
            report_path=report_path,
        )
    except Exception as e:
        logger.error("月次レポート通知失敗: %s", e, exc_info=True)

    # 損益曲線比較チャート（Paper Trading / 配分戦略Bot / S&P500）を生成して送信
    # NON_CRITICAL: 失敗しても月次レポートは成立
    try:
        import os

        from src.infrastructure.yfinance_market_data_adapter import YFinanceMarketDataAdapter
        from src.market_data.backtest_adapter import BacktestMarketDataAdapter
        from src.reporting.discord.discord_utils import send_webhook_file
        from src.reporting.equity_chart import build_equity_chart
        from src.trading.allocation_strategy.equity import get_allocation_equity_curve
        from src.trading.paper_equity import get_paper_equity_curve
        from src.utils.data_path_utils import get_results_dir

        equity = get_paper_equity_curve(days=180)
        if len(equity.dropna()) < 5:
            logger.info("エクイティ系列が不足のため損益曲線チャートをスキップ")
        else:
            start = equity.index[0].strftime("%Y-%m-%d")
            end = equity.index[-1].strftime("%Y-%m-%d")
            benchmark = None
            try:
                bench_df = BacktestMarketDataAdapter().download("^GSPC", start=start, end=end)
                if bench_df is not None and not bench_df.empty and "Close" in bench_df.columns:
                    benchmark = bench_df["Close"]
            except Exception as e:
                logger.warning("S&P500 取得失敗（エクイティのみ描画）: %s", e)

            try:
                allocation_equity = get_allocation_equity_curve(
                    YFinanceMarketDataAdapter(), days=180
                )
            except Exception as e:
                logger.warning("配分戦略エクイティ取得失敗（Paper/S&P500のみ描画）: %s", e)
                allocation_equity = pd.Series(dtype=float)

            chart_dir = os.path.join(get_results_dir(), "monthly")
            os.makedirs(chart_dir, exist_ok=True)
            chart_path = os.path.join(chart_dir, f"{summary.target_month}_equity.png")
            series = _build_monthly_equity_series(equity, allocation_equity, benchmark)
            build_equity_chart(
                series, chart_path, title="Paper Trading vs Allocation Bot vs S&P 500"
            )
            send_webhook_file(chart_path, title="📈 ペーパートレード損益曲線比較（直近180日）")
            logger.info("損益曲線チャート送信完了: %s", chart_path)
    except Exception as e:
        logger.error("損益曲線チャート生成失敗: %s", e, exc_info=True)

    logger.info("=== 月次レポート完了 ===")


def run_nightly_strategy_factory(
    force: bool = False,
    market: Optional[str] = None,
    budget: Optional[int] = None,
    seed: Optional[int] = None,
) -> None:
    """
    毎日実行 (05:00): 戦略ファクトリー夜間バッチ（#369 Phase 1）

    ルール組合せ仮説をサンプリング → 窓分割バックテスト評価 → 過学習ゲート →
    合格仮説のみ results/factory/reports/ へ不変 JSON レポートを出力する。
    Issue 起票は IssueAgent（--factory-intake）の責務。

    Args:
        force: True なら FACTORY_ENABLED=false でも実行（CLI 手動実行用）
        market: 対象マーケット。None なら jp/us を日替わり交互
        budget: 仮説数。None なら FACTORY_NIGHTLY_BUDGET
        seed: サンプラーシード。None なら日付ベース
    """
    from datetime import datetime

    from config.settings import (
        FACTORY_ENABLED,
        FACTORY_LOOKBACK_YEARS,
        FACTORY_N_WINDOWS,
        FACTORY_NIGHTLY_BUDGET,
    )

    if not FACTORY_ENABLED and not force:
        logger.info(
            "戦略ファクトリーはスキップ（FACTORY_ENABLED=false。手動実行は run_strategy_factory.py）"
        )
        return

    if market is None:
        # jp/us を日替わり交互（通日の偶奇）
        market = "jp" if datetime.now().timetuple().tm_yday % 2 == 0 else "us"
    if budget is None:
        budget = FACTORY_NIGHTLY_BUDGET

    logger.info("=== 戦略ファクトリー夜間バッチ開始: market=%s budget=%s ===", market, budget)
    result = None
    try:
        from src.backtest.factory import run_factory_batch
        from src.watchlist.batch_runner import load_target_symbols

        tasks = load_target_symbols()
        symbols = [t.symbol for t in tasks if t.market == market]
        if not symbols:
            logger.warning("対象銘柄なしのため中止: market=%s", market)
            return

        result = run_factory_batch(
            market=market,
            symbols=symbols,
            budget=budget,
            lookback_years=FACTORY_LOOKBACK_YEARS,
            n_windows=FACTORY_N_WINDOWS,
            seed=seed,
        )
        logger.info(
            "=== 戦略ファクトリー完了: 評価=%s 合格=%s ===",
            len(result.candidates),
            len(result.passed),
        )
    except Exception as e:
        logger.error("戦略ファクトリー失敗: %s", e, exc_info=True)

    # Discord 完了通知
    try:
        from src.reporting.discord.discord_utils import send_factory_completion

        if result is not None:
            best = max(result.candidates, key=lambda e: e.sharpe_ratio, default=None)
            send_factory_completion(
                market=market,
                evaluated=len(result.candidates),
                passed=len(result.passed),
                champion_sharpe=result.champion_sharpe,
                pbo=result.pbo,
                best_label=best.hypothesis.label if best else "-",
                best_sharpe=best.sharpe_ratio if best else 0.0,
                report_hashes=[e.hypothesis.hypothesis_hash for e in result.passed],
            )
    except Exception as e:
        logger.error("戦略ファクトリー通知失敗: %s", e, exc_info=True)


_FACTORY_LABELS = frozenset({"strategy-factory", "strategy-factory-idea"})


def run_strategy_promotion_check(force: bool = False) -> None:
    """
    2時間ごと実行: マージ済み戦略ファクトリー由来 PR を検出し strategy_promotions に記録する。

    Args:
        force: True なら STRATEGY_PROMOTION_CHECK_ENABLED=false でも実行（CLI 手動実行用）
    """
    from datetime import datetime, timedelta

    from config.settings import STRATEGY_PROMOTION_CHECK_ENABLED
    from src.backtest.promotion_detection import (
        extract_closing_issue_numbers,
        extract_factory_hash,
        load_gate_baseline,
    )
    from src.utils.db.strategy_promotions import promotion_exists, save_strategy_promotion
    from src.utils.github_api import get_issue, list_recently_merged_pull_requests

    if not STRATEGY_PROMOTION_CHECK_ENABLED and not force:
        logger.info("戦略昇格チェックはスキップ（STRATEGY_PROMOTION_CHECK_ENABLED=false）")
        return

    logger.info("=== 戦略昇格チェック開始 ===")
    detected: list[dict] = []
    try:
        since = datetime.now().astimezone() - timedelta(days=7)
        prs = list_recently_merged_pull_requests(since=since)
        for pr in prs:
            if promotion_exists(pr["number"]):
                continue
            for issue_number in extract_closing_issue_numbers(pr["body"]):
                try:
                    issue = get_issue(issue_number)
                except Exception as e:
                    logger.warning("Issue取得失敗: #%s: %s", issue_number, e)
                    continue
                if not _FACTORY_LABELS.intersection(issue["labels"]):
                    continue
                hypothesis_hash = extract_factory_hash(issue["title"])
                if hypothesis_hash is None:
                    continue
                baseline = load_gate_baseline(hypothesis_hash)
                if baseline is None:
                    logger.warning("baseline未発見のためスキップ: hash=%s", hypothesis_hash)
                    continue
                save_strategy_promotion(
                    pr_number=pr["number"],
                    merge_commit_hash=pr["merge_commit_sha"],
                    rule_or_feature_id=hypothesis_hash,
                    pre_promotion_baseline=baseline,
                )
                detected.append(
                    {
                        "pr_number": pr["number"],
                        "rule_or_feature_id": hypothesis_hash,
                        "pre_promotion_baseline": baseline,
                    }
                )
                break  # 1PRにつき1件のみ記録（複数Issueをcloseする稀なPRは最初の一致のみ）
        logger.info("=== 戦略昇格チェック完了: 新規検出=%s ===", len(detected))
    except Exception as e:
        logger.error("戦略昇格チェック失敗: %s", e, exc_info=True)
        return

    for d in detected:
        try:
            from src.reporting.discord.discord_utils import send_strategy_promotion_detected

            send_strategy_promotion_detected(**d)
        except Exception as e:
            logger.error("戦略昇格通知失敗: %s", e, exc_info=True)


def run_allocation_rebalance_job() -> None:
    """
    配分戦略(TQQQ 80% / 短期債 20%目安、約2年ごとリバランス)のペーパートレードを実行する。

    ON/OFFフラグは設けない。自動実行させない安全装置は run_scheduler.py の
    SCHEDULE_CONFIG 側（auto_schedule: False）にあり、実行するかどうかは
    --run-now allocation_rebalance を叩く人間の判断に委ねる。
    """
    logger.info("=== 配分戦略 実行開始 ===")
    try:
        from src.infrastructure.yfinance_market_data_adapter import YFinanceMarketDataAdapter
        from src.trading.allocation_strategy.service import run_allocation_rebalance

        outcome = run_allocation_rebalance(YFinanceMarketDataAdapter())
    except Exception as e:
        logger.error("配分戦略の実行に失敗しました: %s", e, exc_info=True)
        return
    logger.info("=== 配分戦略 実行完了 ===")

    if outcome is None:
        return

    try:
        from src.reporting.discord.discord_utils import send_allocation_rebalance_report

        send_allocation_rebalance_report(
            action=outcome.action,
            tqqq_price=outcome.tqqq_price,
            shy_price=outcome.shy_price,
            tqqq_qty_before=outcome.tqqq_qty_before,
            shy_qty_before=outcome.shy_qty_before,
            cash_before=outcome.cash_before,
            tqqq_qty_after=outcome.tqqq_qty_after,
            shy_qty_after=outcome.shy_qty_after,
            cash_after=outcome.cash_after,
        )
    except Exception as e:
        logger.error("配分戦略通知失敗: %s", e, exc_info=True)


def run_regime_leverage_weekly_job() -> None:
    """
    レジームレバレッジ戦略(STRATEGY.md 7章、SPY・レバレッジ2.0倍・円建て)の
    週次判定(レジーム転換・新規エントリー)を実行する。
    """
    logger.info("=== レジームレバレッジ戦略(週次) 実行開始 ===")
    try:
        from src.infrastructure.yfinance_market_data_adapter import YFinanceMarketDataAdapter
        from src.trading.regime_leverage_strategy.service import run_regime_leverage_weekly_check

        decision = run_regime_leverage_weekly_check(YFinanceMarketDataAdapter())
        logger.info(
            "レジームレバレッジ戦略(週次): action=%s reason=%s", decision.action, decision.reason
        )
    except Exception as e:
        logger.error("レジームレバレッジ戦略(週次)の実行に失敗しました: %s", e, exc_info=True)
        return
    logger.info("=== レジームレバレッジ戦略(週次) 実行完了 ===")


def run_regime_leverage_daily_margin_job() -> None:
    """
    レジームレバレッジ戦略の日次判定(初期損切り・マージンコール)を実行する。
    未保有の場合は何もしない。
    """
    logger.info("=== レジームレバレッジ戦略(日次) 実行開始 ===")
    try:
        from src.infrastructure.yfinance_market_data_adapter import YFinanceMarketDataAdapter
        from src.trading.regime_leverage_strategy.service import (
            run_regime_leverage_daily_margin_check,
        )

        decision = run_regime_leverage_daily_margin_check(YFinanceMarketDataAdapter())
        if decision is not None:
            logger.info(
                "レジームレバレッジ戦略(日次): action=%s reason=%s maintenance_ratio=%s",
                decision.action,
                decision.reason,
                decision.maintenance_ratio,
            )
    except Exception as e:
        logger.error("レジームレバレッジ戦略(日次)の実行に失敗しました: %s", e, exc_info=True)
        return
    logger.info("=== レジームレバレッジ戦略(日次) 実行完了 ===")
