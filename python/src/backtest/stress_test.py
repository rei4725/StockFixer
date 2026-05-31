"""
ストレステストパイプライン

コロナショック / リーマンショックなど、歴史的暴落期間を対象に
バックテストを実行し MDD・シャープ・連敗数などを算出する。

run_stress_test.py はこのモジュールの関数を呼び出すラッパーとして機能する。
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Optional

import pandas as pd

from src.backtest.pipeline import run_backtest_single
from src.backtest.types import StressTestResult
from src.domain.types import SymbolTask
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# 定数
# ---------------------------------------------------------------------------

STRESS_SCENARIOS: dict[str, dict[str, str]] = {
    "corona": {
        "label": "コロナショック",
        "start_date": "2020-02-01",
        "end_date": "2020-03-31",
    },
    "lehman": {
        "label": "リーマンショック",
        "start_date": "2008-09-01",
        "end_date": "2008-10-31",
    },
}

MDD_THRESHOLD = 0.15


# ---------------------------------------------------------------------------
# 内部ヘルパー
# ---------------------------------------------------------------------------


def _compute_max_consecutive_losses(trade_log: pd.DataFrame) -> int:
    """取引ログから連続損失（連敗）の最大回数を計算する。

    Args:
        trade_log: Backtester.simulate_trading が返す DataFrame

    Returns:
        最大連敗数（取引なしの場合は 0）
    """
    if trade_log is None or trade_log.empty:
        return 0

    sell_actions = {"sell", "final_sell", "stop_loss", "take_profit"}
    buys: list[dict[str, Any]] = []
    max_streak = 0
    current_streak = 0

    for _, row in trade_log.iterrows():
        action = row.get("action", "")
        if action == "buy":
            buys.append({"price": row["price"], "qty": row["qty"]})
        elif action in sell_actions and buys:
            buy = buys.pop(0)
            pnl = (row["price"] - buy["price"]) * buy["qty"]
            if pnl < 0:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0

    return max_streak


# ---------------------------------------------------------------------------
# 公開 API
# ---------------------------------------------------------------------------


def run_stress_test_single(
    market: str,
    symbol: str,
    scenario_name: str,
    model_type: str = "XGBoostModel",
    source: str = "api",
    initial_cash: float = 1_000_000,
    mdd_threshold: float = MDD_THRESHOLD,
) -> Optional[StressTestResult]:
    """1銘柄・1シナリオのストレステストを実行する。

    Args:
        market: マーケット識別子 ("jp", "us" 等)
        symbol: 銘柄シンボル
        scenario_name: シナリオキー ("corona" / "lehman")
        model_type: モデルタイプ
        source: データソース ("api" / "raw" / "file")
        initial_cash: 初期資金
        mdd_threshold: MDD 合格閾値（デフォルト 0.15 = 15%）

    Returns:
        StressTestResult。失敗時は None。
    """
    if scenario_name not in STRESS_SCENARIOS:
        logger.error(f"[stress_test] 不明なシナリオ: {scenario_name}。対応シナリオ: {list(STRESS_SCENARIOS)}")
        return None

    scenario = STRESS_SCENARIOS[scenario_name]
    label = scenario["label"]
    start_date = scenario["start_date"]
    end_date = scenario["end_date"]

    logger.info(f"[stress_test] 開始: {market}/{symbol} シナリオ={label} ({start_date}~{end_date})")

    try:
        result_df, metrics, _ = run_backtest_single(
            market=market,
            symbol=symbol,
            model_type=model_type,
            source=source,
            start_date=start_date,
            end_date=end_date,
            initial_cash=initial_cash,
        )
    except Exception:
        logger.error(
            f"[stress_test] バックテスト失敗: {market}/{symbol} シナリオ={scenario_name}",
            exc_info=True,
        )
        return None

    mdd = metrics.get("max_drawdown", 0.0)
    sharpe = metrics.get("sharpe_ratio", 0.0)
    total_return = metrics.get("total_return", 0.0)
    win_rate = metrics.get("win_rate", 0.0)
    num_trades = metrics.get("num_trades", 0)
    max_consec = _compute_max_consecutive_losses(result_df)
    mdd_pass = abs(mdd) <= mdd_threshold

    result = StressTestResult(
        market=market,
        symbol=symbol,
        scenario_name=scenario_name,
        period_start=start_date,
        period_end=end_date,
        mdd=mdd,
        sharpe_ratio=sharpe,
        total_return=total_return,
        win_rate=win_rate,
        num_trades=num_trades,
        max_consecutive_losses=max_consec,
        mdd_pass=mdd_pass,
    )
    status = "PASS" if mdd_pass else "FAIL"
    logger.info(
        f"[stress_test] 完了: {market}/{symbol} {scenario_name} "
        f"MDD={mdd:.2%} {status}, Sharpe={sharpe:.3f}, 連敗={max_consec}"
    )
    return result


def run_stress_test_batch(
    targets: list[SymbolTask],
    scenarios: Optional[list[str]] = None,
    model_type: str = "XGBoostModel",
    source: str = "api",
    initial_cash: float = 1_000_000,
    mdd_threshold: float = MDD_THRESHOLD,
) -> list[StressTestResult]:
    """複数銘柄・複数シナリオのストレステストをバッチ実行する。

    Args:
        targets: SymbolTask のリスト
        scenarios: 実行するシナリオキーのリスト（None なら全シナリオ）
        model_type: モデルタイプ
        source: データソース
        initial_cash: 初期資金
        mdd_threshold: MDD 合格閾値

    Returns:
        StressTestResult のリスト（失敗銘柄はスキップ）
    """
    if scenarios is None:
        scenarios = list(STRESS_SCENARIOS.keys())

    results: list[StressTestResult] = []
    total = len(targets) * len(scenarios)
    done = 0

    for target in targets:
        market = target.market
        symbol = target.symbol
        for scenario_name in scenarios:
            done += 1
            logger.info(f"[stress_test] バッチ {done}/{total}: {market}/{symbol} {scenario_name}")
            result = run_stress_test_single(
                market=market,
                symbol=symbol,
                scenario_name=scenario_name,
                model_type=model_type,
                source=source,
                initial_cash=initial_cash,
                mdd_threshold=mdd_threshold,
            )
            if result is not None:
                results.append(result)

    logger.info(f"[stress_test] バッチ完了: {len(results)}/{total} 件成功")
    return results


def save_stress_test_results(
    results: list[StressTestResult],
    output_dir: str = "results/stress_test",
) -> str:
    """ストレステスト結果を CSV に保存する。

    Args:
        results: StressTestResult のリスト
        output_dir: 保存先ディレクトリ（python/ からの相対パス）

    Returns:
        保存したファイルのパス文字列
    """
    if not results:
        logger.warning("[stress_test] 保存対象の結果がありません")
        return ""

    os.makedirs(output_dir, exist_ok=True)

    rows = [
        {
            "market": r.market,
            "symbol": r.symbol,
            "scenario_name": r.scenario_name,
            "period_start": r.period_start,
            "period_end": r.period_end,
            "mdd": round(r.mdd, 6),
            "sharpe_ratio": round(r.sharpe_ratio, 4),
            "total_return": round(r.total_return, 6),
            "win_rate": round(r.win_rate, 4),
            "num_trades": r.num_trades,
            "max_consecutive_losses": r.max_consecutive_losses,
            "mdd_pass": r.mdd_pass,
        }
        for r in results
    ]
    df = pd.DataFrame(rows)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(output_dir, f"stress_test_{timestamp}.csv")
    df.to_csv(filepath, index=False, encoding="utf-8-sig")
    logger.info(f"[stress_test] 結果を保存しました: {filepath}")
    return filepath


def print_stress_test_summary(results: list[StressTestResult]) -> None:
    """ストレステスト結果のサマリーをコンソールに出力する。"""
    if not results:
        print("[stress_test] 結果がありません")
        return

    print("\n" + "=" * 70)
    print("ストレステスト結果サマリー")
    print("=" * 70)
    print(
        f"{'銘柄':<12} {'シナリオ':<12} {'MDD':>8} {'判定':>6} "
        f"{'Sharpe':>8} {'トータルリターン':>14} {'勝率':>6} {'取引数':>6} {'連敗':>5}"
    )
    print("-" * 70)

    for r in results:
        status = "PASS" if r.mdd_pass else "FAIL"
        scenario_label = STRESS_SCENARIOS.get(r.scenario_name, {}).get("label", r.scenario_name)
        print(
            f"{r.symbol:<12} {scenario_label:<12} {r.mdd:>8.2%} {status:>6} "
            f"{r.sharpe_ratio:>8.3f} {r.total_return:>14.2%} {r.win_rate:>6.2%} "
            f"{r.num_trades:>6} {r.max_consecutive_losses:>5}"
        )

    pass_count = sum(1 for r in results if r.mdd_pass)
    print("-" * 70)
    print(f"合格: {pass_count}/{len(results)} 件 (MDD ≤ {MDD_THRESHOLD:.0%})")
    print("=" * 70 + "\n")
