"""
外部提供向けデータ取得サービス（R-304）

内部運用データから公開可能なサブセットのみを抽出して返す。

除外情報:
    - paper_orders / paper_positions（注文・ポジション情報）
    - model_metrics / experiment_runs の生データ
    - API キー・内部設定値・秘密情報
"""

from __future__ import annotations

from typing import Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)

# 予測結果の公開フィールド（内部評価値・モデル詳細は除外）
_PUBLIC_PREDICTION_FIELDS = ("market", "symbol", "diff_ratio", "prediction_date")


def get_public_predictions(limit: int = 20) -> list[dict]:
    """最新予測結果の公開用サブセットを返す。

    フィールド: market, symbol, diff_ratio, prediction_date のみ。
    current_price / avg_pred_price などの内部評価値は除外する。
    """
    try:
        from src.prediction.db import load_latest_prediction_timestamp, load_prediction_results

        latest_ts = load_latest_prediction_timestamp()
        if not latest_ts:
            return []

        df = load_prediction_results(predicted_at=latest_ts, limit=limit)
        if df is None or df.empty:
            return []

        results: list[dict] = []
        for _, row in df.iterrows():
            entry: dict = {
                "market": str(row.get("market", "")),
                "symbol": str(row.get("symbol", "")),
                "diff_ratio": (
                    float(row["diff_ratio"]) if row.get("diff_ratio") is not None else None
                ),
                "prediction_date": latest_ts,
            }
            results.append(entry)

        logger.debug("公開予測結果を取得: %d 件", len(results))
        return results

    except Exception:
        logger.error("公開予測結果の取得に失敗", exc_info=True)
        return []


def get_public_monthly_report() -> Optional[dict]:
    """月次 KPI の公開用サブセットを返す。

    フィールド: target_month, net_return, max_drawdown, sharpe_ratio, hit_rate のみ。
    avg_slippage・wf_snapshot_file などの内部運用情報は除外する。
    """
    try:
        from src.reporting.monthly import run_monthly_report

        summary = run_monthly_report()
        return {
            "target_month": summary.target_month,
            "net_return": summary.net_return,
            "max_drawdown": summary.max_drawdown,
            "sharpe_ratio": summary.sharpe_ratio,
            "hit_rate": summary.hit_rate,
        }

    except Exception:
        logger.error("公開月次レポートの取得に失敗", exc_info=True)
        return None


def get_public_symbols() -> list[dict]:
    """ウォッチリスト銘柄一覧を返す（market, symbol のみ）。"""
    try:
        from src.utils.db import get_all_symbols

        pairs = get_all_symbols()
        return [{"market": market, "symbol": symbol} for market, symbol in pairs]

    except Exception:
        logger.error("公開銘柄一覧の取得に失敗", exc_info=True)
        return []
