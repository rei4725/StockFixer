"""
ポートフォリオバックテストサービス

全銘柄を対象に日次シグナルマトリクスを構築し、Top-N 銘柄へ
予測スコア比例配分（softmax）で資金を投じるポートフォリオシミュレーションを行う。

Usage (from run_backtest_portfolio.py):
    results = run_portfolio_backtest(
        market="jp",
        top_n=5,
        rebalance_freq="weekly",
        train_ratio=0.8,
        source="file",
        initial_cash=1_000_000,
        fee_rate=0.001,
    )
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Optional

import numpy as np
import pandas as pd

from config.settings import MAX_SECTOR_POSITIONS
from src.backtest.data_port import get_backtest_data_port
from src.backtest.ports import get_model_manager
from src.utils.logger import get_logger
from src.utils.regime_weights import get_regime_sector_weight
from src.utils.sector_constraints import filter_by_sector_cap, get_symbol_sector

logger = get_logger(__name__)


# ─────────────────────────────────────────
# Public API
# ─────────────────────────────────────────


def run_portfolio_backtest(
    market: Optional[str] = None,
    model_type: str = "XGBoostModel",
    top_n: int = 5,
    rebalance_freq: str = "weekly",
    train_ratio: float = 0.8,
    source: str = "file",
    initial_cash: float = 1_000_000,
    fee_rate: float = 0.001,
    threshold: float = 0.0,
    ensemble: bool = False,
    max_sector_positions: int = MAX_SECTOR_POSITIONS,
    use_sector_rotation: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    """
    ポートフォリオバックテストを実行する。

    Args:
        market: "jp" / "us" / None（全マーケット）
        model_type: "XGBoostModel" or "LightGBMModel"
        top_n: 同時保有する最大銘柄数
        rebalance_freq: "daily" / "weekly" / "monthly"
        train_ratio: 学習データ比率
        source: データソース ("file" / "raw")
        initial_cash: 初期資金
        fee_rate: 片道手数料率
        threshold: 買いシグナル発生の最低予測上昇率（0.0=制限なし）
        ensemble: XGBoost+LightGBM アンサンブルを使用
        max_sector_positions: 同一セクターで許容する最大銘柄数（0 以下で無効）
        use_sector_rotation: True のとき市場レジームに応じてセクターウェイトを切り替える

    Returns:
        (equity_df, metrics, holdings_df)
        - equity_df: 日次ポートフォリオ価値の DataFrame (columns: date, portfolio_value, equal_weight_value)
        - metrics: パフォーマンス指標辞書
        - holdings_df: リバランス日ごとの保有銘柄・ウェイト一覧
    """
    from src.utils.db.stock_features import get_all_symbols

    # 対象銘柄リスト取得
    all_symbols = get_all_symbols()
    if market:
        all_symbols = [(m, s) for m, s in all_symbols if m == market]

    if not all_symbols:
        logger.error(f"対象銘柄が見つかりません: market={market}")
        return pd.DataFrame(), {}, pd.DataFrame()

    logger.info(f"[ポートフォリオ] 対象銘柄数: {len(all_symbols)}")

    # 各銘柄の予測スコアと Close 価格を収集
    score_matrix, close_matrix = _build_signal_matrix(
        all_symbols, model_type, train_ratio, source, threshold, ensemble
    )

    if score_matrix.empty:
        logger.error("[ポートフォリオ] スコアマトリクスが空です")
        return pd.DataFrame(), {}, pd.DataFrame()

    # リバランス日を決定
    rebalance_dates = _get_rebalance_dates(pd.DatetimeIndex(score_matrix.index), rebalance_freq)

    # ポートフォリオシミュレーション
    equity_df, holdings_records = _simulate_portfolio(
        score_matrix=score_matrix,
        close_matrix=close_matrix,
        rebalance_dates=rebalance_dates,
        top_n=top_n,
        initial_cash=initial_cash,
        fee_rate=fee_rate,
        max_sector_positions=max_sector_positions,
        use_sector_rotation=use_sector_rotation,
    )

    metrics = _compute_portfolio_metrics(equity_df, initial_cash)
    equity_df, regime_metrics = _attach_regime_metrics(equity_df, close_matrix)
    if regime_metrics:
        metrics["regime_metrics"] = regime_metrics
    metrics["max_sector_positions"] = max_sector_positions

    holdings_df = pd.DataFrame(holdings_records)

    return equity_df, metrics, holdings_df


def compare_sector_rotation_kpi(
    market: Optional[str] = None,
    model_type: str = "XGBoostModel",
    top_n: int = 5,
    rebalance_freq: str = "weekly",
    train_ratio: float = 0.8,
    source: str = "file",
    initial_cash: float = 1_000_000,
    fee_rate: float = 0.001,
    ensemble: bool = False,
    max_sector_positions: int = MAX_SECTOR_POSITIONS,
) -> dict[str, Any]:
    """
    セクターローテーション有効時/無効時の KPI を比較する。

    シグナルマトリクスを一度だけ構築し、両シミュレーションに再利用する。

    Returns:
        {"rotation_off": metrics, "rotation_on": metrics, "kpi_diff": diff}
        対象銘柄が存在しない場合やデータ不足の場合は空辞書
    """
    from src.utils.db.stock_features import get_all_symbols

    all_symbols = get_all_symbols()
    if market:
        all_symbols = [(m, s) for m, s in all_symbols if m == market]
    if not all_symbols:
        logger.error(f"[セクターローテーション比較] 対象銘柄が見つかりません: market={market}")
        return {}

    score_matrix, close_matrix = _build_signal_matrix(
        all_symbols, model_type, train_ratio, source, 0.0, ensemble
    )
    if score_matrix.empty:
        logger.error("[セクターローテーション比較] スコアマトリクスが空です")
        return {}

    rebalance_dates = _get_rebalance_dates(pd.DatetimeIndex(score_matrix.index), rebalance_freq)

    eq_off, _ = _simulate_portfolio(
        score_matrix,
        close_matrix,
        rebalance_dates,
        top_n,
        initial_cash,
        fee_rate,
        max_sector_positions,
        use_sector_rotation=False,
    )
    eq_on, _ = _simulate_portfolio(
        score_matrix,
        close_matrix,
        rebalance_dates,
        top_n,
        initial_cash,
        fee_rate,
        max_sector_positions,
        use_sector_rotation=True,
    )

    metrics_off = _compute_portfolio_metrics(eq_off, initial_cash)
    metrics_on = _compute_portfolio_metrics(eq_on, initial_cash)

    kpi_diff = {
        "total_return_diff": round(
            metrics_on.get("total_return", 0.0) - metrics_off.get("total_return", 0.0), 6
        ),
        "sharpe_diff": round(
            metrics_on.get("sharpe_ratio", 0.0) - metrics_off.get("sharpe_ratio", 0.0), 4
        ),
        "max_drawdown_diff": round(
            metrics_on.get("max_drawdown", 0.0) - metrics_off.get("max_drawdown", 0.0), 6
        ),
    }

    return {
        "rotation_off": metrics_off,
        "rotation_on": metrics_on,
        "kpi_diff": kpi_diff,
    }


def save_portfolio_results(
    equity_df: pd.DataFrame,
    metrics: dict[str, Any],
    holdings_df: pd.DataFrame,
    market: str,
    top_n: int,
    rebalance_freq: str,
) -> str:
    """結果を CSV に保存する。保存先パスを返す。"""
    from src.utils.data_path_utils import ensure_dir, get_results_dir

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"{get_results_dir()}/backtest/portfolio"
    ensure_dir(out_dir)

    prefix = f"portfolio_{market or 'all'}_{rebalance_freq}_top{top_n}_{ts}"
    equity_path = f"{out_dir}/{prefix}_equity.csv"
    holdings_path = f"{out_dir}/{prefix}_holdings.csv"

    equity_df.to_csv(equity_path, index=False)
    holdings_df.to_csv(holdings_path, index=False)

    print(f"\nエクイティカーブ保存: {equity_path}")
    print(f"保有銘柄推移保存:     {holdings_path}")
    return equity_path


def plot_portfolio(
    equity_df: pd.DataFrame,
    holdings_df: pd.DataFrame,
    metrics: dict[str, Any],
    output_dir: Optional[str] = None,
    market: str = "",
    top_n: int = 5,
    rebalance_freq: str = "weekly",
) -> str:
    """ポートフォリオ結果グラフ（エクイティ + 銘柄寄与 + ターンオーバー）を PNG 保存する。"""
    if equity_df is None or equity_df.empty:
        return ""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib がインストールされていません。")

    equity_df = equity_df.copy()
    equity_df["date"] = pd.to_datetime(equity_df["date"])

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    title = f"ポートフォリオバックテスト ({market or 'ALL'} | Top-{top_n} | {rebalance_freq})"
    fig.suptitle(title, fontsize=13)

    # ── 1. エクイティカーブ ──
    ax1 = axes[0]
    pf_ret = (equity_df["portfolio_value"] / equity_df["portfolio_value"].iloc[0] - 1) * 100
    ew_ret = (equity_df["equal_weight_value"] / equity_df["equal_weight_value"].iloc[0] - 1) * 100
    ax1.plot(
        equity_df["date"], pf_ret, label=f"Top-{top_n} 予測比例", color="steelblue", linewidth=1.5
    )
    ax1.plot(
        equity_df["date"],
        ew_ret,
        label="等分ベンチマーク",
        color="orange",
        linewidth=1.2,
        linestyle="--",
    )
    ax1.axhline(0, color="gray", linewidth=0.7, linestyle=":")
    ax1.set_ylabel("累積リターン (%)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ── 2. 銘柄寄与度（保有銘柄のユニーク出現頻度） ──
    ax2 = axes[1]
    if (
        not holdings_df.empty
        and "symbol" in holdings_df.columns
        and "weight" in holdings_df.columns
    ):
        contrib = (
            holdings_df.groupby("symbol")["weight"].sum().sort_values(ascending=False).head(15)
        )
        ax2.bar(contrib.index, contrib.values, color="mediumseagreen")
        ax2.set_ylabel("累積ウェイト合計")
        ax2.set_xlabel("銘柄")
        ax2.set_title("保有頻度（上位15銘柄）")
        ax2.tick_params(axis="x", rotation=45)
        ax2.grid(True, alpha=0.3, axis="y")
    else:
        ax2.text(0.5, 0.5, "銘柄データなし", ha="center", va="center", transform=ax2.transAxes)

    # ── 3. ターンオーバー率 ──
    ax3 = axes[2]
    if not holdings_df.empty and "turnover" in holdings_df.columns:
        to_df = holdings_df.drop_duplicates(subset=["rebalance_date"])[
            ["rebalance_date", "turnover"]
        ].copy()
        to_df["rebalance_date"] = pd.to_datetime(to_df["rebalance_date"])
        ax3.bar(to_df["rebalance_date"], to_df["turnover"] * 100, color="coral", width=3)
        ax3.set_ylabel("ターンオーバー率 (%)")
        ax3.set_xlabel("リバランス日")
        ax3.set_title("リバランスごとのターンオーバー")
        ax3.xaxis.set_major_formatter(  # type: ignore[no-untyped-call]
            mdates.DateFormatter("%Y-%m")
        )
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=30, ha="right")
        ax3.grid(True, alpha=0.3, axis="y")
    else:
        ax3.text(
            0.5, 0.5, "ターンオーバーデータなし", ha="center", va="center", transform=ax3.transAxes
        )

    # 指標サマリー
    info_parts = [
        f"Total Return: {metrics.get('total_return', 0):.2%}",
        f"Sharpe: {metrics.get('sharpe_ratio', 0):.2f}",
        f"MaxDD: {metrics.get('max_drawdown', 0):.2%}",
    ]
    fig.text(0.5, 0.005, "  |  ".join(info_parts), ha="center", fontsize=9)
    plt.tight_layout(rect=(0, 0.03, 1, 0.96))

    import os

    if output_dir is None:
        from src.utils.data_path_utils import get_results_dir

        output_dir = os.path.join(get_results_dir(), "backtest", "portfolio")
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"portfolio_{market or 'all'}_{rebalance_freq}_top{top_n}_{ts}.png"
    output_path = os.path.join(output_dir, filename)
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return output_path


def print_portfolio_metrics(
    metrics: dict[str, Any], market: str, top_n: int, rebalance_freq: str
) -> None:
    """ポートフォリオメトリクスを標準出力に表示する。"""
    if not metrics:
        return
    label = f"ポートフォリオ ({market or 'ALL'} | Top-{top_n} | {rebalance_freq})"
    print(f"\n{'='*55}")
    print(f" {label}")
    print(f"{'='*55}")
    regime_metrics = metrics.get("regime_metrics", {})
    for k, v in metrics.items():
        if k == "regime_metrics":
            continue
        print(f"  {k:25s}: {v}")
    if regime_metrics:
        print("  regime_metrics:")
        for label in ("all", "bull", "bear", "range"):
            leg = regime_metrics.get(label)
            if not leg:
                continue
            print(
                "    "
                f"{label:5s} days={leg['days']:3d} "
                f"return={leg['total_return']:.4f} "
                f"sharpe={leg['sharpe_ratio']:.4f} "
                f"hit={leg['hit_rate']:.4f}"
            )
    print(f"{'='*55}")


# ─────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────


def _build_signal_matrix(
    symbols: list[tuple[str, str]],
    model_type: str,
    train_ratio: float,
    source: str,
    threshold: float,
    ensemble: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    全銘柄について学習 → 予測し、date × symbol の
    スコアマトリクスと Close 価格マトリクスを返す。

    学習データは train_ratio で分割し、テスト期間の予測のみ収録する。
    """
    from src.backtest.pipeline import _ensemble_predict, load_features

    score_dict: dict[str, pd.Series] = {}
    close_dict: dict[str, pd.Series] = {}

    for market, symbol in symbols:
        key = f"{market}_{symbol}"
        try:
            df = load_features(market, symbol, source)
            if df is None or df.empty or len(df) < 30:
                logger.debug(f"[PF] データ不足スキップ: {key}")
                continue

            # 学習 / テスト分割
            split = int(len(df) * train_ratio)
            if split < 20 or (len(df) - split) < 5:
                logger.debug(f"[PF] 分割後データ不足スキップ: {key}")
                continue

            train_df = df.iloc[:split]
            test_df = df.iloc[split:]

            exclude = {"y", "market", "symbol", "market_encoded", "Close", "close"}
            feature_cols = [c for c in df.columns if c not in exclude]

            X_train = train_df[feature_cols].dropna()
            y_train = train_df.loc[X_train.index, "y"].dropna()
            X_train = X_train.loc[y_train.index]

            X_test = test_df[feature_cols].dropna()

            if X_train.empty or X_test.empty:
                logger.debug(f"[PF] 特徴量空スキップ: {key}")
                continue

            mm = get_model_manager()

            if ensemble:
                pred = _ensemble_predict(mm, X_train, y_train, X_test, f"PF_{key}")
            else:
                mm.create_model(model_type, f"PF_{key}")
                mm.train_model(f"PF_{key}", X_train, y_train)
                raw = mm.predict_with_model(f"PF_{key}", X_test)
                pred = pd.Series(raw, index=X_test.index)

            # threshold 以上の予測値のみシグナルとして扱う（0未満はマスク）
            pred_filtered = pred.where(pred >= threshold, other=np.nan)
            score_dict[key] = pred_filtered

            close_col = "Close" if "Close" in test_df.columns else "close"
            if close_col in test_df.columns:
                close_dict[key] = test_df[close_col].reindex(X_test.index)

            logger.info(f"[PF] 予測完了: {key} ({len(pred)}行)")

        except Exception as e:
            logger.error(f"[PF] 銘柄スキップ ({key}): {e}", exc_info=True)

    if not score_dict:
        return pd.DataFrame(), pd.DataFrame()

    score_matrix = pd.DataFrame(score_dict).sort_index()
    close_matrix = pd.DataFrame(close_dict).sort_index()

    # 共通日付に揃える
    common_idx = score_matrix.index.intersection(close_matrix.index)
    score_matrix = score_matrix.loc[common_idx]
    close_matrix = close_matrix.loc[common_idx]

    return score_matrix, close_matrix


def _get_rebalance_dates(index: pd.DatetimeIndex, freq: str) -> list[Any]:
    """リバランス日のリストを返す。"""
    if freq == "daily":
        return list(index)
    elif freq == "weekly":
        # 各週の最初の取引日（月曜相当）
        week_groups = pd.Series(index, index=index).groupby(pd.Grouper(freq="W-MON"))
        return [g.iloc[0] for _, g in week_groups if len(g) > 0]
    elif freq == "monthly":
        month_groups = pd.Series(index, index=index).groupby(pd.Grouper(freq="MS"))
        return [g.iloc[0] for _, g in month_groups if len(g) > 0]
    else:
        raise ValueError(f"未対応のリバランス頻度: {freq}")


def _softmax_weights(scores: pd.Series) -> pd.Series:
    """正のスコアのみを対象に softmax でウェイトを計算する。"""
    valid = scores.dropna()
    valid = valid[valid > 0]
    if valid.empty:
        return pd.Series(dtype=float)
    exp_s = np.exp(valid - valid.max())  # 数値安定化
    return exp_s / exp_s.sum()


def _apply_sector_rotation(scores: pd.Series, regime: str) -> pd.Series:
    """レジームに応じたセクターウェイト乗数をスコア Series に適用して返す。"""
    result = scores.copy()
    for sym in result.index:
        sector = _get_portfolio_symbol_sector(str(sym))
        result[sym] *= get_regime_sector_weight(regime, sector)
    return result


def _simulate_portfolio(
    score_matrix: pd.DataFrame,
    close_matrix: pd.DataFrame,
    rebalance_dates: list[Any],
    top_n: int,
    initial_cash: float,
    fee_rate: float,
    max_sector_positions: int,
    use_sector_rotation: bool = False,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """
    ポートフォリオシミュレーションを実行する。

    毎リバランス日に Top-N 銘柄をスコア比例配分で保有し、
    日次の portfolio_value と equal_weight_value を記録する。
    """
    # セクターローテーション用に市場レジームを事前計算
    regime_series: Optional[pd.Series] = None
    if use_sector_rotation:
        proxy_df = _build_market_proxy_frame(close_matrix)
        if not proxy_df.empty:
            regime_series = get_backtest_data_port().get_market_regime(proxy_df)

    rebalance_set = set(str(d)[:10] for d in rebalance_dates)

    cash = initial_cash
    # {symbol: {"qty": int, "price": float}}
    holdings: dict[str, dict[str, Any]] = {}
    prev_symbols: set[str] = set()
    holdings_records: list[dict[str, Any]] = []

    equity_rows: list[dict[str, Any]] = []

    # 等分戦略の基準として全銘柄を均等保有したシリーズ（簡易近似）
    eq_cash = initial_cash
    eq_holdings: dict[str, dict[str, Any]] = {}

    for date in score_matrix.index:
        date_str = str(date)[:10]
        prices_today = close_matrix.loc[date].dropna()

        # ─ リバランス ─
        if date_str in rebalance_set:
            scores_today = score_matrix.loc[date].dropna()

            # セクターローテーション: レジームに応じてスコアを調整
            if use_sector_rotation and regime_series is not None:
                date_ts = pd.Timestamp(date_str)
                current_regime = "range"
                if date_ts >= regime_series.index.min():
                    raw = regime_series.asof(date_ts)
                    if not pd.isna(raw):
                        current_regime = str(raw)
                scores_today = _apply_sector_rotation(scores_today, current_regime)

            # 上位 top_n を選択
            top_candidates = _limit_portfolio_candidates_by_sector(
                scores_today.nlargest(top_n * 3),
                max_sector_positions=max_sector_positions,
            ).head(top_n)
            weights = _softmax_weights(top_candidates)

            if not weights.empty:
                new_symbols = set(weights.index)

                # ターンオーバー計算
                if prev_symbols:
                    stayed = prev_symbols & new_symbols
                    turnover = 1.0 - len(stayed) / max(len(prev_symbols), len(new_symbols))
                else:
                    turnover = 1.0

                # 既存保有を全売却
                for sym, pos in holdings.items():
                    if sym in prices_today.index:
                        proceeds = pos["qty"] * prices_today[sym] * (1 - fee_rate)
                        cash += proceeds

                holdings = {}

                # 新規銘柄を購入
                total_budget = cash
                for sym_key, w in weights.items():
                    sym = str(sym_key)
                    if sym not in prices_today.index:
                        continue
                    budget = total_budget * float(w)
                    price = prices_today[sym]
                    if price <= 0:
                        continue
                    qty = int(budget / (price * (1 + fee_rate)))
                    if qty > 0:
                        cost = qty * price * (1 + fee_rate)
                        cash -= cost
                        holdings[sym] = {"qty": qty, "price": price}

                        holdings_records.append(
                            {
                                "rebalance_date": date_str,
                                "symbol": sym,
                                "sector": _get_portfolio_symbol_sector(sym),
                                "weight": float(w),
                                "score": float(top_candidates.get(sym, 0)),
                                "price": price,
                                "qty": qty,
                                "turnover": turnover,
                            }
                        )

                # 等分戦略のリバランス（全有効銘柄に均等配分）
                valid_syms = [s for s in scores_today.index if s in prices_today.index]
                if valid_syms:
                    for sym, pos in eq_holdings.items():
                        if sym in prices_today.index:
                            eq_cash += pos["qty"] * prices_today[sym] * (1 - fee_rate)
                    eq_holdings = {}
                    eq_per = eq_cash / len(valid_syms)
                    for sym in valid_syms:
                        price = prices_today[sym]
                        if price <= 0:
                            continue
                        qty = int(eq_per / (price * (1 + fee_rate)))
                        if qty > 0:
                            eq_cash -= qty * price * (1 + fee_rate)
                            eq_holdings[sym] = {"qty": qty}

                prev_symbols = new_symbols

        # ─ 日次ポートフォリオ価値 ─
        pf_value = cash + sum(
            pos["qty"] * prices_today.get(sym, pos.get("price", 0)) for sym, pos in holdings.items()
        )
        eq_value = eq_cash + sum(
            pos["qty"] * prices_today.get(sym, 0) for sym, pos in eq_holdings.items()
        )

        equity_rows.append(
            {
                "date": date_str,
                "portfolio_value": round(pf_value, 2),
                "equal_weight_value": round(eq_value, 2),
            }
        )

    equity_df = pd.DataFrame(equity_rows)
    return equity_df, holdings_records


def _get_portfolio_symbol_sector(sym_key: str) -> str:
    market, symbol = _split_symbol_key(sym_key)
    return get_symbol_sector(market, symbol)


def _split_symbol_key(sym_key: str) -> tuple[str, str]:
    if "_" not in sym_key:
        return "jp", sym_key
    return tuple(sym_key.split("_", 1))  # type: ignore[return-value]


def _limit_portfolio_candidates_by_sector(
    top_candidates: pd.Series,
    max_sector_positions: int = MAX_SECTOR_POSITIONS,
) -> pd.Series:
    if top_candidates.empty or max_sector_positions <= 0:
        return top_candidates.copy()

    ordered_items = list(top_candidates.items())
    selected_items = filter_by_sector_cap(
        ordered_items,
        max_sector_positions=max_sector_positions,
        sector_getter=lambda item: _get_portfolio_symbol_sector(str(item[0])),
    )
    if not selected_items:
        return top_candidates.iloc[0:0].copy()
    return pd.Series(
        data=[float(score) for _, score in selected_items],
        index=[str(symbol) for symbol, _ in selected_items],
        dtype=float,
    )


def _compute_portfolio_metrics(equity_df: pd.DataFrame, initial_cash: float) -> dict[str, Any]:
    """ポートフォリオ用メトリクスを計算する。"""
    if equity_df is None or equity_df.empty:
        return {}

    pv = equity_df["portfolio_value"]
    final = pv.iloc[-1]
    total_return = (final - initial_cash) / initial_cash

    daily_ret = pv.pct_change().dropna()
    sharpe = 0.0
    if len(daily_ret) >= 2 and daily_ret.std() > 0:
        sharpe = float(daily_ret.mean() / daily_ret.std() * math.sqrt(252))

    roll_max = pv.cummax()
    max_dd = float(((pv - roll_max) / roll_max).min())

    eq_pv = equity_df["equal_weight_value"]
    eq_return = (eq_pv.iloc[-1] - initial_cash) / initial_cash

    return {
        "final_cash": round(final, 2),
        "total_return": round(total_return, 6),
        "equal_weight_return": round(eq_return, 6),
        "alpha_vs_equal": round(total_return - eq_return, 6),
        "sharpe_ratio": round(sharpe, 4),
        "max_drawdown": round(max_dd, 6),
        "days": len(equity_df),
    }


def _attach_regime_metrics(
    equity_df: pd.DataFrame,
    close_matrix: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """ポートフォリオ日次損益にレジーム列を付与し、レジーム別メトリクスを返す。"""
    if equity_df is None or equity_df.empty or close_matrix is None or close_matrix.empty:
        return equity_df, {}

    proxy_df = _build_market_proxy_frame(close_matrix)
    if proxy_df.empty:
        return equity_df, {}

    regime_series = get_backtest_data_port().get_market_regime(proxy_df)
    if regime_series.empty:
        return equity_df, {}

    enriched = equity_df.copy()
    enriched["date"] = pd.to_datetime(enriched["date"])
    if not isinstance(regime_series.index, pd.DatetimeIndex):
        regime_series.index = pd.to_datetime(regime_series.index)

    min_regime_date = regime_series.index.min()
    enriched["regime"] = enriched["date"].map(
        lambda d: regime_series.asof(d) if d >= min_regime_date else None
    )
    regime_metrics = _compute_regime_metrics(enriched)
    enriched["date"] = enriched["date"].dt.strftime("%Y-%m-%d")
    return enriched, regime_metrics


def _build_market_proxy_frame(close_matrix: pd.DataFrame) -> pd.DataFrame:
    """複数銘柄の終値行列から市場全体の簡易プロキシ価格を構築する。"""
    if close_matrix is None or close_matrix.empty:
        return pd.DataFrame()

    aligned = close_matrix.sort_index().ffill()
    proxy_df = pd.DataFrame(index=aligned.index)
    proxy_df["Close"] = aligned.mean(axis=1, skipna=True)
    proxy_df["High"] = aligned.max(axis=1, skipna=True)
    proxy_df["Low"] = aligned.min(axis=1, skipna=True)
    return proxy_df.dropna(subset=["Close"])


def _compute_regime_metrics(equity_df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {"all": _compute_regime_leg_metrics(equity_df)}
    if "regime" not in equity_df.columns:
        return results

    for label in ("bull", "bear", "range"):
        leg = equity_df[equity_df["regime"] == label]
        results[label] = _compute_regime_leg_metrics(leg)
    return results


def _compute_regime_leg_metrics(equity_leg: pd.DataFrame) -> dict[str, Any]:
    if equity_leg is None or equity_leg.empty:
        return {
            "days": 0,
            "total_return": 0.0,
            "sharpe_ratio": 0.0,
            "hit_rate": 0.0,
            "max_drawdown": 0.0,
        }

    portfolio_values = pd.to_numeric(equity_leg["portfolio_value"], errors="coerce").dropna()
    daily_returns = portfolio_values.pct_change().dropna()
    daily_return_values = daily_returns.to_numpy(dtype=float, copy=False)
    total_return = (
        float(np.prod(1.0 + daily_return_values) - 1.0) if len(daily_return_values) else 0.0
    )
    sharpe_ratio = 0.0
    if len(daily_return_values) >= 2:
        daily_std = float(np.std(daily_return_values, ddof=1))
        if daily_std > 0:
            daily_mean = float(np.mean(daily_return_values))
            sharpe_ratio = float(daily_mean / daily_std * math.sqrt(252))

    hit_rate = float(np.mean(daily_return_values > 0)) if len(daily_return_values) else 0.0
    curve = pd.Series(np.cumprod(1.0 + daily_return_values), index=daily_returns.index)
    max_drawdown = 0.0
    if not curve.empty:
        roll_max = curve.cummax()
        max_drawdown = float(((curve - roll_max) / roll_max).min())

    return {
        "days": int(len(equity_leg)),
        "total_return": round(total_return, 6),
        "sharpe_ratio": round(sharpe_ratio, 4),
        "hit_rate": round(hit_rate, 4),
        "max_drawdown": round(max_drawdown, 6),
    }
