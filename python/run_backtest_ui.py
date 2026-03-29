"""
バックテスト簡易UI

Streamlit によるインタラクティブなバックテストUIです。
サイドバーでパラメータを設定し「バックテスト実行」ボタンを押すと
結果をメトリクス・チャート・テーブルで表示します。

起動方法:
    cd python
    streamlit run run_backtest_ui.py
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ──────────────────────────────────────────────
# ページ設定
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="StockFixer Backtest UI",
    page_icon="📈",
    layout="wide",
)

st.title("📈 バックテスト実行 UI")
st.caption("サイドバーでパラメータを設定し「バックテスト実行」ボタンを押してください。")

# メインタブ
tab_single, tab_portfolio, tab_wf, tab_optimize = st.tabs(
    ["📈 単一銘柄", "📊 ポートフォリオ", "🔄 Walk-Forward", "⚡ パラメータ最適化"]
)

# ──────────────────────────────────────────────
# サイドバー: パラメータ設定
# ──────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ パラメータ設定")

    # --- 基本設定 ---
    with st.expander("📌 基本設定", expanded=True):
        market = st.selectbox("マーケット", ["jp", "us"], index=0)
        symbol = st.text_input("銘柄コード", value="7203", help="例: 7203（トヨタ）, AAPL")
        source = st.selectbox(
            "データソース",
            ["file", "api", "raw"],
            index=0,
            help="file=DB特徴量 / api=yfinance直接取得 / raw=DBのOHLCVから再生成",
        )
        col_sd, col_ed = st.columns(2)
        with col_sd:
            start_date = st.date_input("開始日", value=None)
        with col_ed:
            end_date = st.date_input("終了日", value=None)
        train_ratio = st.slider("学習比率", min_value=0.50, max_value=0.95, value=0.80, step=0.05)

    # --- モデル設定 ---
    with st.expander("🤖 モデル設定"):
        model_type = st.selectbox(
            "モデルタイプ",
            ["XGBoostModel", "LightGBMModel"],
            index=0,
        )
        ensemble = st.checkbox(
            "アンサンブル (XGBoost + LightGBM)",
            value=False,
            help="XGBoostとLightGBMの予測平均を使用する",
        )
        threshold = st.slider(
            "シグナル閾値",
            min_value=0.00,
            max_value=0.05,
            value=0.00,
            step=0.005,
            format="%.3f",
            help="予測変化率がこの値を超えた場合にシグナル発生",
        )

    # --- 取引条件 ---
    with st.expander("💰 取引条件"):
        initial_cash = st.number_input(
            "初期資金 (円)",
            min_value=100_000,
            max_value=100_000_000,
            value=1_000_000,
            step=100_000,
        )
        fee_rate = st.number_input(
            "手数料率",
            min_value=0.0,
            max_value=0.01,
            value=0.001,
            step=0.0001,
            format="%.4f",
        )
        slippage = st.number_input(
            "スリッページ",
            min_value=0.0,
            max_value=0.01,
            value=0.0,
            step=0.0001,
            format="%.4f",
        )

    # --- リスク管理 ---
    with st.expander("🛡️ リスク管理"):
        enable_stop_loss = st.checkbox("ストップロスを有効にする", value=False)
        stop_loss_pct = None
        if enable_stop_loss:
            stop_loss_pct = st.slider(
                "ストップロス率",
                min_value=0.01,
                max_value=0.20,
                value=0.05,
                step=0.01,
                format="%.0f%%",
                help="この割合だけ下落したら損切り",
            )

        enable_take_profit = st.checkbox("テイクプロフィットを有効にする", value=False)
        take_profit_pct = None
        if enable_take_profit:
            take_profit_pct = st.slider(
                "テイクプロフィット率",
                min_value=0.01,
                max_value=0.50,
                value=0.10,
                step=0.01,
                format="%.0f%%",
                help="この割合だけ上昇したら利確",
            )

        position_sizing = st.selectbox(
            "ポジションサイジング",
            ["full", "fixed", "confidence", "atr"],
            index=0,
            help="full=全額 / fixed=固定比率 / confidence=予測確信度ベース / atr=ATR連動",
        )
        position_fraction = 0.5
        atr_risk_pct = 0.02
        atr_multiplier = 1.0
        if position_sizing in ("fixed", "confidence"):
            position_fraction = st.slider(
                "ポジション比率",
                min_value=0.1,
                max_value=1.0,
                value=0.5,
                step=0.1,
            )
        elif position_sizing == "atr":
            atr_risk_pct = st.slider(
                "ATR リスク割合",
                min_value=0.005,
                max_value=0.1,
                value=0.02,
                step=0.005,
                format="%.3f",
                help="1トレードあたりのリスク隔52%%",
            )
            atr_multiplier = st.slider(
                "ATR 倍数",
                min_value=0.5,
                max_value=3.0,
                value=1.0,
                step=0.25,
                help="ストップ宽 = ATR × 倍数",
            )

    st.divider()
    run_button = st.button("▶ バックテスト実行", type="primary", use_container_width=True)

    st.divider()

    # ── ポートフォリオ設定 ──
    st.header("📊 ポートフォリオ設定")
    with st.expander("🗂️ ポートフォリオ設定", expanded=True):
        pf_market = st.selectbox(
            "マーケット",
            ["all", "jp", "us"],
            index=0,
            key="pf_market",
            help="all=全マーケット",
        )
        pf_top_n = st.slider("Top-N 銘柄数", min_value=1, max_value=20, value=5, step=1)
        pf_rebalance_freq = st.selectbox(
            "リバランス頻度",
            ["weekly", "monthly", "daily"],
            index=0,
            key="pf_freq",
        )
        pf_train_ratio = st.slider(
            "学習比率",
            min_value=0.50,
            max_value=0.95,
            value=0.80,
            step=0.05,
            key="pf_train_ratio",
        )
        pf_source = st.selectbox(
            "データソース",
            ["file", "raw"],
            index=0,
            key="pf_source",
        )
        pf_initial_cash = st.number_input(
            "初期資金 (円)",
            min_value=100_000,
            max_value=100_000_000,
            value=1_000_000,
            step=100_000,
            key="pf_cash",
        )
        pf_fee_rate = st.number_input(
            "手数料率",
            min_value=0.0,
            max_value=0.01,
            value=0.001,
            step=0.0001,
            format="%.4f",
            key="pf_fee",
        )
        pf_threshold = st.slider(
            "シグナル閾値",
            min_value=0.00,
            max_value=0.05,
            value=0.00,
            step=0.005,
            format="%.3f",
            key="pf_threshold",
        )
        pf_model_type = st.selectbox(
            "モデルタイプ",
            ["XGBoostModel", "LightGBMModel"],
            index=0,
            key="pf_model",
        )
        pf_ensemble = st.checkbox(
            "アンサンブル (XGBoost + LightGBM)",
            value=False,
            key="pf_ensemble",
        )

    pf_run_button = st.button("▶ ポートフォリオ実行", type="primary", use_container_width=True, key="pf_run")

    st.divider()

    # ── Walk-Forward 設定 ──
    st.header("🔄 Walk-Forward 設定")
    with st.expander("🔄 Walk-Forward 設定", expanded=True):
        wf_n_splits = st.slider(
            "フォールド数", min_value=2, max_value=10, value=5, step=1, key="wf_n_splits"
        )

    wf_run_button = st.button(
        "▶ Walk-Forward 実行", type="primary", use_container_width=True, key="wf_run"
    )

    st.divider()

    # ── パラメータ最適化設定 ──
    st.header("⚡ パラメータ最適化設定")
    with st.expander("⚡ 最適化設定", expanded=True):
        opt_threshold_min = st.number_input(
            "閾値 最小値",
            min_value=0.0,
            max_value=0.05,
            value=0.0,
            step=0.001,
            format="%.3f",
            key="opt_thr_min",
        )
        opt_threshold_max = st.number_input(
            "閾値 最大値",
            min_value=0.001,
            max_value=0.05,
            value=0.015,
            step=0.001,
            format="%.3f",
            key="opt_thr_max",
        )
        opt_threshold_step = st.number_input(
            "閾値 ステップ",
            min_value=0.001,
            max_value=0.01,
            value=0.001,
            step=0.001,
            format="%.3f",
            key="opt_thr_step",
        )
        opt_n_splits = st.slider(
            "フォールド数",
            min_value=2,
            max_value=10,
            value=5,
            step=1,
            key="opt_n_splits",
        )
        opt_optimize_risk = st.checkbox(
            "SL/TP もグリッドサーチ",
            value=False,
            key="opt_risk",
            help="有効にするとストップロス・テイクプロフィットの組み合わせも探索します（時間がかかります）",
        )
        opt_sort_by = st.selectbox(
            "ソート基準",
            ["sharpe_ratio", "total_return", "profit_factor", "win_rate"],
            index=0,
            key="opt_sort_by",
        )

    opt_run_button = st.button("▶ 最適化実行", type="primary", use_container_width=True, key="opt_run")


# ──────────────────────────────────────────────
# バックテスト実行ロジック（キャッシュ付き）
# ──────────────────────────────────────────────
@st.cache_data(show_spinner="バックテストを実行中...")
def _run_backtest(
    market,
    symbol,
    model_type,
    ensemble,
    threshold,
    source,
    start_date_str,
    end_date_str,
    train_ratio,
    initial_cash,
    fee_rate,
    slippage,
    stop_loss_pct,
    take_profit_pct,
    position_sizing,
    position_fraction,
    atr_risk_pct=0.02,
    atr_multiplier=1.0,
):
    """パラメータが変わらない限りキャッシュされる。"""
    from src.services.backtest_pipeline import run_backtest_single

    result_df, metrics, price_series = run_backtest_single(
        market=market,
        symbol=symbol,
        model_type=model_type,
        ensemble=ensemble,
        threshold=threshold,
        source=source,
        start_date=start_date_str,
        end_date=end_date_str,
        train_ratio=train_ratio,
        initial_cash=initial_cash,
        fee_rate=fee_rate,
        slippage=slippage,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
        position_sizing=position_sizing,
        position_fraction=position_fraction,
        atr_risk_pct=atr_risk_pct,
        atr_multiplier=atr_multiplier,
    )
    return result_df, metrics, price_series


# ──────────────────────────────────────────────
# グラフ生成
# ──────────────────────────────────────────────
def build_price_signal_chart(
    trade_log: pd.DataFrame, symbol: str, price_series: pd.Series = None
) -> go.Figure:
    """株価ライン + 買い/売りシグナルマーカーのチャートを生成する。"""
    if trade_log is None or "price" not in trade_log.columns or "date" not in trade_log.columns:
        return go.Figure()

    df = trade_log.copy()
    df["date"] = pd.to_datetime(df["date"])

    buys = df[df["action"] == "buy"]

    fig = go.Figure()

    # 全日付の株価ライン（テスト期間全体）
    if price_series is not None and not price_series.empty:
        fig.add_trace(
            go.Scatter(
                x=price_series.index,
                y=price_series.values,
                mode="lines",
                name="株価",
                line=dict(color="#4A90D9", width=1.5),
                opacity=0.85,
            )
        )
    else:
        # フォールバック: 取引日のみの価格ライン
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["price"],
                mode="lines",
                name="取引価格",
                line=dict(color="#4A90D9", width=1.5),
                opacity=0.7,
            )
        )

    # 買いシグナル (▲ 緑)
    if not buys.empty:
        fig.add_trace(
            go.Scatter(
                x=buys["date"],
                y=buys["price"],
                mode="markers",
                name="買い",
                marker=dict(
                    symbol="triangle-up",
                    size=12,
                    color="#2ecc71",
                    line=dict(width=1, color="#27ae60"),
                ),
            )
        )

    # 売りシグナル (▼ 赤)
    action_labels = {
        "sell": "売り",
        "final_sell": "売り(期末)",
        "take_profit": "利確",
        "stop_loss": "損切り",
    }
    for action, label in action_labels.items():
        subset = df[df["action"] == action]
        if subset.empty:
            continue
        color = (
            "#e74c3c"
            if action == "stop_loss"
            else "#e67e22"
            if action == "take_profit"
            else "#e74c3c"
        )
        fig.add_trace(
            go.Scatter(
                x=subset["date"],
                y=subset["price"],
                mode="markers",
                name=label,
                marker=dict(
                    symbol="triangle-down",
                    size=12,
                    color=color,
                    line=dict(width=1, color="#c0392b"),
                ),
            )
        )

    fig.update_layout(
        title=f"{symbol} 株価と売買シグナル",
        xaxis_title="日付",
        yaxis_title="価格",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def build_equity_curve_chart(trade_log: pd.DataFrame, initial_cash: float) -> go.Figure:
    """エクイティカーブ（資産推移）グラフを生成する。"""
    if trade_log is None or trade_log.empty:
        return go.Figure()

    df = trade_log.copy()
    df["date"] = pd.to_datetime(df["date"])

    # 決済アクションの資産推移のみを使用
    sell_actions = ["sell", "final_sell", "stop_loss", "take_profit"]
    equity_df = df[df["action"].isin(sell_actions)][["date", "cash"]].copy()

    if equity_df.empty:
        return go.Figure()

    # 初期資金を先頭に追加
    start_row = pd.DataFrame({"date": [equity_df.iloc[0]["date"]], "cash": [initial_cash]})
    equity_df = pd.concat([start_row, equity_df], ignore_index=True)

    # 基準線（初期資金）
    fig = go.Figure()
    fig.add_hline(
        y=initial_cash,
        line_dash="dash",
        line_color="gray",
        annotation_text="初期資金",
        annotation_position="right",
    )
    fig.add_trace(
        go.Scatter(
            x=equity_df["date"],
            y=equity_df["cash"],
            mode="lines+markers",
            name="資産",
            line=dict(color="#9b59b6", width=2),
            marker=dict(size=6),
            fill="tozeroy",
            fillcolor="rgba(155, 89, 182, 0.1)",
        )
    )
    fig.update_layout(
        title="エクイティカーブ（資産推移）",
        xaxis_title="日付",
        yaxis_title="資産 (円)",
        height=350,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


# ──────────────────────────────────────────────
# メイン: 結果表示
# ──────────────────────────────────────────────
# ──────────────────────────────────────────────
# ポートフォリオ実行ロジック（キャッシュ付き）
# ──────────────────────────────────────────────
@st.cache_data(show_spinner="ポートフォリオバックテストを実行中...", ttl=600)
def _run_portfolio(
    pf_market,
    pf_model_type,
    pf_ensemble,
    pf_top_n,
    pf_rebalance_freq,
    pf_train_ratio,
    pf_source,
    pf_initial_cash,
    pf_fee_rate,
    pf_threshold,
):
    """ポートフォリオバックテストをキャッシュ付きで実行する。"""
    from src.services.portfolio_backtest import run_portfolio_backtest

    market_arg = None if pf_market == "all" else pf_market
    equity_df, metrics, holdings_df = run_portfolio_backtest(
        market=market_arg,
        model_type=pf_model_type,
        top_n=pf_top_n,
        rebalance_freq=pf_rebalance_freq,
        train_ratio=pf_train_ratio,
        source=pf_source,
        initial_cash=pf_initial_cash,
        fee_rate=pf_fee_rate,
        threshold=pf_threshold,
        ensemble=pf_ensemble,
    )
    return equity_df, metrics, holdings_df


def build_portfolio_equity_chart(equity_df: pd.DataFrame, initial_cash: float) -> go.Figure:
    """ポートフォリオ vs 等分ベンチマークのエクイティカーブを生成する。"""
    if equity_df is None or equity_df.empty:
        return go.Figure()

    df = equity_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    pf_ret = (df["portfolio_value"] / df["portfolio_value"].iloc[0] - 1) * 100
    ew_ret = (df["equal_weight_value"] / df["equal_weight_value"].iloc[0] - 1) * 100

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=pf_ret,
            mode="lines",
            name="Top-N 予測比例",
            line=dict(color="#4A90D9", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=ew_ret,
            mode="lines",
            name="等分ベンチマーク",
            line=dict(color="#F5A623", width=1.5, dash="dash"),
        )
    )
    fig.add_hline(y=0, line_dash="dot", line_color="gray")
    fig.update_layout(
        title="ポートフォリオ エクイティカーブ",
        xaxis_title="日付",
        yaxis_title="累積リターン (%)",
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def build_holdings_contrib_chart(holdings_df: pd.DataFrame) -> go.Figure:
    """銘柄寄与度（累積ウェイト）棒グラフを生成する。"""
    if holdings_df is None or holdings_df.empty or "symbol" not in holdings_df.columns:
        return go.Figure()

    contrib = holdings_df.groupby("symbol")["weight"].sum().sort_values(ascending=False).head(15)
    fig = go.Figure(
        go.Bar(
            x=contrib.index,
            y=contrib.values,
            marker_color="#2ecc71",
        )
    )
    fig.update_layout(
        title="銘柄寄与度（上位15銘柄・累積ウェイト合計）",
        xaxis_title="銘柄",
        yaxis_title="累積ウェイト合計",
        height=320,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def build_turnover_chart(holdings_df: pd.DataFrame) -> go.Figure:
    """ターンオーバー率棒グラフを生成する。"""
    if holdings_df is None or holdings_df.empty or "turnover" not in holdings_df.columns:
        return go.Figure()

    to_df = holdings_df.drop_duplicates(subset=["rebalance_date"])[
        ["rebalance_date", "turnover"]
    ].copy()
    to_df["rebalance_date"] = pd.to_datetime(to_df["rebalance_date"])
    fig = go.Figure(
        go.Bar(
            x=to_df["rebalance_date"],
            y=(to_df["turnover"] * 100).round(1),
            marker_color="#e67e22",
        )
    )
    fig.update_layout(
        title="リバランスごとのターンオーバー率",
        xaxis_title="リバランス日",
        yaxis_title="ターンオーバー率 (%)",
        height=280,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


# ──────────────────────────────────────────────
# タブ1: 単一銘柄バックテスト
# ──────────────────────────────────────────────
with tab_single:
    if run_button:
        start_str = start_date.strftime("%Y-%m-%d") if start_date else None
        end_str = end_date.strftime("%Y-%m-%d") if end_date else None

        try:
            result_df, metrics, price_series = _run_backtest(
                market=market,
                symbol=symbol,
                model_type=model_type,
                ensemble=ensemble,
                threshold=threshold,
                source=source,
                start_date_str=start_str,
                end_date_str=end_str,
                train_ratio=train_ratio,
                initial_cash=initial_cash,
                fee_rate=fee_rate,
                slippage=slippage,
                stop_loss_pct=stop_loss_pct,
                take_profit_pct=take_profit_pct,
                position_sizing=position_sizing,
                position_fraction=position_fraction,
                atr_risk_pct=atr_risk_pct,
                atr_multiplier=atr_multiplier,
            )

            # ── 1. メトリクスカード ──
            st.subheader("📊 パフォーマンスサマリー")
            m1, m2, m3, m4, m5, m6, m7 = st.columns(7)

            total_return_pct = metrics.get("total_return", 0) * 100
            win_rate_pct = metrics.get("win_rate", 0) * 100
            max_dd_pct = metrics.get("max_drawdown", 0) * 100
            pf = metrics.get("profit_factor")
            pf_str = f"{pf:.2f}" if pf is not None else "∞"

            m1.metric("総リターン", f"{total_return_pct:+.2f}%", delta=None)
            m2.metric("勝率", f"{win_rate_pct:.1f}%")
            m3.metric("シャープ比", f"{metrics.get('sharpe_ratio', 0):.3f}")
            m4.metric("最大ドローダウン", f"{max_dd_pct:.2f}%")
            m5.metric("取引回数", f"{metrics.get('num_trades', 0)} 回")
            m6.metric("最終資産", f"¥{metrics.get('final_cash', initial_cash):,.0f}")
            m7.metric("プロフィットファクター", pf_str)

            st.divider()

            # ── 2. 株価チャート + シグナル ──
            st.subheader("📉 株価と売買シグナル")
            price_fig = build_price_signal_chart(result_df, symbol, price_series)
            st.plotly_chart(price_fig, use_container_width=True)

            # ── 3. エクイティカーブ ──
            st.subheader("💹 エクイティカーブ")
            eq_fig = build_equity_curve_chart(result_df, initial_cash)
            st.plotly_chart(eq_fig, use_container_width=True)

            # ── 4. 売買ログ ──
            st.subheader("📋 売買ログ")
            if result_df is not None and not result_df.empty:
                log_df = result_df.copy()
                log_df["date"] = pd.to_datetime(log_df["date"]).dt.strftime("%Y-%m-%d")
                log_df["price"] = log_df["price"].map(lambda x: f"{x:,.2f}")
                log_df["cash"] = log_df["cash"].map(lambda x: f"¥{x:,.0f}")

                action_emoji = {
                    "buy": "🟢 買い",
                    "sell": "🔴 売り",
                    "final_sell": "🔴 売り(期末)",
                    "stop_loss": "⛔ 損切り",
                    "take_profit": "✅ 利確",
                }
                log_df["action"] = log_df["action"].map(lambda a: action_emoji.get(a, a))
                log_df.columns = ["日付", "アクション", "価格", "数量", "資産"]
                st.dataframe(log_df, use_container_width=True, height=300)

                # ── 5. CSV エクスポート ──
                col_dl1, col_dl2 = st.columns(2)
                with col_dl1:
                    trade_csv = result_df.to_csv(index=False).encode("utf-8-sig")
                    st.download_button(
                        "📥 売買ログをCSVでダウンロード",
                        trade_csv,
                        file_name=f"backtest_{market}_{symbol}_trades.csv",
                        mime="text/csv",
                    )
                with col_dl2:
                    metrics_csv = pd.DataFrame([metrics]).to_csv(index=False).encode("utf-8-sig")
                    st.download_button(
                        "📥 メトリクスをCSVでダウンロード",
                        metrics_csv,
                        file_name=f"backtest_{market}_{symbol}_metrics.csv",
                        mime="text/csv",
                    )
            else:
                st.info("売買ログがありません。閾値を下げるか、データソースを確認してください。")

        except Exception as e:
            st.error(f"バックテスト実行中にエラーが発生しました:\n\n```\n{e}\n```")
            logger.error("バックテストUIエラー", exc_info=True)

    else:
        # 初期表示
        st.info(
            "👈 サイドバーでパラメータを設定し、「▶ バックテスト実行」ボタンを押してください。\n\n"
            "**ヒント**: データが DB にない銘柄の場合はデータソースを `api` に変更してください。"
        )

        # パラメータ確認テーブル
        with st.expander("現在のパラメータを確認する"):
            params = {
                "マーケット": market,
                "銘柄コード": symbol,
                "データソース": source,
                "開始日": str(start_date) if start_date else "（自動）",
                "終了日": str(end_date) if end_date else "（自動）",
                "学習比率": f"{train_ratio:.0%}",
                "モデルタイプ": "アンサンブル" if ensemble else model_type,
                "シグナル閾値": f"{threshold:.3f}",
                "初期資金": f"¥{initial_cash:,}",
                "手数料率": f"{fee_rate:.4f}",
                "スリッページ": f"{slippage:.4f}",
                "ストップロス": f"{stop_loss_pct:.0%}" if stop_loss_pct else "無効",
                "テイクプロフィット": f"{take_profit_pct:.0%}" if take_profit_pct else "無効",
                "ポジションサイジング": position_sizing,
                "ポジション比率": f"{position_fraction:.0%}" if position_sizing != "full" else "—",
            }
            st.table(pd.DataFrame(params.items(), columns=["パラメータ", "値"]))


# ──────────────────────────────────────────────
# タブ2: ポートフォリオバックテスト
# ──────────────────────────────────────────────
with tab_portfolio:
    if pf_run_button:
        try:
            equity_df, pf_metrics, holdings_df = _run_portfolio(
                pf_market=pf_market,
                pf_model_type=pf_model_type,
                pf_ensemble=pf_ensemble,
                pf_top_n=pf_top_n,
                pf_rebalance_freq=pf_rebalance_freq,
                pf_train_ratio=pf_train_ratio,
                pf_source=pf_source,
                pf_initial_cash=pf_initial_cash,
                pf_fee_rate=pf_fee_rate,
                pf_threshold=pf_threshold,
            )

            if equity_df.empty:
                st.warning("結果が空です。DBにデータが存在するか確認してください。")
            else:
                # ── 1. メトリクスカード ──
                st.subheader("📊 ポートフォリオ パフォーマンスサマリー")
                pm1, pm2, pm3, pm4, pm5, pm6 = st.columns(6)
                tr = pf_metrics.get("total_return", 0) * 100
                ew = pf_metrics.get("equal_weight_return", 0) * 100
                alpha = pf_metrics.get("alpha_vs_equal", 0) * 100
                sharpe = pf_metrics.get("sharpe_ratio", 0)
                maxdd = pf_metrics.get("max_drawdown", 0) * 100
                final = pf_metrics.get("final_cash", pf_initial_cash)
                pm1.metric("総リターン", f"{tr:+.2f}%")
                pm2.metric("等分BM リターン", f"{ew:+.2f}%")
                pm3.metric("アルファ", f"{alpha:+.2f}%")
                pm4.metric("シャープ比", f"{sharpe:.3f}")
                pm5.metric("最大ドローダウン", f"{maxdd:.2f}%")
                pm6.metric("最終資産", f"¥{final:,.0f}")

                st.divider()

                # ── 2. エクイティカーブ ──
                st.subheader("💹 エクイティカーブ（vs 等分ベンチマーク）")
                pf_eq_fig = build_portfolio_equity_chart(equity_df, pf_initial_cash)
                st.plotly_chart(pf_eq_fig, use_container_width=True)

                # ── 3. 銘柄寄与度 ──
                st.subheader("📌 銘柄寄与度（上位15銘柄）")
                contrib_fig = build_holdings_contrib_chart(holdings_df)
                st.plotly_chart(contrib_fig, use_container_width=True)

                # ── 4. ターンオーバー ──
                st.subheader("🔄 ターンオーバー率")
                to_fig = build_turnover_chart(holdings_df)
                st.plotly_chart(to_fig, use_container_width=True)

                # ── 5. 保有銘柄推移テーブル ──
                st.subheader("📋 保有銘柄推移")
                if not holdings_df.empty:
                    display_df = holdings_df.copy()
                    display_df["weight"] = display_df["weight"].map(lambda x: f"{x:.1%}")
                    display_df["score"] = display_df["score"].map(lambda x: f"{x:.4f}")
                    display_df["price"] = display_df["price"].map(lambda x: f"{x:,.2f}")
                    display_df["turnover"] = display_df["turnover"].map(lambda x: f"{x:.1%}")
                    display_df = display_df.rename(
                        columns={
                            "rebalance_date": "リバランス日",
                            "symbol": "銘柄",
                            "weight": "ウェイト",
                            "score": "スコア",
                            "price": "価格",
                            "qty": "数量",
                            "turnover": "ターンオーバー",
                        }
                    )
                    st.dataframe(display_df, use_container_width=True, height=400)

                    # ── 6. CSV エクスポート ──
                    col_pf_dl1, col_pf_dl2 = st.columns(2)
                    with col_pf_dl1:
                        eq_csv = equity_df.to_csv(index=False).encode("utf-8-sig")
                        st.download_button(
                            "📥 エクイティカーブをCSVでダウンロード",
                            eq_csv,
                            file_name=f"portfolio_equity_{pf_market}.csv",
                            mime="text/csv",
                        )
                    with col_pf_dl2:
                        hold_csv = holdings_df.to_csv(index=False).encode("utf-8-sig")
                        st.download_button(
                            "📥 保有銘柄をCSVでダウンロード",
                            hold_csv,
                            file_name=f"portfolio_holdings_{pf_market}.csv",
                            mime="text/csv",
                        )
                else:
                    st.info("保有銘柄データがありません。")

        except Exception as e:
            st.error(f"ポートフォリオバックテスト実行中にエラーが発生しました:\n\n```\n{e}\n```")
            logger.error("ポートフォリオバックテストUIエラー", exc_info=True)

    else:
        st.info(
            "👈 サイドバーの「📊 ポートフォリオ設定」でパラメータを設定し、"
            "「▶ ポートフォリオ実行」ボタンを押してください。\n\n"
            "**動作概要**: DB内の全銘柄（またはマーケット絞り込み）を対象に、"
            "各銘柄の予測スコアを計算し、上位 Top-N 銘柄へソフトマックス比例配分で投資します。"
        )
        with st.expander("ポートフォリオ設定を確認する"):
            pf_params = {
                "マーケット": pf_market,
                "Top-N": pf_top_n,
                "リバランス頻度": pf_rebalance_freq,
                "学習比率": f"{pf_train_ratio:.0%}",
                "データソース": pf_source,
                "モデルタイプ": "アンサンブル" if pf_ensemble else pf_model_type,
                "シグナル閾値": f"{pf_threshold:.3f}",
                "初期資金": f"¥{pf_initial_cash:,}",
                "手数料率": f"{pf_fee_rate:.4f}",
            }
            st.table(pd.DataFrame(pf_params.items(), columns=["パラメータ", "値"]))


# ──────────────────────────────────────────────
# Walk-Forward 実行ロジック（キャッシュ付き）
# ──────────────────────────────────────────────
@st.cache_data(show_spinner="Walk-Forward 検証中...")
def _run_walk_forward(
    market,
    symbol,
    model_type,
    ensemble,
    threshold,
    source,
    n_splits,
    initial_cash,
    fee_rate,
    slippage,
    stop_loss_pct,
    take_profit_pct,
    position_sizing,
    position_fraction,
    atr_risk_pct=0.02,
    atr_multiplier=1.0,
):
    """Walk-Forward 検証をキャッシュ付きで実行する。"""
    from src.services.backtest_pipeline import run_backtest_walk_forward

    _, _, wf_df = run_backtest_walk_forward(
        market=market,
        symbol=symbol,
        model_type=model_type,
        threshold=threshold,
        source=source,
        n_splits=n_splits,
        initial_cash=initial_cash,
        fee_rate=fee_rate,
        slippage=slippage,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
        position_sizing=position_sizing,
        position_fraction=position_fraction,
        atr_risk_pct=atr_risk_pct,
        atr_multiplier=atr_multiplier,
        ensemble=ensemble,
    )
    return wf_df


# ──────────────────────────────────────────────
# パラメータ最適化 実行ロジック（キャッシュ付き）
# ──────────────────────────────────────────────
@st.cache_data(show_spinner="パラメータ最適化中（時間がかかります）...", ttl=1800)
def _run_optimization(
    market,
    symbol,
    model_type,
    ensemble,
    source,
    n_splits,
    initial_cash,
    fee_rate,
    slippage,
    threshold_min,
    threshold_max,
    threshold_step,
    optimize_risk,
):
    """パラメータ最適化をキャッシュ付きで実行する。"""
    from src.services.backtest_optimize_pipeline import run_optimization

    result_df = run_optimization(
        market=market,
        symbol=symbol,
        model_type=model_type,
        ensemble=ensemble,
        source=source,
        n_splits=n_splits,
        initial_cash=initial_cash,
        fee_rate=fee_rate,
        slippage=slippage,
        threshold_min=threshold_min,
        threshold_max=threshold_max,
        threshold_step=threshold_step,
        optimize_risk=optimize_risk,
    )
    return result_df


# ──────────────────────────────────────────────
# Walk-Forward グラフ
# ──────────────────────────────────────────────
def build_wf_fold_chart(wf_df: pd.DataFrame) -> go.Figure:
    """Fold 別リターン（棒）+ シャープ比（折れ線・第2Y軸）のチャートを生成する。"""
    if wf_df is None or wf_df.empty:
        return go.Figure()

    df = wf_df.copy()
    folds = df["fold"].astype(str)
    returns_pct = df["total_return"] * 100

    fig = go.Figure()

    # リターン棒グラフ (左Y軸)
    bar_colors = ["#2ecc71" if r >= 0 else "#e74c3c" for r in returns_pct]
    fig.add_trace(
        go.Bar(
            x=folds,
            y=returns_pct,
            name="リターン (%)",
            marker_color=bar_colors,
            yaxis="y1",
        )
    )

    # シャープ比折れ線 (右Y軸)
    fig.add_trace(
        go.Scatter(
            x=folds,
            y=df["sharpe_ratio"],
            mode="lines+markers",
            name="シャープ比",
            line=dict(color="#9b59b6", width=2),
            marker=dict(size=8),
            yaxis="y2",
        )
    )

    fig.add_hline(y=0, line_dash="dot", line_color="gray", yref="y1")

    fig.update_layout(
        title="Fold 別 リターン & シャープ比",
        xaxis_title="Fold",
        yaxis=dict(title="リターン (%)", side="left"),
        yaxis2=dict(title="シャープ比", side="right", overlaying="y", showgrid=False),
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


def build_optimize_heatmap(result_df: pd.DataFrame, sort_by: str) -> go.Figure:
    """threshold × stop_loss の 2D ヒートマップ（optimize_risk=True 時のみ使用）。"""
    if result_df is None or result_df.empty:
        return go.Figure()
    if "stop_loss_pct" not in result_df.columns:
        return go.Figure()

    pivot = result_df.pivot_table(
        index="stop_loss_pct", columns="threshold", values=sort_by, aggfunc="mean"
    )
    if pivot.empty:
        return go.Figure()

    fig = go.Figure(
        go.Heatmap(
            z=pivot.values,
            x=[f"{v:.3f}" for v in pivot.columns],
            y=[str(v) if v is not None else "なし" for v in pivot.index],
            colorscale="RdYlGn",
            colorbar=dict(title=sort_by),
            hovertemplate="閾値: %{x}<br>SL: %{y}<br>%{z:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"ヒートマップ: {sort_by}（threshold × stop_loss_pct）",
        xaxis_title="シグナル閾値",
        yaxis_title="ストップロス率",
        height=400,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    return fig


# ──────────────────────────────────────────────
# タブ3: Walk-Forward バックテスト
# ──────────────────────────────────────────────
with tab_wf:
    if wf_run_button:
        start_str_wf = start_date.strftime("%Y-%m-%d") if start_date else None
        end_str_wf = end_date.strftime("%Y-%m-%d") if end_date else None

        try:
            wf_df = _run_walk_forward(
                market=market,
                symbol=symbol,
                model_type=model_type,
                ensemble=ensemble,
                threshold=threshold,
                source=source,
                n_splits=wf_n_splits,
                initial_cash=initial_cash,
                fee_rate=fee_rate,
                slippage=slippage,
                stop_loss_pct=stop_loss_pct,
                take_profit_pct=take_profit_pct,
                position_sizing=position_sizing,
                position_fraction=position_fraction,
                atr_risk_pct=atr_risk_pct,
                atr_multiplier=atr_multiplier,
            )

            if wf_df is None or wf_df.empty:
                st.warning("Walk-Forward 結果が空です。データソースとパラメータを確認してください。")
            else:
                # ── 1. サマリー（全 fold 平均） ──
                st.subheader("📊 Walk-Forward サマリー（全 Fold 平均）")
                wf_mean = wf_df.mean(numeric_only=True)
                wf_std = wf_df.std(numeric_only=True)

                wm1, wm2, wm3, wm4, wm5, wm6 = st.columns(6)
                wm1.metric("平均リターン", f"{wf_mean.get('total_return', 0) * 100:+.2f}%")
                wm2.metric("平均シャープ比", f"{wf_mean.get('sharpe_ratio', 0):.3f}")
                wm3.metric("平均最大DD", f"{wf_mean.get('max_drawdown', 0) * 100:.2f}%")
                wm4.metric("平均勝率", f"{wf_mean.get('win_rate', 0) * 100:.1f}%")
                wm5_pf = wf_mean.get("profit_factor")
                wm5.metric("平均PF", f"{wm5_pf:.2f}" if wm5_pf is not None else "∞")
                wm6.metric("平均取引回数", f"{wf_mean.get('num_trades', 0):.1f} 回")

                # ── 2. 安定性評価 ──
                ret_std = wf_std.get("total_return", 0)
                if ret_std < 0.05:
                    stability_color = "success"
                    stability_msg = f"✅ 安定（リターン標準偏差 {ret_std * 100:.2f}% — 低リスク）"
                elif ret_std < 0.15:
                    stability_color = "warning"
                    stability_msg = f"⚠️ やや不安定（リターン標準偏差 {ret_std * 100:.2f}%）"
                else:
                    stability_color = "error"
                    stability_msg = f"❌ 不安定（リターン標準偏差 {ret_std * 100:.2f}% — 過学習の可能性）"

                getattr(st, stability_color)(stability_msg)

                st.divider()

                # ── 3. Fold 別グラフ ──
                st.subheader("📉 Fold 別 リターン & シャープ比")
                wf_fig = build_wf_fold_chart(wf_df)
                st.plotly_chart(wf_fig, use_container_width=True)

                # ── 4. Fold 別テーブル ──
                st.subheader("📋 Fold 別 詳細メトリクス")
                wf_display = wf_df.copy()
                for col in ["total_return", "win_rate", "max_drawdown"]:
                    if col in wf_display.columns:
                        wf_display[col] = wf_display[col].map(lambda x: f"{x * 100:+.2f}%")
                for col in ["sharpe_ratio"]:
                    if col in wf_display.columns:
                        wf_display[col] = wf_display[col].map(lambda x: f"{x:.3f}")
                for col in ["profit_factor"]:
                    if col in wf_display.columns:
                        wf_display[col] = wf_display[col].map(
                            lambda x: f"{x:.2f}" if x is not None else "∞"
                        )
                col_rename = {
                    "fold": "Fold",
                    "val_start": "検証開始",
                    "val_end": "検証終了",
                    "train_rows": "学習行数",
                    "val_rows": "検証行数",
                    "total_return": "リターン",
                    "sharpe_ratio": "シャープ比",
                    "max_drawdown": "最大DD",
                    "win_rate": "勝率",
                    "profit_factor": "PF",
                    "num_trades": "取引回数",
                }
                wf_display = wf_display.rename(
                    columns={k: v for k, v in col_rename.items() if k in wf_display.columns}
                )
                st.dataframe(wf_display, use_container_width=True)

                # ── 5. CSV エクスポート ──
                wf_csv = wf_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    "📥 Walk-Forward 結果を CSV でダウンロード",
                    wf_csv,
                    file_name=f"walk_forward_{market}_{symbol}.csv",
                    mime="text/csv",
                )

        except Exception as e:
            st.error(f"Walk-Forward 実行中にエラーが発生しました:\n\n```\n{e}\n```")
            logger.error("Walk-Forward UIエラー", exc_info=True)

    else:
        st.info(
            "👈 サイドバーの共有パラメータ（基本設定・モデル設定・取引条件・リスク管理）と"
            "「🔄 Walk-Forward 設定」でフォールド数を設定し、「▶ Walk-Forward 実行」を押してください。\n\n"
            "**Walk-Forward とは**: データを時系列で複数のフォールドに分割し、学習→検証を繰り返すことで"
            "モデルの汎化性能を評価します。単一バックテストより過学習リスクを排除できます。"
        )
        with st.expander("現在のパラメータを確認する"):
            wf_params = {
                "マーケット": market,
                "銘柄コード": symbol,
                "データソース": source,
                "モデルタイプ": "アンサンブル" if ensemble else model_type,
                "シグナル閾値": f"{threshold:.3f}",
                "フォールド数": wf_n_splits,
                "初期資金": f"¥{initial_cash:,}",
                "手数料率": f"{fee_rate:.4f}",
                "ストップロス": f"{stop_loss_pct:.0%}" if stop_loss_pct else "無効",
                "テイクプロフィット": f"{take_profit_pct:.0%}" if take_profit_pct else "無効",
                "ポジションサイジング": position_sizing,
            }
            st.table(pd.DataFrame(wf_params.items(), columns=["パラメータ", "値"]))


# ──────────────────────────────────────────────
# タブ4: パラメータ最適化
# ──────────────────────────────────────────────
with tab_optimize:
    if opt_run_button:
        try:
            opt_df = _run_optimization(
                market=market,
                symbol=symbol,
                model_type=model_type,
                ensemble=ensemble,
                source=source,
                n_splits=opt_n_splits,
                initial_cash=initial_cash,
                fee_rate=fee_rate,
                slippage=slippage,
                threshold_min=opt_threshold_min,
                threshold_max=opt_threshold_max,
                threshold_step=opt_threshold_step,
                optimize_risk=opt_optimize_risk,
            )

            if opt_df is None or opt_df.empty:
                st.warning("最適化結果が空です。パラメータ範囲を確認してください。")
            else:
                sorted_df = opt_df.sort_values(opt_sort_by, ascending=False).reset_index(drop=True)
                best = sorted_df.iloc[0]

                # ── 1. ベストパラメータ表示 ──
                st.subheader("🏆 最良パラメータ")
                best_sl = best.get("stop_loss_pct")
                best_tp = best.get("take_profit_pct")
                best_msg = (
                    f"**閾値**: `{best['threshold']:.3f}` | "
                    f"**SL**: `{f'{best_sl:.0%}' if best_sl else 'なし'}` | "
                    f"**TP**: `{f'{best_tp:.0%}' if best_tp else 'なし'}` | "
                    f"**{opt_sort_by}**: `{best[opt_sort_by]:.4f}`"
                )
                st.success(best_msg)

                # ── 2. 適用ボタン ──
                if st.button("📋 このパラメータを単一銘柄タブに適用", key="opt_apply"):
                    st.session_state["apply_threshold"] = float(best["threshold"])
                    if best_sl is not None:
                        st.session_state["apply_stop_loss"] = float(best_sl)
                    if best_tp is not None:
                        st.session_state["apply_take_profit"] = float(best_tp)
                    st.success("✅ パラメータを適用しました。単一銘柄タブで確認してください。")
                    st.info(
                        "ℹ️ Streamlit の仕様上、サイドバーのウィジェット値は次回再読み込み時に反映されます。"
                        "「▶ バックテスト実行」を押すと最新値が使われます。"
                    )

                # ── 3. 保存ボタン ──
                if st.button("💾 最適パラメータを JSON に保存", key="opt_save"):
                    import json
                    from pathlib import Path

                    params_path = Path(__file__).parent / "config" / "optimal_params.json"
                    params_path.parent.mkdir(exist_ok=True)
                    existing = {}
                    if params_path.exists():
                        try:
                            existing = json.loads(params_path.read_text(encoding="utf-8"))
                        except Exception:
                            existing = {}

                    key = f"{market}_{symbol}"
                    existing[key] = {
                        "threshold": float(best["threshold"]),
                        "stop_loss_pct": float(best_sl) if best_sl is not None else None,
                        "take_profit_pct": float(best_tp) if best_tp is not None else None,
                        "sort_by": opt_sort_by,
                        "score": float(best[opt_sort_by]),
                        "updated_at": pd.Timestamp.now().isoformat(),
                    }
                    params_path.write_text(
                        json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8"
                    )
                    st.success(f"✅ 保存しました: `{params_path}`")

                st.divider()

                # ── 4. ヒートマップ（optimize_risk=True 時のみ） ──
                if opt_optimize_risk:
                    st.subheader(f"🗺️ ヒートマップ: {opt_sort_by}")
                    hmap_fig = build_optimize_heatmap(sorted_df, opt_sort_by)
                    st.plotly_chart(hmap_fig, use_container_width=True)

                # ── 5. 全結果テーブル ──
                st.subheader(f"📋 全組み合わせ結果（{opt_sort_by} 降順）")
                display_opt = sorted_df.copy()
                for col in ["total_return", "win_rate", "max_drawdown"]:
                    if col in display_opt.columns:
                        display_opt[col] = display_opt[col].map(lambda x: f"{x * 100:+.2f}%")
                for col in ["sharpe_ratio"]:
                    if col in display_opt.columns:
                        display_opt[col] = display_opt[col].map(lambda x: f"{x:.3f}")
                for col in ["profit_factor"]:
                    if col in display_opt.columns:
                        display_opt[col] = display_opt[col].map(
                            lambda x: f"{x:.2f}" if x is not None else "∞"
                        )
                st.dataframe(display_opt, use_container_width=True, height=400)

                # ── 6. CSV エクスポート ──
                opt_csv = opt_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    "📥 最適化結果を CSV でダウンロード",
                    opt_csv,
                    file_name=f"optimize_{market}_{symbol}.csv",
                    mime="text/csv",
                )

        except Exception as e:
            st.error(f"パラメータ最適化中にエラーが発生しました:\n\n```\n{e}\n```")
            logger.error("最適化UIエラー", exc_info=True)

    else:
        st.info(
            "👈 サイドバーの「⚡ 最適化設定」でパラメータ範囲を設定し、「▶ 最適化実行」ボタンを押してください。\n\n"
            "**動作概要**: 指定した閾値レンジをグリッドサーチし、各組み合わせで Walk-Forward 検証を実行します。\n"
            "最良パラメータは `python/config/optimal_params.json` に保存でき、単一銘柄タブへの適用も可能です。"
        )
        with st.expander("現在の最適化設定を確認する"):
            opt_params_display = {
                "マーケット": market,
                "銘柄コード": symbol,
                "データソース": source,
                "モデルタイプ": "アンサンブル" if ensemble else model_type,
                "フォールド数": opt_n_splits,
                "閾値 最小": f"{opt_threshold_min:.3f}",
                "閾値 最大": f"{opt_threshold_max:.3f}",
                "閾値 ステップ": f"{opt_threshold_step:.3f}",
                "SL/TP グリッドサーチ": "有効" if opt_optimize_risk else "無効",
                "ソート基準": opt_sort_by,
            }
            st.table(pd.DataFrame(opt_params_display.items(), columns=["パラメータ", "値"]))
