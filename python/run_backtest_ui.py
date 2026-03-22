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
tab_single, tab_portfolio = st.tabs(["📈 単一銘柄", "📊 ポートフォリオ"])

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
            ["full", "fixed", "confidence"],
            index=0,
            help="full=全額 / fixed=固定比率 / confidence=予測確信度ベース",
        )
        position_fraction = 0.5
        if position_sizing in ("fixed", "confidence"):
            position_fraction = st.slider(
                "ポジション比率",
                min_value=0.1,
                max_value=1.0,
                value=0.5,
                step=0.1,
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
            )

            # ── 1. メトリクスカード ──
            st.subheader("📊 パフォーマンスサマリー")
            m1, m2, m3, m4, m5, m6 = st.columns(6)

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
