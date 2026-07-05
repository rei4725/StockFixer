from typing import Any, Callable, Optional

import pandas as pd

from src.backtest.data_port import get_backtest_data_port
from src.backtest.metrics import compute_cost_comparison_metrics
from src.backtest.task import BacktestTask, ReturnRegressionTask

# 動的スリッページの平均出来高算出ウィンドウ（バー数）— #494
_AVG_VOL_WINDOW = 20


class Backtester:
    def __init__(
        self,
        model_manager: Any,
        signal_generator: Any,
        data_loader: Any,
        start_date: Any,
        end_date: Any,
        market: str,
        symbol: str,
        initial_cash: float = 1_000_000,
        fee_rate: float = 0.0,
        slippage: float = 0.0,
        stop_loss_pct: Optional[float] = None,
        take_profit_pct: Optional[float] = None,
        position_sizing: str = "full",
        position_fraction: float = 0.5,
        atr_risk_pct: float = 0.02,
        atr_multiplier: float = 1.0,
        atr_min_fraction: float = 0.1,
        atr_max_fraction: float = 1.0,
        enable_short: bool = False,
        slippage_fn: Optional[Callable[[int, float, float], float]] = None,
        execution_lag: int = 1,
        include_monte_carlo: bool = False,
    ):
        self.model_manager = model_manager
        self.signal_generator = signal_generator
        self.data_loader = data_loader
        self.start_date = start_date
        self.end_date = end_date
        self.market = market
        self.symbol = symbol
        self.initial_cash = initial_cash
        self.fee_rate = fee_rate
        self.slippage = slippage
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.position_sizing = position_sizing
        self.position_fraction = position_fraction
        self.atr_risk_pct = atr_risk_pct
        self.atr_multiplier = atr_multiplier
        self.atr_min_fraction = max(0.0, atr_min_fraction)
        self.atr_max_fraction = min(1.0, max(0.0, atr_max_fraction))
        self.enable_short = enable_short
        self.slippage_fn = slippage_fn  # R-210: 動的スリッページ関数 (qty, price, avg_vol) -> rate
        # #493: シグナル約定の実行ラグ（バー数）。1=翌バー約定（引け後シグナル→翌寄り）。
        # 0 で従来の同バー(当日Close)約定に戻す（後方互換）。
        self.execution_lag = max(0, int(execution_lag))
        # #374: Monte Carlo リスク統計（ブートストラップ）を metrics に含めるか
        self.include_monte_carlo = include_monte_carlo

    def _get_slippage(self, qty: int, price: float, volume: float = 0.0) -> float:
        """有効スリッページ率を返す（#494）。

        基準スプレッド（フラット self.slippage）に、slippage_fn が設定されていれば
        サイズ・流動性依存のマーケットインパクト（R-210 平方根モデル）を加算する。
        出来高不明（volume<=0）や数量0の場合はフラット基準のみ。
        """
        base = self.slippage
        if self.slippage_fn is not None and volume > 0 and qty > 0:
            return base + self.slippage_fn(qty, price, volume)
        return base

    def run(
        self,
        model_name: str,
        model_type: Optional[str] = None,
        source: str = "file",
        task: Optional[BacktestTask] = None,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """
        バックテストを実行する。

        Args:
            model_name: 使用するモデル名
            model_type: モデルタイプ（任意）
            source: データソース "file"(DB特徴量), "api"(yfinance直接), "raw"(DBのOHLCVから再生成)
            task: BacktestTask 実装（Noneなら ReturnRegressionTask()）

        Returns:
            (result_df, metrics) のタプル
        """
        if task is None:
            task = ReturnRegressionTask()

        # 1. データ取得
        if source == "raw":
            df = self._load_from_raw()
        else:
            df = self.data_loader.get_stock_data_auto(
                self.market, self.symbol, self.start_date, self.end_date, source=source
            )

        # 2. 特徴量生成（テクニカル指標付与）
        df = get_backtest_data_port().add_technical_indicators(df)

        # 3. ラベル生成（タスクに委譲）
        df[task.label_col] = task.make_labels(df)
        df = df.dropna(subset=[task.label_col])

        # 4. モデル学習・予測
        feature_cols = [c for c in df.columns if c not in (task.label_col, "Close", "close")]
        X = df[feature_cols].dropna()
        model = self.model_manager.get_model(model_name)
        prediction = pd.Series(model.predict(X), index=X.index)

        # 5. シグナル生成（タスクに委譲）
        signal = task.make_signal(prediction)

        # 6. 仮想売買シミュレーション
        result_df, metrics = self.simulate_trading(df.loc[X.index], signal)
        return result_df, metrics

    def simulate_trading(
        self, df: pd.DataFrame, signal: pd.Series, pred: Optional[pd.Series] = None
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """
        仮想売買シミュレーションを実行する。

        ストップロス・テイクプロフィット・ポジションサイジングに対応。

        Args:
            df: Close 列を含む DataFrame
            signal: シグナル Series (1=buy, -1=sell, 0=hold)
            pred: 予測値 Series（ポジションサイジング "confidence" モードで使用）

        Returns:
            (result_df, metrics) のタプル
        """
        cash = self.initial_cash
        cash_gross = self.initial_cash
        position = 0
        position_price = 0.0
        short_position = 0
        short_price = 0.0
        short_trade_log: list[dict[str, Any]] = []
        trade_log = []
        # 各バーの mark-to-market equity 記録 (date, equity_net, equity_gross) — #492
        equity_records: list[tuple[Any, float, float]] = []

        close_col = "Close" if "Close" in df.columns else "close"
        # #494: 動的スリッページ用の平均出来高系列（Volume 列が無ければ全て0でフラットのみ適用）
        if "Volume" in df.columns:
            avg_vol = df["Volume"].rolling(_AVG_VOL_WINDOW, min_periods=1).mean()
        else:
            avg_vol = pd.Series(0.0, index=df.index)
        # #493: 実行ラグ。シグナル/確信度を execution_lag バー遅延させ、翌バーで約定する。
        # 約定価格は翌バー始値(Open)。Open 列が無ければ Close にフォールバック。
        # SL/TP・mark-to-market は従来どおり当バー Close 基準（判定・約定とも Close で整合）。
        # 最終バーまでに約定先バーが無いシグナルは shift により自然に破棄される。
        exec_col = "Open" if (self.execution_lag > 0 and "Open" in df.columns) else close_col
        if self.execution_lag > 0:
            exec_signal = signal.reindex(df.index).shift(self.execution_lag)
            exec_pred = (
                pred.reindex(df.index).shift(self.execution_lag) if pred is not None else None
            )
        else:
            exec_signal = signal
            exec_pred = pred

        for date in df.index:
            price = df.loc[date, close_col]
            exec_price = df.loc[date, exec_col]
            vol = float(avg_vol.loc[date]) if date in avg_vol.index else 0.0
            raw_sig = exec_signal.get(date, 0) if date in exec_signal.index else 0
            sig = 0 if pd.isna(raw_sig) else int(raw_sig)

            # ストップロス / テイクプロフィット判定（シグナルより先に評価）
            if position > 0 and position_price > 0:
                change_from_entry = (price - position_price) / position_price

                if self.stop_loss_pct is not None and change_from_entry <= -self.stop_loss_pct:
                    slip = self._get_slippage(position, price, vol)
                    proceeds = position * price * (1 - self.fee_rate - slip)
                    proceeds_gross = position * price
                    cash += proceeds
                    cash_gross += proceeds_gross
                    trade_log.append(
                        {
                            "date": date,
                            "action": "stop_loss",
                            "price": price,
                            "qty": position,
                            "cash": cash,
                            "cash_gross": cash_gross,
                        }
                    )
                    position = 0
                    position_price = 0.0
                    equity_records.append(
                        self._mark_to_market(
                            date, cash, cash_gross, position, price, short_position, short_price
                        )
                    )
                    continue

                if self.take_profit_pct is not None and change_from_entry >= self.take_profit_pct:
                    slip = self._get_slippage(position, price, vol)
                    proceeds = position * price * (1 - self.fee_rate - slip)
                    proceeds_gross = position * price
                    cash += proceeds
                    cash_gross += proceeds_gross
                    trade_log.append(
                        {
                            "date": date,
                            "action": "take_profit",
                            "price": price,
                            "qty": position,
                            "cash": cash,
                            "cash_gross": cash_gross,
                        }
                    )
                    position = 0
                    position_price = 0.0
                    equity_records.append(
                        self._mark_to_market(
                            date, cash, cash_gross, position, price, short_position, short_price
                        )
                    )
                    continue

            # シグナルに基づく売買
            # ショートカバー: 買いシグナル + ショートポジション保有
            if sig == 1 and self.enable_short and short_position > 0:
                # エントリー売り: short_price * qty * (1 - fee - slip)
                # カバー買い: exec_price * qty * (1 + fee + slip)
                slip = self._get_slippage(short_position, exec_price, vol)
                net_pnl = short_price * short_position * (
                    1 - self.fee_rate - slip
                ) - exec_price * short_position * (1 + self.fee_rate + slip)
                pnl = (short_price - exec_price) * short_position
                cash += net_pnl
                cash_gross += pnl
                short_trade_log.append(
                    {
                        "entry_price": short_price,
                        "exit_price": exec_price,
                        "qty": short_position,
                        "pnl": net_pnl,
                    }
                )
                trade_log.append(
                    {
                        "date": date,
                        "action": "short_cover",
                        "price": exec_price,
                        "qty": short_position,
                        "cash": cash,
                        "cash_gross": cash_gross,
                    }
                )
                short_position = 0
                short_price = 0.0
            elif sig == 1 and position == 0:
                # Buy（翌バー始値 exec_price で約定）
                atr_value = df.loc[date, "atr"] if "atr" in df.columns else None
                position_details = self._calc_position_details(
                    cash,
                    exec_price,
                    exec_pred.get(date) if exec_pred is not None else None,
                    atr_value=atr_value,
                )
                qty = position_details["qty"]
                if qty > 0:
                    slip = self._get_slippage(qty, exec_price, vol)
                    cost = qty * exec_price * (1 + self.fee_rate + slip)
                    cost_gross = qty * exec_price
                    cash -= cost
                    cash_gross -= cost_gross
                    position += qty
                    position_price = exec_price
                    trade_log.append(
                        {
                            "date": date,
                            "action": "buy",
                            "price": exec_price,
                            "qty": qty,
                            "cash": cash,
                            "cash_gross": cash_gross,
                            "position_fraction": position_details["position_fraction"],
                            "position_value": round(cost_gross, 2),
                            "sizing_mode": position_details["sizing_mode"],
                            "atr_value": position_details["atr_value"],
                            "atr_stop_distance": position_details["atr_stop_distance"],
                            "atr_risk_amount": position_details["atr_risk_amount"],
                            "atr_fallback_used": position_details["atr_fallback_used"],
                        }
                    )
            elif sig == -1 and position > 0:
                # Sell (ロング決済・翌バー始値 exec_price で約定)
                slip = self._get_slippage(position, exec_price, vol)
                proceeds = position * exec_price * (1 - self.fee_rate - slip)
                proceeds_gross = position * exec_price
                cash += proceeds
                cash_gross += proceeds_gross
                trade_log.append(
                    {
                        "date": date,
                        "action": "sell",
                        "price": exec_price,
                        "qty": position,
                        "cash": cash,
                        "cash_gross": cash_gross,
                    }
                )
                position = 0
                position_price = 0.0
            elif sig == -1 and self.enable_short and position == 0 and short_position == 0:
                # ショートエントリー（翌バー始値 exec_price で約定）
                atr_value = df.loc[date, "atr"] if "atr" in df.columns else None
                position_details = self._calc_position_details(
                    cash,
                    exec_price,
                    exec_pred.get(date) if exec_pred is not None else None,
                    atr_value=atr_value,
                )
                qty = position_details["qty"]
                if qty > 0:
                    short_position = qty
                    short_price = exec_price
                    trade_log.append(
                        {
                            "date": date,
                            "action": "short",
                            "price": exec_price,
                            "qty": qty,
                            "cash": cash,
                            "cash_gross": cash_gross,
                        }
                    )

            # 各バー末の mark-to-market equity を記録（保有中の含み損益込み・#492）
            equity_records.append(
                self._mark_to_market(
                    date, cash, cash_gross, position, price, short_position, short_price
                )
            )

        # 最終日に未決済ポジションを強制決済
        final_vol = float(avg_vol.iloc[-1]) if len(avg_vol) else 0.0
        if position > 0:
            price = df.iloc[-1][close_col]
            slip = self._get_slippage(position, price, final_vol)
            proceeds = position * price * (1 - self.fee_rate - slip)
            proceeds_gross = position * price
            cash += proceeds
            cash_gross += proceeds_gross
            trade_log.append(
                {
                    "date": df.index[-1],
                    "action": "final_sell",
                    "price": price,
                    "qty": position,
                    "cash": cash,
                    "cash_gross": cash_gross,
                }
            )
            position = 0

        # 最終日に未決済ショートを強制返済
        if self.enable_short and short_position > 0:
            price = df.iloc[-1][close_col]
            # エントリー売り: short_price * qty * (1 - fee - slip)
            # カバー買い: price * qty * (1 + fee + slip)
            slip = self._get_slippage(short_position, price, final_vol)
            net_pnl = short_price * short_position * (
                1 - self.fee_rate - slip
            ) - price * short_position * (1 + self.fee_rate + slip)
            pnl = (short_price - price) * short_position
            cash += net_pnl
            cash_gross += pnl
            short_trade_log.append(
                {
                    "entry_price": short_price,
                    "exit_price": price,
                    "qty": short_position,
                    "pnl": net_pnl,
                }
            )
            trade_log.append(
                {
                    "date": df.index[-1],
                    "action": "final_short_cover",
                    "price": price,
                    "qty": short_position,
                    "cash": cash,
                    "cash_gross": cash_gross,
                }
            )
            short_position = 0

        result_df = pd.DataFrame(trade_log)
        equity_net, equity_gross = self._build_equity_curves(equity_records)
        metrics = compute_cost_comparison_metrics(
            result_df,
            self.initial_cash,
            equity_net=equity_net,
            equity_gross=equity_gross,
            include_monte_carlo=self.include_monte_carlo,
        )
        # ショートメトリクスを追加
        if short_trade_log:
            metrics["short_return"] = round(
                sum(t["pnl"] for t in short_trade_log) / self.initial_cash, 6
            )
            metrics["short_num_trades"] = len(short_trade_log)
        else:
            metrics["short_return"] = 0.0
            metrics["short_num_trades"] = 0
        return result_df, metrics

    def _mark_to_market(
        self,
        date: Any,
        cash: float,
        cash_gross: float,
        position: int,
        price: float,
        short_position: int,
        short_price: float,
    ) -> tuple[Any, float, float]:
        """各バーの mark-to-market equity（net / gross）を返す（#492）。

        保有中のロング時価とショート未実現損益を現金に加算し、決済を待たずに
        含み損益を資産曲線へ反映する。これにより max_drawdown の過小評価を防ぐ。
        - long_value : 保有株の時価（コスト無視。清算コストは決済時に計上）
        - short_net  : 現時点でカバーした場合の純損益（往復コスト込み）
        - short_gross: コスト無視のショート未実現損益
        """
        long_value = position * price
        if short_position > 0:
            short_net = short_price * short_position * (
                1 - self.fee_rate - self.slippage
            ) - price * short_position * (1 + self.fee_rate + self.slippage)
            short_gross = (short_price - price) * short_position
        else:
            short_net = 0.0
            short_gross = 0.0
        equity_net = cash + long_value + short_net
        equity_gross = cash_gross + long_value + short_gross
        return date, equity_net, equity_gross

    @staticmethod
    def _build_equity_curves(
        equity_records: list[tuple[Any, float, float]],
    ) -> tuple[pd.Series, pd.Series]:
        """記録した (date, equity_net, equity_gross) 列から日次 equity Series を構築する。

        記録が空の場合は空 Series を返し、metrics 側で従来の決済時点キャッシュ
        ベースの equity にフォールバックさせる（後方互換）。
        """
        if not equity_records:
            empty = pd.Series(dtype=float)
            return empty, empty
        dates = [r[0] for r in equity_records]
        net = pd.Series([r[1] for r in equity_records], index=dates)
        gross = pd.Series([r[2] for r in equity_records], index=dates)
        return net, gross

    def _calc_qty(
        self,
        cash: float,
        price: float,
        pred_value: Optional[float] = None,
        atr_value: Optional[float] = None,
    ) -> int:
        return int(self._calc_position_details(cash, price, pred_value, atr_value)["qty"])

    def _calc_position_details(
        self,
        cash: float,
        price: float,
        pred_value: Optional[float] = None,
        atr_value: Optional[float] = None,
    ) -> dict[str, Any]:
        """
        ポジションサイジングに基づいて購入数量を算出する。

        Args:
            cash: 利用可能な現金
            price: 現在の株価
            pred_value: 予測値（confidence モードで使用）
            atr_value: ATR値（atr モードで使用）

        Returns:
            購入数量と補助情報
        """
        unit_cost = price * (1 + self.fee_rate + self.slippage)
        fallback_used: bool = False
        if unit_cost <= 0 or cash <= 0:
            return {
                "qty": 0,
                "position_fraction": 0.0,
                "sizing_mode": self.position_sizing,
                "atr_value": atr_value,
                "atr_stop_distance": None,
                "atr_risk_amount": None,
                "atr_fallback_used": False,
            }

        if self.position_sizing == "fixed":
            available = cash * self.position_fraction
        elif self.position_sizing == "confidence" and pred_value is not None:
            # 予測確信度（|pred|）に比例して資金を配分
            # |pred| = 0.01 (1%) → fraction ~0.5, |pred| = 0.02+ → fraction ~1.0
            confidence = min(abs(pred_value) * 50, 1.0)
            min_frac = 0.2
            max_frac = 1.0
            fraction = min_frac + confidence * (max_frac - min_frac)
            available = cash * fraction
        elif self.position_sizing == "atr" and atr_value is not None and atr_value > 0:
            # ATR連動ポジションサイジング
            # リスク額 = equity × atr_risk_pct
            # ストップ幅 = ATR × atr_multiplier（1ATR分の価格変動をリスク上限とみなす）
            # 購入株数 = リスク額 / (ATR × atr_multiplier)
            risk_amount = cash * self.atr_risk_pct
            stop_distance = atr_value * self.atr_multiplier
            qty_by_risk = risk_amount / stop_distance
            min_fraction = min(self.atr_min_fraction, self.atr_max_fraction)
            max_fraction = max(self.atr_min_fraction, self.atr_max_fraction)
            min_qty = int((cash * min_fraction) // unit_cost)
            max_qty_by_fraction = int((cash * max_fraction) // unit_cost)
            max_affordable_qty = int(cash // unit_cost)
            effective_max_qty = min(max_affordable_qty, max_qty_by_fraction)
            if effective_max_qty <= 0:
                qty = 0
            else:
                qty = int(min(qty_by_risk, effective_max_qty))
                if min_qty > effective_max_qty:
                    min_qty = effective_max_qty
                qty = max(min_qty, qty)
            return self._build_position_details(
                qty=qty,
                cash=cash,
                unit_cost=unit_cost,
                sizing_mode="atr",
                atr_value=atr_value,
                atr_stop_distance=round(stop_distance, 6),
                atr_risk_amount=round(risk_amount, 6),
            )
        else:
            # "full" モード（デフォルト: 全額投入）
            available = cash
            fallback_used = self.position_sizing == "atr"

        qty = int(available // unit_cost)
        return self._build_position_details(
            qty=qty,
            cash=cash,
            unit_cost=unit_cost,
            sizing_mode="full" if self.position_sizing == "atr" else self.position_sizing,
            atr_value=atr_value,
            atr_fallback_used=fallback_used if self.position_sizing == "atr" else False,
        )

    @staticmethod
    def _build_position_details(
        qty: int,
        cash: float,
        unit_cost: float,
        sizing_mode: str,
        atr_value: Optional[float],
        atr_stop_distance: Optional[float] = None,
        atr_risk_amount: Optional[float] = None,
        atr_fallback_used: bool = False,
    ) -> dict[str, Any]:
        position_value = max(0, qty) * unit_cost
        position_fraction = position_value / cash if cash > 0 else 0.0
        return {
            "qty": max(0, qty),
            "position_fraction": round(position_fraction, 6),
            "sizing_mode": sizing_mode,
            "atr_value": atr_value,
            "atr_stop_distance": atr_stop_distance,
            "atr_risk_amount": atr_risk_amount,
            "atr_fallback_used": atr_fallback_used,
        }

    def _load_from_raw(self) -> pd.DataFrame:
        """market_data_raw テーブルからOHLCVを取得する"""
        df = get_backtest_data_port().get_raw_ohlcv_from_db(
            self.market, self.symbol, self.start_date, self.end_date
        )
        if df is None or df.empty:
            raise ValueError(
                f"market_data_rawにデータがありません: {self.market}/{self.symbol} "
                f"({self.start_date} ～ {self.end_date})\n"
                "先に run_data_creation.py を実行してください。"
            )
        return df
