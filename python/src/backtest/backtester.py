from typing import Optional

import pandas as pd

from src.backtest.metrics import compute_cost_comparison_metrics
from src.backtest.task import BacktestTask, ReturnRegressionTask


class Backtester:
    def __init__(
        self,
        model_manager,
        signal_generator,
        data_loader,
        start_date,
        end_date,
        market,
        symbol,
        initial_cash=1_000_000,
        fee_rate=0.0,
        slippage=0.0,
        stop_loss_pct: Optional[float] = None,
        take_profit_pct: Optional[float] = None,
        position_sizing: str = "full",
        position_fraction: float = 0.5,
        atr_risk_pct: float = 0.02,
        atr_multiplier: float = 1.0,
        atr_min_fraction: float = 0.1,
        atr_max_fraction: float = 1.0,
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

    def run(
        self,
        model_name: str,
        model_type: Optional[str] = None,
        source: str = "file",
        task: Optional[BacktestTask] = None,
    ):
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
        from src.features.technical_analysis import add_technical_indicators

        df = add_technical_indicators(df)

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
    ):
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
        trade_log = []

        close_col = "Close" if "Close" in df.columns else "close"

        for date in df.index:
            price = df.loc[date, close_col]
            sig = signal.get(date, 0) if date in signal.index else 0

            # ストップロス / テイクプロフィット判定（シグナルより先に評価）
            if position > 0 and position_price > 0:
                change_from_entry = (price - position_price) / position_price

                if self.stop_loss_pct is not None and change_from_entry <= -self.stop_loss_pct:
                    proceeds = position * price * (1 - self.fee_rate - self.slippage)
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
                    continue

                if self.take_profit_pct is not None and change_from_entry >= self.take_profit_pct:
                    proceeds = position * price * (1 - self.fee_rate - self.slippage)
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
                    continue

            # シグナルに基づく売買
            if sig == 1 and position == 0:
                # Buy
                atr_value = df.loc[date, "atr"] if "atr" in df.columns else None
                position_details = self._calc_position_details(
                    cash,
                    price,
                    pred.get(date) if pred is not None else None,
                    atr_value=atr_value,
                )
                qty = position_details["qty"]
                if qty > 0:
                    cost = qty * price * (1 + self.fee_rate + self.slippage)
                    cost_gross = qty * price
                    cash -= cost
                    cash_gross -= cost_gross
                    position += qty
                    position_price = price
                    trade_log.append(
                        {
                            "date": date,
                            "action": "buy",
                            "price": price,
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
                # Sell
                proceeds = position * price * (1 - self.fee_rate - self.slippage)
                proceeds_gross = position * price
                cash += proceeds
                cash_gross += proceeds_gross
                trade_log.append(
                    {
                        "date": date,
                        "action": "sell",
                        "price": price,
                        "qty": position,
                        "cash": cash,
                        "cash_gross": cash_gross,
                    }
                )
                position = 0
                position_price = 0.0

        # 最終日に未決済ポジションを強制決済
        if position > 0:
            price = df.iloc[-1][close_col]
            proceeds = position * price * (1 - self.fee_rate - self.slippage)
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

        result_df = pd.DataFrame(trade_log)
        metrics = compute_cost_comparison_metrics(result_df, self.initial_cash)
        return result_df, metrics

    def _calc_qty(
        self,
        cash: float,
        price: float,
        pred_value: Optional[float] = None,
        atr_value: Optional[float] = None,
    ) -> int:
        return self._calc_position_details(cash, price, pred_value, atr_value)["qty"]

    def _calc_position_details(
        self,
        cash: float,
        price: float,
        pred_value: Optional[float] = None,
        atr_value: Optional[float] = None,
    ) -> dict:
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
    ) -> dict:
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
        from src.data.data_loader import get_raw_ohlcv_from_db

        df = get_raw_ohlcv_from_db(self.market, self.symbol, self.start_date, self.end_date)
        if df is None or df.empty:
            raise ValueError(
                f"market_data_rawにデータがありません: {self.market}/{self.symbol} "
                f"({self.start_date} ～ {self.end_date})\n"
                "先に run_data_creation.py を実行してください。"
            )
        return df
