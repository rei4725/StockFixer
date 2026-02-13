import pandas as pd

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

    def run(self, model_name, model_type=None, source="file"):
        # 1. データ取得（API/ファイル切替対応）
        df = self.data_loader.get_stock_data_auto(
            self.market, self.symbol, self.start_date, self.end_date, source=source
        )
        # 2. 特徴量生成（テクニカル指標付与）
        from src.features.technical_analysis import add_technical_indicators
        df = add_technical_indicators(df)
        # 3. モデル予測
        model = self.model_manager.get_model(model_name)
        X = df.dropna().drop(columns=["Close"], errors="ignore")
        prediction = model.predict(X)
        # 4. シグナル生成
        signal = self.signal_generator.generate_signal(df.loc[X.index], prediction)
        # 5. 仮想売買シミュレーション
        result_df, metrics = self.simulate_trading(df.loc[X.index], signal)
        return result_df, metrics

    def simulate_trading(self, df, signal):
        cash = self.initial_cash
        position = 0
        position_price = 0
        trade_log = []
        for date, sig in signal.items():
            price = df.loc[date, "Close"]
            if sig == 1 and position == 0:
                # Buy
                qty = cash // (price * (1 + self.fee_rate + self.slippage))
                if qty > 0:
                    cost = qty * price * (1 + self.fee_rate + self.slippage)
                    cash -= cost
                    position += qty
                    position_price = price
                    trade_log.append({"date": date, "action": "buy", "price": price, "qty": qty, "cash": cash})
            elif sig == -1 and position > 0:
                # Sell
                proceeds = position * price * (1 - self.fee_rate - self.slippage)
                cash += proceeds
                trade_log.append({"date": date, "action": "sell", "price": price, "qty": position, "cash": cash})
                position = 0
                position_price = 0
        # 最終日に未決済ポジションを決済
        if position > 0:
            price = df.iloc[-1]["Close"]
            proceeds = position * price * (1 - self.fee_rate - self.slippage)
            cash += proceeds
            trade_log.append({"date": df.index[-1], "action": "final_sell", "price": price, "qty": position, "cash": cash})
            position = 0
        # 結果集計
        result_df = pd.DataFrame(trade_log)
        total_return = (cash - self.initial_cash) / self.initial_cash
        metrics = {
            "final_cash": cash,
            "total_return": total_return,
            "num_trades": len(result_df) // 2,
        }
        return result_df, metrics
