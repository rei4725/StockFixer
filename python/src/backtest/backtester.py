import pandas as pd
from typing import Optional

from src.backtest.task import BacktestTask, ReturnRegressionTask
from src.backtest.metrics import compute_metrics


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
        from src.features.technical_analysis import add_technical_indicators, create_basic_lag_features
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

    def simulate_trading(self, df: pd.DataFrame, signal: pd.Series):
        """
        仮想売買シミュレーションを実行する。

        Args:
            df: Close 列を含む DataFrame
            signal: シグナル Series (1=buy, -1=sell, 0=hold)

        Returns:
            (result_df, metrics) のタプル
        """
        cash = self.initial_cash
        position = 0
        position_price = 0.0
        trade_log = []

        close_col = "Close" if "Close" in df.columns else "close"

        for date, sig in signal.items():
            if date not in df.index:
                continue
            price = df.loc[date, close_col]
            if sig == 1 and position == 0:
                # Buy
                qty = int(cash // (price * (1 + self.fee_rate + self.slippage)))
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
                position_price = 0.0

        # 最終日に未決済ポジションを強制決済
        if position > 0:
            price = df.iloc[-1][close_col]
            proceeds = position * price * (1 - self.fee_rate - self.slippage)
            cash += proceeds
            trade_log.append({"date": df.index[-1], "action": "final_sell", "price": price, "qty": position, "cash": cash})
            position = 0

        result_df = pd.DataFrame(trade_log)
        metrics = compute_metrics(result_df, self.initial_cash)
        return result_df, metrics

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
