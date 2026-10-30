from typing import Any, Callable, Optional

import pandas as pd

from src.backtest.data_port import get_backtest_data_port
from src.backtest.execution import ExecutionModel, TradingCosts, lot_size_for
from src.backtest.limit_fill import LimitFillModel
from src.backtest.mark_to_market import build_equity_curves, mark_to_market
from src.backtest.metrics import compute_cost_comparison_metrics
from src.backtest.position_sizing import calc_position_details
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
        # 手数料・スリッページの演算はすべて ExecutionModel に委譲する（Phase 1 / PR-1）。
        self.execution = ExecutionModel(
            TradingCosts(fee_rate=fee_rate, slippage_rate=slippage, lot_size=lot_size_for(market)),
            slippage_fn=slippage_fn,
        )
        # #493: シグナル約定の実行ラグ（バー数）。1=翌バー約定（引け後シグナル→翌寄り）。
        # 0 で従来の同バー(当日Close)約定に戻す（後方互換）。
        self.execution_lag = max(0, int(execution_lag))
        self.include_monte_carlo = include_monte_carlo  # #374: MCリスク統計を含めるか

    def _get_slippage(self, qty: int, price: float, volume: float = 0.0) -> float:
        """有効スリッページ率を返す（#494）。

        実装は ExecutionModel.effective_slippage にある。既存の呼び出し元と
        テストのために薄い委譲として残している。
        """
        return self.execution.effective_slippage(qty, price, volume)

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
        self,
        df: pd.DataFrame,
        signal: pd.Series,
        pred: Optional[pd.Series] = None,
        collect_equity: bool = False,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """
        仮想売買シミュレーションを実行する。

        ストップロス・テイクプロフィット・ポジションサイジングに対応。

        Args:
            df: Close 列を含む DataFrame
            signal: シグナル Series (1=buy, -1=sell, 0=hold)
            pred: 予測値 Series（ポジションサイジング "confidence" モードで使用）
            collect_equity: True のとき metrics["equity_curve"] に日次 mark-to-market
                equity（net）を載せる。複数銘柄のポートフォリオ equity を合成する
                用途向け。既定 False では従来どおり載せない（metrics を JSON/DB に
                そのまま流す既存経路に Series が混入しないようにするため）。

        Returns:
            (result_df, metrics) のタプル
        """
        cash = self.initial_cash
        cash_gross = self.initial_cash
        position = 0
        position_price = 0.0
        short_position = 0
        short_price = 0.0
        # #801: ショートのエントリー売り代金は、エントリーの時点で分かる平均出来高で決める
        short_entry_vol = 0.0
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
        # SL/TP の判定と mark-to-market は当バー Close 基準。SL/TP の約定は翌バー始値（#964）。
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
        # シグナル約定（execution_lag > 0 なら翌バー始値）のコスト・サイジングには、
        # 約定時点で既知の前バーまでの平均出来高を使う（当バー出来高の先読みを防ぐ）。
        # 先頭バーは前バーが無いため 0（フラットのみ適用）。SL/TP の翌バー決済も同じ exec_avg_vol を使う。
        if self.execution_lag > 0:
            exec_avg_vol = avg_vol.shift(1).fillna(0.0)
        else:
            exec_avg_vol = avg_vol
        # #800: ATR サイジングも約定時点で既知の前バーまでの ATR を使う。
        # atr_lag1 があれば各行の前バー ATR としてそのまま使い（file/api/raw で同じ意味）、
        # 無ければ当バー高安引けを含む atr を 1 本ずらす。同バー約定は当バー atr のまま。
        exec_atr: Optional[pd.Series] = None
        if self.execution_lag > 0:
            if "atr_lag1" in df.columns:
                exec_atr = df["atr_lag1"]
            elif "atr" in df.columns:
                exec_atr = df["atr"].shift(1)
        elif "atr" in df.columns:
            exec_atr = df["atr"]

        # #964: execution_lag > 0 では SL/TP を当バー Close で判定し、決済は翌バーの
        # exec_price（始値、無ければ終値）と exec_vol で行う。判定済みの決済の種類を持ち越す。
        # 最終バーで判定したものは約定先バーが無いため、最終日の強制決済に任せる。
        pending_exit: Optional[str] = None
        # #1418: 本番で指値になる日は、翌バーが指値に届いたときだけ指値で約定する
        limit_fill = LimitFillModel(df, close_col, self.market, self.execution_lag > 0)

        for date in df.index:
            price = df.loc[date, close_col]
            exec_price = df.loc[date, exec_col]
            vol = float(avg_vol.loc[date]) if date in avg_vol.index else 0.0
            exec_vol = float(exec_avg_vol.loc[date]) if date in exec_avg_vol.index else 0.0
            raw_sig = exec_signal.get(date, 0) if date in exec_signal.index else 0
            sig = 0 if pd.isna(raw_sig) else int(raw_sig)

            # 前バーで判定した SL/TP の決済（この足のシグナルは評価しない＝二重に建てない・売らない）
            if pending_exit is not None and position > 0:
                proceeds = self.execution.sell_proceeds(position, exec_price, exec_vol)
                proceeds_gross = position * exec_price
                cash += proceeds
                cash_gross += proceeds_gross
                trade_log.append(
                    {
                        "date": date,
                        "action": pending_exit,
                        "price": exec_price,
                        "qty": position,
                        "cash": cash,
                        "cash_gross": cash_gross,
                    }
                )
                position = 0
                position_price = 0.0
                pending_exit = None
                equity_records.append(
                    mark_to_market(
                        self.execution,
                        date,
                        cash,
                        cash_gross,
                        position,
                        price,
                        short_position,
                        short_price,
                        short_entry_vol,
                        exec_vol,
                    )
                )
                continue
            pending_exit = None

            # 同バー約定（execution_lag == 0）のストップロス / テイクプロフィット判定
            # （シグナルより先に評価）。execution_lag > 0 の判定はシグナルの後で行う（#1361）
            if self.execution_lag == 0 and position > 0 and position_price > 0:
                change_from_entry = (price - position_price) / position_price

                if self.stop_loss_pct is not None and change_from_entry <= -self.stop_loss_pct:
                    proceeds = self.execution.sell_proceeds(position, price, vol)
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
                        mark_to_market(
                            self.execution,
                            date,
                            cash,
                            cash_gross,
                            position,
                            price,
                            short_position,
                            short_price,
                            short_entry_vol,
                            exec_vol,
                        )
                    )
                    continue

                if self.take_profit_pct is not None and change_from_entry >= self.take_profit_pct:
                    proceeds = self.execution.sell_proceeds(position, price, vol)
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
                        mark_to_market(
                            self.execution,
                            date,
                            cash,
                            cash_gross,
                            position,
                            price,
                            short_position,
                            short_price,
                            short_entry_vol,
                            exec_vol,
                        )
                    )
                    continue

            # シグナルに基づく売買。#1418: 本番で指値になる日は、指値に届かなければ約定させず、
            # 届けば exec_price を指値の値段に置き換える（SL/TP は本番も成行なので置き換えない）
            fill_price = limit_fill.fill_price(date, sig == 1, exec_price) if sig else exec_price
            sig = 0 if fill_price is None else sig
            exec_price = exec_price if fill_price is None else fill_price
            # ショートカバー: 買いシグナル + ショートポジション保有
            if sig == 1 and self.enable_short and short_position > 0:
                # エントリー売り: short_price * qty * (1 - fee - slip)（エントリー時点の出来高）
                # カバー買い: exec_price * qty * (1 + fee + slip)
                net_pnl = self.execution.sell_proceeds(
                    short_position, short_price, short_entry_vol
                ) - self.execution.buy_cost(short_position, exec_price, exec_vol)
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
                short_entry_vol = 0.0
            elif sig == 1 and position == 0:
                # Buy（翌バー始値 exec_price で約定）
                atr_value = exec_atr.loc[date] if exec_atr is not None else None
                position_details = self._calc_position_details(
                    cash,
                    exec_price,
                    exec_pred.get(date) if exec_pred is not None else None,
                    atr_value=atr_value,
                    volume=exec_vol,
                )
                qty = position_details["qty"]
                if qty > 0:
                    cost = self.execution.buy_cost(qty, exec_price, exec_vol)
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
                proceeds = self.execution.sell_proceeds(position, exec_price, exec_vol)
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
                atr_value = exec_atr.loc[date] if exec_atr is not None else None
                position_details = self._calc_position_details(
                    cash,
                    exec_price,
                    exec_pred.get(date) if exec_pred is not None else None,
                    atr_value=atr_value,
                    volume=exec_vol,
                )
                qty = position_details["qty"]
                if qty > 0:
                    short_position = qty
                    short_price = exec_price
                    short_entry_vol = exec_vol
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

            # #1361: execution_lag > 0 では、この足の寄りで約定するシグナルを先に処理し、
            # まだ保有が残っていれば当バー Close で SL/TP を判定して翌バーの決済を立てる。
            # 寄りの約定を当バーの終値で取り消さない（終値は寄りの時点で分からない）。
            if self.execution_lag > 0 and position > 0 and position_price > 0:
                change_from_entry = (price - position_price) / position_price
                if self.stop_loss_pct is not None and change_from_entry <= -self.stop_loss_pct:
                    pending_exit = "stop_loss"
                elif self.take_profit_pct is not None and change_from_entry >= self.take_profit_pct:
                    pending_exit = "take_profit"

            # 各バー末の mark-to-market equity を記録（保有中の含み損益込み・#492）
            equity_records.append(
                mark_to_market(
                    self.execution,
                    date,
                    cash,
                    cash_gross,
                    position,
                    price,
                    short_position,
                    short_price,
                    short_entry_vol,
                    exec_vol,
                )
            )

        # 最終日に未決済ポジションを強制決済。execution_lag > 0 では最後の足の出来高を
        # 先読みしないよう、前の足までの平均出来高（mark-to-market と同じ値）を使う（#1347）
        final_vol = float(exec_avg_vol.iloc[-1]) if len(exec_avg_vol) else 0.0
        if position > 0:
            price = df.iloc[-1][close_col]
            proceeds = self.execution.sell_proceeds(position, price, final_vol)
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
            # エントリー売り: short_price * qty * (1 - fee - slip)（エントリー時点の出来高）
            # カバー買い: price * qty * (1 + fee + slip)
            net_pnl = self.execution.sell_proceeds(
                short_position, short_price, short_entry_vol
            ) - self.execution.buy_cost(short_position, price, final_vol)
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
        equity_net, equity_gross = build_equity_curves(equity_records)
        metrics = compute_cost_comparison_metrics(
            result_df,
            self.initial_cash,
            equity_net=equity_net,
            equity_gross=equity_gross,
            include_monte_carlo=self.include_monte_carlo,
        )
        if collect_equity:
            metrics["equity_curve"] = equity_net
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
        volume: float = 0.0,
    ) -> dict[str, Any]:
        """ポジションサイジングに基づいて購入数量を算出する（実装: position_sizing.py）。"""
        return calc_position_details(
            cash=cash,
            price=price,
            position_sizing=self.position_sizing,
            position_fraction=self.position_fraction,
            atr_risk_pct=self.atr_risk_pct,
            atr_multiplier=self.atr_multiplier,
            atr_min_fraction=self.atr_min_fraction,
            atr_max_fraction=self.atr_max_fraction,
            execution=self.execution,
            pred_value=pred_value,
            atr_value=atr_value,
            volume=volume,
        )

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
