"""
Walk-Forward Validation

時系列分割による学習/検証を繰り返し、各期間のメトリクスを返す。
推論タスクの比較実験や過学習の検出に使用する。
"""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from src.backtest.task import BacktestTask, ReturnRegressionTask
from src.utils.logger import get_logger

logger = get_logger(__name__)


class WalkForwardValidator:
    """
    Walk-Forward Validation（時系列ローリング分割）。

    学習期間で特徴量を構築・モデル学習を行い、
    検証期間でシグナル生成・仮想売買をシミュレートする。
    これを n_splits 回繰り返し、各期間のメトリクスを収集する。

    Usage::

        wfv = WalkForwardValidator(
            market="jp", symbol="7203",
            model_manager=..., signal_generator=...,
            initial_cash=1_000_000, fee_rate=0.001,
            n_splits=5,
        )
        results = wfv.run(model_name="StockXGBoostModel", task=ReturnRegressionTask())
        print(results)  # DataFrame: 各期間のメトリクス
    """

    def __init__(
        self,
        market: str,
        symbol: str,
        model_manager: Any,
        signal_generator: Any,
        initial_cash: float = 1_000_000,
        fee_rate: float = 0.0,
        slippage: float = 0.0,
        n_splits: int = 5,
        source: str = "file",
        stop_loss_pct: Optional[float] = None,
        take_profit_pct: Optional[float] = None,
        position_sizing: str = "full",
        position_fraction: float = 0.5,
        atr_risk_pct: float = 0.02,
        atr_multiplier: float = 1.0,
        atr_min_fraction: float = 0.1,
        atr_max_fraction: float = 1.0,
        ensemble: bool = False,
        enable_short: bool = False,
    ):
        self.market = market
        self.symbol = symbol
        self.model_manager = model_manager
        self.signal_generator = signal_generator
        self.initial_cash = initial_cash
        self.fee_rate = fee_rate
        self.slippage = slippage
        self.n_splits = n_splits
        self.source = source
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.position_sizing = position_sizing
        self.position_fraction = position_fraction
        self.atr_risk_pct = atr_risk_pct
        self.atr_multiplier = atr_multiplier
        self.atr_min_fraction = atr_min_fraction
        self.atr_max_fraction = atr_max_fraction
        self.ensemble = ensemble
        self.enable_short = enable_short

    def run(
        self,
        model_name: str,
        model_type: Optional[str] = None,
        task: Optional[BacktestTask] = None,
    ) -> pd.DataFrame:
        """
        Walk-Forward 検証を実行する。

        Args:
            model_name: モデル名
            model_type: モデルタイプ（任意）
            task: BacktestTask 実装（Noneなら ReturnRegressionTask()）

        Returns:
            各分割期間のメトリクスをまとめた DataFrame
        """
        if task is None:
            task = ReturnRegressionTask()

        # 全データ取得
        df = self._load_features()
        if df is None or df.empty:
            raise ValueError(f"データが取得できませんでした: {self.market}/{self.symbol}")

        # ラベル列を付与
        df[task.label_col] = task.make_labels(df)
        df = df.dropna(subset=[task.label_col])

        tss = TimeSeriesSplit(n_splits=self.n_splits)
        all_results = []

        for fold_idx, (train_idx, val_idx) in enumerate(tss.split(df)):
            train_df = df.iloc[train_idx]
            val_df = df.iloc[val_idx]

            fold_start = val_df.index[0]
            fold_end = val_df.index[-1]
            logger.info(
                "[walk_forward] Fold %d/%d 検証期間: %s → %s",
                fold_idx + 1,
                self.n_splits,
                fold_start,
                fold_end,
            )

            try:
                metrics = self._run_fold(
                    train_df=train_df,
                    val_df=val_df,
                    model_name=model_name,
                    model_type=model_type,
                    task=task,
                )
                metrics["fold"] = fold_idx + 1
                metrics["val_start"] = str(fold_start)
                metrics["val_end"] = str(fold_end)
                metrics["train_rows"] = len(train_df)
                metrics["val_rows"] = len(val_df)
                all_results.append(metrics)
            except Exception as e:
                logger.error("[walk_forward] Fold %d エラー", fold_idx + 1, exc_info=True)
                all_results.append(
                    {
                        "fold": fold_idx + 1,
                        "val_start": str(fold_start),
                        "val_end": str(fold_end),
                        "error": str(e),
                    }
                )

        result_df = pd.DataFrame(all_results)
        self._print_summary(result_df)
        return result_df

    def _run_fold(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        model_name: str,
        model_type: Optional[str],
        task: BacktestTask,
    ) -> dict[str, Any]:
        """1折分の学習→予測→シミュレーションを実行する"""
        from src.backtest.backtester import Backtester

        feature_cols = [
            c
            for c in train_df.columns
            if c not in (task.label_col, "market", "symbol", "market_encoded")
        ]

        X_train = train_df[feature_cols].dropna()
        y_train = train_df.loc[X_train.index, task.label_col]
        X_val = val_df[feature_cols].dropna()

        if self.ensemble:
            from src.backtest.pipeline import _ensemble_predict

            pred = _ensemble_predict(
                self.model_manager,
                X_train,
                y_train,
                X_val,
                model_name,
            )
        else:
            # 学習 (BaseModel の train() メソッドを使用)
            model = self.model_manager.get_model(model_name)
            model.train(X_train, y_train)
            pred = pd.Series(model.predict(X_val), index=X_val.index, name="pred")

        # シグナル生成
        signal = task.make_signal(pred)

        # シミュレーション
        backtester = Backtester(
            model_manager=self.model_manager,
            signal_generator=self.signal_generator,
            data_loader=None,
            start_date=None,
            end_date=None,
            market=self.market,
            symbol=self.symbol,
            initial_cash=self.initial_cash,
            fee_rate=self.fee_rate,
            slippage=self.slippage,
            stop_loss_pct=self.stop_loss_pct,
            take_profit_pct=self.take_profit_pct,
            position_sizing=self.position_sizing,
            position_fraction=self.position_fraction,
            atr_risk_pct=self.atr_risk_pct,
            atr_multiplier=self.atr_multiplier,
            atr_min_fraction=self.atr_min_fraction,
            atr_max_fraction=self.atr_max_fraction,
            enable_short=self.enable_short,
        )
        result_df, metrics = backtester.simulate_trading(val_df, signal, pred=pred)
        return metrics

    def _load_features(self) -> Optional[pd.DataFrame]:
        """特徴量データを読み込む（source に応じてDB/API/Rawを使い分ける）"""
        from src.backtest.pipeline import load_features

        return load_features(self.market, self.symbol, self.source)

    @staticmethod
    def _print_summary(result_df: pd.DataFrame) -> None:
        """検証結果のサマリーを表示する"""
        print("\n" + "=" * 60)
        print("Walk-Forward 検証 サマリー")
        print("=" * 60)
        numeric_cols = [
            "total_return",
            "sharpe_ratio",
            "max_drawdown",
            "gross_total_return",
            "gross_sharpe_ratio",
            "gross_max_drawdown",
            "cost_impact_return",
            "cost_impact_cash",
            "win_rate",
            "profit_factor",
            "avg_position_fraction",
            "max_position_fraction",
            "avg_position_value",
            "atr_fallback_trades",
        ]
        available = [c for c in numeric_cols if c in result_df.columns]
        if available:
            print(result_df[["fold", "val_start", "val_end"] + available].to_string(index=False))
            print("\n--- 平均 ---")
            # None値をnp.nanに変換して平均計算可能にする
            result_numeric = result_df[available].copy()
            for col in result_numeric.columns:
                result_numeric[col] = pd.to_numeric(result_numeric[col], errors="coerce")
            print(result_numeric.mean().round(4).to_string())
        print("=" * 60)
