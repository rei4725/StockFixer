"""長期コホート・バックテストの設定値。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from src.backtest.execution import DEFAULT_FEE_RATE, TradingCosts
from src.domain.types import HoldRules

# 市場ごとのベンチマークの既定（紙の週の成績の WEEKLY_BENCHMARKS と同じ指数）。
# 未知の市場は従来どおり ^GSPC。
DEFAULT_BENCHMARKS = {"jp": "1306.T", "us": "^GSPC"}
_FALLBACK_BENCHMARK = "^GSPC"


def default_benchmark_ticker(market: str) -> str:
    """market に対応するベンチマークの既定のティッカーを返す。"""
    return DEFAULT_BENCHMARKS.get(market, _FALLBACK_BENCHMARK)


@dataclass(frozen=True)
class LongtermBacktestConfig:
    """1 回の長期バックテストを定義する不変の設定。

    execution_lag: エントリー約定をリスクリーン日から何営業日ずらすか。
                   スクリーン（as_of）は day loop 側がリスクリーン日で行い、
                   `enter_candidates` はそのスクリーン済み候補リストを受け取って
                   約定するだけである。価格の引き当てと `simulate_position` の
                   起点はスクリーン日からこの値だけ後ろの営業日
                   （`resolve_entry_date`）になる。0 なら従来通りスクリーン
                   当日の終値で約定する。リスクリーン日がカレンダー末尾から
                   lag 営業日以内の場合、その回のエントリーは見送られる。
    n_trials:      DSR（過学習ガード）の試行回数。`compute_longterm_metrics` が
                   これが 0 より大きいときだけ `metrics["dsr"]` を計算する
                   （既定の 0 では DSR は出力されない）。エンジン側では機能するが、
                   CLI からこの値を指定するフラグは未提供（呼び出し元コードから
                   設定する必要がある）。
    benchmark_ticker: 比べる指数。空なら market の既定（`DEFAULT_BENCHMARKS`）を使う。
    """

    market: str = "us"
    start: str = "2021-01-01"
    end: str = "2026-01-01"
    rescreen_freq: str = "quarterly"
    top_n: int = 30
    initial_cash: float = 1_000_000.0
    max_positions: int = 10
    execution_lag: int = 1
    benchmark_ticker: str = ""
    n_trials: int = 0
    costs: TradingCosts = field(default_factory=TradingCosts)
    rules: HoldRules = field(default_factory=HoldRules)

    def __post_init__(self) -> None:
        if self.execution_lag < 0:
            raise ValueError(f"execution_lag は 0 以上である必要があります: {self.execution_lag}")
        if not self.benchmark_ticker:
            object.__setattr__(self, "benchmark_ticker", default_benchmark_ticker(self.market))

    @classmethod
    def build(
        cls,
        *,
        market: str = "us",
        fee_rate: float = DEFAULT_FEE_RATE,
        slippage: Optional[float] = None,
        rules: Optional[HoldRules] = None,
        benchmark_ticker: Optional[str] = None,
        **kwargs: Any,
    ) -> "LongtermBacktestConfig":
        """市場別スリッページ・ベンチマーク既定の解決を含む組み立て。CLI はこれを呼ぶ。"""
        return cls(
            market=market,
            costs=TradingCosts.for_market(market, fee_rate, slippage),
            rules=rules or HoldRules(),
            benchmark_ticker=benchmark_ticker or "",
            **kwargs,
        )
