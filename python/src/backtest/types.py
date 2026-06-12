"""
backtest BC の型定義。

バックテスト・ストレステスト・Walk-Forward・戦略ファクトリーで使用するドメイン型。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field


@dataclass(frozen=True)
class FactoryHypothesis:
    """戦略ファクトリー（#369）の評価対象1件。

    rule_spec は再帰構造:
        atomic: {"type": "atomic", "rule": "<rule_name>", "params": {...}}
        合成:   {"type": "and"|"or", "rules": [<atomic>, ...]}
    """

    rule_spec: dict
    market: str
    lookback_years: int = 2
    is_control: bool = False

    @property
    def hypothesis_hash(self) -> str:
        """スペックの正規化 JSON から決定的に導出するハッシュ（先頭12桁）。

        is_control は同一スペックの対照/候補を同一視するため含めない。
        """
        canonical = json.dumps(
            {
                "rule_spec": self.rule_spec,
                "market": self.market,
                "lookback_years": self.lookback_years,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]

    @property
    def label(self) -> str:
        """人間可読の短いラベル（Issue タイトル等に使用）。"""
        return _spec_label(self.rule_spec)


def _spec_label(spec: dict) -> str:
    if spec.get("type") == "atomic":
        params = spec.get("params") or {}
        if params:
            inner = ",".join(f"{k}={v}" for k, v in sorted(params.items()))
            return f"{spec['rule']}({inner})"
        return str(spec["rule"])
    children = ", ".join(_spec_label(s) for s in spec.get("rules", []))
    return f"{spec.get('type', '?')}({children})"


@dataclass
class FactoryEvaluation:
    """1仮説の評価結果（全期間メトリクス + 窓別リターン + ゲート判定）。"""

    hypothesis: FactoryHypothesis
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    num_trades: int = 0
    max_drawdown: float = 0.0
    total_return: float = 0.0
    window_returns: list[float] = field(default_factory=list)
    n_symbols: int = 0
    dsr: float = float("nan")
    pbo: float = float("nan")
    gate_passed: bool = False
    gate_reasons: list[str] = field(default_factory=list)
    report_path: str | None = None


@dataclass
class FactoryBatchResult:
    """夜間バッチ1回分の結果サマリー（Discord 通知・ログ用）。"""

    market: str
    evaluated: list[FactoryEvaluation] = field(default_factory=list)
    champion_sharpe: float = float("nan")
    pbo: float = float("nan")

    @property
    def candidates(self) -> list[FactoryEvaluation]:
        return [e for e in self.evaluated if not e.hypothesis.is_control]

    @property
    def passed(self) -> list[FactoryEvaluation]:
        return [e for e in self.candidates if e.gate_passed]


@dataclass
class StressTestResult:
    """ストレステスト1銘柄・1シナリオの結果。"""

    market: str
    symbol: str
    scenario_name: str  # "corona" / "lehman"
    period_start: str  # "2020-02-01"
    period_end: str  # "2020-03-31"
    mdd: float  # 最大ドローダウン（負の値）
    sharpe_ratio: float
    total_return: float
    win_rate: float
    num_trades: int
    max_consecutive_losses: int
    mdd_pass: bool  # abs(mdd) <= MDD_THRESHOLD (0.15)
