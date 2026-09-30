"""戦略ファクトリーの合格ゲートを、正解ラベル付きの合成戦略で統計的に検証する。

ゲートの「論理的な正しさ」を、次の分類仕様として固定する:

- 真のポートフォリオ Sharpe がチャンピオン以下の候補 → 合格率 NEGATIVE_MAX_PASS_RATE 以下
- 真のポートフォリオ Sharpe がチャンピオンの POSITIVE_MIN_RATIO 倍以上 → 合格率
  POSITIVE_MIN_PASS_RATE 以上
- その間は推定誤差で揺れるグレーゾーンとして要求しない

グレーゾーンの幅は、2 年分の日次データで試行 1000 回分の多重比較補正（DSR）を掛けた
ときの統計的な限界である。実測した検出力（seed 150 本）:

    真の実力（チャンピオン比）  1.3x   1.6x   2.0x   2.5x
    合格率                      23%    79%    99%    100%

すなわち「年率 Sharpe がおよそ 3 を超えない戦略は、偶然と見分けられない」。

さらに、チャンピオン（対照群）自体が弱い夜を別に検証する（WEAK_CHAMPION_CASES）。
チャンピオン比較だけでは弱い対照に勝つだけで通ってしまうため、多重比較の補正（DSR）が
「真の年率 Sharpe が UNDETECTABLE_SHARPE 未満の戦略は通さない」ことを保証する。
取引単位で DSR を算出していた旧実装では、真の Sharpe 0.47・取引 5 倍の戦略が 17% 合格した。これは
意図した厳しさであり（2026-09-30 判断）、緩める場合は FACTORY_LOOKBACK_YEARS を延ばして
観測日数を増やすのが筋である（試行数の数え方を緩めると多重比較の補正が外れる）。

「真の Sharpe」は生成パラメータから理論的に決まる等金額ポートフォリオの日次 Sharpe
（factory_gate_synthetic.true_portfolio_sharpe）。各クラスの真値はテスト内で検算し、
ラベル自体の取り違えも防ぐ。

生成条件は本番の夜間バッチに寄せている: 200 銘柄 × 2 年、銘柄あたり約 10 取引、
取引あたり Sharpe 約 0.18（本番の合格候補と同水準で DSR が飽和する領域）、市場ファクター
beta=1（実際の株と同じく、保有銘柄を増やしても市場リスクは分散されない）。
"""

from __future__ import annotations

import functools

import pytest
from tests.unit.backtest.factory_gate_synthetic import (
    Leg,
    StrategySpec,
    market_path,
    simulate,
    to_evaluation,
    true_portfolio_sharpe,
)

from src.backtest.factory_gate import apply_gate
from src.backtest.types import FactoryEvaluation

N_SEEDS = 200
N_TRIALS = 1000  # DSR の累計試行数（本番台帳の件数と同程度）
NEGATIVE_MAX_PASS_RATE = 0.05
POSITIVE_MIN_PASS_RATE = 0.90
POSITIVE_MIN_RATIO = 2.0
# 2 年・試行 1000 回では確かめられない真の年率 Sharpe（方針: 通さない）
UNDETECTABLE_SHARPE = 1.0

N_SYMBOLS = 200
HOLD = 5
SIGMA = 0.02
MU = 0.0018


def _spec(*legs: Leg, beta: float = 1.0) -> StrategySpec:
    return StrategySpec(tuple(legs), n_symbols=N_SYMBOLS, beta=beta)


CHAMPION = _spec(Leg(10, HOLD, MU, SIGMA))
# 対照群が振るわない夜（チャンピオン自体が期待値ゼロ）
WEAK_CHAMPION = _spec(Leg(10, HOLD, 0.0, SIGMA))

# (名前, 候補, 期待される分類)。"negative" は不合格にすべき、"positive" は合格にすべき。
CASES = [
    # チャンピオンと同じ実力。推定誤差だけで勝つ候補を通してはならない
    ("same_as_champion", _spec(Leg(10, HOLD, MU, SIGMA)), "negative"),
    # 取引の質は半分・頻度 4 倍。市場ファクターが分散できないため真の実力はチャンピオン未満
    ("half_quality_4x_trades", _spec(Leg(40, HOLD, MU / 2, SIGMA)), "negative"),
    # チャンピオンに期待値ゼロの取引を足したもの（OR 合成でノイズ側の子が鳴る状況）
    ("champion_plus_noise", _spec(Leg(10, HOLD, MU, SIGMA), Leg(10, HOLD, 0.0, SIGMA)), "negative"),
    # 期待値ゼロの戦略を高頻度で回す
    ("pure_noise_5x_trades", _spec(Leg(50, HOLD, 0.0, SIGMA)), "negative"),
    # 同頻度で質が劣る
    ("worse_quality", _spec(Leg(10, HOLD, MU * 0.7, SIGMA)), "negative"),
    # 同頻度で質が 2.1 倍（真の Sharpe がチャンピオンの 2 倍を超えるように）
    ("better_quality", _spec(Leg(10, HOLD, MU * 2.1, SIGMA)), "positive"),
    # 質は同じで市場感応度が低い（市場リスクを抑えた分だけポートフォリオの質が高い）
    ("lower_market_beta", _spec(Leg(10, HOLD, MU, SIGMA), beta=0.3), "positive"),
]


# チャンピオンが弱い夜。候補はどれも真の Sharpe が UNDETECTABLE_SHARPE 未満で、不合格にすべき
WEAK_CHAMPION_CASES = [
    # 市場に連動するだけの期待値ゼロ戦略を高頻度で回す
    ("noise_5x_trades_vs_weak_champion", _spec(Leg(50, HOLD, 0.0, SIGMA))),
    # ごく小さな優位を高頻度で回す（真の Sharpe 約 0.47）
    ("tiny_edge_5x_trades_vs_weak_champion", _spec(Leg(50, HOLD, 0.0003, SIGMA))),
]


@functools.lru_cache(maxsize=None)
def _market(seed: int):
    return market_path(CHAMPION.n_days, CHAMPION.market_sigma, seed * 3 + 2)


_CHAMPIONS = {"normal": CHAMPION, "weak": WEAK_CHAMPION}


@functools.lru_cache(maxsize=None)
def _champion(seed: int, kind: str = "normal") -> FactoryEvaluation:
    return to_evaluation(
        simulate(_CHAMPIONS[kind], seed * 3 + 1, _market(seed)), N_TRIALS, is_control=True
    )


def _pass_rate(candidate: StrategySpec, champion_kind: str = "normal") -> float:
    passed = 0
    for seed in range(N_SEEDS):
        evaluation = to_evaluation(
            simulate(candidate, seed * 3, _market(seed)), N_TRIALS, is_control=False
        )
        apply_gate(evaluation, _champion(seed, champion_kind))
        passed += evaluation.gate_passed
    return passed / N_SEEDS


@pytest.mark.parametrize("name,candidate,label", CASES, ids=[c[0] for c in CASES])
def test_labels_match_ground_truth(name, candidate, label):
    """各クラスのラベルが、理論上の真の Sharpe と矛盾しないこと（テスト自体の検算）。"""
    champion_sr = true_portfolio_sharpe(CHAMPION)
    candidate_sr = true_portfolio_sharpe(candidate)
    if label == "negative":
        assert candidate_sr <= champion_sr + 1e-9, (name, candidate_sr, champion_sr)
    else:
        assert candidate_sr >= champion_sr * POSITIVE_MIN_RATIO, (name, candidate_sr, champion_sr)


@pytest.mark.parametrize("name,candidate,label", CASES, ids=[c[0] for c in CASES])
def test_gate_classifies_synthetic_strategies(name, candidate, label):
    """ゲートが正解ラベルどおりに仕分けること（seed を変えた N_SEEDS 回の合格率で判定）。"""
    rate = _pass_rate(candidate)
    champion_sr = true_portfolio_sharpe(CHAMPION)
    candidate_sr = true_portfolio_sharpe(candidate)
    detail = (
        f"{name}: pass_rate={rate:.3f} true_sharpe={candidate_sr:.2f} "
        f"champion_true_sharpe={champion_sr:.2f}"
    )
    if label == "negative":
        assert rate <= NEGATIVE_MAX_PASS_RATE, detail
    else:
        assert rate >= POSITIVE_MIN_PASS_RATE, detail


@pytest.mark.parametrize(
    "name,candidate", WEAK_CHAMPION_CASES, ids=[c[0] for c in WEAK_CHAMPION_CASES]
)
def test_undetectable_strategies_fail_even_against_weak_champion(name, candidate):
    """チャンピオンが弱い夜でも、確かめようのない戦略は通さない（多重比較の補正）。"""
    candidate_sr = true_portfolio_sharpe(candidate)
    assert candidate_sr < UNDETECTABLE_SHARPE, (name, candidate_sr)
    assert true_portfolio_sharpe(WEAK_CHAMPION) <= candidate_sr + 1e-9

    rate = _pass_rate(candidate, champion_kind="weak")
    assert (
        rate <= NEGATIVE_MAX_PASS_RATE
    ), f"{name}: pass_rate={rate:.3f} true_sharpe={candidate_sr:.2f}"
