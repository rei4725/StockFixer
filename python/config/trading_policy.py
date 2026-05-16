"""
トレード方針・リスクプロファイルの集約

RISK_PROFILE 環境変数でプロファイルを選択できる。
各値は個別の環境変数で上書き可能。
パースエラーは ValueError を raise する（settings.py の silent fallback とは異なる）。
"""
import os

_VALID_PROFILES = {"conservative", "moderate", "aggressive"}

_PROFILE_DEFAULTS: dict[str, dict[str, float]] = {
    "conservative": {
        "MAX_ACCEPTABLE_DRAWDOWN": 0.10,
        "KELLY_CAP": 0.50,
        "HIGH_CONFIDENCE_POSITION_CAP": 0.20,
        "MIN_SHARPE_TO_TRADE": 0.80,
    },
    "moderate": {
        "MAX_ACCEPTABLE_DRAWDOWN": 0.20,
        "KELLY_CAP": 0.75,
        "HIGH_CONFIDENCE_POSITION_CAP": 0.30,
        "MIN_SHARPE_TO_TRADE": 0.50,
    },
    "aggressive": {
        "MAX_ACCEPTABLE_DRAWDOWN": 0.30,
        "KELLY_CAP": 1.00,
        "HIGH_CONFIDENCE_POSITION_CAP": 0.40,
        "MIN_SHARPE_TO_TRADE": 0.30,
    },
}


def _strict_float(
    env: str,
    default: float,
    min_val: float = 0.0,
    max_val: float = float("inf"),
) -> float:
    import math

    val = os.getenv(env, "").strip()
    if not val:
        return default
    try:
        result = float(val)
    except ValueError:
        raise ValueError(f"環境変数 {env}='{val}' を float に変換できません")
    if math.isnan(result) or math.isinf(result) or not (min_val <= result <= max_val):
        raise ValueError(f"環境変数 {env}={result} は範囲外です（{min_val} ≤ x ≤ {max_val}）")
    return result


_raw_profile = os.getenv("RISK_PROFILE", "moderate").strip().lower()
if _raw_profile not in _VALID_PROFILES:
    raise ValueError(f"RISK_PROFILE='{_raw_profile}' は無効です。" f"有効値: {sorted(_VALID_PROFILES)}")

RISK_PROFILE: str = _raw_profile
_defaults = _PROFILE_DEFAULTS[RISK_PROFILE]

MAX_ACCEPTABLE_DRAWDOWN: float = _strict_float(
    "MAX_ACCEPTABLE_DRAWDOWN", _defaults["MAX_ACCEPTABLE_DRAWDOWN"], min_val=0.0, max_val=1.0
)
KELLY_CAP: float = _strict_float("KELLY_CAP", _defaults["KELLY_CAP"], min_val=0.0, max_val=1.0)
HIGH_CONFIDENCE_POSITION_CAP: float = _strict_float(
    "HIGH_CONFIDENCE_POSITION_CAP",
    _defaults["HIGH_CONFIDENCE_POSITION_CAP"],
    min_val=0.0,
    max_val=1.0,
)
MIN_SHARPE_TO_TRADE: float = _strict_float(
    "MIN_SHARPE_TO_TRADE", _defaults["MIN_SHARPE_TO_TRADE"], min_val=0.0, max_val=10.0
)

# Kelly 入力パラメータ（BT実績値が取得できない場合のフォールバック）
DEFAULT_WIN_RATE: float = 0.55
DEFAULT_AVG_WIN: float = 0.015
DEFAULT_AVG_LOSS: float = 0.008

# VIX テールリスクヘッジ設定（R-406）
# VIX がこの閾値を超えたときにポジション縮小を適用する
VIX_SPIKE_THRESHOLD: float = _strict_float("VIX_SPIKE_THRESHOLD", 30.0, min_val=10.0, max_val=100.0)
# VIX 急騰時のポジション縮小係数（1.0 = 縮小なし, 0.5 = 半減）
VIX_POSITION_SCALE: float = _strict_float("VIX_POSITION_SCALE", 0.5, min_val=0.0, max_val=1.0)
# インバース ETF ヘッジ発注オプション（SH, SQQQ 等への連動発注を有効化）
VIX_HEDGE_ENABLED: bool = os.getenv("VIX_HEDGE_ENABLED", "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
