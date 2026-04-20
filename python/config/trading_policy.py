"""
トレード方針・リスクプロファイルの集約

RISK_PROFILE 環境変数でプロファイルを選択できる。
各値は個別の環境変数で上書き可能。
パースエラーは ValueError を raise する（settings.py の silent fallback とは異なる）。
"""
import os

_VALID_PROFILES = {"conservative", "moderate", "aggressive"}

_PROFILE_DEFAULTS: dict[str, dict] = {
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


def _strict_float(env: str, default: float) -> float:
    val = os.getenv(env, "").strip()
    if not val:
        return default
    try:
        return float(val)
    except ValueError:
        raise ValueError(f"環境変数 {env}='{val}' を float に変換できません")


_raw_profile = os.getenv("RISK_PROFILE", "moderate").strip().lower()
if _raw_profile not in _VALID_PROFILES:
    raise ValueError(f"RISK_PROFILE='{_raw_profile}' は無効です。" f"有効値: {sorted(_VALID_PROFILES)}")

RISK_PROFILE: str = _raw_profile
_defaults = _PROFILE_DEFAULTS[RISK_PROFILE]

MAX_ACCEPTABLE_DRAWDOWN: float = _strict_float(
    "MAX_ACCEPTABLE_DRAWDOWN", _defaults["MAX_ACCEPTABLE_DRAWDOWN"]
)
KELLY_CAP: float = _strict_float("KELLY_CAP", _defaults["KELLY_CAP"])
HIGH_CONFIDENCE_POSITION_CAP: float = _strict_float(
    "HIGH_CONFIDENCE_POSITION_CAP", _defaults["HIGH_CONFIDENCE_POSITION_CAP"]
)
MIN_SHARPE_TO_TRADE: float = _strict_float("MIN_SHARPE_TO_TRADE", _defaults["MIN_SHARPE_TO_TRADE"])
