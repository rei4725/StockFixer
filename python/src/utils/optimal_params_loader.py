"""
バックテスト最適化で生成された optimal_params.json を読み込むユーティリティ。
SignalGenerator / RiskManager から参照する。
"""
import json
from pathlib import Path
from typing import Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)

# python/src/utils/optimal_params_loader.py -> python/config/optimal_params.json
OPTIMAL_PARAMS_PATH = Path(__file__).parents[2] / "config" / "optimal_params.json"


def get_optimal_params(market: str, symbol: str) -> Optional[dict]:
    """
    optimal_params.json から market/symbol に対応するパラメータを返す。
    ファイルが存在しない、またはキーが見つからない場合は None を返す。

    Args:
        market: マーケット識別子（例: "jp", "us"）
        symbol: 銘柄シンボル（例: "7203", "AAPL"）

    Returns:
        最適パラメータ辞書（threshold, stop_loss_pct, take_profit_pct 等を含む）、
        ファイル未存在またはキー未登録の場合は None
    """
    if not OPTIMAL_PARAMS_PATH.exists():
        logger.debug(f"optimal_params.json が見つかりません: {OPTIMAL_PARAMS_PATH}")
        return None
    try:
        with OPTIMAL_PARAMS_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
        key = f"{market}_{symbol}"
        params = data.get(key)
        if params is None:
            logger.debug(f"optimal_params.json にキー '{key}' が存在しません")
        return params
    except (json.JSONDecodeError, OSError) as e:
        logger.warning(f"optimal_params.json の読み込みに失敗しました: {e}")
        return None
