"""
バックテスト最適化で生成された optimal_params.json を読み込むユーティリティ。
SignalGenerator / RiskManager から参照する。

週の最適化の結果は results/optimal_params.json に書く（本番でホストにマウントされ、
デプロイの起き直しで消えない）。まだ無いときは config/ の初めの値を読む（#1081）。
"""

import json
from pathlib import Path
from typing import Optional

from src.utils.data_path_utils import get_results_dir
from src.utils.logger import get_logger

logger = get_logger(__name__)

# 初めの値（リポジトリが持ち、イメージに焼かれる）。最適化の結果はここに書かない
# python/src/utils/optimal_params_loader.py -> python/config/optimal_params.json
OPTIMAL_PARAMS_PATH = Path(__file__).parents[2] / "config" / "optimal_params.json"


def optimal_params_results_path() -> Path:
    """週の最適化が書く置き場（results/optimal_params.json）を返す。"""
    return Path(get_results_dir()) / "optimal_params.json"


def optimal_params_read_path() -> Path:
    """読む置き場を返す。最適化の結果があればそれを、無ければ初めの値を読む。"""
    results_path = optimal_params_results_path()
    return results_path if results_path.exists() else OPTIMAL_PARAMS_PATH


class OptimalParamsUnreadableError(RuntimeError):
    """optimal_params.json が在るのに読めない、または最上位が dict でないことを表す。

    未登録（None）と区別し、既定の閾値・Kelly 値で黙って進めないために投げる。
    """


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

    Raises:
        OptimalParamsUnreadableError: ファイルが在るのに読めない・JSON が壊れている・
            最上位が dict でない場合
    """
    path = optimal_params_read_path()
    if not path.exists():
        logger.debug(f"optimal_params.json が見つかりません: {path}")
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as e:
        logger.error(f"optimal_params.json が在るのに読めません: {path}", exc_info=True)
        raise OptimalParamsUnreadableError(f"optimal_params.json を読めません: {path}: {e}") from e
    if not isinstance(data, dict):
        logger.error(
            f"optimal_params.json の最上位が dict ではありません: "
            f"{path} ({type(data).__name__})"
        )
        raise OptimalParamsUnreadableError(
            f"optimal_params.json の最上位が dict ではありません: "
            f"{path} ({type(data).__name__})"
        )
    key = f"{market}_{symbol}"
    params = data.get(key)
    if params is None:
        logger.debug(f"optimal_params.json にキー '{key}' が存在しません")
    return params
