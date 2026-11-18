"""
バックテスト最適化で生成された optimal_params.json を読み込むユーティリティ。
SignalGenerator / RiskManager から参照する。

週の最適化の結果は results/optimal_params.json に書く（本番でホストにマウントされ、
デプロイの起き直しで消えない）。まだ無いときは config/ の初めの値を読む（#1081）。
"""

import json
import threading
import time
from pathlib import Path
from typing import Any, Optional, Set

from filelock import FileLock, Timeout

from src.utils.data_path_utils import get_results_dir
from src.utils.logger import get_logger

logger = get_logger(__name__)

# 初めの値（リポジトリが持ち、イメージに焼かれる）。最適化の結果はここに書かない
# python/src/utils/optimal_params_loader.py -> python/config/optimal_params.json
OPTIMAL_PARAMS_PATH = Path(__file__).parents[2] / "config" / "optimal_params.json"

# 書き手（週の最適化の保存）がロックを持つのは読み直しと置き換えの間だけ（#1996）
_LOCK_TIMEOUT_SEC = 30.0
# 一度在った results/ が無く見えたときに読み直す回数と間隔（マウントの上の置き換えの一瞬）
_VANISHED_RETRIES = 20
_VANISHED_RETRY_WAIT_SEC = 0.05

# このプロセスで在ると見た results/ の置き場。無く見えても初めの値に落とさない
_seen_results_paths: Set[str] = set()
_seen_guard = threading.Lock()


def optimal_params_results_path() -> Path:
    """週の最適化が書く置き場（results/optimal_params.json）を返す。"""
    return Path(get_results_dir()) / "optimal_params.json"


def optimal_params_lock_path(results_path: Any) -> str:
    """書き手と読み手が共に取るロックのファイルの名前を返す。"""
    return f"{results_path}.lock"


class OptimalParamsUnreadableError(RuntimeError):
    """optimal_params.json が在るのに読めない、または最上位が dict でないことを表す。

    未登録（None）と区別し、既定の閾値・Kelly 値で黙って進めないために投げる。
    """


def _results_lock(results_path: Path) -> Optional[FileLock]:
    """書き手と同じロックを返す。置き場のフォルダが無ければ結果も無いので None。"""
    if not results_path.parent.is_dir():
        return None
    return FileLock(optimal_params_lock_path(results_path), timeout=_LOCK_TIMEOUT_SEC)


def _mark_seen(results_path: Path) -> None:
    with _seen_guard:
        _seen_results_paths.add(str(results_path))


def _was_seen(results_path: Path) -> bool:
    with _seen_guard:
        return str(results_path) in _seen_results_paths


def _resolve_read_path_locked(results_path: Path) -> Path:
    """ロックの内で読む置き場を決める。

    一度在った results/ が無く見えたときは、初めの値に落とさず短く待って見直す。
    待っても無ければ results/ の置き場を返す（呼び手が「無い」として扱う）。
    """
    if results_path.exists():
        _mark_seen(results_path)
        return results_path
    if not _was_seen(results_path):
        return OPTIMAL_PARAMS_PATH
    for _ in range(_VANISHED_RETRIES):
        time.sleep(_VANISHED_RETRY_WAIT_SEC)
        if results_path.exists():
            return results_path
    return results_path


def optimal_params_read_path() -> Path:
    """読む置き場を返す。最適化の結果があればそれを、無ければ初めの値を読む。

    書き手と同じロックの内で在るかを見る。ロックを待ちきれないときは、
    一度在った（または今在る）なら results/ を、そうでなければ初めの値を返す。
    """
    results_path = optimal_params_results_path()
    lock = _results_lock(results_path)
    if lock is None:
        return OPTIMAL_PARAMS_PATH
    try:
        with lock:
            return _resolve_read_path_locked(results_path)
    except Timeout:
        logger.error(
            f"optimal_params.json のロックを {_LOCK_TIMEOUT_SEC} 秒待ちきれません: {results_path}"
        )
        if _was_seen(results_path) or results_path.exists():
            return results_path
        return OPTIMAL_PARAMS_PATH


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
            最上位が dict でない・書き手のロックを待ちきれない・一度在った results/ が
            待っても無い場合
    """
    results_path = optimal_params_results_path()
    lock = _results_lock(results_path)
    if lock is None:
        return _read_params(OPTIMAL_PARAMS_PATH, market, symbol)
    try:
        with lock:
            path = _resolve_read_path_locked(results_path)
            if path == results_path and not path.exists():
                logger.error(f"一度在った optimal_params.json が待っても無いままです: {path}")
                raise OptimalParamsUnreadableError(
                    f"一度在った optimal_params.json が無くなりました: {path}"
                )
            return _read_params(path, market, symbol)
    except Timeout as e:
        logger.error(
            f"optimal_params.json のロックを {_LOCK_TIMEOUT_SEC} 秒待ちきれません: {results_path}"
        )
        raise OptimalParamsUnreadableError(
            f"optimal_params.json のロックを待ちきれません: {results_path}"
        ) from e


def _read_params(path: Path, market: str, symbol: str) -> Optional[dict]:
    """置き場を読み、market/symbol のパラメータを返す（無い・未登録は None）。"""
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
