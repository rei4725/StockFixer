"""
仮の入れ替えの状態・予想のコードの版・ロック・入れ替えと戻し。

本番の統合モデル（XGBoost・LightGBM）のファイルをバックテストの候補と入れ替え、
戻すときは写しから戻す。入れ替えも戻しも、全部のファイルがそろうか、何も変わらないかの
どちらかにする。週次の学習と同時に動かないよう、同じロックのファイルを使う。
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path

from filelock import FileLock

from src.prediction.types import BacktestPromotionState
from src.utils import data_path_utils
from src.utils.japan_time import now_jst
from src.utils.logger import get_logger

logger = get_logger(__name__)

CANDIDATE_NAMES = ["UnifiedStockXGBoost_candidate", "UnifiedStockLightGBM_candidate"]
SWAP_PRODUCTION_NAMES = ["UnifiedStockXGBoost", "UnifiedStockLightGBM"]

# python/ の置き場（CWD ではなく、このファイルの場所から決める）
_PYTHON_DIR = Path(__file__).resolve().parents[3]
# 予想のコードの版に入れるコードの置き場（python/ からの相対。フォルダか .py のファイル）
CODE_VERSION_SCOPE = ["src/prediction", "src/market_data", "config/settings.py"]

_LOCK_FILE_NAME = ".unified_models.lock"
# 同じパスのロックは1つの FileLock を使い回す（同じプロセスの入れ子で詰まらないように）
_LOCKS: dict[str, FileLock] = {}


def _unified_dir() -> str:
    # テストが data_path_utils.get_models_dir を差し替えられるよう、呼ぶたびに引く
    return os.path.join(data_path_utils.get_models_dir(), "unified")


def results_dir() -> Path:
    """仮の入れ替えの結果と状態の置き場。"""
    return Path(data_path_utils.get_results_dir()) / "backtest_promotion"


def _state_path() -> Path:
    return results_dir() / "state.json"


def _features_version_path() -> Path:
    return results_dir() / "features_version.json"


def _write_json_atomic(path: Path, data: dict) -> None:
    """同じ置き場の一時ファイルに書いてから置き換える。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.stem}_", suffix=".json.tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


# ---------------------------------------------------------------------------
# 状態のファイル
# ---------------------------------------------------------------------------


def load_state() -> BacktestPromotionState:
    """状態を読む。ファイルが無ければ空の状態を返す。"""
    path = _state_path()
    if not path.exists():
        return BacktestPromotionState()
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return BacktestPromotionState(
        last_evaluated_version=data.get("last_evaluated_version"),
        rolled_back_versions=list(data.get("rolled_back_versions") or []),
        watch=data.get("watch"),
    )


def save_state(state: BacktestPromotionState) -> None:
    """状態を書く。同じ置き場の一時ファイルに書いてから置き換える。"""
    _write_json_atomic(_state_path(), asdict(state))


def clear_watch(reason: str) -> None:
    """見張りがあれば消し、理由をログに出す。"""
    state = load_state()
    if state.watch is None:
        return
    version = state.watch.get("version")
    state.watch = None
    save_state(state)
    logger.info(f"仮の入れ替えの見張りを終えた（版 {version}）: {reason}")


# ---------------------------------------------------------------------------
# 特徴量を作り直したコードの版の印
# ---------------------------------------------------------------------------


def save_features_version(version: str) -> None:
    """stock_features を作り直したコードの版を印のファイルに書く（日次の道が呼ぶ）。"""
    _write_json_atomic(
        _features_version_path(), {"version": version, "written_at": now_jst().isoformat()}
    )


def load_features_version() -> str | None:
    """stock_features を作り直したコードの版。印のファイルが無ければ None。"""
    path = _features_version_path()
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    version = data.get("version") if isinstance(data, dict) else None
    return None if version is None else str(version)


# ---------------------------------------------------------------------------
# 予想のコードの版
# ---------------------------------------------------------------------------


def prediction_code_version() -> str:
    """予想に関わるコード（CODE_VERSION_SCOPE の .py）のハッシュの先頭12桁。

    相対パスの順に、パスと中身を sha256 に入れる。__pycache__ は除く。
    改行は LF にそろえる（チェックアウトの違いで版が変わらないように）。
    """
    files = []
    for rel in CODE_VERSION_SCOPE:
        base = _PYTHON_DIR / rel
        if base.is_file():
            files.append((base.relative_to(_PYTHON_DIR).as_posix(), base))
            continue
        if not base.exists():
            continue
        for path in base.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            files.append((path.relative_to(_PYTHON_DIR).as_posix(), path))
    digest = hashlib.sha256()
    for rel, path in sorted(files):
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
        digest.update(b"\0")
    return digest.hexdigest()[:12]


# ---------------------------------------------------------------------------
# ロック
# ---------------------------------------------------------------------------


def unified_model_lock(timeout_s: float = 3 * 3600) -> FileLock:
    """統合モデルのファイルを触る間に持つロック（models/unified/.unified_models.lock）。

    同じパスには同じ FileLock を返す。FileLock は同じオブジェクトなら同じスレッドで
    入れ子に取れるので、同じプロセスの中で二重に取っても詰まらない。
    別のスレッドや別のプロセスとは排他になる。timeout_s は呼ぶたびに置き換える。
    thread_local=True（既定）なので timeout はスレッドごとに持つ。別のスレッドで呼ぶときも
    この関数を通せば、そのスレッドの timeout が timeout_s になる。
    """
    unified = _unified_dir()
    os.makedirs(unified, exist_ok=True)
    path = os.path.join(unified, _LOCK_FILE_NAME)
    lock = _LOCKS.get(path)
    if lock is None:
        lock = FileLock(path, timeout=timeout_s)
        _LOCKS[path] = lock
    else:
        lock.timeout = timeout_s
    return lock


# ---------------------------------------------------------------------------
# 入れ替えと戻し
# ---------------------------------------------------------------------------


def _backup_path(dst: str, stamp: str) -> str:
    """dst の写しの名前 `<名前>_<stamp>.joblib`。もうあれば `_2` から番号を足す。"""
    folder = os.path.dirname(dst)
    name = os.path.splitext(os.path.basename(dst))[0]
    path = os.path.join(folder, f"{name}_{stamp}.joblib")
    n = 2
    while os.path.exists(path):
        path = os.path.join(folder, f"{name}_{stamp}_{n}.joblib")
        n += 1
    return path


def _copy_replace(src: str, dst: str) -> None:
    """src を dst と同じ置き場の一時ファイルに写してから dst に置き換える。

    途中で落ちても dst は前の中身のまま残り、書きかけの dst を読む人も出ない。
    """
    fd, tmp = tempfile.mkstemp(
        prefix=f".{os.path.basename(dst)}.", suffix=".tmp", dir=os.path.dirname(dst) or "."
    )
    os.close(fd)
    try:
        shutil.copy2(src, tmp)
        os.replace(tmp, dst)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def copy_with_backup(pairs: list[tuple[str, str]], stamp: str) -> list[str]:
    """(src, dst) の組ごとに src を dst に写す。先に今の dst を全部写しておく。

    写しに1つでも失敗したら、どの dst も上書きせずに例外を投げる（#824）。
    コピーは一時ファイルに写してから置き換えるので、1つの dst が書きかけになることはない。
    コピーの途中で落ちたら、写しから全部の dst を戻し（前に無かった dst は消し）、
    例外を投げ直す。

    Returns:
        写しのパス（前からあった dst の分だけ）
    """
    backups: list[str] = []
    originals: list[tuple[str, str | None]] = []
    for _, dst in pairs:
        if os.path.exists(dst):
            backup = _backup_path(dst, stamp)
            shutil.copy2(dst, backup)
            backups.append(backup)
            originals.append((dst, backup))
            logger.info(f"上書きの前に今のファイルを写した: {backup}")
        else:
            originals.append((dst, None))

    try:
        for src, dst in pairs:
            _copy_replace(src, dst)
    except BaseException:
        logger.error("コピーの途中で落ちた。写しから元に戻す", exc_info=True)
        for dst, saved in originals:
            try:
                if saved is None:
                    if os.path.exists(dst):
                        os.remove(dst)
                else:
                    _copy_replace(saved, dst)
            except Exception:
                # 1つ戻せなくても残りは戻し、元の例外を投げ直す
                logger.error(f"写しから戻せなかった: {dst}（写し {saved}）", exc_info=True)
        raise
    return backups


def swap_in_candidate() -> list[str]:
    """バックテストの候補のファイルを本番の名前で置く。

    候補が1つでも無ければ、何も触らずに FileNotFoundError を投げる。

    Returns:
        写した前の本番のファイルのパス
    """
    unified = _unified_dir()
    pairs = []
    for candidate, production in zip(CANDIDATE_NAMES, SWAP_PRODUCTION_NAMES):
        src = os.path.join(unified, f"{candidate}.joblib")
        if not os.path.exists(src):
            raise FileNotFoundError(f"候補のモデルが見つかりません: {src}")
        pairs.append((src, os.path.join(unified, f"{production}.joblib")))
    stamp = now_jst().strftime("%Y%m%d_%H%M%S")
    backups = copy_with_backup(pairs, stamp)
    logger.info(f"候補を本番に入れ替えた: {SWAP_PRODUCTION_NAMES}（写し {backups}）")
    return backups


def _production_for_backup(backup: str) -> str:
    """写しの名前 `<本番の名前>_YYYYMMDD_HHMMSS(_N)?.joblib` から本番の名前を返す。"""
    name = os.path.basename(backup)
    for production in SWAP_PRODUCTION_NAMES:
        if re.fullmatch(rf"{re.escape(production)}_\d{{8}}_\d{{6}}(_\d+)?\.joblib", name):
            return production
    raise ValueError(f"本番のモデルの写しではありません: {backup}")


def restore_backups(backups: list[str]) -> None:
    """写しから本番のファイルに戻す。戻す前に今の本番のファイルも写す。

    写しが無い・本番の写しでないものがあれば、何も触らずに例外を投げる。
    """
    unified = _unified_dir()
    pairs = []
    for backup in backups:
        production = _production_for_backup(backup)
        if not os.path.exists(backup):
            raise FileNotFoundError(f"戻す写しが見つかりません: {backup}")
        pairs.append((backup, os.path.join(unified, f"{production}.joblib")))
    stamp = now_jst().strftime("%Y%m%d_%H%M%S")
    copy_with_backup(pairs, stamp)
    logger.info(f"写しから本番のモデルに戻した: {backups}")
