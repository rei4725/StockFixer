"""
仮の入れ替えの状態・コードの版・ロック・入れ替えと戻しのテスト

モデルと結果の置き場は tmp_path に向ける。本番の models/ と results/ には触らない。
"""

import json
import os
import re
import shutil
import threading

import pytest
from filelock import Timeout

from src.prediction.backtest_promotion import state as st
from src.prediction.types import BacktestPromotionState


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    models = tmp_path / "models"
    results = tmp_path / "results"
    (models / "unified").mkdir(parents=True)
    results.mkdir()
    monkeypatch.setattr("src.utils.data_path_utils.get_models_dir", lambda: str(models))
    monkeypatch.setattr("src.utils.data_path_utils.get_results_dir", lambda: str(results))
    return models / "unified", results


def _write(path, data: bytes):
    with open(path, "wb") as f:
        f.write(data)


def _read(path) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def _make_models(unified):
    files = {}
    for prod, cand in zip(st.SWAP_PRODUCTION_NAMES, st.CANDIDATE_NAMES):
        _write(unified / f"{prod}.joblib", f"old {prod}".encode())
        _write(unified / f"{cand}.joblib", f"new {prod}".encode())
        files[prod] = (unified / f"{prod}.joblib", unified / f"{cand}.joblib")
    return files


# ---------------------------------------------------------------------------
# 入れ替えと戻し
# ---------------------------------------------------------------------------


def test_swap_failure_on_second_copy_restores_and_raises(dirs, monkeypatch):
    """2本目のコピーで落ちたら、両方の本番のファイルを元の中身に戻して例外を投げ直す"""
    unified, _ = dirs
    files = _make_models(unified)
    real_copy2 = shutil.copy2
    calls = {"n": 0}

    def _copy2(src, dst, *a, **k):
        # 写し2本のあとの、候補→本番のコピーの2本目で落とす
        if "_candidate" in os.path.basename(str(src)):
            calls["n"] += 1
            if calls["n"] == 2:
                raise OSError("disk full")
        return real_copy2(src, dst, *a, **k)

    monkeypatch.setattr("shutil.copy2", _copy2)
    with pytest.raises(OSError, match="disk full"):
        st.swap_in_candidate()

    for prod, (prod_path, _) in files.items():
        assert _read(prod_path) == f"old {prod}".encode()


def test_backup_failure_leaves_production_unchanged(dirs, monkeypatch):
    """写しで落ちたら、どの本番のファイルも上書きしない"""
    unified, _ = dirs
    files = _make_models(unified)
    real_copy2 = shutil.copy2

    def _copy2(src, dst, *a, **k):
        if os.path.basename(str(src)) == "UnifiedStockLightGBM.joblib":
            raise OSError("no space")
        return real_copy2(src, dst, *a, **k)

    monkeypatch.setattr("shutil.copy2", _copy2)
    with pytest.raises(OSError, match="no space"):
        st.swap_in_candidate()

    for prod, (prod_path, _) in files.items():
        assert _read(prod_path) == f"old {prod}".encode()


def test_copy_failure_removes_dst_that_did_not_exist(tmp_path, monkeypatch):
    """前に無かった dst は、コピーの途中で落ちたら消す（全部か無しか）"""
    src_a, src_b = tmp_path / "a_src.joblib", tmp_path / "b_src.joblib"
    dst_a, dst_b = tmp_path / "A.joblib", tmp_path / "B.joblib"
    _write(src_a, b"a")
    _write(src_b, b"b")
    _write(dst_b, b"old b")
    real_copy2 = shutil.copy2

    def _copy2(src, dst, *a, **k):
        if str(src) == str(src_b):
            raise OSError("boom")
        return real_copy2(src, dst, *a, **k)

    monkeypatch.setattr("shutil.copy2", _copy2)
    with pytest.raises(OSError):
        st.copy_with_backup([(str(src_a), str(dst_a)), (str(src_b), str(dst_b))], "20261103_120000")

    assert not dst_a.exists()
    assert _read(dst_b) == b"old b"


def test_swap_missing_candidate_raises_without_touching(dirs):
    unified, _ = dirs
    files = _make_models(unified)
    os.remove(unified / "UnifiedStockLightGBM_candidate.joblib")

    with pytest.raises(FileNotFoundError):
        st.swap_in_candidate()

    for prod, (prod_path, _) in files.items():
        assert _read(prod_path) == f"old {prod}".encode()
    assert sorted(os.listdir(unified)) == sorted(
        [
            "UnifiedStockXGBoost.joblib",
            "UnifiedStockXGBoost_candidate.joblib",
            "UnifiedStockLightGBM.joblib",
        ]
    )


def test_swap_then_restore_round_trip(dirs):
    """入れ替えて戻すと元の中身。同じ秒に戻しても写しを潰さない"""
    unified, _ = dirs
    files = _make_models(unified)

    backups = st.swap_in_candidate()

    assert len(backups) == 2
    for prod, (prod_path, cand_path) in files.items():
        assert _read(prod_path) == _read(cand_path)
        (backup,) = [b for b in backups if os.path.basename(b).startswith(prod + "_")]
        assert _read(backup) == f"old {prod}".encode()

    st.restore_backups(backups)

    for prod, (prod_path, _) in files.items():
        assert _read(prod_path) == f"old {prod}".encode()
    # 戻す前の今のファイル（候補の中身）も写してある
    contents = sorted(_read(unified / n) for n in os.listdir(unified))
    assert contents.count(b"new UnifiedStockXGBoost") == 2  # 候補と、戻す前の写し
    assert contents.count(b"old UnifiedStockXGBoost") == 2  # 戻した本番と、入れ替えの写し


def test_backup_name_matches_promote_format(dirs):
    unified, _ = dirs
    _make_models(unified)
    backups = st.swap_in_candidate()
    for b in backups:
        assert os.path.dirname(b) == str(unified)
    names = sorted(os.path.basename(b) for b in backups)
    assert re.fullmatch(r"UnifiedStockLightGBM_\d{8}_\d{6}\.joblib", names[0])
    assert re.fullmatch(r"UnifiedStockXGBoost_\d{8}_\d{6}\.joblib", names[1])


def test_restore_unknown_backup_raises_without_touching(dirs, tmp_path):
    unified, _ = dirs
    files = _make_models(unified)
    stray = tmp_path / "Other_20261103_120000.joblib"
    _write(stray, b"x")

    with pytest.raises(ValueError):
        st.restore_backups([str(stray)])
    with pytest.raises(FileNotFoundError):
        st.restore_backups([str(unified / "UnifiedStockXGBoost_20261103_120000.joblib")])

    for prod, (prod_path, _) in files.items():
        assert _read(prod_path) == f"old {prod}".encode()


def test_copy_failure_in_temp_leaves_dst_byte_identical(dirs, monkeypatch):
    """一時ファイルへの写しが書きかけで落ちても、本番のファイルは元のまま。一時ファイルも残らない"""
    unified, _ = dirs
    files = _make_models(unified)
    real_copy2 = shutil.copy2
    targets = []

    def _copy2(src, dst, *a, **k):
        if "_candidate" in os.path.basename(str(src)):
            targets.append(str(dst))
            _write(dst, b"half")  # 書きかけ
            raise OSError("killed")
        return real_copy2(src, dst, *a, **k)

    monkeypatch.setattr("shutil.copy2", _copy2)
    with pytest.raises(OSError, match="killed"):
        st.swap_in_candidate()

    for prod, (prod_path, _) in files.items():
        assert _read(prod_path) == f"old {prod}".encode()
    assert not [n for n in os.listdir(unified) if n.endswith(".tmp")]
    # 本番のファイルへ直に書いていない（同じ置き場の一時ファイルに書いた）
    assert targets
    for target in targets:
        assert os.path.dirname(target) == str(unified)
        assert target not in {str(p) for p, _ in files.values()}


def test_rollback_continues_after_non_oserror(dirs, monkeypatch):
    """戻しの1つが OSError 以外で落ちても残りは戻し、元の例外を投げ直す"""
    unified, _ = dirs
    files = _make_models(unified)
    real_copy2 = shutil.copy2
    state = {"n": 0}

    def _copy2(src, dst, *a, **k):
        name = os.path.basename(str(src))
        if "_candidate" in name:
            state["n"] += 1
            if state["n"] == 2:
                raise OSError("disk full")
        elif name.startswith("UnifiedStockXGBoost_2"):
            raise RuntimeError("restore broke")  # XGBoost の戻しだけ落とす
        return real_copy2(src, dst, *a, **k)

    monkeypatch.setattr("shutil.copy2", _copy2)
    with pytest.raises(OSError, match="disk full"):
        st.swap_in_candidate()

    # LightGBM は戻せている
    assert _read(files["UnifiedStockLightGBM"][0]) == b"old UnifiedStockLightGBM"


@pytest.mark.parametrize(
    "name",
    [
        "UnifiedStockXGBoost_candidate.joblib",
        "UnifiedStockXGBoost_challenger.joblib",
        "UnifiedStockXGBoost.joblib",
        "UnifiedStockXGBoost_20261103.joblib",
        "UnifiedStockXGBoost_20261103_120000.joblib.tmp",
    ],
)
def test_restore_rejects_non_backup_names(dirs, name):
    unified, _ = dirs
    files = _make_models(unified)
    _write(unified / "UnifiedStockXGBoost_challenger.joblib", b"challenger")
    _write(unified / "UnifiedStockXGBoost_20261103.joblib", b"x")
    _write(unified / "UnifiedStockXGBoost_20261103_120000.joblib.tmp", b"x")
    before = sorted(os.listdir(unified))

    with pytest.raises(ValueError):
        st.restore_backups([str(unified / name)])

    assert sorted(os.listdir(unified)) == before
    for prod, (prod_path, _) in files.items():
        assert _read(prod_path) == f"old {prod}".encode()


def test_restore_accepts_numbered_backup(dirs):
    unified, _ = dirs
    files = _make_models(unified)
    backup = unified / "UnifiedStockXGBoost_20261103_120000_2.joblib"
    _write(backup, b"numbered")
    st.restore_backups([str(backup)])
    assert _read(files["UnifiedStockXGBoost"][0]) == b"numbered"


# ---------------------------------------------------------------------------
# 状態のファイル
# ---------------------------------------------------------------------------


def test_results_dir_under_results(dirs):
    _, results = dirs
    assert st.results_dir() == results / "backtest_promotion"


def test_state_round_trip_and_atomic_write(dirs, monkeypatch):
    _, results = dirs
    assert st.load_state() == BacktestPromotionState()

    state = BacktestPromotionState(
        last_evaluated_version="abc123def456",
        rolled_back_versions=["000000000000"],
        watch={
            "version": "abc123def456",
            "swapped_at": "2026-11-03T21:00:00+09:00",
            "swap_date": "2026-11-03",
            "backups": ["/m/UnifiedStockXGBoost_20261103_210000.joblib"],
            "backtest_candidate": {"hit_rate_minus_majority": 0.05},
            "backtest_old_production": {"hit_rate_minus_majority": 0.0},
        },
    )
    st.save_state(state)
    path = results / "backtest_promotion" / "state.json"
    assert json.loads(path.read_text(encoding="utf-8"))["last_evaluated_version"] == "abc123def456"
    assert st.load_state() == state

    # 置き換えで落ちたら、前の中身が残り、一時ファイルも残らない
    def _boom(*a, **k):
        raise OSError("replace failed")

    monkeypatch.setattr("os.replace", _boom)
    with pytest.raises(OSError):
        st.save_state(BacktestPromotionState(last_evaluated_version="other"))
    assert st.load_state() == state
    assert os.listdir(results / "backtest_promotion") == ["state.json"]


def test_features_version_round_trip(dirs):
    _, results = dirs
    assert st.load_features_version() is None

    st.save_features_version("abc123def456")

    assert st.load_features_version() == "abc123def456"
    saved = json.loads(
        (results / "backtest_promotion" / "features_version.json").read_text("utf-8")
    )
    assert saved["version"] == "abc123def456"
    assert "written_at" in saved
    # 一時ファイルは残さない
    assert [p.name for p in (results / "backtest_promotion").iterdir()] == ["features_version.json"]


def test_clear_watch(dirs):
    st.save_state(
        BacktestPromotionState(last_evaluated_version="v1", watch={"version": "v1", "backups": []})
    )
    st.clear_watch("週次の昇格が起きた")
    loaded = st.load_state()
    assert loaded.watch is None
    assert loaded.last_evaluated_version == "v1"

    # 見張りが無ければ何もしない
    st.clear_watch("もう一度")
    assert st.load_state().watch is None


# ---------------------------------------------------------------------------
# コードの版
# ---------------------------------------------------------------------------


def test_code_version_changes_only_with_prediction_or_market_data_code(tmp_path, monkeypatch):
    root = tmp_path / "python"
    for rel in [
        "src/prediction/a.py",
        "src/prediction/sub/b.py",
        "src/market_data/c.py",
        "src/reporting/d.py",
    ]:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(f"# {rel}\n", encoding="utf-8")
    monkeypatch.setattr(st, "_PYTHON_DIR", root)

    base = st.prediction_code_version()
    assert len(base) == 12
    assert st.prediction_code_version() == base

    # 関係の無いコードと __pycache__ と .py 以外は版を変えない
    (root / "src/reporting/d.py").write_text("changed\n", encoding="utf-8")
    (root / "src/prediction/__pycache__").mkdir()
    (root / "src/prediction/__pycache__/a.py").write_text("cache\n", encoding="utf-8")
    (root / "src/prediction/notes.txt").write_text("x\n", encoding="utf-8")
    assert st.prediction_code_version() == base

    (root / "src/market_data/c.py").write_text("changed\n", encoding="utf-8")
    after_md = st.prediction_code_version()
    assert after_md != base

    (root / "src/prediction/sub/b.py").write_text("changed\n", encoding="utf-8")
    assert st.prediction_code_version() != after_md

    # ファイルの名前を変えるだけでも版は変わる
    os.rename(root / "src/prediction/a.py", root / "src/prediction/a2.py")
    assert st.prediction_code_version() != after_md


def test_code_version_includes_settings(tmp_path, monkeypatch):
    root = tmp_path / "python"
    for rel in ["src/prediction/a.py", "config/settings.py", "config/other.py"]:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(f"# {rel}\n", encoding="utf-8")
    monkeypatch.setattr(st, "_PYTHON_DIR", root)
    base = st.prediction_code_version()

    # config/ のほかのファイルは版を変えない
    (root / "config/other.py").write_text("changed\n", encoding="utf-8")
    assert st.prediction_code_version() == base

    (root / "config/settings.py").write_text("changed\n", encoding="utf-8")
    assert st.prediction_code_version() != base
    assert "config/settings.py" in st.CODE_VERSION_SCOPE


def test_code_version_uses_module_location():
    """既定の置き場は python/ （CWD ではない）"""
    assert (st._PYTHON_DIR / "src" / "prediction" / "backtest_promotion" / "state.py").exists()
    assert all((st._PYTHON_DIR / rel).exists() for rel in st.CODE_VERSION_SCOPE)
    assert len(st.prediction_code_version()) == 12


# ---------------------------------------------------------------------------
# ロック
# ---------------------------------------------------------------------------


def test_unified_model_lock_is_shared_and_reentrant(dirs):
    unified, _ = dirs
    lock = st.unified_model_lock()
    assert lock is st.unified_model_lock(timeout_s=1)
    assert os.path.normcase(lock.lock_file) == os.path.normcase(
        str(unified / ".unified_models.lock")
    )
    with st.unified_model_lock():
        with st.unified_model_lock(timeout_s=0):
            assert lock.is_locked
    assert not lock.is_locked


def test_unified_model_lock_excludes_other_thread(dirs):
    """別のスレッドが持っている間は、timeout=0 で取ろうとすると Timeout"""
    held, release = threading.Event(), threading.Event()

    def _hold():
        with st.unified_model_lock():
            held.set()
            release.wait(10)

    t = threading.Thread(target=_hold)
    t.start()
    try:
        assert held.wait(10)
        with pytest.raises(Timeout):
            st.unified_model_lock(timeout_s=0).acquire()
    finally:
        release.set()
        t.join(10)
    with st.unified_model_lock(timeout_s=0):
        pass


# ---------------------------------------------------------------------------
# promote_unified_challenger は振る舞いを変えない
# ---------------------------------------------------------------------------


def test_promote_unified_challenger_unchanged_behavior(dirs):
    from src.prediction.shadow_evaluation import promote_unified_challenger

    unified, _ = dirs
    for name in ["UnifiedStockXGBoost", "UnifiedStockLightGBM"]:
        _write(unified / f"{name}.joblib", f"old {name}".encode())
        _write(unified / f"{name}_challenger.joblib", b"challenger")

    dry = promote_unified_challenger(dry_run=True)
    assert dry == {
        "promoted": ["UnifiedStockXGBoost", "UnifiedStockLightGBM"],
        "skipped": ["UnifiedStockTransformer_challenger"],
        "backups": [],
        "dry_run": True,
    }

    result = promote_unified_challenger()
    assert result["promoted"] == ["UnifiedStockXGBoost", "UnifiedStockLightGBM"]
    assert result["skipped"] == ["UnifiedStockTransformer_challenger"]
    assert result["dry_run"] is False
    assert len(result["backups"]) == 2
    for name in ["UnifiedStockXGBoost", "UnifiedStockLightGBM"]:
        assert _read(unified / f"{name}.joblib") == b"challenger"
