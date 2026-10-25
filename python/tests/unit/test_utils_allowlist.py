"""utils/ に業務のモジュールが戻らないよう止める（2026-10-25 設計書 §6）。

utils/ に置いてよいのは logger・DB・パス・リトライ・外部 API の薄いクライアントなどの道具だけ。
売買の判断・表・暦は domain/、1つの BC しか使わない処理はその BC へ置く。
utils/ に新しいモジュールを足すときは、ALLOWED_UTILS_MODULES に理由を書いて PR のレビューを通す。
"""

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"

_DB = "DB の接続・テーブル操作"
_LATER = "外とのやり取りを持つため残す。後続: 設計書 §8"

ALLOWED_UTILS_MODULES: dict[str, str] = {
    "alert_service.py": "運用アラートの送り分け（通知の道具）",
    "data_path_utils.py": "ファイルパスとティッカーの組み立て",
    "db/__init__.py": _DB,
    "db/_bulk.py": _DB,
    "db/_connection.py": _DB,
    "db/_read.py": _DB,
    "db/compact.py": _DB,
    "db/experiment.py": _DB,
    "db/factory_runs.py": _DB,
    "db/index_membership.py": _DB,
    "db/market_data.py": _DB,
    "db/migration_runner.py": _DB,
    "db/quality_log.py": _DB,
    "db/retention.py": _DB,
    "db/rule_results.py": _DB,
    "db/stock_features.py": _DB,
    "db/stock_fundamentals.py": _DB,
    "db/strategy_promotions.py": _DB,
    "db/symbol_master.py": _DB,
    "db/system_config.py": _DB,
    "df_to_string.py": "DataFrame の文字列化",
    "github_api.py": "GitHub REST API の薄いクライアント",
    "health_check.py": "死活の判定の部品",
    "heartbeat.py": "外部の死活監視への ping",
    "japan_time.py": "日本時間の取得と整形",
    "logger.py": "統一ロガー",
    "ml_base.py": "ML モデルの基底クラス（joblib の保存と読み込み）。売買の判断ではない",
    "ollama_client.py": "Ollama の薄いクライアント",
    "optimal_params_loader.py": _LATER,
    "results_io.py": "結果 CSV の保存",
    "retry_helper.py": "リトライ",
    "run_context.py": "run_id の受け渡し",
    "scheduler_types.py": "スケジューラーの共有型",
    "sector_constraints.py": _LATER,
    "signal_generator.py": _LATER,
}
TEMPORARY_SHIMS: dict[str, str] = {
    "regime_weights.py": "転送: PR 5 で消す",
    "jp_market_holidays.py": "転送: PR 5 で消す",
}


def _utils_modules() -> set[str]:
    root = SRC / "utils"
    return {
        p.relative_to(root).as_posix() for p in root.rglob("*.py") if "__pycache__" not in p.parts
    }


def test_no_unlisted_module_in_utils():
    unlisted = _utils_modules() - ALLOWED_UTILS_MODULES.keys() - TEMPORARY_SHIMS.keys()
    assert not unlisted, f"utils/ に許可リストに無いモジュールがある: {sorted(unlisted)}"


def test_allowlist_has_no_stale_entries():
    stale = (ALLOWED_UTILS_MODULES.keys() | TEMPORARY_SHIMS.keys()) - _utils_modules()
    assert not stale, f"許可リストに消えたモジュールが残っている: {sorted(stale)}"


def _absolute_imports(tree: ast.AST) -> list[str]:
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.append(node.module)
    return names


def test_domain_imports_nothing_outside_domain():
    bad = []
    for path in (SRC / "domain").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for name in _absolute_imports(tree):
            top = name.split(".")[0]
            if top == "config" or (top == "src" and not name.startswith("src.domain")):
                bad.append(f"{path.name}: {name}")
    assert not bad, f"domain が外を import している: {bad}"
