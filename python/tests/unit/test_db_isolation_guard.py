"""_isolate_db（tests/unit/conftest.py）の分離契約を検証する回帰テスト。

DB接続をテストセッション全体で使い回す方式（接続確立コストの削減が目的）に
変更した際、「あるテストで書き込んだ値が、次のテストへ漏れ出さないこと」が
壊れていないかを機械的に確認する。

このファイル内の2関数は必ず test_a → test_b の順で実行される想定。
pytest はデフォルトでファイル内の定義順にテストを収集するため、この2関数名は
アルファベット順・定義順のどちらで並んでも同じ順序になるよう
`test_a_`/`test_b_` というプレフィックスを付けている。リネームする場合は
この順序依存を壊さないよう注意すること。
"""

from src.utils.db.system_config import get_config_value, set_config_value

_MARKER_KEY = "_isolation_guard_marker"


def test_a_write_marker():
    set_config_value(_MARKER_KEY, "should_not_persist")


def test_b_marker_not_visible_in_next_test():
    value = get_config_value(_MARKER_KEY)
    assert value is None, (
        "前のテスト(test_a_write_marker)で書き込んだ値がロールバックされずに"
        "残っている。DB分離(_isolate_db)が壊れている可能性がある。"
    )
