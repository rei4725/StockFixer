"""_isolate_db（tests/unit/conftest.py）の分離契約を検証する回帰テスト。

DB接続をテストセッション全体で使い回す方式（接続確立コストの削減が目的）に
変更した際、「あるテストで書き込んだ値が、次のテストへ漏れ出さないこと」が
壊れていないかを機械的に確認する。

このファイル内の2関数は必ず test_a → test_b の順で実行される想定。
pytest はデフォルトでファイル内の定義順にテストを収集するため、この2関数名は
アルファベット順・定義順のどちらで並んでも同じ順序になるよう
`test_a_`/`test_b_` というプレフィックスを付けている。リネームする場合は
この順序依存を壊さないよう注意すること。

注意: この回帰テストは「test_a → test_b が同一プロセス内で直列に実行される」
ことを前提にしている。pytest-xdist の既定の分散方式（--dist load）はテストを
ワーカーへ個別に散らすため、test_a と test_b が別ワーカー（＝別のDB接続）に
割り振られると、test_b は「他ワーカーが書いていない」だけの理由で見かけ上
成功してしまい、分離チェックとして機能しなくなる（false negativeにはならないが
検知能力を失う）。xdist を導入する際は `--dist loadfile` を使うか、
`pytest-xdist` の `xdist_group` マーカーでこの2関数を同一ワーカーへ固定する
必要がある。
"""

from src.utils.db.system_config import get_config_value, set_config_value

_MARKER_KEY = "_isolation_guard_marker"


def test_a_write_marker():
    set_config_value(_MARKER_KEY, "should_not_persist")
    # 書き込みパス自体が機能していることを確認する（ポジティブコントロール）。
    # これが無いと、set_config_value/get_config_value の対応が将来壊れても
    # test_b は「たまたま両方とも動いていない」だけで通ってしまいかねない。
    assert get_config_value(_MARKER_KEY) == "should_not_persist"


def test_b_marker_not_visible_in_next_test():
    value = get_config_value(_MARKER_KEY)
    try:
        assert value is None, (
            "前のテスト(test_a_write_marker)で書き込んだ値がロールバックされずに"
            "残っている。DB分離(_isolate_db)が壊れている可能性がある。"
        )
    finally:
        if value is not None:
            # 分離が壊れて実DBへ永続してしまった場合のベストエフォート自己修復。
            # これが無いと、原因を修正した後もこのテストが失敗し続ける。
            set_config_value(_MARKER_KEY, "")
