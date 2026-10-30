"""本物の models/ を読まないための fixture（#1408）。

ドリフトのチェックの _is_recently_retrained は呼ぶ都度
`from src.utils.data_path_utils import get_models_dir` するため、本物の
python/models/ にテストと同じ銘柄名のフォルダが7日以内に更新されていると、
その銘柄を再学習から外し、デプロイの前のテストが落ちる（#1402）。

run_daily_drift_check() を呼ぶテストのファイルは、
`from tests.unit.models_dir_isolation import isolate_models_dir  # noqa: F401`
と import すれば、そのファイルのテスト全部に autouse で効く。
独自の場所を使いたいテストは、当該テスト内で改めて patch すればそちらが勝つ。
"""

import pytest


@pytest.fixture(autouse=True)
def isolate_models_dir(monkeypatch, tmp_path):
    """get_models_dir を、テストごとの空の一時フォルダ（名前は models）に向ける。

    定義元を上書きするので、遅延 import する消費先と get_models_subdir に効く。
    """
    models_dir = str(tmp_path / "models")
    monkeypatch.setattr("src.utils.data_path_utils.get_models_dir", lambda: models_dir)
    return models_dir
