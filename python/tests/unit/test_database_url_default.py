"""get_database_url の既定値が IPv4 のループバックを指すことの確認。

docker-compose.yml は DB のポートを 127.0.0.1 にだけ開く（PR 901）。既定値が
localhost だと、Windows では先に ::1 を試して接続のたびに約5秒待ち、unit テストの
fixture が時間切れになる。
"""

from src.utils.data_path_utils import get_database_url


def test_default_database_url_uses_ipv4_loopback(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert get_database_url() == "postgresql://stockfixer:stockfixer_dev@127.0.0.1:5433/stockfixer"
