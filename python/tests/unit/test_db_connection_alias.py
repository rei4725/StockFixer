"""ユニットテスト: src.utils.db.db_connection 公開別名。

infrastructure/persistence/ のアダプタが私的シンボル _db_connection を
直接参照せずに済むよう、公開別名が同一実体を指していることを保証する。
"""

import unittest


class TestDbConnectionAlias(unittest.TestCase):
    def test_alias_is_the_same_object(self):
        from src.utils.db import db_connection
        from src.utils.db._connection import _db_connection

        self.assertIs(db_connection, _db_connection)

    def test_alias_is_importable_from_package_root(self):
        import src.utils.db as db_module

        self.assertTrue(hasattr(db_module, "db_connection"))


if __name__ == "__main__":
    unittest.main()
