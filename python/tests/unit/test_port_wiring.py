"""wire_ports() が screening ポートを注入すること。"""

import unittest

from src.backtest import screening_port as sp
from src.orchestration import port_wiring


class TestWirePorts(unittest.TestCase):
    def setUp(self):
        self._saved_port = sp._port
        self._saved_wired = port_wiring._wired

    def tearDown(self):
        sp._port = self._saved_port
        port_wiring._wired = self._saved_wired

    def test_injects_screening_port(self):
        sp._port = None
        port_wiring._wired = False
        port_wiring.wire_ports()
        self.assertIsNotNone(sp.get_backtest_screening_port())

    def test_is_idempotent(self):
        port_wiring._wired = False
        port_wiring.wire_ports()
        first = sp.get_backtest_screening_port()
        port_wiring.wire_ports()
        self.assertIs(sp.get_backtest_screening_port(), first)

    def test_force_reinjects(self):
        port_wiring._wired = False
        port_wiring.wire_ports()
        first = sp.get_backtest_screening_port()
        port_wiring.wire_ports(force=True)
        self.assertIsNot(sp.get_backtest_screening_port(), first)


if __name__ == "__main__":
    unittest.main()
