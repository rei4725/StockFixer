"""utils/ の転送が domain の本体と同じ物を返すことを確かめる（2026-10-25 設計書 §5）。

転送は PR 5 で消し、このファイルも一緒に消す。
"""


def test_regime_weights_shim_reexports_domain_objects():
    import src.domain.regime_weights as new
    import src.utils.regime_weights as old

    assert old.REGIME_SECTOR_WEIGHTS is new.REGIME_SECTOR_WEIGHTS
    assert old.get_regime_sector_weight is new.get_regime_sector_weight


def test_jp_market_holidays_shim_reexports_domain_functions():
    import src.domain.jp_market_holidays as new
    import src.utils.jp_market_holidays as old

    assert old.is_jp_market_holiday is new.is_jp_market_holiday
    assert old.is_jp_business_day is new.is_jp_business_day
