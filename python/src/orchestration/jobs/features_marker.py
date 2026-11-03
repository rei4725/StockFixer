"""特徴量の版の印（夜間の仮の入れ替えが、今のコードの版の特徴量かを見るため）。

日次の道は印を書けなくても止めない。
"""

from src.utils.logger import get_logger

logger = get_logger(__name__)


def current_code_version() -> str | None:
    """予想のコードの版。出せなければ None（日次の道は止めない）。"""
    from src.prediction.backtest_promotion import prediction_code_version

    try:
        return prediction_code_version()
    except Exception:
        logger.error("予想のコードの版を出せなかった（特徴量の版の印は書かない）", exc_info=True)
        return None


def record_features_version(version: str | None) -> None:
    """stock_features を作り直したコードの版の印を書く。失敗しても日次の道は止めない。"""
    if version is None:
        return
    from src.prediction.backtest_promotion import save_features_version

    try:
        save_features_version(version)
        logger.info("特徴量の版の印を書いた: %s", version)
    except Exception:
        logger.error("特徴量の版の印を書けなかった", exc_info=True)
