"""統合モデルの全銘柄予想の完了のログ（#1118）。

予想を出せなかった銘柄を、特徴量が古い・例外・そのほかの理由（現在価格が取れない等）に
分けて出す。例外で落ちた銘柄は WARNING にも並べる。
"""

from src.utils.logger import get_logger

# 完了の行は prediction_pipeline から切り出したもの。本番のログで同じ名前のまま追えるようにする
logger = get_logger("src.prediction.prediction_pipeline")


def log_unified_prediction_summary(
    n_rows: int,
    n_total: int,
    stale_keys: list[tuple[str, str]],
    error_keys: list[tuple[str, str]],
) -> None:
    """予想を出した数と、出せなかった数を理由ごとに分けてログへ出す。"""
    n_other = n_total - n_rows - len(stale_keys) - len(error_keys)
    logger.info(
        "並列予測完了（統合モデル）: %d/%d銘柄, 特徴量が古いため予想なし: %d銘柄, "
        "例外で予想なし: %d銘柄, そのほかの理由で予想なし: %d銘柄",
        n_rows,
        n_total,
        len(stale_keys),
        len(error_keys),
        n_other,
    )
    if error_keys:
        logger.warning(
            "例外で予想を出せなかった銘柄: %s",
            ", ".join(f"{market}_{symbol}" for market, symbol in sorted(error_keys)),
        )
