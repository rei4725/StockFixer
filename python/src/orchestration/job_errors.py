"""スケジューラのジョブが投げる、やり直さない失敗。

scheduler_queue.py の行数の上限（600 行）を超えないよう、例外の型だけ分けた。
"""


class NonRetryableJobError(RuntimeError):
    """やり直しても結果が変わらない失敗。scheduler_queue はリトライせずに失敗にする。

    ジョブが自分の内で知らせを送ってから投げる失敗に使う。リトライすると同じ処理と
    同じ知らせが max_retries 回繰り返される（週次ルール評価の 4 通の重複）。
    """


class LockWaitExpiredJobError(NonRetryableJobError):
    """ロックを待ち切って落ちた失敗。この回からは recovery_min_gap_minutes を数えない。

    待ちの間に間はもう空いていて、障害の明けを待つ理由も無い（#1682）。
    """
