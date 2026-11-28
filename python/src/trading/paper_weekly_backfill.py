"""紙の週の成績のファイルの読み書きと、前の週の見直し（#1695）。

週の回は対象の週を書いた後に前の3週を見て、ファイルの無い週と、超過の騰落率が null か
部分の週の市場がある週を計算し直す。kind:strategy の判定は4週の超過の騰落率を足すため、
抜けた週や null の週を次の回で埋める。それでも残る週は1度だけ知らせる。
"""

from __future__ import annotations

import json
import os
import re
from datetime import date, timedelta
from typing import Any, Callable, Optional, Sequence

from src.utils.logger import get_logger

logger = get_logger(__name__)

# 対象の週の前に見る週の数
WEEKS_TO_REVIEW = 3

# 超過の騰落率の計算の印。これの無い週のファイルは、見る範囲に限らず全部を計算し直す（#1792）
EXCESS_BASIS = "exposure_scaled"

# 入金のある週の投資比率を、入金後の元手で割って書いた印（#2177）。入金のある週だけに書く。
# 累計を分けないよう excess_basis の値は変えず、別の欄にする
EXPOSURE_DEPOSIT_KEY = "exposure_deposit_basis"
EXPOSURE_DEPOSIT_BASIS = "after_deposits"

ComputeWeek = Callable[[date], dict[str, Any]]
Notify = Callable[[str], bool]

_WEEK_FILE = re.compile(r"^\d{4}-W\d{2}\.json$")
# 知らせた行の記録。週のファイルの本数に数えられないよう .json にしない
_NOTICE_FILE = ".backfill_notices.txt"


def week_name(week_end: date) -> str:
    """週の終わりの金曜から ISO 週の名前（YYYY-Www）を返す。"""
    iso = week_end.isocalendar()
    return f"{iso[0]}-W{iso[1]:02d}"


def write_week_file(output_dir: str, result: dict[str, Any]) -> str:
    """週の成績を output_dir/YYYY-Www.json に書き、そのパスを返す。"""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{result['week']}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    logger.info("紙の売買の週の成績を書いた: %s", path)
    return path


def _read_week_file(path: str) -> Optional[dict[str, Any]]:
    """週のファイルを読む。無いか、読めないか、市場の値が無ければ None（書き直す）。"""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        logger.warning("紙の週の成績のファイルが読めないため書き直す: %s", path, exc_info=True)
        return None
    if not isinstance(data, dict) or not isinstance(data.get("markets"), dict):
        logger.warning("紙の週の成績のファイルの形が違うため書き直す: %s", path)
        return None
    return data if data["markets"] else None


def _week_defects(result: dict[str, Any]) -> dict[tuple[str, str], str]:
    """週の結果の欠け（(市場, 種類) -> 理由の文）を返す。"""
    defects: dict[tuple[str, str], str] = {}
    for market, values in sorted(result["markets"].items()):
        if values.get("excess_return") is None:
            defects[(market, "excess")] = f"{market} 超過の騰落率が null"
        if values.get("partial_week"):
            symbols = ", ".join(values.get("missing_week_end_close") or [])
            defects[(market, "partial")] = f"{market} 部分の週（{symbols}）"
    return defects


def write_week_file_unless_worse(
    output_dir: str, result: dict[str, Any]
) -> tuple[str, dict[str, Any]]:
    """欠けが今のファイルより増えないときだけ週の成績を書き、パスと残った中身を返す。

    取れていた値を、取得の失敗の回の null で消さない（#1695・#1804）。今のファイルが
    無いか読めないとき、新しい結果の欠けが今のファイルの欠けと同じか少ないときは書く。
    """
    path = os.path.join(output_dir, f"{result['week']}.json")
    current = _read_week_file(path)
    if current is not None and not _week_defects(result).keys() <= _week_defects(current).keys():
        logger.warning("紙の週の成績の書き直しで欠けが増えるため元のまま残す: %s", result["week"])
        return path, current
    return write_week_file(output_dir, result), result


def _is_old_basis(result: dict[str, Any]) -> bool:
    """超過の騰落率が今の計算の印で書かれていない市場があれば True。"""
    return any(
        not isinstance(values, dict) or values.get("excess_basis") != EXCESS_BASIS
        for values in result["markets"].values()
    )


def has_deposit_in_week(week_end: date, deposit_days: Sequence[date]) -> bool:
    """前の週の金曜より後から week_end の金曜までに入金があれば True（土日の入金は次の週）。"""
    return any(week_end - timedelta(weeks=1) < day <= week_end for day in deposit_days)


def _is_old_deposit_basis(
    result: dict[str, Any], week_end: date, deposit_days: Sequence[date]
) -> bool:
    """入金のある週なのに、投資比率が入金後の元手で割った印で書かれていなければ True（#2177）。"""
    return (
        has_deposit_in_week(week_end, deposit_days)
        and result.get(EXPOSURE_DEPOSIT_KEY) != EXPOSURE_DEPOSIT_BASIS
    )


def _week_end_of(week: str) -> date:
    """ISO 週の名前（YYYY-Www）からその週の金曜を返す。"""
    year, number = week.split("-W")
    return date.fromisocalendar(int(year), int(number), 5)


def _review_week(
    week: str,
    week_end: date,
    compute_week: ComputeWeek,
    output_dir: str,
    deposit_days: Sequence[date] = (),
) -> Optional[str]:
    """1つの週を見て、要れば計算し直して書く。欠けが残れば理由を返す。"""
    path = os.path.join(output_dir, f"{week}.json")
    current = _read_week_file(path)
    if (
        current is not None
        and not _week_defects(current)
        and not _is_old_basis(current)
        and not _is_old_deposit_basis(current, week_end, deposit_days)
    ):
        return None

    # 土曜を as_of にすると、直前に終わった週はその金曜の週になる
    result = compute_week(week_end + timedelta(days=1))
    _, kept = write_week_file_unless_worse(output_dir, result)
    return "、".join(_week_defects(kept).values()) or None


def _review_frozen_week(
    week: str,
    week_end: date,
    compute_week: ComputeWeek,
    output_dir: str,
    base_balance: Optional[float],
) -> None:
    """元手の切り替えより前の週を見る。前の計算の印のファイルだけを計算し直す（#2202）。

    今の印で書かれた週は、欠けがあっても書き直さない（#1877）。計算し直した週の元手が
    入金を足さない元手 base_balance と違えば書かない。base_balance が None なら見ない。
    欠けが増えるときは書かない決まり（write_week_file_unless_worse）は同じ。
    """
    if base_balance is None:
        return
    current = _read_week_file(os.path.join(output_dir, f"{week}.json"))
    if current is None or not _is_old_basis(current):
        return
    result = compute_week(week_end + timedelta(days=1))
    balance = result.get("initial_balance")
    if not isinstance(balance, (int, float)) or abs(float(balance) - base_balance) > 1e-6:
        logger.warning(
            "切り替えの前の週の元手が入金を足さない元手と違うため元のまま残す: %s %s",
            week,
            balance,
        )
        return
    write_week_file_unless_worse(output_dir, result)


def _review_weeks(
    target_week_end: date,
    compute_week: ComputeWeek,
    output_dir: str,
    frozen_before: Optional[date] = None,
    deposit_days: Sequence[date] = (),
    base_balance: Optional[float] = None,
) -> tuple[list[str], dict[str, str]]:
    """対象の週の前の週を見て、見た週の名前と、欠けが残る週の理由を返す。"""
    names = [name[:-5] for name in os.listdir(output_dir) if _WEEK_FILE.match(name)]
    checked: list[str] = []
    remaining: dict[str, str] = {}
    if not names:
        return checked, remaining

    oldest = min(names)
    weeks: dict[str, date] = {}
    for back in range(WEEKS_TO_REVIEW, 0, -1):
        week_end = target_week_end - timedelta(weeks=back)
        week = week_name(week_end)
        if week >= oldest:
            weeks[week] = week_end
    # 前の計算の印のファイルは、見る範囲の外の週も計算し直す（#1792・#2177）
    target = week_name(target_week_end)
    for week in names:
        if week < target and week not in weeks:
            current = _read_week_file(os.path.join(output_dir, f"{week}.json"))
            if current is None:
                continue
            week_end = _week_end_of(week)
            if _is_old_basis(current) or _is_old_deposit_basis(current, week_end, deposit_days):
                weeks[week] = week_end
    for week, week_end in sorted(weeks.items()):
        if frozen_before is not None and week_end < frozen_before:
            # 元手の切り替えより前の週は、前の計算の印のファイルだけを計算し直す（#1877・#2202）
            try:
                _review_frozen_week(week, week_end, compute_week, output_dir, base_balance)
            except Exception:
                logger.error("切り替えの前の紙の週の成績の書き直し失敗: %s", week, exc_info=True)
            continue
        checked.append(week)
        try:
            reason = _review_week(week, week_end, compute_week, output_dir, deposit_days)
        except Exception as e:
            logger.error("紙の週の成績の書き直し失敗: %s", week, exc_info=True)
            reason = f"書き直しが落ちた（{type(e).__name__}）"
        if reason:
            logger.warning("書き直しても紙の週の成績に欠けが残る: %s %s", week, reason)
            remaining[week] = reason
    return checked, remaining


def _read_notice_lines(path: str) -> list[str]:
    try:
        with open(path, encoding="utf-8") as f:
            return f.read().splitlines()
    except FileNotFoundError:
        return []


def _notify_remaining(
    checked: list[str], remaining: dict[str, str], output_dir: str, notify: Optional[Notify]
) -> None:
    """残る週のうち、まだ知らせていない行を送り、送れた行を記録する。

    補完の回が同じ週を見ても、残る週が見る範囲から外れるまでの毎週の回でも、
    週と理由が同じなら2度は送らない。送れなかった行は次の回に送る。
    """
    if notify is None:
        return
    path = os.path.join(output_dir, _NOTICE_FILE)
    sent = _read_notice_lines(path)
    lines = [f"{week}: {reason}" for week, reason in remaining.items()]
    new_lines = [line for line in lines if line not in sent]
    if not new_lines:
        return
    try:
        delivered = notify("⚠️ 紙の週の成績に、書き直しても残る週がある\n" + "\n".join(new_lines))
    except Exception:
        logger.error("紙の週の成績の残る週の通知失敗", exc_info=True)
        return
    if not delivered:
        logger.error("紙の週の成績の残る週の通知が送れなかった: %s", new_lines)
        return
    # 見る範囲から外れた週の行は、もう要らないので捨てる
    kept = [line for line in sent if line.split(":", 1)[0] in checked]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(kept + new_lines) + "\n")


def backfill_previous_weeks(
    target_week_end: date,
    compute_week: ComputeWeek,
    output_dir: str,
    notify: Optional[Notify] = None,
    frozen_before: Optional[date] = None,
    deposit_days: Sequence[date] = (),
    base_balance: Optional[float] = None,
) -> None:
    """対象の週の前の週を見直す。例外は出さない（対象の週の書き出しの成否を変えない）。

    今ある一番古いファイルより前の週は作らない。金曜が frozen_before（元手の切り替えの日）
    より前の週は、今の印で書かれていれば計算し直さず、ファイルも書き直さない（#1877）。
    前の計算の印の週だけを計算し直し、元手が base_balance のときに書く（#2202）。

    Args:
        target_week_end: 対象の週の金曜
        compute_week: as_of -> その直前に終わった週の結果
        output_dir: 週のファイルの置き場
        notify: 知らせの文を送り、送れたかを返す関数。None なら知らせない
        frozen_before: 元手の切り替えの日
        deposit_days: 入金の日。入金のある週で印の無いファイルを計算し直す（#2177）
        base_balance: 入金を足さない元手。None なら切り替えの前の週は書き直さない
    """
    try:
        checked, remaining = _review_weeks(
            target_week_end,
            compute_week,
            output_dir,
            frozen_before,
            deposit_days,
            base_balance,
        )
        _notify_remaining(checked, remaining, output_dir, notify)
    except Exception:
        logger.error("紙の週の成績の前の週の見直し失敗", exc_info=True)
