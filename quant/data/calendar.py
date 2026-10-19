"""美股（NYSE）交易日历：按规则推算的固定休市日，不依赖外部库。

用途：休市日（周末/节假日）每日运行照常拉数据，但不发日报（2026-10-19 用户要求）。
只覆盖规则性假日；国丧日、飓风等临时休市（如 2012-10-29/30、2018-12-05、2025-01-09）
无法事先推算——那几天会照常发一份"无新信号"的日报，无害。
"""

from __future__ import annotations

from datetime import date, timedelta


def _easter(year: int) -> date:
    """公历复活节（Anonymous Gregorian algorithm）。"""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l_ = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l_) // 451
    month, day = divmod(h + l_ - 7 * m + 114, 31)
    return date(year, month, day + 1)


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """某月第 n 个星期几（weekday: 周一=0）；n=-1 表示最后一个。"""
    if n > 0:
        d = date(year, month, 1)
        d += timedelta(days=(weekday - d.weekday()) % 7)
        return d + timedelta(weeks=n - 1)
    d = date(year + (month == 12), month % 12 + 1, 1) - timedelta(days=1)
    return d - timedelta(days=(d.weekday() - weekday) % 7)


def _observed(d: date) -> date:
    """周六的假日提前到周五、周日的顺延到周一。"""
    if d.weekday() == 5:
        return d - timedelta(days=1)
    if d.weekday() == 6:
        return d + timedelta(days=1)
    return d


def nyse_holidays(year: int) -> set[date]:
    hol = {
        _nth_weekday(year, 1, 0, 3),               # 马丁·路德·金纪念日
        _nth_weekday(year, 2, 0, 3),               # 总统日
        _easter(year) - timedelta(days=2),          # 耶稣受难日
        _nth_weekday(year, 5, 0, -1),              # 阵亡将士纪念日
        _observed(date(year, 7, 4)),                # 独立日
        _nth_weekday(year, 9, 0, 1),               # 劳动节
        _nth_weekday(year, 11, 3, 4),              # 感恩节
        _observed(date(year, 12, 25)),              # 圣诞节
    }
    # 元旦落在周六时 NYSE 不在前一年 12-31 补休
    ny = date(year, 1, 1)
    if ny.weekday() != 5:
        hol.add(_observed(ny))
    if year >= 2022:                                # 六月节
        hol.add(_observed(date(year, 6, 19)))
    return hol


def is_trading_day(d: date) -> bool:
    return d.weekday() < 5 and d not in nyse_holidays(d.year)
