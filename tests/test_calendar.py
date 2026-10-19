"""NYSE 交易日历：规则假日推算（休市日不发日报）。"""

from datetime import date

from quant.data.calendar import _easter, is_trading_day, nyse_holidays


def test_easter_known_years():
    assert _easter(2024) == date(2024, 3, 31)
    assert _easter(2026) == date(2026, 4, 5)
    assert _easter(2027) == date(2027, 3, 28)


def test_known_closures_and_observed_rules():
    closed = [date(2026, 1, 1), date(2026, 1, 19), date(2026, 2, 16), date(2026, 4, 3),
              date(2026, 5, 25), date(2026, 6, 19), date(2026, 7, 3),   # 7-4 周六 → 周五补休
              date(2026, 9, 7), date(2026, 11, 26), date(2026, 12, 25),
              date(2027, 12, 24),                                       # 圣诞周六 → 周五
              date(2023, 1, 2)]                                         # 元旦周日 → 周一
    for d in closed:
        assert not is_trading_day(d), d
    # 元旦落在周六（2022-01-01）不在前一年 12-31 补休；2021 年还没有六月节
    assert is_trading_day(date(2021, 12, 31))
    assert date(2021, 6, 18) not in nyse_holidays(2021)
    assert not is_trading_day(date(2026, 10, 17))   # 周六
    assert is_trading_day(date(2026, 10, 19))
    assert is_trading_day(date(2026, 11, 27))       # 感恩节次日半天也算开盘
