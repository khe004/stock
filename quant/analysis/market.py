"""市场概览页用到的纯计算：52 周区间位置、行业宽度、收益率曲线利差。不含 UI。"""

import pandas as pd


def range_position(prices: pd.Series, window: int = 252) -> float | None:
    """最新价在过去 window 个交易日 [低点, 高点] 区间中的位置，0~1。
    数据不足或区间退化（高==低）时返回 None。"""
    recent = prices.iloc[-window:]
    if len(recent) < 2:
        return None
    lo, hi = float(recent.min()), float(recent.max())
    if hi <= lo:
        return None
    return (float(prices.iloc[-1]) - lo) / (hi - lo)


def sector_breadth(sector_closes: dict[str, pd.Series], ma: int = 200) -> dict:
    """行业 ETF 中收盘价站上 ma 日均线的只数与参与统计的总数。"""
    above, total = 0, 0
    for close in sector_closes.values():
        if len(close) < ma:
            continue
        total += 1
        if float(close.iloc[-1]) >= float(close.rolling(ma).mean().iloc[-1]):
            above += 1
    return {"above": above, "total": total}


def yield_curve_spread(long_yield: pd.Series, short_yield: pd.Series) -> float | None:
    """长端减短端收益率利差（百分点），负值代表倒挂。yfinance 的 ^TNX/^IRX
    已是百分比数值（如 4.5 代表 4.5%），直接相减即可。"""
    if long_yield.empty or short_yield.empty:
        return None
    return float(long_yield.iloc[-1]) - float(short_yield.iloc[-1])
