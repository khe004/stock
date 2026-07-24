"""市场概览页用到的纯计算：52 周区间位置、行业宽度、收益率曲线利差。不含 UI。"""

import pandas as pd

# ETF/标的中文名（面板与邮件共用，避免只显示 XLK 这种记不住的代码）
ETF_NAMES = {
    # 11 个 SPDR 行业
    "XLK": "科技", "XLV": "医疗", "XLF": "金融", "XLY": "可选消费",
    "XLP": "必需消费", "XLE": "能源", "XLI": "工业", "XLB": "材料",
    "XLU": "公用事业", "XLRE": "房地产", "XLC": "通信",
    # 大盘/成长/主题/资产
    "SPY": "标普500", "QQQ": "纳指100", "IWM": "罗素2000", "DIA": "道指",
    "SMH": "半导体", "SOXX": "半导体", "IGV": "软件", "XBI": "生物科技",
    "TLT": "长债", "GLD": "黄金", "IBIT": "比特币",
}


def etf_label(sym: str) -> str:
    """代码 + 中文名，如 'XLK 科技'；无名则原样返回。"""
    name = ETF_NAMES.get(sym)
    return f"{sym} {name}" if name else sym


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
