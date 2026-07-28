"""策略基类与信号数据结构。"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd

BUY = "buy"
SELL = "sell"


def price_series(df: pd.DataFrame) -> pd.Series:
    """收益计算用的总回报价格序列：优先 adj_close（含分红再投资），回退 close。
    TLT 等标的收益大头在分红，用 close 会严重低估长期回报。"""
    if "adj_close" in df.columns and df["adj_close"].notna().any():
        return df["adj_close"]
    return df["close"]


def month_anchors(index: pd.DatetimeIndex, offset: int = 0) -> pd.DatetimeIndex:
    """月度调仓锚点：每月的第 (offset+1) 个交易日。

    offset=0（默认）= 每月首个交易日，即全平台月频策略的既定行为，改动前后完全一致。

    offset>0 是为【调仓日 timing luck 检验】准备的（Newfound / Hoffstein-Faber-Braun）：
    "锚在月首日"本身是个从未被检验的隐含选择，把锚点错开到第 6/11/16 个交易日
    （offset=5/10/15，≈4 个周度错峰 tranche）重跑，结果散布就是 timing luck 的幅度。
    实测本平台年化跨度 133~569bp，最低的都超过文献典型值 100bp——所以这不是理论问题。
    详见 quant/analysis/robustness.py 与 CLAUDE.md 关键设计决策 #9。

    当月交易日不足 offset+1 天的月份自动跳过（月末几天开市的残月）。
    """
    if index.empty:
        return index[:0]
    grouped = pd.Series(range(len(index)), index=index).groupby(
        [index.year, index.month]
    )
    picks = grouped.nth(offset) if offset else grouped.head(1)
    return index[picks.to_numpy()]


@dataclass
class Signal:
    date: str        # YYYY-MM-DD
    symbol: str
    strategy: str
    direction: str   # buy / sell
    price: float     # 信号日收盘价
    strength: float  # 0~1
    reason: str      # 人话解释，推送与面板直接展示


class Strategy(ABC):
    """策略对全部历史生成信号；每日运行时由调用方筛选出最新一天的信号，
    回测时使用完整信号序列。"""

    name = "base"

    @abstractmethod
    def generate(self, prices: dict[str, pd.DataFrame]) -> list[Signal]:
        """prices: symbol -> 日线 DataFrame（含 close 列，DatetimeIndex 升序）。"""
