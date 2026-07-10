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
