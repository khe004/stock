"""策略基类与信号数据结构。"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd

BUY = "buy"
SELL = "sell"


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
