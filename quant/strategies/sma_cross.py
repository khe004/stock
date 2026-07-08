"""双均线策略：快线上穿慢线买入，下穿卖出。"""

import pandas as pd

from quant.strategies.base import BUY, SELL, Signal, Strategy


class SmaCross(Strategy):
    name = "sma_cross"

    def __init__(self, fast: int = 20, slow: int = 60, **_):
        self.fast = fast
        self.slow = slow

    def generate(self, prices: dict[str, pd.DataFrame]) -> list[Signal]:
        signals: list[Signal] = []
        for symbol, df in prices.items():
            if len(df) < self.slow + 1:
                continue
            close = df["close"]
            fast_ma = close.rolling(self.fast).mean()
            slow_ma = close.rolling(self.slow).mean()
            above = fast_ma > slow_ma
            valid = slow_ma.notna() & slow_ma.shift(1).notna()
            cross_up = above & ~above.shift(1, fill_value=False) & valid
            cross_dn = ~above & above.shift(1, fill_value=False) & valid
            # 强度用快线近 5 日斜率衡量：拐头越快越强
            slope = fast_ma.pct_change(5).abs()
            for ts in df.index[cross_up]:
                signals.append(self._make(symbol, ts, close, slope, BUY, "上穿"))
            for ts in df.index[cross_dn]:
                signals.append(self._make(symbol, ts, close, slope, SELL, "下穿"))
        return signals

    def _make(self, symbol, ts, close, slope, direction, verb) -> Signal:
        strength = round(min(1.0, max(0.1, float(slope.get(ts, 0) or 0) * 20)), 2)
        return Signal(
            date=ts.strftime("%Y-%m-%d"),
            symbol=symbol,
            strategy=self.name,
            direction=direction,
            price=round(float(close[ts]), 2),
            strength=strength,
            reason=f"{symbol}：{self.fast}日均线{verb}{self.slow}日均线，收盘 ${close[ts]:.2f}",
        )
