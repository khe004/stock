"""智能定投：每月定投提醒，死叉期暂停积攒、金叉恢复补投。"""

import pandas as pd

from quant.strategies.base import BUY, Signal, Strategy, price_series


class SmartDca(Strategy):
    name = "smart_dca"

    def __init__(self, symbol: str = "SPY", fast: int = 20, slow: int = 60, **_):
        self.symbol = symbol
        self.fast = fast
        self.slow = slow

    def generate(self, prices: dict[str, pd.DataFrame]) -> list[Signal]:
        df = prices.get(self.symbol)
        if df is None or len(df) < self.slow + 1:
            return []
        close = df["close"]
        px = price_series(df)  # 趋势判断用总回报口径，与回测一致
        fast_ma = px.rolling(self.fast).mean()
        slow_ma = px.rolling(self.slow).mean()
        regime = (fast_ma >= slow_ma) | slow_ma.isna()
        month_firsts = set(close.groupby([close.index.year, close.index.month]).head(1).index)

        signals: list[Signal] = []
        pending = 0
        prev_r = True
        for ts, price in close.items():
            r = bool(regime[ts])
            resumed = r and not prev_r
            if ts in month_firsts:
                if r:
                    if pending:
                        reason = (f"{self.symbol}：定投日，趋势已恢复（MA{self.fast}≥MA{self.slow}），"
                                  f"正常定投并补投暂停期间累积的 {pending} 份")
                        pending = 0
                    else:
                        reason = f"{self.symbol}：定投日，MA{self.fast}≥MA{self.slow} 趋势向上，正常定投一份"
                    signals.append(self._buy(ts, float(price), reason))
                else:
                    pending += 1
            elif resumed and pending:
                signals.append(self._buy(
                    ts, float(price),
                    f"{self.symbol}：金叉恢复，补投暂停期间累积的 {pending} 份定投款",
                ))
                pending = 0
            prev_r = r
        return signals

    def _buy(self, ts, price: float, reason: str) -> Signal:
        return Signal(
            date=ts.strftime("%Y-%m-%d"), symbol=self.symbol, strategy=self.name,
            direction=BUY, price=round(price, 2), strength=0.5, reason=reason,
        )
