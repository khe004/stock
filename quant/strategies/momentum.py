"""动量轮动策略：组内按近 N 日收益排名，进入前 top_n 买入，跌出卖出。"""

import pandas as pd

from quant.strategies.base import BUY, SELL, Signal, Strategy


class Momentum(Strategy):
    name = "momentum"

    def __init__(self, lookback_days: int = 63, top_n: int = 3, **_):
        self.lookback = lookback_days
        self.top_n = top_n

    def generate(self, prices: dict[str, pd.DataFrame]) -> list[Signal]:
        if len(prices) <= self.top_n:
            return []
        closes = pd.DataFrame({s: df["close"] for s, df in prices.items()})
        rets = closes.pct_change(self.lookback, fill_method=None)
        ranks = rets.rank(axis=1, ascending=False)
        in_top = (ranks <= self.top_n) & rets.notna()

        signals: list[Signal] = []
        for symbol in closes.columns:
            top = in_top[symbol]
            prev_valid = rets[symbol].shift(1).notna()  # 首个有效日不发信号
            entered = top & ~top.shift(1, fill_value=False) & prev_valid
            exited = ~top & top.shift(1, fill_value=False) & rets[symbol].notna()
            for ts in closes.index[entered]:
                signals.append(self._make(symbol, ts, closes, rets, ranks, BUY))
            for ts in closes.index[exited]:
                signals.append(self._make(symbol, ts, closes, rets, ranks, SELL))
        signals.sort(key=lambda s: s.date)
        return signals

    def _make(self, symbol, ts, closes, rets, ranks, direction) -> Signal:
        ret = float(rets.loc[ts, symbol])
        rank = int(ranks.loc[ts, symbol])
        price = float(closes.loc[ts, symbol])
        if direction == BUY:
            reason = f"{symbol}：近{self.lookback}日收益 {ret:+.1%}，进入动量前{self.top_n}名（第{rank}名）"
        else:
            reason = f"{symbol}：近{self.lookback}日收益 {ret:+.1%}，跌出动量前{self.top_n}名（第{rank}名）"
        return Signal(
            date=ts.strftime("%Y-%m-%d"),
            symbol=symbol,
            strategy=self.name,
            direction=direction,
            price=round(price, 2),
            strength=round(min(1.0, max(0.1, abs(ret) * 5)), 2),
            reason=reason,
        )
