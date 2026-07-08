"""RSI 反转策略：RSI 从超卖区回升买入，从超买区回落卖出。"""

import pandas as pd

from quant.strategies.base import BUY, SELL, Signal, Strategy


def wilder_rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rsi = 100 - 100 / (1 + avg_gain / avg_loss)
    return rsi.where(avg_loss != 0, 100.0)


class RsiReversal(Strategy):
    name = "rsi_reversal"

    def __init__(self, period: int = 14, oversold: float = 30, overbought: float = 70, **_):
        self.period = period
        self.oversold = oversold
        self.overbought = overbought

    def generate(self, prices: dict[str, pd.DataFrame]) -> list[Signal]:
        signals: list[Signal] = []
        for symbol, df in prices.items():
            if len(df) < self.period + 2:
                continue
            close = df["close"]
            rsi = wilder_rsi(close, self.period)
            prev = rsi.shift(1)
            buy = (prev < self.oversold) & (rsi >= self.oversold)
            sell = (prev > self.overbought) & (rsi <= self.overbought)
            for ts in df.index[buy.fillna(False)]:
                # 前一日 RSI 越低，超卖越深，信号越强
                strength = min(1.0, max(0.1, (self.oversold - float(prev[ts])) / self.oversold + 0.3))
                signals.append(Signal(
                    date=ts.strftime("%Y-%m-%d"), symbol=symbol, strategy=self.name,
                    direction=BUY, price=round(float(close[ts]), 2), strength=round(strength, 2),
                    reason=f"{symbol}：RSI({self.period}) 从超卖区回升至 {rsi[ts]:.0f}，收盘 ${close[ts]:.2f}",
                ))
            for ts in df.index[sell.fillna(False)]:
                strength = min(1.0, max(0.1, (float(prev[ts]) - self.overbought) / (100 - self.overbought) + 0.3))
                signals.append(Signal(
                    date=ts.strftime("%Y-%m-%d"), symbol=symbol, strategy=self.name,
                    direction=SELL, price=round(float(close[ts]), 2), strength=round(strength, 2),
                    reason=f"{symbol}：RSI({self.period}) 从超买区回落至 {rsi[ts]:.0f}，收盘 ${close[ts]:.2f}",
                ))
        return signals
