"""VIX 情绪/风险提醒：恐慌区进入与消退、自满区、期限结构倒挂与解除。

信号标的为 ^VIX（price 字段是 VIX 点位），本身不可交易；buy/sell 表示对
股票仓位的方向建议（sell=风险预警减仓，buy=恐慌消退回补）。
"""

import pandas as pd

from quant.strategies.base import BUY, SELL, Signal, Strategy


class VixRegime(Strategy):
    name = "vix_regime"

    def __init__(self, vix: str = "^VIX", vix3m: str = "^VIX3M",
                 panic: float = 30, complacency: float = 15,
                 trade_symbol: str = "SPY", **_):
        self.vix = vix
        self.vix3m = vix3m
        self.panic = panic
        self.complacency = complacency
        self.trade_symbol = trade_symbol

    def generate(self, prices: dict[str, pd.DataFrame]) -> list[Signal]:
        vdf = prices.get(self.vix)
        if vdf is None or len(vdf) < 2:
            return []
        v = vdf["close"]
        prev = v.shift(1)
        signals: list[Signal] = []

        up_panic = ((prev < self.panic) & (v >= self.panic)).fillna(False)
        dn_panic = ((prev > self.panic) & (v <= self.panic)).fillna(False)
        dn_compl = ((prev > self.complacency) & (v <= self.complacency)).fillna(False)

        for ts in v.index[up_panic]:
            signals.append(self._sig(ts, float(v[ts]), SELL,
                f"VIX 上穿 {self.panic:.0f} 进入恐慌区（现 {v[ts]:.1f}），波动加剧，注意控制仓位",
                strength=min(1.0, float(v[ts]) / 50)))
        for ts in v.index[dn_panic]:
            signals.append(self._sig(ts, float(v[ts]), BUY,
                f"VIX 回落穿 {self.panic:.0f}（现 {v[ts]:.1f}），恐慌消退，历史上常是分批回补窗口",
                strength=0.7))
        for ts in v.index[dn_compl]:
            signals.append(self._sig(ts, float(v[ts]), SELL,
                f"VIX 跌破 {self.complacency:.0f}（现 {v[ts]:.1f}），市场进入自满区，防范突发回调",
                strength=0.3))

        v3df = prices.get(self.vix3m)
        if v3df is not None and len(v3df) >= 2:
            spread = (v - v3df["close"]).dropna()
            sp_prev = spread.shift(1)
            invert = ((sp_prev < 0) & (spread >= 0)).fillna(False)
            restore = ((sp_prev >= 0) & (spread < 0)).fillna(False)
            for ts in spread.index[invert]:
                signals.append(self._sig(ts, float(v[ts]), SELL,
                    f"VIX 期限结构倒挂（VIX {v[ts]:.1f} ≥ VIX3M），近期恐惧超过远期，可靠性较高的风险预警",
                    strength=0.8))
            for ts in spread.index[restore]:
                signals.append(self._sig(ts, float(v[ts]), BUY,
                    f"VIX 期限结构倒挂解除（VIX {v[ts]:.1f} < VIX3M），风险预警撤除",
                    strength=0.6))

        signals.sort(key=lambda s: s.date)
        return signals

    def _sig(self, ts, level: float, direction: str, reason: str, strength: float) -> Signal:
        return Signal(
            date=ts.strftime("%Y-%m-%d"), symbol=self.vix, strategy=self.name,
            direction=direction, price=round(level, 2),
            strength=round(strength, 2), reason=reason,
        )
