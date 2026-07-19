"""低波动因子策略：横截面按已实现波动率排名，持有波动最低的前 top_n 只。

低波动异象（Low Volatility Anomaly）：传统金融理论认为高风险应有高回报，
但实证研究（Baker, Bradley & Wurgler 2011; Ang et al. 2006）发现低波动资产的
长期风险调整后收益反而更好。原因包括彩票偏好（投资者高估高波动股的上涨潜力）、
杠杆约束（机构不能杠杆买低波动所以它被低估）、以及基准追踪导致的系统性忽视。

本策略是平台首个非动量因子，目的是与动量家族（momentum / dual_momentum /
stock_momentum，三者相关性约 0.51）提供低相关的分散来源。

方法论：
- 波动率口径：近 lookback_days 日已实现波动率 = 日收益率标准差 × √252（年化）
- 收益率用总回报口径 price_series(df)（复权价，含分红再投资）计算波动率
- 信号展示价 Signal.price 用原始 close
- 月度调仓（每月首个交易日），先卖后买（参考 dual_momentum/stock_momentum 模式）
- 新进入最低波动前 top_n → BUY；跌出 → SELL
"""

import numpy as np
import pandas as pd

from quant.strategies.base import BUY, SELL, Signal, Strategy, price_series


class LowVol(Strategy):
    """低波动因子：横截面按近 N 日年化波动率排名，持有波动最低的前 top_n 只，月度调仓。

    这是平台首个非动量因子策略。动量选「涨得最猛的」，低波动选「波动最低的」，
    二者选择标准正交，理论上低相关——可到「策略相关性」页验证实际相关系数。
    """

    name = "low_vol"

    def __init__(self, lookback_days: int = 90, top_n: int = 3, **_):
        self.lookback = lookback_days
        self.top_n = top_n

    def generate(self, prices: dict[str, pd.DataFrame]) -> list[Signal]:
        """对全量历史生成低波动选股信号。

        Args:
            prices: symbol -> 日线 DataFrame（含 close / adj_close 列，DatetimeIndex 升序）

        Returns:
            按日期升序的信号列表。月度调仓，先卖后买。
        """
        if len(prices) <= self.top_n:
            return []

        # 收益率用总回报口径（adj_close）计算波动率；信号展示价用原始 close
        closes = pd.DataFrame({s: df["close"] for s, df in prices.items()}).sort_index()
        adj = pd.DataFrame({s: price_series(df) for s, df in prices.items()}).sort_index()
        daily_ret = adj.pct_change(fill_method=None)

        # 滚动年化波动率 = 日收益率的 lookback 日滚动标准差 × √252
        vol = daily_ret.rolling(self.lookback).std() * np.sqrt(252)

        # 月度调仓日：每月首个交易日
        month_firsts = closes.groupby(
            [closes.index.year, closes.index.month]
        ).head(1).index

        signals: list[Signal] = []
        held: set[str] = set()  # 当前持有的标的

        for ts in month_firsts:
            # 取当日各标的的年化波动率，跳过 NaN（窗口不足）
            row = vol.loc[ts].dropna()
            if len(row) <= self.top_n:
                continue

            # 按波动率升序排名（最低=第1名），取前 top_n
            ranked = row.sort_values()
            top_syms = set(ranked.index[:self.top_n])

            # 先卖后买（与 dual_momentum / stock_momentum 一致）
            # 卖出：原来持有但本月不在最低波动组合中的
            for sym in list(held):
                if sym not in top_syms:
                    if pd.notna(closes.at[ts, sym]):
                        sell_vol = float(vol.at[ts, sym]) if pd.notna(vol.at[ts, sym]) else None
                        reason = (f"{sym}：波动升高跌出最低波动前{self.top_n}名，"
                                  f"调出低波组合")
                        if sell_vol is not None:
                            reason = (f"{sym}：近{self.lookback}日年化波动 {sell_vol:.1%}，"
                                      f"跌出最低波动前{self.top_n}名，调出低波组合")
                        signals.append(self._sig(
                            ts, sym, closes, vol, ranked, SELL, reason))
                    held.discard(sym)

            # 买入：本月在最低波动组合中但之前没持有的
            for sym in ranked.index[:self.top_n]:
                if sym not in held:
                    if pd.notna(closes.at[ts, sym]):
                        sym_vol = float(ranked[sym])
                        rank = list(ranked.index).index(sym) + 1
                        reason = (f"{sym}：近{self.lookback}日年化波动 {sym_vol:.1%}，"
                                  f"为最低波动前{self.top_n}名（第{rank}名），纳入低波组合")
                        signals.append(self._sig(
                            ts, sym, closes, vol, ranked, BUY, reason))
                    held.add(sym)

        return signals

    def _sig(self, ts, symbol, closes, vol, ranked, direction, reason) -> Signal:
        """构造 Signal，strength 按波动率映射到 0~1。

        映射逻辑：波动越低 → 强度越高（更值得持有）。
        用 1 - clip(vol / 0.5, 0, 1) 映射：
        - 年化波动 0%   → strength 1.0（极低波动，最强信号）
        - 年化波动 25%  → strength 0.5（中等波动）
        - 年化波动 ≥50% → strength 0.0（高波动，最弱）
        最终 clip 到 [0.1, 1.0] 避免极端值。
        """
        sym_vol = float(vol.at[ts, symbol]) if pd.notna(vol.at[ts, symbol]) else 0.25
        # 波动越低 → 强度越高
        strength = 1.0 - min(1.0, sym_vol / 0.5)
        strength = round(min(1.0, max(0.1, strength)), 2)

        return Signal(
            date=ts.strftime("%Y-%m-%d"),
            symbol=symbol,
            strategy=self.name,
            direction=direction,
            price=round(float(closes.at[ts, symbol]), 2),
            strength=strength,
            reason=reason,
        )
