"""行业 12-1 月度动量轮动：每月首个交易日按 12-1 动量横截面排名，持有前 top_n 的板块。

12-1 动量口径：近 lookback_days 收益但跳过最近 skip_days，避开短期反转/买在山顶。
旧版（63日回看 + 每日进出）实测跑输板块等权基准（whipsaw + 短期反转所致），
改为 252/skip21 月度调仓后拿得更稳、交易次数大幅下降、总收益与 Calmar 显著改善。
"""

import pandas as pd

from quant.strategies.base import BUY, SELL, Signal, Strategy, price_series


class Momentum(Strategy):
    name = "momentum"

    def __init__(self, lookback_days: int = 252, skip_days: int = 21,
                 top_n: int = 3, **_):
        self.lookback = lookback_days
        self.skip = skip_days
        self.top_n = top_n

    def generate(self, prices: dict[str, pd.DataFrame]) -> list[Signal]:
        if len(prices) <= self.top_n:
            return []
        # 排名收益用总回报口径（adj_close），信号展示价用原始收盘价
        closes = pd.DataFrame({s: df["close"] for s, df in prices.items()}).sort_index()
        adj = pd.DataFrame({s: price_series(df) for s, df in prices.items()}).sort_index()

        # 12-1 动量：t-skip 相对 t-lookback 的收益（参考 stock_momentum 的 mom 计算）
        mom = adj.shift(self.skip) / adj.shift(self.lookback) - 1

        # 月度调仓日：每月首个交易日
        month_firsts = closes.groupby(
            [closes.index.year, closes.index.month]
        ).head(1).index

        signals: list[Signal] = []
        held: set[str] = set()  # 当前持有的标的

        for ts in month_firsts:
            # 取当日各标的的 12-1 动量值，跳过 NaN（窗口不足）
            row = mom.loc[ts].dropna()
            if len(row) <= self.top_n:
                continue

            # 按 12-1 动量降序排名，取前 top_n
            ranked = row.sort_values(ascending=False)
            top_syms = set(ranked.index[:self.top_n])

            # 先卖后买（与 dual_momentum / stock_momentum / low_vol 一致）
            # 卖出：原来持有但本月跌出前 top_n 的
            for sym in list(held):
                if sym not in top_syms:
                    if pd.notna(closes.at[ts, sym]):
                        sym_mom = float(mom.at[ts, sym]) if pd.notna(mom.at[ts, sym]) else 0.0
                        rank = list(ranked.index).index(sym) + 1 if sym in ranked.index else len(ranked)
                        reason = (f"{sym}：12-1 动量 {sym_mom:+.1%}，"
                                  f"跌出行业动量前{self.top_n}名（第{rank}名），调出组合")
                        signals.append(self._sig(ts, sym, closes, mom, SELL, reason, sym_mom))
                    held.discard(sym)

            # 买入：本月在前 top_n 但之前没持有的
            for sym in ranked.index[:self.top_n]:
                if sym not in held:
                    if pd.notna(closes.at[ts, sym]):
                        sym_mom = float(ranked[sym])
                        rank = list(ranked.index).index(sym) + 1
                        reason = (f"{sym}：12-1 动量 {sym_mom:+.1%}，"
                                  f"行业动量第{rank}名，纳入轮动组合")
                        signals.append(self._sig(ts, sym, closes, mom, BUY, reason, sym_mom))
                    held.add(sym)

        return signals

    def _sig(self, ts, symbol, closes, mom, direction, reason, mom_val) -> Signal:
        """构造 Signal，strength 用动量值映射到 0~1。

        映射逻辑：min(1.0, max(0.1, abs(mom_val) * 2))
        - 动量 ±50% 以上 → strength 1.0
        - 动量 ±5%      → strength 0.1
        参考 stock_momentum 的 strength 映射。
        """
        return Signal(
            date=ts.strftime("%Y-%m-%d"),
            symbol=symbol,
            strategy=self.name,
            direction=direction,
            price=round(float(closes.at[ts, symbol]), 2),
            strength=round(min(1.0, max(0.1, abs(mom_val) * 2)), 2),
            reason=reason,
        )
