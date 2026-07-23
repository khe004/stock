"""跨资产 12-1 月度动量轮动 + 绝对动量开关：在多类资产间做横截面动量排名。

宇宙：SPY/QQQ/VEA/VWO/TLT/HYG/GLD/DBC/XLRE（9 类，横跨美股/国际/债/信用/金/商品/REITs）。
选择宇宙的理由：资产类别间低相关、离散度高，动量才有真信息——对比板块 ETF 或个股
（高相关宇宙里动量信号大部分来自共同市场 beta，排名噪音大、轮动收益低）。

动量口径：12-1 月度（lookback=252, skip=21），与 momentum / stock_momentum 一致。
每月首个交易日调仓，横截面取动量最高 top_n。

绝对动量开关（abs_momentum）：
- True（默认）：只买入动量为正的 picks——动量为负的位置空着 = 持现金。
  top_n 名额可能不满，极端时全部为负 = 全持现金。
  参考 dual_momentum 的绝对动量思想，在全市场下行时自动减仓。
- False：永远满仓 top_n，不管动量正负。

实测结论（2015-2026，top3 + abs_momentum=True）：
- 总收益 +237%，夏普 0.80，回撤 -21.5%，Calmar 0.52
- 跑赢"等权全资产"公平基准（+172%/0.39）= 动量真加了信息
- raw 收益跑输 SPY 长持（美股独大，分散必然拖累绝对收益），但风险调整后更优
- 绝对动量开关在这段牛市拖累了收益（和 dual_momentum 的 TLT 开关同理），是熊市保险

先卖后买 emit（参考 momentum / low_vol），reason 含人话数值。
"""

import pandas as pd

from quant.strategies.base import BUY, SELL, Signal, Strategy, price_series


class CrossAssetMomentum(Strategy):
    """跨资产 12-1 月度动量轮动：在股/债/金/商品/REITs/国际等多类资产间做横截面动量排名。

    宇宙含 9 类低相关资产，离散度高于板块/个股宇宙——动量信号含真信息。
    可选绝对动量开关（abs_momentum=True）：动量为负的资产不买入，持现金等待，
    熊市时自动减仓。
    """

    name = "cross_asset_mom"

    def __init__(self, lookback_days: int = 252, skip_days: int = 21,
                 top_n: int = 3, abs_momentum: bool = True, **_):
        self.lookback = lookback_days
        self.skip = skip_days
        self.top_n = top_n
        self.abs_momentum = abs_momentum

    def generate(self, prices: dict[str, pd.DataFrame]) -> list[Signal]:
        if len(prices) <= self.top_n:
            return []
        # 排名收益用总回报口径（adj_close），信号展示价用原始收盘价
        closes = pd.DataFrame({s: df["close"] for s, df in prices.items()}).sort_index()
        adj = pd.DataFrame({s: price_series(df) for s, df in prices.items()}).sort_index()

        # 12-1 动量：t-skip 相对 t-lookback 的收益（与 momentum / stock_momentum 口径一致）
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

            # 绝对动量开关：只买入动量为正的 picks
            if self.abs_momentum:
                picks = set(s for s in ranked.index[:self.top_n]
                            if ranked[s] > 0)
            else:
                picks = set(ranked.index[:self.top_n])

            # 先卖后买（与 momentum / dual_momentum / low_vol 一致）
            # 卖出：原来持有但本月跌出 picks 的
            for sym in list(held):
                if sym not in picks:
                    if pd.notna(closes.at[ts, sym]):
                        sym_mom = float(mom.at[ts, sym]) if pd.notna(mom.at[ts, sym]) else 0.0
                        rank = list(ranked.index).index(sym) + 1 if sym in ranked.index else len(ranked)
                        if self.abs_momentum and sym_mom <= 0:
                            reason = (f"{sym}：12-1 动量 {sym_mom:+.1%} 转负，"
                                      f"绝对动量开关触发，调出组合持现金")
                        else:
                            reason = (f"{sym}：12-1 动量 {sym_mom:+.1%}，"
                                      f"跌出前{self.top_n}名（第{rank}名），调出组合")
                        signals.append(self._sig(ts, sym, closes, mom, SELL, reason, sym_mom))
                    held.discard(sym)

            # 全部风险资产动量转负时的特殊说明
            if self.abs_momentum and not picks and held:
                # held 已被上面清空了，这里只是辅助记录
                pass

            # 买入：本月在 picks 中但之前没持有的
            for sym in ranked.index[:self.top_n]:
                if sym in picks and sym not in held:
                    if pd.notna(closes.at[ts, sym]):
                        sym_mom = float(ranked[sym])
                        rank = list(ranked.index).index(sym) + 1
                        reason = (f"{sym}：12-1 动量 {sym_mom:+.1%}，"
                                  f"跨资产动量第{rank}名，纳入组合")
                        signals.append(self._sig(ts, sym, closes, mom, BUY, reason, sym_mom))
                    held.add(sym)

        return signals

    def _sig(self, ts, symbol, closes, mom, direction, reason, mom_val) -> Signal:
        """构造 Signal，strength 用动量值映射到 0~1。

        映射逻辑：min(1.0, max(0.1, abs(mom_val) * 2))
        - 动量 ±50% 以上 → strength 1.0
        - 动量 ±5%      → strength 0.1
        与 momentum / stock_momentum 的 strength 映射一致。
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
