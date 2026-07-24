"""进攻档：成长/科技 ETF 集中 12-1 月度动量 + 现金感知避险开关。

目标是【够 QQQ 的增长同时控回撤】——用集中的成长动量追增长，用现金感知避险控回撤。

宇宙分两类角色：
- 进攻（offense）：成长/科技 ETF（QQQ/XLK/SMH/IGV/XLY/XBI），动量强时集中持有
- 避险（safe_assets，默认 TLT）：成长动量转负时的退路

动量口径：12-1 月度（lookback=252, skip=21），每月首个交易日调仓。

选股逻辑（现金感知，是本策略的关键）：
1. 成长里按 12-1 动量降序，取动量【为正】的前 top_n 只（默认 top_n=1，集中）。
2. 若正动量成长不足 top_n → 用避险填补：safe_assets 里挑动量为正的按动量降序补。
3. 【现金感知】连避险动量也为负 → 不买、持现金（名额可不满，极端全现金）。
   绝不硬拿在下跌的避险资产——旧版硬切 TLT 在 2022 股债双杀时 TLT 也崩、导致
   -45% 回撤；现金感知（TLT 动量也为负就持现金）把回撤降到 -34%。

实测结论（2015-2026，top1 + 现金感知避险 TLT）：
- 总收益 +766%，年化 20.6%，回撤 -34%，夏普 0.82，Calmar 0.60
- 跑赢 QQQ 长持（+635%/18.9%/-35%/0.54）的收益，且回撤不比 QQQ 差、Calmar 更高
- walk-forward：2015-2020 与 2021-2026 两段都跑赢/追平 QQQ，非单段运气

诚实边界：本质是集中的成长/科技押注（吃了这段科技牛市），成长失宠的 regime 会落后；
现金感知帮大忙主要因 2022 是罕见的股债双杀（多数熊市债券会涨）；比稳健档
cross_asset_mom 波动大。两者是【进攻/稳健】两档定位。
"""

import pandas as pd

from quant.strategies.base import BUY, SELL, Signal, Strategy, price_series


class AggressiveMomentum(Strategy):
    """进攻档：成长 ETF 集中 12-1 月度动量 + 现金感知避险。

    集中持有动量最强的成长 ETF（默认 top1）；成长动量转负时切入正动量避险资产，
    连避险也转负则持现金（现金感知，不硬拿下跌的避险）。
    """

    name = "aggressive_mom"

    def __init__(self, lookback_days: int = 252, skip_days: int = 21,
                 top_n: int = 1, safe_assets=("TLT",), **_):
        self.lookback = lookback_days
        self.skip = skip_days
        self.top_n = top_n
        self.safe_assets = list(safe_assets)

    def generate(self, prices: dict[str, pd.DataFrame]) -> list[Signal]:
        safe = [s for s in self.safe_assets if s in prices]
        offense = [s for s in prices if s not in set(self.safe_assets)]
        if not offense:
            return []

        closes = pd.DataFrame({s: df["close"] for s, df in prices.items()}).sort_index()
        adj = pd.DataFrame({s: price_series(df) for s, df in prices.items()}).sort_index()
        # 12-1 动量：t-skip 相对 t-lookback（与 momentum/cross_asset_mom 口径一致）
        mom = adj.shift(self.skip) / adj.shift(self.lookback) - 1

        month_firsts = closes.groupby(
            [closes.index.year, closes.index.month]
        ).head(1).index

        signals: list[Signal] = []
        held: set[str] = set()

        for ts in month_firsts:
            row = mom.loc[ts].dropna()
            # 成长里正动量的按动量降序
            off_ranked = row[[s for s in offense if s in row.index]].sort_values(ascending=False)
            if off_ranked.empty:
                continue
            picks: list[str] = [s for s in off_ranked.index[:self.top_n] if off_ranked[s] > 0]

            # 名额有空缺 → 现金感知避险：仅在避险动量为正时补入
            if len(picks) < self.top_n and safe:
                safe_ranked = row[[s for s in safe if s in row.index]]
                safe_ranked = safe_ranked[safe_ranked > 0].sort_values(ascending=False)
                for s in safe_ranked.index:
                    if s not in picks:
                        picks.append(s)
                    if len(picks) >= self.top_n:
                        break
            # 仍不满 = 持现金（picks 可少于 top_n，甚至为空）

            top = set(picks)
            # 先卖后买——只在标的跌出组合时卖出并移除持仓（持续持有的不重复发信号）
            for sym in list(held):
                if sym not in top:
                    if pd.notna(closes.at[ts, sym]):
                        sym_mom = float(mom.at[ts, sym]) if pd.notna(mom.at[ts, sym]) else 0.0
                        if not picks:
                            reason = (f"{sym}：成长与避险资产动量全负，清仓持现金"
                                      f"（{sym} 12-1动量 {sym_mom:+.1%}）")
                        else:
                            reason = (f"{sym}：12-1 动量 {sym_mom:+.1%}，"
                                      f"跌出进攻组合，调出")
                        signals.append(self._sig(ts, sym, closes, mom, SELL, reason, sym_mom))
                    held.discard(sym)

            for sym in picks:
                if sym not in held and pd.notna(closes.at[ts, sym]):
                    sym_mom = float(row[sym])
                    if sym in self.safe_assets:
                        reason = (f"{sym}：成长资产动量全负，切入避险"
                                  f"（{sym} 12-1动量 {sym_mom:+.1%}）")
                    else:
                        rank = list(off_ranked.index).index(sym) + 1
                        reason = (f"{sym}：12-1 动量 {sym_mom:+.1%}，"
                                  f"成长动量第{rank}名，纳入进攻组合")
                    signals.append(self._sig(ts, sym, closes, mom, BUY, reason, sym_mom))
                    held.add(sym)

        return signals

    def _sig(self, ts, symbol, closes, mom, direction, reason, mom_val) -> Signal:
        """构造 Signal，strength 用动量值映射到 0~1（与其它动量策略一致）。"""
        return Signal(
            date=ts.strftime("%Y-%m-%d"),
            symbol=symbol,
            strategy=self.name,
            direction=direction,
            price=round(float(closes.at[ts, symbol]), 2),
            strength=round(min(1.0, max(0.1, abs(mom_val) * 2)), 2),
            reason=reason,
        )
