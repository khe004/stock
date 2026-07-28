"""进攻档：成长/科技 ETF 集中 12-1 月度动量 + 现金感知避险开关。

目标是【够 QQQ 的增长同时控回撤】——用集中的成长动量追增长，用现金感知避险控回撤。

宇宙分两类角色：
- 进攻（offense）：成长/科技 ETF（QQQ/XLK/SMH/IGV/XLY/XBI），动量强时集中持有
- 避险（safe_assets，默认 TLT）：成长动量转负时的退路

动量口径：12-1 月度（lookback=252, skip=21），每月首个交易日调仓。

选股逻辑（现金感知，是本策略的关键）：
1. 成长里按 12-1 动量降序，取动量【为正】的前 top_n 只（默认 top_n=1，集中）。
2. 若正动量成长不足 top_n → 用避险填补：safe_assets 里挑动量为正的按动量降序补。
3. 【现金感知】连避险动量也为负 → 空缺名额持现金等价 BIL（1-3月短债，吃无风险利率，
   替代 0% 现金；BIL 数据缺失才回落到真持现金）。绝不硬拿在下跌的避险资产——旧版硬
   切 TLT 在 2022 股债双杀时 TLT 也崩、导致 -45% 回撤；现金感知把回撤降到 -34%。
   top_n=1 全进全出，BIL 建模精确：该持现金的整段就是 100% BIL。

实测结论（2015-2026，top1 + 现金感知避险 TLT + BIL 现金等价）：
- 总收益 +737%，年化 20.2%，回撤 -34.3%，夏普 0.81，Calmar 0.59（月首日调仓口径）
- （不接 BIL、防御段持 0% 现金时略低；BIL 吃利息略增厚）
- **调仓日 timing luck 检验最稳的策略**：4-tranche 错峰口径 +717%/年化20.0%/回撤-33.1%/
  夏普0.81/Calmar0.60，与月首日口径几乎一致 → 优势不靠调仓日运气（对比 cross_asset_mom
  跨度 569bp、结论被推翻）。以下比较一律用错峰口径。

【公平基准对比 2026-07-27】按平台标准（CLAUDE.md 决策 #3），判断"排名有没有加信息"的
可信对比是【与策略共享候选名单的等权持有】，**不是 QQQ**（QQQ 是这十年的事后赢家）。
成长池等权 = 6 只进攻腿 ETF 每日等权。

  全历史        总收益   年化    回撤     夏普   Calmar
  策略（错峰）    +717%  +20.0%  -33.1%  0.81   0.60
  ★成长池等权     +606%  +18.5%  -37.4%  0.85   0.49
  全宇宙等权(含TLT) +374%  +14.4%  -32.1%  0.87   0.45
  QQQ 长持       +621%  +18.7%  -35.1%  0.89   0.53

- 全历史赢池子等权的收益/回撤/Calmar，但**夏普输**（0.81 vs 0.85）；且**四个口径里
  策略夏普最低**（0.81 < 0.85 < 0.87 < 0.89）。这是集中押注的标准签名：买到更高的
  绝对回报和更好的回撤形状，不是更高的风险调整效率（见 memory: stock-core-principle）。
- **walk-forward 分裂，超额只在后半段**：
    2015-2020  策略 +231%/夏普0.96/Calmar0.67  vs 池子等权 +232%/1.01/0.71 → 收益打平、
               夏普与 Calmar 都输；回撤也更差（-33.1% vs -31.1%）。这六年排名没加信息。
    2021-2026  策略 +138%/0.68/0.52  vs 池子等权 +114%/0.69/0.39 → 收益与 Calmar 赢，
               夏普仍打平。回撤优势也只在这半段（-32.8% vs -37.4%）。
- 与 backlog 的 4 段拆解（超额集中在 2024-26 AI/半导体行情）互相印证。

口径必须这么说才诚实：
- ✅ 跑赢 QQQ 收益、Calmar 更高 —— 真的，**但 QQQ 不是公平基准**
- ⚠️ 跑赢自己池子的等权 —— **只在 2021-2026 成立，2015-2020 平局偏输**
- ❌ 夏普跑赢任一基准 —— **从未**，全历史是四个口径里最低的
→ 所以"12-1 排名加了信息"在本策略上**未被证明**；它站得住的是【集中成长押注 +
  有效的避险腿】，以及不吃调仓日运气这一点。

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
                 top_n: int = 1, safe_assets=("TLT",), cash_asset: str = "BIL", **_):
        self.lookback = lookback_days
        self.skip = skip_days
        self.top_n = top_n
        self.safe_assets = list(safe_assets)
        self.cash_asset = cash_asset

    def generate(self, prices: dict[str, pd.DataFrame]) -> list[Signal]:
        safe = [s for s in self.safe_assets if s in prices]
        # 进攻池排除避险与现金等价（BIL），二者只作退路不参与成长排名
        exclude = set(self.safe_assets) | {self.cash_asset}
        offense = [s for s in prices if s not in exclude]
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
            # 仍有空缺 → 现金感知：持现金等价 BIL 吃短债利率（替代 0% 现金）。
            # top_n=1 时 picks 为空 → 全仓 BIL，建模精确。
            if (len(picks) < self.top_n and self.cash_asset in prices
                    and self.cash_asset not in picks
                    and pd.notna(closes.at[ts, self.cash_asset])):
                picks.append(self.cash_asset)

            top = set(picks)
            # 先卖后买——只在标的跌出组合时卖出并移除持仓（持续持有的不重复发信号）
            for sym in list(held):
                if sym not in top:
                    if pd.notna(closes.at[ts, sym]):
                        sym_mom = float(mom.at[ts, sym]) if pd.notna(mom.at[ts, sym]) else 0.0
                        if sym == self.cash_asset:
                            reason = (f"{sym}：成长/避险动量转正，现金等价调出，"
                                      f"资金切回进攻组合")
                        elif picks == [self.cash_asset]:
                            reason = (f"{sym}：成长与避险动量全负，切换至现金等价 "
                                      f"{self.cash_asset}（吃短债利率）")
                        elif not picks:
                            reason = (f"{sym}：成长与避险资产动量全负，清仓持现金"
                                      f"（{sym} 12-1动量 {sym_mom:+.1%}）")
                        else:
                            reason = (f"{sym}：12-1 动量 {sym_mom:+.1%}，"
                                      f"跌出进攻组合，调出")
                        signals.append(self._sig(ts, sym, closes, mom, SELL, reason, sym_mom))
                    held.discard(sym)

            for sym in picks:
                if sym not in held and pd.notna(closes.at[ts, sym]):
                    sym_mom = float(mom.at[ts, sym]) if pd.notna(mom.at[ts, sym]) else 0.0
                    if sym == self.cash_asset:
                        reason = (f"{sym}：成长与避险资产动量全负，切入现金等价"
                                  f"（吃短债利率，替代 0% 现金）")
                    elif sym in self.safe_assets:
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
