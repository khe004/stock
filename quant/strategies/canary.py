"""哨兵动量（Canary / Breadth Momentum）：把「何时防守」与「防守时持什么」拆开。

来源：Keller & Keuning 的 VAA(2017) → DAA(2018) → BAA(2022) → HAA(2023) 系列
（SSRN 论文 + Allocate Smartly 第三方复现）。**先验强**——公开发表、独立复现过，
不是我们自己在数据里翻出来的（这一点对怎么读 walk-forward 结果很关键，见
quant/analysis/robustness.py 文档头）。

【与平台现有防守逻辑的区别，这是本策略存在的理由】
- 现有 dual_momentum / cross_asset_mom / aggressive_mom 的防守是**自指的**：
  每个资产用它自己的绝对动量决定要不要持有。问题在 backlog 里反复出现——
  「绝对动量开关在牛市里拖收益」：要等我持有的资产自己跌到动量转负才减仓，
  往往已经跌了一段；而虚晃一枪后又要在更高价买回来。
- 哨兵架构把决策拆成两层：一个**与持仓无关**的小资产集合（canary universe）
  当预警器，只负责「现在该不该冒险」；持仓选择照旧。论文里 DAA 相对 VAA 的核心
  改进就是**平均现金仓位降到不到一半，而收益/风险持平甚至更好**——正对我们的毛病。

【两层用不同速度的动量，这是刻意的】
- **选择层（慢）**：12-1 月度动量（lookback=252, skip=21），与平台其它动量策略一致。
  我们自己测过，短窗口选择在 walk-forward 前半段崩掉（见 memory: stock-backlog）。
- **触发层（快）**：13612W —— 1/3/6/12 个月收益各自年化后取平均
  ((12·r1 + 4·r3 + 2·r6 + 1·r12) / 4)，Keller 系列的标准哨兵口径。预警器要反应快，
  和选择层要不要反应快是两个独立问题；我们过去一直用同一把尺子量这两件事。

规则（每月调仓一次）：
1. 算哨兵资产的 13612W 动量。**任一哨兵为负 → 全体风险关闭**，
   持防守腿里动量最高的那只（IEF/BIL，都为负则持 BIL 吃短债利率）。
2. 哨兵全正 → 按 12-1 动量取进攻池前 top_n。
3. abs_momentum（默认 False）：是否在进攻腿内**额外**叠加自指的绝对动量过滤。
   默认关掉——本策略的假设就是「哨兵能替代自指开关」，叠上去就测不出这个假设了。
   设 True 可退化成「哨兵 + 自指」的双保险（HAA 原版是双保险）。

【实测 2026-07-27，一律走 robustness.py 的三根轴】进攻池 = cross_asset_mom 的 9 类资产，
top_n=3，与 cross_asset_mom 同宇宙同参数以便 A/B。策略数字均为**去 timing luck 的错峰口径**。

  全历史        总收益  年化   回撤    波动   夏普  Calmar
  策略（错峰）    +208%  10.2%  -19.8%  10.7%  0.97  0.52
  ★进攻池等权     +171%   9.0%  -23.0%  11.4%  0.81  0.39
  → 超额年化 +1.2%，夏普 +0.15，回撤好 3.3pt —— **三项全赢**

  walk-forward： 2015-2020 超额 +1.1% / 夏普 +0.21 / 回撤 +3.3pt  ← 赢
                 2020-2026 超额 +0.7% / 夏普 +0.07 / 回撤 +7.5pt  ← 赢
  调仓日 timing luck：年化跨度 194bp（中等），月首日既非最优也非最差。

**这是本平台第一个在 walk-forward 两段都跑赢自己公平基准的策略。**（板块动量输板块等权、
个股动量超额全在 NVDA、cross_asset_mom 两段超额均为负、aggressive_mom 只赢后半段。）
夏普 0.97 也高于 SPY 长持(0.81) 与 QQQ 长持(0.89)。

A/B 支持了「哨兵替代自指开关」这个假设：
  哨兵单保险（本策略）        超额 +1.2%  夏普差 +0.15   ← 唯一两段都正
  哨兵 + 自指（HAA 原版双保险） 超额 -0.8%  夏普差 -0.08   ← 叠上去反而变差
  仅自指（现有 cross_asset_mom）超额 +0.4%  夏普差 -0.07   ← 两段超额均为负(-1.8%/-1.8%)
  完全不防守                  超额 +0.5%  夏普差 -0.07   ← 两段亦均为负

诚实边界（这些必须和上面的数字一起讲）：
- **机制故事没有论文那么漂亮。** 实际是 21 段风险关闭、其中 20 段只持续 1 个月——
  更像「频繁的短暂减仓」而不是「预判大崩盘」。而且它**没躲开 2020 年 3 月的新冠崩盘**
  （2020-04 才转防守，已在底部之后）。所以别把它讲成危机预警器。
- **abs_momentum=False 是我们在自己数据上选的**，偏离了 HAA 的公开规格（原版是双保险）。
  「哨兵架构」有强先验（论文+第三方复现），「去掉自指过滤」没有——这一半是 in-sample 选择。
- 我们只有 2015 年起的数据，这段里**没有 2008 级别的熊市**；哨兵架构的卖点恰恰是大级别
  下跌的保护，样本内几乎没机会体现。论文的长样本（1970/1925 起）不在我们手上。
- 超额主要来自**风险端**（波动 10.7% vs 11.4%、回撤好 3.3pt），收益端优势只有 +1.2%/年。
- 绝对收益仍**大幅跑输 SPY 长持**（+208% vs +335%）——和 cross_asset_mom 同理，
  分散到非美/债/商品在这段美股独大的历史里必然拖累绝对收益。
→ 2026-07-28 起 `notify: true`：它已进入 config 的推荐模型组合 model_portfolio
  （换掉了 cross_asset_mom），推荐一个策略却不推它的信号是自相矛盾的。但请记住本平台
  只发信号不下单，而它**至今没有任何样本外记录**——收到信号 ≠ 该照做。
"""

import pandas as pd

from quant.strategies.base import (BUY, SELL, Signal, Strategy,
                                   month_anchors, price_series)

# 13612W 的四段回看（交易日）与年化权重：1/3/6/12 个月
_SPANS = ((21, 12.0), (63, 4.0), (126, 2.0), (252, 1.0))


def momentum_13612w(adj: pd.DataFrame) -> pd.DataFrame:
    """13612W：1/3/6/12 个月收益各自年化后取平均，Keller 系列的哨兵动量口径。

    (12·r1 + 4·r3 + 2·r6 + 1·r12) / 4 —— 年化系数让最近一个月的变化主导得分，
    所以它比 12-1 反应快得多。**只用来判正负（风险开关），不用来排名。**
    """
    parts = [w * adj.pct_change(span, fill_method=None) for span, w in _SPANS]
    return sum(parts) / 4.0


class CanaryMomentum(Strategy):
    """哨兵动量：哨兵资产动量转负 → 全体防守；否则按 12-1 动量持进攻池前 top_n。"""

    name = "canary_mom"

    def __init__(self, lookback_days: int = 252, skip_days: int = 21,
                 top_n: int = 3, canary_assets=("TIP",),
                 defense_assets=("IEF", "BIL"), cash_asset: str = "BIL",
                 abs_momentum: bool = False, rebalance_offset: int = 0, **_):
        self.lookback = lookback_days
        self.skip = skip_days
        self.top_n = top_n
        self.canary_assets = list(canary_assets)
        self.defense_assets = list(defense_assets)
        self.cash_asset = cash_asset
        self.abs_momentum = abs_momentum
        self.rebalance_offset = rebalance_offset  # 调仓日错峰（timing luck 检验用）

    def generate(self, prices: dict[str, pd.DataFrame]) -> list[Signal]:
        canary = [s for s in self.canary_assets if s in prices]
        defense = [s for s in self.defense_assets if s in prices]
        if not canary or not defense:
            return []
        # 哨兵只当开关，不参与持仓排名；防守腿也不进攻击池
        exclude = set(self.canary_assets) | set(self.defense_assets) | {self.cash_asset}
        offense = [s for s in prices if s not in exclude]
        if len(offense) <= self.top_n:
            return []

        closes = pd.DataFrame({s: df["close"] for s, df in prices.items()}).sort_index()
        adj = pd.DataFrame({s: price_series(df) for s, df in prices.items()}).sort_index()
        # 选择层：12-1（慢）。触发层：13612W（快）
        mom = adj.shift(self.skip) / adj.shift(self.lookback) - 1
        fast = momentum_13612w(adj)

        signals: list[Signal] = []
        held: set[str] = set()

        for ts in month_anchors(closes.index, self.rebalance_offset):
            can_row = fast.loc[ts, canary].dropna()
            if len(can_row) < len(canary):
                continue                      # 哨兵窗口未满，本月不动
            risk_off = bool((can_row <= 0).any())

            if risk_off:
                worst = can_row.idxmin()
                def_row = mom.loc[ts, defense].dropna()
                pick = None
                if not def_row.empty and float(def_row.max()) > 0:
                    pick = def_row.idxmax()
                elif self.cash_asset in prices and pd.notna(closes.at[ts, self.cash_asset]):
                    pick = self.cash_asset
                picks = {pick} if pick else set()
                reason_buy = (f"{{sym}}：哨兵 {worst} 的 13612W 动量 "
                              f"{float(can_row[worst]):+.1%} 转负 → 风险关闭，切入防守腿")
                reason_sell = (f"{{sym}}：哨兵 {worst} 动量 {float(can_row[worst]):+.1%} 转负，"
                               f"风险关闭，调出进攻组合")
            else:
                row = mom.loc[ts].drop(labels=[c for c in exclude if c in mom.columns],
                                       errors="ignore").dropna()
                if len(row) <= self.top_n:
                    continue
                ranked = row.sort_values(ascending=False)
                cand = list(ranked.index[:self.top_n])
                if self.abs_momentum:
                    cand = [s for s in cand if ranked[s] > 0]
                picks = set(cand)
                weakest = float(can_row.min())
                reason_buy = (f"{{sym}}：哨兵全为正（最弱 {weakest:+.1%}）→ 风险开启，"
                              f"12-1 动量排名纳入组合")
                reason_sell = "{sym}：跌出进攻组合前{n}名，调出".format(
                    sym="{sym}", n=self.top_n)

            for sym in sorted(held - picks):
                if pd.notna(closes.at[ts, sym]):
                    val = float(mom.at[ts, sym]) if pd.notna(mom.at[ts, sym]) else 0.0
                    signals.append(self._sig(ts, sym, closes, SELL,
                                             reason_sell.format(sym=sym), val))
                held.discard(sym)
            for sym in sorted(picks - held):
                if pd.notna(closes.at[ts, sym]):
                    val = float(mom.at[ts, sym]) if pd.notna(mom.at[ts, sym]) else 0.0
                    detail = reason_buy.format(sym=sym)
                    if not risk_off:
                        detail += f"（12-1 动量 {val:+.1%}）"
                    signals.append(self._sig(ts, sym, closes, BUY, detail, val))
                    held.add(sym)

        return signals

    def _sig(self, ts, symbol, closes, direction, reason, mom_val) -> Signal:
        return Signal(
            date=ts.strftime("%Y-%m-%d"),
            symbol=symbol,
            strategy=self.name,
            direction=direction,
            price=round(float(closes.at[ts, symbol]), 2),
            strength=round(min(1.0, max(0.1, abs(mom_val) * 2)), 2),
            reason=reason,
        )
