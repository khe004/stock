"""行业 12-1 月度动量轮动：每月首个交易日按 12-1 动量横截面排名，持有前 top_n 的板块。

12-1 动量口径：近 lookback_days 收益但跳过最近 skip_days，避开短期反转/买在山顶。
旧版（63日回看 + 每日进出）实测跑输板块等权基准（whipsaw + 短期反转所致），
改为 252/skip21 月度调仓后拿得更稳、交易次数大幅下降、总收益与 Calmar 显著改善。

【top_n 与简单避险，2026-07-30 用户追问，走完整三轴复测】用户两个问题：
①top1/top2 有没有测过（2026-07-26 测过，但当时没有 timing luck 工具）；
②能不能加个简单避险——负动量的板块不买，空出的名额要么空着（等权引擎会集中到
剩下的正动量赢家上）要么切现金等价 BIL。新增 `abs_momentum`/`cash_asset` 两个参数
（默认 False/None，不改变现状行为，已用真实信号逐字节回归验证）实现第②问，
配合已有的 top_n 复测第①问。全部走 robustness.py 三轴（错峰口径 + walk-forward
两段 + timing luck + ★板块等权公平基准）。

  全历史（错峰口径）    总收益   夏普  Calmar  超额年化  夏普差  timing luck跨度
  top1 无避险           +452%  0.79   0.51    +4.4%    +0.05      750bp（！）
  top1 负动量不买        +452%  0.79   0.51    +4.4%    +0.05      750bp（与无避险完全相同）
  top2 无避险           +338%  0.76   0.39    +2.1%    +0.02      175bp
  top2 负动量不买        +332%  0.76   0.38    +2.0%    +0.02      261bp（变宽了）
  top3 无避险（现状）    +266%  0.74   0.36    +0.4%    -0.01      142bp（全平台最窄）
  top3 负动量不买        +257%  0.72   0.35    +0.1%    -0.02      274bp（变宽了）
  （"+BIL"变体几乎和"负动量不买"数字一样，见下）

**① top_n**：复现且加固了 2026-07-26 的结论——**继续持有 top_n=3**。top1 全历史数字
最好看，但 timing luck 跨度 750bp 是全平台最宽的（3~5 倍于 top2/top3），点估计本身
就不可信；且 walk-forward 两段一好一坏（前段夏普 0.63 勉强赢基准，后段收益赢但夏普
反而**输**基准 -0.12），跟 aggressive_mom 的"赢 Calmar 输夏普"是同一个集中押注签名。
top3（现状）的 timing luck 最窄、最不依赖调仓日运气，代价是相对板块等权基本打平
（超额+0.4%/夏普-0.01——"排名没加信息"，这个结论本来就没变过）。

**② 简单避险（负动量不买）：测了但没用，多数情况下还轻微变差，不采纳。**
- **top1 几乎是个 no-op**：全历史 139 个月首日里，排名第一的板块动量为负的次数只有
  **1 次**（2020-05，XLK -0.5%，恰在新冠崩盘反弹初期）。11 个板块里最强的那个几乎
  从不是负的——除非全市场同时下跌到"最强板块也扛不住"的程度，这在样本内约等于
  没发生过。**"负动量不买"这个过滤器只对多选（top2/top3）有意义**，因为多选里较弱
  的那个位置更容易先转负。
- **top2/top3 加了这个过滤器后全历史与前半段(2015-2020)都轻微变差**（夏普/Calmar
  双双略降），只有后半段(2020-2026)基本打平或略好——不是稳定的改善，是噪音级别
  的扰动，不构成"加了这条能变好"的证据。
- **"空着让等权引擎集中" vs "空出的名额切 BIL"，这次几乎没区别**（top2/top3 两个
  变体数字几乎一样）——这跟 cross_asset_mom 那次（BIL 反而把 +237% 拉到 +200%，
  差异很大）不同，因为板块动量的 top2/top3 里很少同时两个位置一起转负，"计划外
  集中"的规模本来就小，切不切现金对结果影响有限。
- timing luck 还都变宽了（top2 175→261bp，top3 142→274bp）——多一层过滤器等于
  多一层依赖调仓日撞上哪个板块转负，稳健性反而更差。
→ 因此 `abs_momentum`/`cash_asset` 默认关闭，不采纳。跟 canary_mom 的
  `risk_off_threshold` 是同一类结论：机制听起来合理，但这个宇宙规模上没有信息可提取。
"""

import pandas as pd

from quant.strategies.base import (BUY, SELL, Signal, Strategy,
                                  month_anchors, price_series)
from quant.strategies.selectors import momentum_return, momentum_strength


class Momentum(Strategy):
    name = "momentum"

    def __init__(self, lookback_days: int = 252, skip_days: int = 21,
                 top_n: int = 3, rebalance_offset: int = 0,
                 abs_momentum: bool = False, cash_asset: str | None = None, **_):
        self.lookback = lookback_days
        self.skip = skip_days
        self.top_n = top_n
        self.rebalance_offset = rebalance_offset  # 调仓日错峰（timing luck 检验用）
        # 简单避险实验（2026-07-30，默认关闭，实测未采纳——见文件文档头）：
        # abs_momentum=True 时，负动量的板块不买，空出的名额要么空着（cash_asset=None，
        # 等权引擎会把资金集中到剩下的正动量赢家上）要么切现金等价（cash_asset="BIL"）。
        self.abs_momentum = abs_momentum
        self.cash_asset = cash_asset

    def generate(self, prices: dict[str, pd.DataFrame]) -> list[Signal]:
        if len(prices) <= self.top_n:
            return []
        # 排名收益用总回报口径（adj_close），信号展示价用原始收盘价
        closes = pd.DataFrame({s: df["close"] for s, df in prices.items()}).sort_index()
        adj = pd.DataFrame({s: price_series(df) for s, df in prices.items()}).sort_index()

        # 12-1 动量：t-skip 相对 t-lookback 的收益
        mom = momentum_return(adj, self.lookback, self.skip)

        # 月度调仓日：每月第 (rebalance_offset+1) 个交易日（默认月首日）
        month_firsts = month_anchors(closes.index, self.rebalance_offset)

        signals: list[Signal] = []
        held: set[str] = set()  # 当前持有的标的

        for ts in month_firsts:
            # 取当日各标的的 12-1 动量值，跳过 NaN（窗口不足）
            row = mom.loc[ts].dropna()
            if len(row) <= self.top_n:
                continue

            # 按 12-1 动量降序排名，取前 top_n
            ranked = row.sort_values(ascending=False)
            cand = list(ranked.index[:self.top_n])
            if self.abs_momentum:
                cand = [s for s in cand if ranked[s] > 0]
            top_syms = set(cand)
            cash = self.cash_asset if self.cash_asset in prices else None
            if (self.abs_momentum and cash and len(top_syms) < self.top_n
                    and pd.notna(closes.at[ts, cash])):
                top_syms.add(cash)

            # 先卖后买（与 dual_momentum / stock_momentum / low_vol 一致）
            # 卖出：原来持有但本月跌出前 top_n（或动量转负、或现金调出）的
            for sym in list(held):
                if sym not in top_syms:
                    if pd.notna(closes.at[ts, sym]):
                        sym_mom = float(mom.at[ts, sym]) if pd.notna(mom.at[ts, sym]) else 0.0
                        if sym == cash:
                            reason = f"{sym}：有正动量板块补位，现金等价调出，资金切回轮动组合"
                        elif self.abs_momentum and sym_mom <= 0:
                            reason = f"{sym}：12-1 动量 {sym_mom:+.1%} 转负，绝对动量开关触发，调出组合"
                        else:
                            rank = list(ranked.index).index(sym) + 1 if sym in ranked.index else len(ranked)
                            reason = (f"{sym}：12-1 动量 {sym_mom:+.1%}，"
                                      f"跌出行业动量前{self.top_n}名（第{rank}名），调出组合")
                        signals.append(self._sig(ts, sym, closes, mom, SELL, reason, sym_mom))
                    held.discard(sym)

            # 买入：本月在前 top_n（含现金补位）但之前没持有的
            for sym in cand + ([cash] if cash in top_syms and cash not in cand else []):
                if sym not in held:
                    if pd.notna(closes.at[ts, sym]):
                        if sym == cash:
                            reason = "{}：正动量板块不足 {} 个，空缺名额切现金等价（吃短债利率）".format(
                                sym, self.top_n)
                            sym_mom = 0.0
                        else:
                            sym_mom = float(ranked[sym])
                            rank = list(ranked.index).index(sym) + 1
                            reason = (f"{sym}：12-1 动量 {sym_mom:+.1%}，"
                                      f"行业动量第{rank}名，纳入轮动组合")
                        signals.append(self._sig(ts, sym, closes, mom, BUY, reason, sym_mom))
                    held.add(sym)

        return signals

    def _sig(self, ts, symbol, closes, mom, direction, reason, mom_val) -> Signal:
        """构造 Signal，strength 用动量值映射到 0~1（见 selectors.momentum_strength）。"""
        return Signal(
            date=ts.strftime("%Y-%m-%d"),
            symbol=symbol,
            strategy=self.name,
            direction=direction,
            price=round(float(closes.at[ts, symbol]), 2),
            strength=round(momentum_strength(mom_val), 2),
            reason=reason,
        )
