"""双动量 GEM：每月选动量最强的风险资产，动量转负切换到避险资产（现金感知）。

现金感知避险：风险资产全部动量转负时切避险资产（TLT），但若避险资产自己动量也为负
则不硬抱在下跌的避险资产上。实测比无条件切 TLT 全面更优（尤其 2022 加息年 TLT 也崩，
切 TLT 亏 -25% vs 现金感知 -14%）。

现金等价（cash_asset，默认 BIL）：避险资产也转负时，不再持 0% 现金，而是持 BIL
（1-3月短债 ETF，近零波动、零久期）吃无风险利率——2015-2021 利率≈0 时 BIL≈现金，
2022 起利率升到 ~5% 时 BIL 每年多赚几个点。top_n=1 全进全出，建模精确：该持现金的
整段就是 100% BIL。cash_asset 数据缺失时回退到 0% 现金（target=None）。

三步演进（2015-2026，SPY/QQQ + TLT，均为月首日调仓口径）：无条件切 TLT +367%/夏普0.78/回撤-31%
→ 现金感知(0%现金) +403%/0.84/-29% → 现金感知+BIL吃利息 +419%/0.85/-29%（2022 -13%）。
与 aggressive_mom 的避险逻辑一致（cross_asset_mom 因 top3 空槽建模不干净，不接 BIL）。

【调仓日 timing luck 提醒 2026-07-27】上面的 +419% 偏乐观：月首日恰好是四个调仓锚点
（每月第 1/6/11/16 个交易日）里**最好**的一个，另外三天只有 +319%~+330%，年化跨度 209bp。
去掉这份运气的公平估计 = 4-tranche 错峰组合 **+347%/年化13.9%/夏普0.82/回撤-28.7%**。
好消息是错峰后夏普（0.82）仍接近最优、回撤明显好于最差单日（-37.1%）——策略本身站得住，
只是绝对收益数字该按 +347% 而非 +419% 来预期。
"""

import pandas as pd

from quant.strategies.base import (BUY, SELL, Signal, Strategy,
                                  month_anchors, price_series)
from quant.strategies.selectors import momentum_return, momentum_strength


class DualMomentum(Strategy):
    name = "dual_momentum"

    def __init__(self, lookback_days: int = 252, risk_assets=("SPY", "QQQ"),
                 safe_asset: str = "TLT", cash_asset: str = "BIL",
                 rebalance_offset: int = 0, **_):
        self.lookback = lookback_days
        self.risk_assets = list(risk_assets)
        self.safe_asset = safe_asset
        self.cash_asset = cash_asset
        self.rebalance_offset = rebalance_offset  # 调仓日错峰（timing luck 检验用）

    def generate(self, prices: dict[str, pd.DataFrame]) -> list[Signal]:
        risk = [a for a in self.risk_assets if a in prices]
        if not risk or self.safe_asset not in prices:
            return []
        cash = self.cash_asset if self.cash_asset in prices else None
        cols = risk + [self.safe_asset] + ([cash] if cash else [])
        # 动量用总回报口径（adj_close）——TLT/BIL 收益大头在票息，close 会严重低估；
        # 信号展示价用原始收盘价
        closes = pd.DataFrame(
            {s: prices[s]["close"] for s in cols}
        ).sort_index()
        adj = pd.DataFrame({s: price_series(prices[s]) for s in cols}).sort_index()
        rets = momentum_return(adj, self.lookback)  # skip=0：近 lookback 日收益，无跳过
        # 月度调仓日：每月第 (rebalance_offset+1) 个交易日（默认月首日）
        month_firsts = month_anchors(closes.index, self.rebalance_offset)

        signals: list[Signal] = []
        held: str | None = None
        for ts in month_firsts:
            row = rets.loc[ts, risk].dropna()
            if row.empty:
                continue
            best = row.idxmax()
            best_ret = float(row[best])
            if best_ret > 0:
                target = best
            else:
                # 风险资产全负 → 避险，但【现金感知】：避险资产自己动量也为负则不硬抱，
                # 退到现金等价 BIL 吃短债利率（BIL 也不可用才回落到 0% 现金）。
                safe_mom = rets.at[ts, self.safe_asset]
                if pd.notna(safe_mom) and safe_mom > 0:
                    target = self.safe_asset
                elif cash is not None and pd.notna(closes.at[ts, cash]):
                    target = cash
                else:
                    target = None
            if target == held:
                continue
            if target is not None and pd.isna(closes.at[ts, target]):
                continue
            # 先卖旧仓
            if held is not None:
                if target is None:
                    sell_reason = f"{held}：风险与避险资产动量全负，清仓持现金"
                elif target == cash:
                    sell_reason = f"{held}：风险与避险动量全负，切换至现金等价 {cash}"
                else:
                    sell_reason = f"{held}：双动量月度调仓，切换至 {target}"
                signals.append(self._sig(ts, held, closes, SELL, sell_reason, best_ret))
            # 再买新仓（target=None 表示持现金，不买入）
            if target is not None:
                if target == cash:
                    buy_reason = (f"{cash}：风险资产动量全部转负（最强 {best} {best_ret:+.1%}）"
                                  f"且避险 {self.safe_asset} 也转负，切入现金等价（吃短债利率）")
                elif target == self.safe_asset:
                    safe_mom = float(rets.at[ts, self.safe_asset])
                    buy_reason = (f"{self.safe_asset}：风险资产动量全部转负"
                                  f"（最强 {best} {best_ret:+.1%}），切入避险"
                                  f"（{self.safe_asset} 动量 {safe_mom:+.1%} 为正）")
                else:
                    buy_reason = (f"{target}：近{self.lookback}日动量 {best_ret:+.1%}，"
                                  f"为风险资产最强且为正，持有")
                signals.append(self._sig(ts, target, closes, BUY, buy_reason, best_ret))
            held = target
        return signals

    def _sig(self, ts, symbol, closes, direction, reason, best_ret) -> Signal:
        return Signal(
            date=ts.strftime("%Y-%m-%d"), symbol=symbol, strategy=self.name,
            direction=direction, price=round(float(closes.at[ts, symbol]), 2),
            strength=round(momentum_strength(best_ret), 2), reason=reason,
        )
