"""双动量 GEM：每月选动量最强的风险资产，动量转负切换到避险资产（现金感知）。

现金感知避险：风险资产全部动量转负时切避险资产（TLT），但若避险资产自己动量也为负
则持现金——不硬抱在下跌的避险资产上。实测比无条件切 TLT 全面更优（尤其 2022 加息年
TLT 也崩，切 TLT 亏 -25% vs 现金感知 -13%；全程 +367%→+415%、夏普 0.78→0.85）。
与 aggressive_mom / cross_asset_mom 的避险逻辑一致。
"""

import pandas as pd

from quant.strategies.base import BUY, SELL, Signal, Strategy, price_series


class DualMomentum(Strategy):
    name = "dual_momentum"

    def __init__(self, lookback_days: int = 252, risk_assets=("SPY", "QQQ"),
                 safe_asset: str = "TLT", **_):
        self.lookback = lookback_days
        self.risk_assets = list(risk_assets)
        self.safe_asset = safe_asset

    def generate(self, prices: dict[str, pd.DataFrame]) -> list[Signal]:
        risk = [a for a in self.risk_assets if a in prices]
        if not risk or self.safe_asset not in prices:
            return []
        # 动量用总回报口径（adj_close）——TLT 等收益大头在票息，close 会严重低估；
        # 信号展示价用原始收盘价
        closes = pd.DataFrame(
            {s: prices[s]["close"] for s in risk + [self.safe_asset]}
        ).sort_index()
        rets = pd.DataFrame(
            {s: price_series(prices[s]) for s in risk + [self.safe_asset]}
        ).sort_index().pct_change(self.lookback, fill_method=None)
        month_firsts = closes.groupby([closes.index.year, closes.index.month]).head(1).index

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
                # 风险资产全负 → 避险，但【现金感知】：避险资产自己动量也为负则持现金，
                # 不硬抱在下跌的避险资产上（如 2022 加息年 TLT 也崩，切 TLT 不如持现金）。
                safe_mom = rets.at[ts, self.safe_asset]
                target = self.safe_asset if (pd.notna(safe_mom) and safe_mom > 0) else None
            if target == held:
                continue
            if target is not None and pd.isna(closes.at[ts, target]):
                continue
            # 先卖旧仓
            if held is not None:
                sell_reason = (f"{held}：双动量月度调仓，切换至 {target}" if target
                               else f"{held}：风险与避险资产动量全负，清仓持现金")
                signals.append(self._sig(ts, held, closes, SELL, sell_reason, best_ret))
            # 再买新仓（target=None 表示持现金，不买入）
            if target is not None:
                if target == self.safe_asset:
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
            strength=round(min(1.0, max(0.1, abs(best_ret) * 2)), 2), reason=reason,
        )
