"""个股横截面动量：动态流动性池 + 12-1 动量选股 + 大盘 regime 过滤。

设计要点（详见策略说明页）：
- 选股池逐月用"当时"的成交额重建（point-in-time，避免用今天的权重名单回测过去）；
  候选超集来自 universe 文件，其幸存者偏差通过与"池子等权"基准共享而近似抵消
- 动量用 12-1 口径：近 lookback 收益但跳过最近 skip 天，避开短期反转
- 大盘跌破 regime_ma 日均线时整体切换避险资产，防动量崩溃
"""

import pandas as pd
import yaml

from quant.config import ROOT
from quant.strategies.base import BUY, SELL, Signal, Strategy, price_series


class StockMomentum(Strategy):
    name = "stock_momentum"

    def __init__(self, universe_file: str = "universe_sp500.yaml",
                 pool_size: int = 100, liquidity_window: int = 20,
                 lookback_days: int = 252, skip_days: int = 21,
                 top_n: int = 6, max_per_sector: int = 2,
                 regime_symbol: str = "SPY", regime_ma: int = 200,
                 safe_asset: str = "TLT",
                 universe: list[str] | None = None,
                 sectors: dict[str, str] | None = None,
                 exclude: list[str] | None = None, **_):
        if universe is None:
            with open(ROOT / universe_file, encoding="utf-8") as f:
                grouped = yaml.safe_load(f)
            sectors = {sym: sector for sector, syms in grouped.items() for sym in syms}
            universe = list(sectors)
        if exclude:   # 敏感性检验：剔除指定标的（如大赢家），看超额是否依赖它们
            universe = [s for s in universe if s not in set(exclude)]
        self.universe = universe
        self.sectors = sectors or {}
        self.pool_size = pool_size
        self.liquidity_window = liquidity_window
        self.lookback = lookback_days
        self.skip = skip_days
        self.top_n = top_n
        self.max_per_sector = max_per_sector
        self.regime_symbol = regime_symbol
        self.regime_ma = regime_ma
        self.safe_asset = safe_asset

    def monthly_pools(self, prices: dict[str, pd.DataFrame]) -> dict[pd.Timestamp, list[str]]:
        """每月首个交易日，按近 liquidity_window 日平均成交额取前 pool_size 名。"""
        uni = [s for s in self.universe if s in prices]
        if not uni:
            return {}
        dollar = pd.DataFrame({
            s: prices[s]["close"] * prices[s]["volume"] for s in uni
        }).sort_index().rolling(self.liquidity_window).mean()
        month_firsts = dollar.groupby([dollar.index.year, dollar.index.month]).head(1).index
        pools = {}
        for ts in month_firsts:
            dv = dollar.loc[ts].dropna()
            if not dv.empty:
                pools[ts] = list(dv.nlargest(self.pool_size).index)
        return pools

    def generate(self, prices: dict[str, pd.DataFrame]) -> list[Signal]:
        uni = [s for s in self.universe if s in prices]
        if len(uni) < self.top_n or self.regime_symbol not in prices:
            return []
        adj = pd.DataFrame({s: price_series(prices[s]) for s in uni}).sort_index()
        # 12-1 动量：t-skip 相对 t-lookback 的收益
        mom = adj.shift(self.skip) / adj.shift(self.lookback) - 1
        spy = price_series(prices[self.regime_symbol])
        regime = (spy > spy.rolling(self.regime_ma).mean()).reindex(adj.index).ffill()
        pools = self.monthly_pools(prices)

        signals: list[Signal] = []
        held: list[str] = []
        for ts, pool in pools.items():
            r = regime.get(ts)
            if pd.isna(r):        # 大盘均线还没长够，跳过
                continue
            if not r:
                target = [self.safe_asset] if self._tradable(prices, self.safe_asset, ts) else []
                picks = {}
            else:
                m = mom.loc[ts, [s for s in pool if s in mom.columns]].dropna()
                if m.empty:
                    continue
                picks = self._select(m.sort_values(ascending=False))
                target = [s for s in picks if self._tradable(prices, s, ts)]
            self._emit(signals, prices, ts, held, target, picks, bool(r))
            held = target
        return signals

    def _select(self, ranked: pd.Series) -> dict[str, tuple[int, float]]:
        """按排名依次入选，单行业最多 max_per_sector 只；行业未知不设限。"""
        picks: dict[str, tuple[int, float]] = {}
        counts: dict[str, int] = {}
        for rank, (sym, ret) in enumerate(ranked.items(), start=1):
            sector = self.sectors.get(sym)
            if sector is not None:
                if counts.get(sector, 0) >= self.max_per_sector:
                    continue
                counts[sector] = counts.get(sector, 0) + 1
            picks[sym] = (rank, float(ret))
            if len(picks) >= self.top_n:
                break
        return picks

    @staticmethod
    def _tradable(prices, sym: str, ts) -> bool:
        return sym in prices and ts in prices[sym].index and pd.notna(prices[sym]["close"].get(ts))

    def _emit(self, signals, prices, ts, held, target, picks, risk_on):
        for sym in held:
            if sym in target:
                continue
            if not self._tradable(prices, sym, ts):
                continue
            if not risk_on:
                reason = f"{sym}：大盘跌破 {self.regime_ma} 日均线，风险关闭，清仓换入 {self.safe_asset}"
            elif sym == self.safe_asset:
                reason = f"{self.safe_asset}：大盘重回 {self.regime_ma} 日均线上方，切回个股动量组合"
            else:
                reason = f"{sym}：月度调仓，跌出动量前 {self.top_n} 或行业限额让位"
            signals.append(self._sig(prices, ts, sym, SELL, reason, 0.5))
        for sym in target:
            if sym in held:
                continue
            if sym == self.safe_asset:
                reason = f"{self.safe_asset}：大盘跌破 {self.regime_ma} 日均线，风险关闭切换避险"
                strength = 0.8
            else:
                rank, ret = picks[sym]
                reason = f"{sym}：12-1 动量 {ret:+.1%}，流动性池内第 {rank} 名，入选前 {self.top_n}"
                strength = min(1.0, max(0.1, abs(ret) * 2))
            signals.append(self._sig(prices, ts, sym, BUY, reason, strength))

    def _sig(self, prices, ts, sym, direction, reason, strength) -> Signal:
        return Signal(
            date=ts.strftime("%Y-%m-%d"), symbol=sym, strategy=self.name,
            direction=direction, price=round(float(prices[sym]["close"].get(ts)), 2),
            strength=round(strength, 2), reason=reason,
        )
