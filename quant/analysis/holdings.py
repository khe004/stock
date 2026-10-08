"""模型持仓：把策略的全历史信号回放成"现在该持有什么"（纯计算）。

平台只出信号，不知道用户账户里有什么；这里推导的是**按信号机械执行会持有的标的**，
用来对账——漏了一次推送、或隔了几周没看，不用翻信号历史就能知道策略当前的目标持仓。
它不是用户的真实账户，也不随月中排名变化而变：只在策略发出信号的调仓日变。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from quant.strategies.base import BUY

# 持仓可由买卖信号回放得到的策略（进出成对）。smart_dca 只发定投买入、没有对应的卖出，
# 回放不出"持仓"，单独按定投状态描述。
REPLAYABLE = {"momentum", "dual_momentum", "low_vol", "cross_asset_mom",
              "aggressive_mom", "canary_mom", "stock_momentum"}


@dataclass
class ModelHolding:
    strategy: str
    symbols: list[str] = field(default_factory=list)   # 当前目标持仓（按买入先后）
    since: dict[str, str] = field(default_factory=dict)  # 标的 → 买入信号日
    last_change: str | None = None                     # 最近一次持仓变动的信号日
    replayable: bool = True
    note: str = ""

    def describe(self) -> str:
        if not self.replayable:
            return self.note
        if not self.symbols:
            return "空仓（现金）" + (f"，自 {self.last_change}" if self.last_change else "")
        return " / ".join(self.symbols) + (f"（上次调仓 {self.last_change}）" if self.last_change else "")


def replay_holdings(strategy: str, signals: list, as_of: str | None = None) -> ModelHolding:
    """按日期顺序回放该策略的信号：买入加入持仓、卖出移出；同日先卖后买。"""
    own = [s for s in signals if s.strategy == strategy and (as_of is None or s.date <= as_of)]
    if strategy not in REPLAYABLE:
        last = max(own, key=lambda s: s.date, default=None)
        note = (f"定投型，无固定持仓；最近信号 {last.date}：{last.reason}" if last
                else "定投型，无固定持仓；暂无信号")
        return ModelHolding(strategy, replayable=False, note=note,
                            last_change=last.date if last else None)
    since: dict[str, str] = {}
    last_change = None
    for s in sorted(own, key=lambda s: (s.date, s.direction == BUY)):
        if s.direction == BUY:
            since.setdefault(s.symbol, s.date)
        else:
            since.pop(s.symbol, None)
        last_change = s.date
    return ModelHolding(strategy, list(since), since, last_change)


def look_through_weights(holdings: list[ModelHolding], hold_assets: list[str]) -> dict[str, float]:
    """模型组合的穿透权重：各成分等权，成分内各持仓等权；空仓与定投型成分记为「现金/定投」。"""
    legs = len(holdings) + len(hold_assets)
    if legs == 0:
        return {}
    weights: dict[str, float] = {}
    for h in holdings:
        if h.replayable and h.symbols:
            for sym in h.symbols:
                weights[sym] = weights.get(sym, 0.0) + 1 / legs / len(h.symbols)
        else:
            key = "现金" if h.replayable else f"{h.strategy}（定投）"
            weights[key] = weights.get(key, 0.0) + 1 / legs
    for sym in hold_assets:
        weights[sym] = weights.get(sym, 0.0) + 1 / legs
    return dict(sorted(weights.items(), key=lambda kv: -kv[1]))


def holdings_report(signals_by_strategy: dict[str, list], strategies: list[str],
                    hold_assets: list[str], as_of: str | None = None) -> str:
    """日报里的「模型持仓」段：逐成分一行 + 穿透权重一行。"""
    holdings = [replay_holdings(name, signals_by_strategy.get(name, []), as_of)
                for name in strategies if name in signals_by_strategy]
    if not holdings and not hold_assets:
        return ""
    lines = ["———— 模型组合目标持仓（按信号机械执行，非真实账户）————"]
    lines += [f"{h.strategy}: {h.describe()}" for h in holdings]
    lines += [f"{sym}: 买入持有" for sym in hold_assets]
    weights = look_through_weights(holdings, hold_assets)
    if weights:
        lines.append("穿透权重: " + " | ".join(f"{k} {v:.0%}" for k, v in weights.items()))
    return "\n".join(lines)
