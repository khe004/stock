"""稳健性检验：调仓日 timing luck + walk-forward 分段 + 公平基准（池子等权）。

【这套检验回答什么、不回答什么】（2026-07-27 定调，改动前务必读）

我们只有 11 年月频数据：名义 ~130 次调仓，但收益按 regime 高度自相关，实际独立观测
大概只有 2-3 个。**这个样本量不可能回答"这个策略有没有 alpha"**——任何检验都没有统计
功效。所以本模块的定位不是判生死，而是**校准期望值**，回答"我报出来的数字有没有虚高"：

- **调仓日 timing luck**（Newfound / Hoffstein-Faber-Braun）：这个数字是不是恰好挑中了
  最好的调仓日？全平台月频策略都默认锚在"每月首个交易日"，这是个从未被检验的隐含选择。
  把锚点错开到第 6/11/16 个交易日重跑，散布就是 timing luck 的幅度。实测本平台年化跨度
  133~569bp，最低的都超过文献典型值 100bp。
  "去掉运气"的公平估计 = 4 个 tranche 各 1/4 资金独立跑再求和（tranching）。
- **walk-forward 分段**：这个数字是不是靠单一 regime 撑起来的？
- **公平基准（池子等权）**：这个数字是不是宇宙本身好、而非选择能力？
  基准 = 与策略共享候选名单的等权持有。**别拿 QQQ/XLK 这类事后赢家当基准。**

**输出是期望区间，不是通过/不通过。** 要求一个策略在每一段都跑赢基准是不可能的门槛
（价值因子连输 13 年、动量 2009 年崩过），按那个标准连有几十年文献支撑的因子都会被
误杀。walk-forward 本身也分不出"真效应坐冷板凳"和"假象恰好在某段好看"——分开它们靠的
是**先验**（12-1 动量有跨市场跨资产的独立验证 → 先验强；"top2 比 top3 好"是我们自己在
数据里翻出来的 → 先验为零）。所以本模块只给区间，判断留给读的人。

**读法**：点估计取错峰口径与各段的平均，不是取最大；最坏那一段是你真的要扛的东西。
既然策略有时效性、事前又分不清哪个在当季，实操解药是同时持有几个低相关策略
（specification risk），不是找那个永远有效的策略。
"""

import pandas as pd

from quant import strategies
from quant.backtest.engine import equity_metrics, run_portfolio_backtest
from quant.strategies.base import price_series

# 4 个周度错峰锚点：每月第 1/6/11/16 个交易日
DEFAULT_OFFSETS = (0, 5, 10, 15)


def defensive_symbols(params: dict) -> set[str]:
    """策略的避险腿/现金等价/哨兵——都不属于"候选池"，公平基准应把它们排除。

    哨兵（canary_assets）尤其要排除：它只当风险开关、从不持有，
    把它算进等权基准等于让基准持有一个策略永远不会买的东西。
    """
    out: set[str] = set()
    if params.get("safe_asset"):
        out.add(params["safe_asset"])
    out.update(params.get("safe_assets") or [])
    out.update(params.get("canary_assets") or [])
    out.update(params.get("defense_assets") or [])
    if params.get("cash_asset"):
        out.add(params["cash_asset"])
    return out


def equal_weight_equity(prices: dict[str, pd.DataFrame], initial_cash: float,
                        subset: set[str] | None = None) -> pd.Series | None:
    """等权基准：宇宙内标的每日等权持有（日度再平衡，不计成本）。

    去掉"事后挑中赢家"的偏差——比起拿 XLK/QQQ 长持（十几年里恰好封神的那只）当基准，
    与策略共享候选名单的等权才是判断"选择有没有加信息"的公平对照。
    subset 用于只等权候选池（排除避险腿），None = 全宇宙。
    """
    cols = {s: price_series(df) for s, df in prices.items()
            if subset is None or s in subset}
    if not cols:
        return None
    adj = pd.DataFrame(cols).sort_index()
    if adj.empty:
        return None
    daily = adj.pct_change(fill_method=None).mean(axis=1)
    return (initial_cash * (1 + daily.fillna(0.0)).cumprod()).rename("equal_weight")


def _signals(strategy_name: str, params: dict, prices_full: dict[str, pd.DataFrame],
             offset: int) -> list:
    """在【全量历史】上出信号（保证动量回看窗口完整），窗口截取交给调用方。"""
    return strategies.build(strategy_name, {**params, "rebalance_offset": offset}).generate(
        prices_full)


def strategy_equity(strategy_name: str, params: dict,
                    prices_full: dict[str, pd.DataFrame],
                    prices_window: dict[str, pd.DataFrame],
                    offset: int, initial: float, cost_bps: float,
                    start: str | None = None, end: str | None = None) -> pd.Series:
    sigs = _signals(strategy_name, params, prices_full, offset)
    if start is not None:
        sigs = [s for s in sigs if start <= s.date <= end]
    return run_portfolio_backtest(prices_window, sigs, strategy_name,
                                  initial, cost_bps).equity


def tranched_equity(strategy_name: str, params: dict,
                    prices_full: dict[str, pd.DataFrame],
                    prices_window: dict[str, pd.DataFrame],
                    initial: float, cost_bps: float,
                    offsets=DEFAULT_OFFSETS,
                    start: str | None = None, end: str | None = None) -> pd.Series:
    """错峰组合（tranching）：资金等分成 len(offsets) 份，各自独立按不同调仓日跑，再求和。

    这是"去掉调仓日运气"的公平估计——文献说它不牺牲收益、纯降方差，实测也确实普遍拿到
    "接近平均的收益 + 明显好于最差单日的回撤"。
    """
    share = initial / len(offsets)
    eqs = [strategy_equity(strategy_name, params, prices_full, prices_window,
                           o, share, cost_bps, start, end) for o in offsets]
    combined = pd.concat(eqs, axis=1).ffill()
    return combined.sum(axis=1).dropna().rename("tranched")


def timing_luck_sweep(strategy_name: str, params: dict,
                      prices: dict[str, pd.DataFrame],
                      initial: float, cost_bps: float,
                      offsets=DEFAULT_OFFSETS) -> dict:
    """调仓日散布：逐个 offset 跑一遍 + 错峰组合，返回指标与年化跨度。"""
    rows = []
    for o in offsets:
        eq = strategy_equity(strategy_name, params, prices, prices, o, initial, cost_bps)
        m = equity_metrics(eq, initial)
        m["offset"] = o
        m["label"] = f"第 {o + 1} 个交易日调仓"
        rows.append(m)
    tr = equity_metrics(
        tranched_equity(strategy_name, params, prices, prices, initial, cost_bps, offsets),
        initial)
    cagrs = [r["cagr"] for r in rows]
    return {
        "per_offset": rows,
        "tranched": tr,
        "spread_cagr_bps": (max(cagrs) - min(cagrs)) * 10_000,
        "current_is_best": rows[0]["cagr"] == max(cagrs),
        "current_is_worst": rows[0]["cagr"] == min(cagrs),
    }


def split_windows(index: pd.DatetimeIndex, n: int = 2) -> list[tuple[str, str, str]]:
    """把历史等分成 n 段（默认前后两半），返回 [(标签, 起, 止)]。"""
    if index.empty or n < 1:
        return []
    bounds = [index[int(round(i * (len(index) - 1) / n))] for i in range(n + 1)]
    out = []
    for i in range(n):
        s, e = bounds[i], bounds[i + 1]
        out.append((f"{s.year}-{e.year}", s.strftime("%Y-%m-%d"), e.strftime("%Y-%m-%d")))
    return out


def robustness_report(strategy_name: str, params: dict,
                      prices: dict[str, pd.DataFrame],
                      initial: float, cost_bps: float,
                      offsets=DEFAULT_OFFSETS, n_windows: int = 2) -> dict:
    """完整稳健性报告：调仓日散布 + 分段的（策略错峰口径 vs 池子等权 vs 全宇宙等权）。

    返回 dict：
      timing_luck: timing_luck_sweep 的结果
      windows: [{label, strategy, pool_ew, universe_ew, excess_cagr}]
      excess_range: (最小超额年化, 最大超额年化)  ← 期望区间，报告的主角
    """
    sweep = timing_luck_sweep(strategy_name, params, prices, initial, cost_bps, offsets)

    defensive = defensive_symbols(params)
    pool = {s for s in prices if s not in defensive} or set(prices)

    idx = pd.DatetimeIndex(sorted({t for df in prices.values() for t in df.index}))
    windows = [("全历史", None, None)] + list(split_windows(idx, n_windows))

    rows = []
    for label, start, end in windows:
        pw = prices if start is None else {
            s: df.loc[start:end] for s, df in prices.items()}
        pw = {s: df for s, df in pw.items() if not df.empty}
        if not pw:
            continue
        strat = equity_metrics(
            tranched_equity(strategy_name, params, prices, pw, initial, cost_bps,
                            offsets, start, end), initial)
        pool_eq = equal_weight_equity(pw, initial, pool)
        uni_eq = equal_weight_equity(pw, initial)
        pool_m = equity_metrics(pool_eq, initial) if pool_eq is not None else None
        uni_m = equity_metrics(uni_eq, initial) if uni_eq is not None else None
        rows.append({
            "label": label,
            "strategy": strat,
            "pool_ew": pool_m,
            "universe_ew": uni_m,
            "excess_cagr": (strat["cagr"] - pool_m["cagr"]) if pool_m else None,
        })

    seg_excess = [r["excess_cagr"] for r in rows[1:] if r["excess_cagr"] is not None]
    return {
        "timing_luck": sweep,
        "windows": rows,
        "pool_symbols": sorted(pool),
        "defensive_symbols": sorted(defensive),
        "excess_range": (min(seg_excess), max(seg_excess)) if seg_excess else None,
    }
