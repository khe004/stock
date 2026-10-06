"""个股/指数卖 put（CSP）前向检验：用自己攒的期权链快照选合约、到期后按收盘价结算。

为什么需要它：yfinance 不给历史期权链，个股 CSP 没法像 ETF 策略那样回测。
``fetcher.update_option_snapshots`` 每天存一份 point-in-time 的 put 报价，这里负责：
1. ``annotate_quotes``：给每条报价算透明指标（DTE、下跌缓冲、BS delta、年化权利金、财报是否落在期内）；
2. ``pick_csp``：按**固定机械规则**每个标的每天选一张（默认 ~30 天、|Δ|≈0.25、以买价成交）；
3. ``settle_csp``：到期后用库内收盘价结算，算以担保本金为分母的单笔收益；
4. ``forward_summary``：汇总，并按指数 ETF / 个股分组——回答"个股高 IV 是否真给了更多 VRP"。

**故意不做综合打分排名**（PutFinder 那类"质量×适配×财报门"）：在积累到足够样本之前，
排名有没有加信息无从验证（决策 #3/#9）。等数据够了，可信的检验是"排名前 N vs 同日全部候选等权"。

诚实边界：
- 以**买价**成交、到期按收盘价现金结算——没有提前指派、没有滑点外的冲击成本、没有税。
- 每天都选一张会产生**重叠持仓**，日度样本高度自相关；``monthly_only=True`` 只取每月首个快照日，
  得到不重叠的月度序列，统计时以它为准。
- 卖 put 的胜率天然很高，**单看胜率没有意义**；要看最差单笔和"一次亏损吃掉几笔权利金"。
  前向样本没经历过崩盘之前，任何数字都系统性偏乐观。
"""

import math

import numpy as np
import pandas as pd

from quant.analysis.putwrite import bs_put_delta

DAYS_PER_YEAR = 365


def annotate_quotes(quotes: pd.DataFrame, rate: float = 0.0) -> pd.DataFrame:
    """给 ``store.load_option_quotes`` 的结果补透明指标列。

    - dte：快照日到到期日的日历天
    - buffer：下跌缓冲 = 1 − K/S（正数 = 虚值多少）
    - delta：BS delta（用该合约自身 IV，期限 dte/365）
    - prem_yield / ann_yield：买价 / K，及其按 365/dte 年化
    - earnings_in_window：下次财报是否落在到期日之前（含当天）
    """
    q = quotes.copy()
    if q.empty:
        return q
    snap = pd.to_datetime(q["snapshot_date"])
    exp = pd.to_datetime(q["expiration"])
    q["dte"] = (exp - snap).dt.days
    q["buffer"] = 1 - q["strike"] / q["spot"]
    q["delta"] = [
        bs_put_delta(s, k, d / DAYS_PER_YEAR, v, rate)
        if pd.notna(v) and v > 0 and d > 0 else np.nan
        for s, k, d, v in zip(q["spot"], q["strike"], q["dte"], q["iv"])
    ]
    q["prem_yield"] = q["bid"] / q["strike"]
    q["ann_yield"] = q["prem_yield"] * DAYS_PER_YEAR / q["dte"].where(q["dte"] > 0)
    earn = pd.to_datetime(q["next_earnings"], errors="coerce")
    q["earnings_in_window"] = earn.notna() & (earn <= exp)
    return q


def pick_csp(annotated: pd.DataFrame, target_dte: int = 30, dte_range: tuple[int, int] = (20, 45),
             target_delta: float = 0.25, min_bid: float = 0.05) -> pd.DataFrame:
    """每个 (快照日, 标的) 按机械规则选一张 put：

    1. 到期日：dte 落在 dte_range 内、离 target_dte 最近的那个；
    2. 行权价：该到期日里 bid ≥ min_bid 且有 IV 的合约中，|delta| 离 target_delta 最近的那个。
    """
    q = annotated
    if q.empty:
        return q
    q = q[(q["dte"] >= dte_range[0]) & (q["dte"] <= dte_range[1])
          & (q["bid"] >= min_bid) & q["delta"].notna()].copy()
    if q.empty:
        return q
    q["_dte_gap"] = (q["dte"] - target_dte).abs()
    best_exp = q.groupby(["snapshot_date", "symbol"])["_dte_gap"].transform("min")
    q = q[q["_dte_gap"] == best_exp]
    # 同等远近的两个到期日取较早的那个，保证每 (日, 标的) 只剩一个到期日
    first_exp = q.groupby(["snapshot_date", "symbol"])["expiration"].transform("min")
    q = q[q["expiration"] == first_exp].copy()
    q["_delta_gap"] = (q["delta"].abs() - target_delta).abs()
    idx = q.groupby(["snapshot_date", "symbol"])["_delta_gap"].idxmin()
    return q.loc[idx].drop(columns=["_dte_gap", "_delta_gap"]).reset_index(drop=True)


def settle_csp(picks: pd.DataFrame, closes: dict[str, pd.Series]) -> pd.DataFrame:
    """到期结算：S_T = 到期日（或之前最近一个交易日）的收盘价。

    库内行情还没覆盖到期日的合约 status='open'（未结算）。收益以担保本金 K 为分母：
    ret = (bid − max(K − S_T, 0)) / K；不含担保现金利息（与合成指数回测的 cycle 口径一致）。
    """
    if picks.empty:
        return picks.assign(status=[], spot_end=[], payoff_pct=[], ret=[], assigned=[])
    out = picks.copy()
    status, spot_end, payoff_pct, ret = [], [], [], []
    for _, r in out.iterrows():
        c = closes.get(r["symbol"])
        exp = pd.Timestamp(r["expiration"])
        if c is None or c.empty or c.index.max() < exp:
            status.append("open")
            spot_end.append(np.nan)
            payoff_pct.append(np.nan)
            ret.append(np.nan)
            continue
        s_t = float(c.loc[:exp].iloc[-1])
        pay = max(r["strike"] - s_t, 0.0) / r["strike"]
        status.append("settled")
        spot_end.append(s_t)
        payoff_pct.append(pay)
        ret.append(r["prem_yield"] - pay)
    out["status"] = status
    out["spot_end"] = spot_end
    out["payoff_pct"] = payoff_pct
    out["ret"] = ret
    out["assigned"] = out["payoff_pct"] > 0
    return out


def monthly_first(df: pd.DataFrame) -> pd.DataFrame:
    """只保留每个日历月第一个快照日的记录 → 不重叠的月度样本。"""
    if df.empty:
        return df
    dates = pd.to_datetime(df["snapshot_date"])
    first = dates.groupby(dates.dt.to_period("M")).transform("min")
    return df[dates == first]


def forward_summary(settled: pd.DataFrame, index_symbols: set[str] | None = None) -> pd.DataFrame:
    """按分组汇总已结算的单笔：笔数/胜率/被行权率/平均权利金/平均收益/最差/最差吃掉几笔权利金。

    index_symbols 给出时多出「指数 ETF」与「个股」两行，外加「全部」。
    """
    s = settled[settled["status"] == "settled"] if "status" in settled else settled
    if s.empty:
        return pd.DataFrame()
    groups = {"全部": s}
    if index_symbols:
        is_idx = s["symbol"].isin(index_symbols)
        groups["指数 ETF"] = s[is_idx]
        groups["个股"] = s[~is_idx]
    rows = []
    for name, g in groups.items():
        if g.empty:
            continue
        avg_prem = float(g["prem_yield"].mean())
        worst = float(g["ret"].min())
        rows.append({
            "分组": name,
            "笔数": int(len(g)),
            "标的数": int(g["symbol"].nunique()),
            "胜率": float((g["ret"] > 0).mean()),
            "被行权率": float(g["assigned"].mean()),
            "平均权利金": avg_prem,
            "平均单笔收益": float(g["ret"].mean()),
            "最差单笔": worst,
            "最差≈几笔权利金": abs(worst) / avg_prem if avg_prem > 0 and worst < 0 else math.nan,
        })
    return pd.DataFrame(rows)
