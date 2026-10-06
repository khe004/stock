"""合成指数卖 put 回测（SPY + VIX，Black-Scholes 定价）：纯计算模块，不含 UI。

回答的问题：**每月卖一张 SPY 看跌期权、现金全额担保（CSP），能不能给平台加东西？**
来源是 2026-10 评估 PutFinder（个股 CSP 排行榜）的结论：个股期权没有历史数据无法回测，
但"卖指数波动率赚 VRP（波动率风险溢价）"有几十年文献支撑（CBOE PUT 指数），
且我们手里已有 SPY + ^VIX + BIL 全历史，可以合成出来先看指数这条腿。

机制（与 CBOE PUT 指数同构，可调到虚值）：
- 锚点 = ``month_anchors(index, offset)``（默认每月首个交易日，与平台月频策略一致；
  offset 用于 timing luck 检验）。在锚点收盘卖出一张到期日 = 下一个锚点的 put。
- 行权价：``target_delta=None`` → 平值（ATM，K=S，即 PUT 指数口径）；否则按 BS delta 反解，
  如 0.30 = 卖 30-delta 虚值 put（PutFinder 一类工具的常见区间 0.2~0.3）。
- 张数 n = 当时权益 / K：**全额现金担保、零杠杆**，被行权时现金正好够接货。
- 权利金 = BS 理论价 × (1 − premium_haircut)，haircut 模拟以买价成交吃掉的价差。
- 每日盯市：权益 = 担保现金 − n × 期权当前 BS 价值（用当日 VIX、剩余交易日）；
  担保现金按 ``cash_ret``（BIL 日总回报）计息，缺省按 rate/252。
- 到期（下一锚点收盘）按内在价值 max(K − S, 0) 现金结算，然后立刻卖下一张。

**口径与偏差（必须同时说清，宁可低估不可虚高）**：
1. IV 用 VIX：VIX 是 30 天方差互换的报价，含虚值 put 的偏斜，通常**高于平值 IV 1~3 个点**、
   与 25~30-delta put 的 IV 大致相当。所以 ATM 口径的权利金**偏高（虚高）**，30-delta 口径
   大致中性；更深虚值（<0.2 delta）真实偏斜更陡，VIX 口径反而偏低。``iv_mult`` 供敏感性检验。
2. 到期日不是真实的第三个周五，期限也不是精确 30 天（锚点间约 19~23 个交易日）。
3. 用 SPY 原始 close（期权按价格结算，不含分红）；担保现金不持 SPY，所以不该拿 SPY 分红。
4. 没有提前行权（美式 SPY put 深度实值时可能被提前指派，影响很小）、没有保证金/税务。
5. 盯市用当日 VIX 给所有剩余期限定价：近到期时 VIX 不再代表该期权 IV，日波动的形状
   （进而夏普/回撤）只是近似，月度结算值才是"真"的。

公平基准（决策 #3 的精神）：卖 put ≈ 半仓股票 + 卖掉上方收益。拿它比 SPY 长持是苹果比橘子，
可信对比是 **β 匹配混合** —— 按其实测 β 持 SPY、其余持 BIL（``beta_matched_blend``）。
跑赢 β 混合 = 真赚到了 VRP；只跑赢 BIL 或只是回撤小于 SPY，都可能只是"少持股票"。
"""

import math
from statistics import NormalDist

import numpy as np
import pandas as pd

from quant.backtest.engine import TRADING_DAYS
from quant.strategies.base import month_anchors

_N = NormalDist()


def bs_put(spot: float, strike: float, t: float, sigma: float, rate: float = 0.0) -> float:
    """欧式 put 的 Black-Scholes 价格（无分红）。t 以年计；t<=0 或 sigma<=0 时返回内在价值。"""
    if t <= 0 or sigma <= 0:
        return max(strike - spot, 0.0)
    vs = sigma * math.sqrt(t)
    d1 = (math.log(spot / strike) + (rate + 0.5 * sigma * sigma) * t) / vs
    d2 = d1 - vs
    return strike * math.exp(-rate * t) * _N.cdf(-d2) - spot * _N.cdf(-d1)


def bs_put_delta(spot: float, strike: float, t: float, sigma: float, rate: float = 0.0) -> float:
    """put 的 BS delta（负数，-1~0）。"""
    if t <= 0 or sigma <= 0:
        return -1.0 if spot < strike else 0.0
    d1 = (math.log(spot / strike) + (rate + 0.5 * sigma * sigma) * t) / (sigma * math.sqrt(t))
    return _N.cdf(d1) - 1.0


def strike_for_delta(spot: float, t: float, sigma: float, delta: float, rate: float = 0.0) -> float:
    """反解 |delta| = ``delta`` 的 put 行权价（如 0.30 → 30-delta 虚值 put）。"""
    if not 0 < delta < 1:
        raise ValueError("delta 须在 (0,1) 之间，例如 0.30")
    d1 = _N.inv_cdf(1.0 - delta)
    vs = sigma * math.sqrt(t)
    return spot * math.exp(-(d1 * vs - (rate + 0.5 * sigma * sigma) * t))


def simulate_putwrite(
    spot: pd.Series,
    iv: pd.Series,
    rate: pd.Series | None = None,
    cash_ret: pd.Series | None = None,
    target_delta: float | None = None,
    offset: int = 0,
    premium_haircut: float = 0.05,
    iv_mult: float = 1.0,
    initial: float = 10_000.0,
) -> tuple[pd.Series, pd.DataFrame]:
    """逐日模拟"每月卖一张全额担保 put"的权益曲线。

    参数：
    - spot：标的原始收盘价（SPY close，不复权——期权按价格结算）。
    - iv：年化隐含波动率，小数（^VIX close / 100）。缺失日前向填充。
    - rate：年化无风险利率，小数（^IRX close / 100），用于 BS 折现；缺省 0。
    - cash_ret：担保现金的日收益率（BIL adj_close 的 pct_change）；缺省按 rate/252。
    - target_delta：None=平值；0.30=30-delta 虚值。
    - offset：锚点偏移（0=每月首个交易日），给 timing luck 检验用。
    - premium_haircut：权利金折价比例，模拟以买价成交。
    - iv_mult：IV 乘数，做"VIX 口径偏高/偏低"的敏感性检验。

    返回 (equity, cycles)：
    - equity：从第一个锚点开始的每日权益（首值 = initial）。
    - cycles：每期一行（卖出日/到期日/S/K/IV/权利金率/结算赔付率/当期收益率 等），
      收益率都以担保本金 K 为分母，不含现金利息。
    """
    df = pd.DataFrame({"S": spot.astype(float)})
    df["iv"] = iv.reindex(df.index).ffill() * iv_mult
    df["r"] = (rate.reindex(df.index).ffill().fillna(0.0) if rate is not None
               else pd.Series(0.0, index=df.index))
    if cash_ret is not None:
        df["cr"] = cash_ret.reindex(df.index).fillna(0.0)
    else:
        df["cr"] = df["r"] / TRADING_DAYS
    df = df.dropna(subset=["S", "iv"])
    df = df[df["iv"] > 0]
    if df.empty:
        return pd.Series(dtype=float, name="putwrite"), pd.DataFrame()

    idx = df.index
    anchors = month_anchors(idx, offset)
    if len(anchors) == 0:
        return pd.Series(dtype=float, name="putwrite"), pd.DataFrame()
    pos = {ts: i for i, ts in enumerate(idx)}
    anchor_pos = [pos[a] for a in anchors]
    start = anchor_pos[0]
    # 每个锚点卖出的那张 put 在下一个锚点到期；最后一个锚点的 put 按假想 21 日期限盯市
    expiry_of = {p: (anchor_pos[k + 1] if k + 1 < len(anchor_pos) else p + 21)
                 for k, p in enumerate(anchor_pos)}
    is_anchor = set(anchor_pos)

    S = df["S"].to_numpy()
    sig = df["iv"].to_numpy()
    r = df["r"].to_numpy()
    cr = df["cr"].to_numpy()

    cash = float(initial)
    n = 0.0          # 卖出张数（以 1 份标的计）
    K = 0.0
    exp_pos = -1
    cur: dict | None = None
    values = np.empty(len(idx) - start)
    cycles: list[dict] = []

    for i in range(start, len(idx)):
        if i > start:
            cash *= 1.0 + cr[i]
        if i in is_anchor:
            # 先结算上一张（到期 = 今天）
            if n:
                payoff = max(K - S[i], 0.0)
                cash -= n * payoff
                cur.update(expiry=idx[i], spot_end=S[i], payoff_pct=payoff / K,
                           ret_pct=cur["premium_pct"] - payoff / K,
                           assigned=payoff > 0)
                cycles.append(cur)
            # 再卖下一张
            exp_pos = expiry_of[i]
            t = (exp_pos - i) / TRADING_DAYS
            K = S[i] if target_delta is None else strike_for_delta(S[i], t, sig[i], target_delta, r[i])
            prem = bs_put(S[i], K, t, sig[i], r[i]) * (1.0 - premium_haircut)
            n = cash / K
            cash += n * prem
            cur = {"sell_date": idx[i], "spot": S[i], "strike": K, "moneyness": K / S[i] - 1,
                   "iv": sig[i], "dte": exp_pos - i, "premium_pct": prem / K,
                   "delta": bs_put_delta(S[i], K, t, sig[i], r[i])}
            liability = n * bs_put(S[i], K, t, sig[i], r[i])
        else:
            t_rem = max(exp_pos - i, 0) / TRADING_DAYS
            liability = n * bs_put(S[i], K, t_rem, sig[i], r[i])
        values[i - start] = cash - liability

    equity = pd.Series(values, index=idx[start:], name="putwrite")
    return equity, pd.DataFrame(cycles)


def regression_beta(ret: pd.Series, bench_ret: pd.Series, freq: str = "ME") -> float:
    """策略对基准的 β。默认用月度复利收益估计：日度盯市只是近似（见模块头第 5 条），
    月度结算值才可靠；月度也避开了日度 β 被非同步噪声压低的问题。"""
    both = pd.concat([ret, bench_ret], axis=1).dropna()
    if freq:
        both = (1 + both).resample(freq).prod() - 1
    if len(both) < 3 or both.iloc[:, 1].var() == 0:
        return float("nan")
    return float(both.iloc[:, 0].cov(both.iloc[:, 1]) / both.iloc[:, 1].var())


def beta_matched_blend(equity_ret: pd.Series, cash_ret: pd.Series, beta: float,
                       initial: float = 10_000.0, cost_bps: float = 0.0) -> pd.Series:
    """β 匹配混合基准：β 份持 SPY（总回报）、1−β 份持现金等价，每日再平衡到固定权重。

    卖 put 的公平对照——"它赚的到底是 VRP，还是只是少持了一半股票"。
    cost_bps 只在首日收一次建仓成本（每日再平衡的微小换手不计，略偏向基准）。
    """
    beta = min(max(beta, 0.0), 1.0)
    both = pd.concat([equity_ret, cash_ret], axis=1).fillna(0.0)
    r = beta * both.iloc[:, 0] + (1 - beta) * both.iloc[:, 1]
    r.iloc[0] = 0.0
    out = initial * (1 - beta * cost_bps / 1e4) * (1 + r).cumprod()
    out.name = "beta_blend"
    return out


def cycle_summary(cycles: pd.DataFrame) -> dict:
    """按期统计：被行权率、平均权利金率、平均/最差单期收益、盈亏比。

    **胜率不是卖 put 的好指标**（大概率小赚、小概率大亏是收益结构本身）——
    这里同时给出最差单期和"一次亏损吃掉几期权利金"，让尾部一眼可见。
    """
    if cycles.empty or "ret_pct" not in cycles:
        return {}
    ret = cycles["ret_pct"]
    wins, losses = ret[ret > 0], ret[ret <= 0]
    avg_prem = float(cycles["premium_pct"].mean())
    worst = float(ret.min())
    return {
        "periods": int(len(ret)),
        "win_rate": float((ret > 0).mean()),
        "assigned_rate": float(cycles["assigned"].mean()),
        "avg_premium": avg_prem,
        "avg_ret": float(ret.mean()),
        "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) else 0.0,
        "worst": worst,
        "worst_date": cycles.loc[ret.idxmin(), "sell_date"],
        # 最差一期亏掉的钱 ≈ 多少期的平均权利金
        "worst_in_premiums": abs(worst) / avg_prem if avg_prem > 0 else float("nan"),
    }


def excess_sharpe(equity: pd.Series, cash_ret: pd.Series) -> float:
    """扣掉无风险收益后的夏普。平台的 ``equity_metrics`` 夏普不扣无风险利率——
    对 ETF 轮动策略影响不大，但卖 put / β 混合这类**半仓现金**的东西会被它系统性抬高
    （纯持 BIL 的"夏普"能到两位数），所以在这里另给一个扣现金的口径。"""
    r = equity.pct_change().dropna()
    ex = r - cash_ret.reindex(r.index).fillna(0.0)
    if len(ex) < 2 or ex.std() == 0:
        return float("nan")
    return float(ex.mean() / ex.std() * math.sqrt(TRADING_DAYS))
