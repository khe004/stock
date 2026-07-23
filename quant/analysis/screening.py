"""市场筛选页的纯计算：个股/板块当前强弱快照。不含 UI。

多因子综合分 = 【动量分 · 50% + 价值分 · 50%】（提供 fundamentals 时；否则退化为纯动量分）：

- 动量分 = 三个趋势维度【横截面百分位排名】的等权平均：
  - 12-1 动量：近 lookback 日收益但跳过最近 skip 日（避短期反转），中长期趋势强度
  - 52 周区间位置：现价在过去 range_window 日 [低,高] 区间的位置（0~1，越高越强）
  - 距均线：现价相对 ma 日均线的偏离（正=站上均线，趋势向上）
- 价值分 = 盈利收益率 E/P（= 1/forward_pe，反映分析师预期盈利/增长趋势；forward 缺失
  或非正时回退 trailing_pe）的【行业内】百分位（同行业内越便宜越高）。用 forward 而非
  trailing：成长股 trailing PE 常被畸高误判为"贵"（如 AMD 164 vs forward 37）。
  负盈利/无 P/E 的标的价值分缺失，其综合分退回只用动量分（不倒扣）。
  注意 forward 依赖分析师估计，可能偏乐观/被修正——它是"预期便宜"不等于"真便宜"。

动量与价值理念相反（动量买贵的赢家、价值买便宜的），50/50 融合是刻意的多因子折中；
各维仍用百分位等权、不做权重优化（避免落入 stock_momentum 那类过拟合陷阱）。

口径提醒：这是【当前快照】，非回测；价值用当前基本面快照（yfinance 现值，非
point-in-time 历史，无法回测）；个股宇宙是今天的成分快照含幸存者偏差；12-1 动量
买的是已涨完的强者，短期有反转/买在山顶风险。仅作强弱参考，不构成交易建议。
"""

import pandas as pd

from quant.analysis.market import range_position
from quant.strategies.base import price_series

STRENGTH_DIMS = ("mom", "pos_52w", "dist_ma")


def compute_strength(
    prices: dict[str, pd.DataFrame],
    fundamentals: pd.DataFrame | None = None,
    sectors: dict | pd.Series | None = None,
    pe_field: str = "forward_pe",
    pe_fallback: str | None = "trailing_pe",
    lookback: int = 252,
    skip: int = 21,
    ma: int = 200,
    range_window: int = 252,
) -> pd.DataFrame:
    """为每个标的算强弱快照（截至各自最新一日）。

    参数：
        prices: symbol -> 日线 DataFrame。
        fundamentals: 可选，索引=symbol、含 trailing_pe 列的当前基本面快照。
            提供时综合分融入价值分（动量半+价值半）；否则综合分为纯动量分。
        sectors: 可选，symbol -> 行业。提供时价值分做【行业内中性化】——盈利收益率
            在同行业内排百分位，消除科技高PE/银行低PE的结构性偏差（否则"价值"沦为
            "做多低PE行业"的行业押注）。不提供则价值分为全市场横截面百分位。

    返回 DataFrame（索引=symbol，按综合分降序），列：
    mom（12-1动量）、pos_52w（52周位置 0~1）、dist_ma（距均线偏离）、above_ma、
    trend_score（动量分 0~1）、composite（综合分 0~1）；提供 fundamentals 时另有
    pe（trailing PE）、earn_yield（盈利收益率）、value_score（价值分 0~1，行业内中性化）。
    历史不足 lookback+1 日的标的被跳过。
    """
    adj = pd.DataFrame({s: price_series(df) for s, df in prices.items()}).sort_index()
    records = []
    for s in adj.columns:
        a = adj[s].dropna()
        if len(a) < lookback + 1:
            continue
        mom = float(a.iloc[-1 - skip] / a.iloc[-1 - lookback] - 1)
        pos = range_position(a, range_window)
        ma_val = float(a.rolling(ma).mean().iloc[-1]) if len(a) >= ma else None
        dist = float(a.iloc[-1] / ma_val - 1) if ma_val and ma_val > 0 else None
        records.append({
            "symbol": s, "mom": mom, "pos_52w": pos, "dist_ma": dist,
            "above_ma": bool(dist is not None and dist > 0),
        })
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records).set_index("symbol")

    # 动量分：三维趋势的横截面百分位等权平均
    trend = pd.concat([df[c].rank(pct=True) for c in STRENGTH_DIMS], axis=1).mean(axis=1)
    df["trend_score"] = trend

    # 价值维度：盈利收益率 E/P = 1/PE。默认用 forward PE（反映分析师预期盈利=增长趋势），
    # forward 缺失/非正时回退 trailing PE（成长股 trailing 常被畸高，如 AMD 164→forward 37）。
    pe = None
    if fundamentals is not None:
        def _col(name):
            if name and name in fundamentals.columns:
                return pd.to_numeric(fundamentals[name].reindex(df.index), errors="coerce")
            return pd.Series(index=df.index, dtype="float64")
        pe = _col(pe_field)
        if pe_fallback:
            pe = pe.where(pe > 0, _col(pe_fallback))  # forward 非正/缺失 → 回退 trailing
    if pe is not None and pe.where(pe > 0).notna().any():
        df["pe"] = pe
        df["earn_yield"] = (1.0 / pe.where(pe > 0))  # 负盈利/无PE → NaN
        if sectors is not None:
            # 行业内中性化：盈利收益率在【同行业】内排百分位，消除行业结构性 PE 差异
            sec = pd.Series(sectors).reindex(df.index).fillna("其他")
            value = df.groupby(sec)["earn_yield"].rank(pct=True)
        else:
            value = df["earn_yield"].rank(pct=True)
        df["value_score"] = value
        # 动量半 + 价值半；价值缺失（无PE）的标的按 skipna 退回只用动量分
        df["composite"] = pd.concat([trend, value], axis=1).mean(axis=1)
    else:
        df["composite"] = trend

    return df.sort_values("composite", ascending=False)


def market_regime(spy_df: pd.DataFrame, ma: int = 200) -> dict:
    """大盘趋势判断：现价相对 ma 日均线。risk_on=站上均线，dist=偏离幅度。
    数据不足返回 risk_on=None。"""
    a = price_series(spy_df).dropna()
    if len(a) < ma:
        return {"risk_on": None, "dist": None, "ma": ma}
    ma_val = float(a.rolling(ma).mean().iloc[-1])
    price = float(a.iloc[-1])
    return {"risk_on": price >= ma_val, "dist": price / ma_val - 1, "ma": ma}
