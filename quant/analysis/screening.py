"""市场筛选页的纯计算：个股/板块当前强弱快照。不含 UI。

对每个标的算三个"强弱"维度，再横截面合成综合分：
- 12-1 动量：近 lookback 日收益但跳过最近 skip 日（避短期反转），衡量中长期趋势强度
- 52 周区间位置：现价在过去 range_window 日 [低,高] 区间的位置（0~1，越高越强）
- 距均线：现价相对 ma 日均线的偏离（正=站上均线，趋势向上）
综合分 = 三维【横截面百分位排名】的等权平均——刻意简单透明，不做权重优化
（避免过拟合；权重一旦去拟合历史就落入 stock_momentum 那类陷阱）。

口径提醒：这是【当前快照】，非回测；个股宇宙是今天的成分快照，含幸存者偏差；
12-1 动量买的是已涨完的强者，短期有反转/买在山顶风险，仅作强弱参考不构成建议。
"""

import pandas as pd

from quant.analysis.market import range_position
from quant.strategies.base import price_series

STRENGTH_DIMS = ("mom", "pos_52w", "dist_ma")


def compute_strength(
    prices: dict[str, pd.DataFrame],
    lookback: int = 252,
    skip: int = 21,
    ma: int = 200,
    range_window: int = 252,
) -> pd.DataFrame:
    """为每个标的算强弱快照（截至各自最新一日）。

    返回 DataFrame（索引=symbol，按综合分降序），列：
    mom（12-1 动量）、pos_52w（52周位置 0~1）、dist_ma（距均线偏离）、
    above_ma（是否站上均线）、composite（综合分 0~1，横截面百分位均值）。
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
    # 横截面百分位排名合成综合分（三维等权，缺失维度按可用维度平均）
    ranks = pd.concat([df[c].rank(pct=True) for c in STRENGTH_DIMS], axis=1)
    df["composite"] = ranks.mean(axis=1)
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
