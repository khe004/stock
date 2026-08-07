"""AI 基建个股页面的纯计算层：增长/盈利指标、赛道统治力、赛道汇总。不含 UI。

这是一个**观察/研究页面**的计算后端，不产生任何交易信号、不接入任何策略、不进模型组合。

增长指标从 financials 表（年度利润表）算出，口径与 fundamentals 表的 TTM 快照不同：
- 营收 3 年 CAGR：(最新财年营收 / 3年前营收)^(1/3) - 1。年报不足 4 期时按实际期数降级
  （如只有 3 期就算 2 年 CAGR），并在返回值里标注实际用了几年。
- 毛利率/净利率：最新财年 gross_profit/net_income / revenue（不用 fundamentals.gross_margins
  的 TTM 口径，避免同一页两个毛利率打架）。

赛道统治力用的是**公司整体市值份额，不是 AI 业务份额**——yfinance 拿不到业务分部数据，
无法量化"AI 纯度"。这个页面回答的是"这条赛道里谁体量大"，不是"谁最纯粹受益于 AI"。
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd


# ---------------------------------------------------------------------------
# 2.1 增长与盈利指标（每只股）
# ---------------------------------------------------------------------------

@dataclass
class GrowthMetrics:
    """单只个股的增长与盈利指标。"""
    symbol: str
    revenue_cagr: float | None       # 营收 CAGR（小数，如 1.0 = +100%）
    cagr_years: int | None           # CAGR 实际使用年数（3 或降级后的值）
    revenue_yoy: float | None        # 营收最近一年同比（小数）
    gross_margin: float | None       # 毛利率（小数，如 0.71 = 71%）
    net_margin: float | None         # 净利率（小数）


def compute_growth_metrics(fin_df: pd.DataFrame, symbol: str) -> GrowthMetrics:
    """从 financials 表的数据计算单只股的增长指标。

    fin_df: 该 symbol 的年度财报 DataFrame（需含 fiscal_date, revenue,
            gross_profit, net_income 列），按 fiscal_date 升序。

    边界处理（必须写测试覆盖）：
    - 营收为 0 或负 → 无法算 CAGR/同比，返回 None
    - 缺失年份 / 只有 1 期 → 无法算增长，返回 None
    - 分拆上市导致的历史断层（如 SNDK 只有 3 期）→ 降级为 2 年 CAGR 并标注
    - 缺失一律返回 None，**不用 0 填充**
    """
    result = GrowthMetrics(symbol=symbol, revenue_cagr=None, cagr_years=None,
                           revenue_yoy=None, gross_margin=None, net_margin=None)

    if fin_df.empty:
        return result

    # 按 fiscal_date 升序排列，取有 revenue 的行
    df = fin_df.sort_values("fiscal_date").reset_index(drop=True)
    rev_rows = df.dropna(subset=["revenue"])
    rev_rows = rev_rows[rev_rows["revenue"] > 0]  # 营收为 0 或负排除

    # 毛利率：最新财年
    latest = df.iloc[-1]
    rev_latest = latest.get("revenue")
    gp_latest = latest.get("gross_profit")
    ni_latest = latest.get("net_income")

    if rev_latest is not None and _is_positive(rev_latest):
        if gp_latest is not None and not _is_nan(gp_latest):
            result.gross_margin = float(gp_latest) / float(rev_latest)
        if ni_latest is not None and not _is_nan(ni_latest):
            result.net_margin = float(ni_latest) / float(rev_latest)

    if len(rev_rows) < 2:
        # 只有 1 期或 0 期，无法算增长
        return result

    # 营收最近一年同比
    latest_rev = float(rev_rows.iloc[-1]["revenue"])
    prev_rev = float(rev_rows.iloc[-2]["revenue"])
    if prev_rev > 0:
        result.revenue_yoy = latest_rev / prev_rev - 1

    # 营收 CAGR：尝试 3 年，不足则降级
    n_periods = len(rev_rows)
    target_years = 3
    actual_years = min(target_years, n_periods - 1)

    if actual_years >= 1:
        start_rev = float(rev_rows.iloc[-1 - actual_years]["revenue"])
        end_rev = float(rev_rows.iloc[-1]["revenue"])
        if start_rev > 0 and end_rev > 0:
            result.revenue_cagr = (end_rev / start_rev) ** (1 / actual_years) - 1
            result.cagr_years = actual_years

    return result


def _is_nan(v) -> bool:
    """检查值是否为 NaN（兼容 numpy 和 Python float）。"""
    try:
        return math.isnan(float(v))
    except (TypeError, ValueError):
        return True


def _is_positive(v) -> bool:
    """检查值是否为正数（非 NaN、非 None、> 0）。"""
    if v is None:
        return False
    try:
        f = float(v)
        return not math.isnan(f) and f > 0
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# 2.2 赛道统治力（市值份额）
# ---------------------------------------------------------------------------

def compute_lane_market_share(
    lane_symbols: list[str],
    market_caps: dict[str, float | None],
) -> dict[str, float | None]:
    """计算赛道内各成分的市值份额。

    market_caps: symbol -> market_cap（None 或缺失表示该股市值未知）。
    市值缺失的成分从分母里剔除（不当 0 处理）。

    返回 symbol -> 市值份额（0~1），市值缺失的返回 None。
    """
    valid = {s: market_caps[s] for s in lane_symbols
             if s in market_caps and market_caps[s] is not None
             and _is_positive(market_caps[s])}
    total = sum(valid.values())

    result: dict[str, float | None] = {}
    for s in lane_symbols:
        if s in valid and total > 0:
            result[s] = valid[s] / total
        else:
            result[s] = None
    return result


# ---------------------------------------------------------------------------
# 2.3 赛道汇总
# ---------------------------------------------------------------------------

def _weighted_by_cap(valid_caps: dict[str, float],
                     values: dict[str, float | None]) -> float | None:
    """按市值加权求均值，跳过取值缺失的成分（其市值一并不计入分母）。"""
    w_sum = 0.0
    cap_sum = 0.0
    for s, cap in valid_caps.items():
        v = values.get(s)
        if v is not None and not _is_nan(v):
            w_sum += cap * v
            cap_sum += cap
    return w_sum / cap_sum if cap_sum > 0 else None


def compute_lane_summary(
    lane_name: str,
    lane_symbols: list[str],
    market_caps: dict[str, float | None],
    growth_metrics: dict[str, GrowthMetrics],
    returns_1y: dict[str, float | None],
    momentum: dict[str, float | None] | None = None,
    returns_3y: dict[str, float | None] | None = None,
    returns_5y: dict[str, float | None] | None = None,
    revenue_growth_q: dict[str, float | None] | None = None,
) -> dict:
    """计算单条赛道的汇总数据。

    returns_1y/3y/5y: symbol -> 各期涨幅（小数），市值加权成赛道涨幅。
    momentum:   symbol -> 12-1 动量（小数），同样市值加权——与涨幅同口径，
                便于横向对比"已经涨完的"（历史涨幅）与"当前动量强弱"（12-1，跳过最近1个月）。
    三个可选参数传 None 时不输出对应列（向后兼容）。

    **短历史成分的处理**：某成分该期涨幅缺失（上市/分拆晚于窗口）时，其市值不计入该列的
    加权分母——赛道读数只由有完整历史的成分决定，不会被次新股的短期暴涨污染。代价是
    该列实际覆盖的成分可能少于赛道全部成分，跨列比较时需注意（页面 caption 已注明）。

    返回 dict：{赛道, 成分数, 合计市值, 近1/3/5年涨幅(市值加权), 12-1动量(市值加权),
                营收增长中位数}
    """
    # 成分数
    n = len(lane_symbols)

    # 合计市值（跳过缺失）
    valid_caps = {s: market_caps[s] for s in lane_symbols
                  if s in market_caps and market_caps[s] is not None
                  and _is_positive(market_caps[s])}
    total_cap = sum(valid_caps.values()) if valid_caps else None

    # 赛道各期涨幅 / 12-1 动量（均为市值加权，同口径）
    weighted_return = None
    weighted_3y = weighted_5y = weighted_mom = None
    if valid_caps and total_cap and total_cap > 0:
        weighted_return = _weighted_by_cap(valid_caps, returns_1y)
        if returns_3y is not None:
            weighted_3y = _weighted_by_cap(valid_caps, returns_3y)
        if returns_5y is not None:
            weighted_5y = _weighted_by_cap(valid_caps, returns_5y)
        if momentum is not None:
            weighted_mom = _weighted_by_cap(valid_caps, momentum)

    # 赛道营收增长中位数。优先用最新季度同比（revenue_growth_q）——年报同比滞后 6~12 个月
    # （最新财年早已结束），会出现"赛道股价涨 900% 但营收增长只显示 40%"这种自相矛盾的读数。
    # 未提供季度数据时回落到年报同比（向后兼容）。
    if revenue_growth_q is not None:
        yoy_values = [v for s in lane_symbols
                      if (v := revenue_growth_q.get(s)) is not None and not _is_nan(v)]
    else:
        yoy_values = [gm.revenue_yoy for s in lane_symbols
                      if (gm := growth_metrics.get(s)) and gm.revenue_yoy is not None]
    median_growth = float(pd.Series(yoy_values).median()) if yoy_values else None

    out = {
        "赛道": lane_name,
        "成分数": n,
        "合计市值": total_cap,
        "近1年涨幅(市值加权)": weighted_return,
    }
    if returns_3y is not None:
        out["近3年涨幅(市值加权)"] = weighted_3y
    if returns_5y is not None:
        out["近5年涨幅(市值加权)"] = weighted_5y
    if momentum is not None:
        out["12-1动量(市值加权)"] = weighted_mom
    out["营收增长中位数"] = median_growth
    return out
