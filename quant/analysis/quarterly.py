"""季度经营指标：只在报告期连续且字段完整时计算同比与 TTM。"""

from dataclasses import dataclass

import pandas as pd


@dataclass
class QuarterlyMetrics:
    reported_latest_period: str | None = None
    latest_period: str | None = None
    latest_period_incomplete: bool = False
    year_ago_period: str | None = None
    revenue: float | None = None
    revenue_yoy: float | None = None
    operating_income_yoy: float | None = None
    gross_margin: float | None = None
    gross_margin_year_ago: float | None = None
    operating_margin: float | None = None
    operating_margin_year_ago: float | None = None
    operating_cash_flow: float | None = None
    capital_expenditure: float | None = None
    free_cash_flow: float | None = None
    cash: float | None = None
    total_debt: float | None = None
    ttm_revenue: float | None = None
    ttm_operating_income: float | None = None
    ttm_operating_cash_flow: float | None = None
    ttm_capital_expenditure: float | None = None
    ttm_free_cash_flow: float | None = None


def _number(row: pd.Series, name: str) -> float | None:
    value = row.get(name)
    return float(value) if value is not None and pd.notna(value) else None


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def _year_ago(index: pd.DatetimeIndex, latest: pd.Timestamp) -> pd.Timestamp | None:
    candidates = [(abs((latest - date).days - 365), date) for date in index
                  if 330 <= (latest - date).days <= 400]
    return min(candidates)[1] if candidates else None


def _continuous_four(index: pd.DatetimeIndex, latest: pd.Timestamp) -> list[pd.Timestamp]:
    dates = list(index[index <= latest])[-4:]
    if len(dates) != 4:
        return []
    gaps = [(b - a).days for a, b in zip(dates, dates[1:])]
    return dates if all(70 <= gap <= 120 for gap in gaps) else []


def compute_quarterly_metrics(frame: pd.DataFrame) -> QuarterlyMetrics:
    """计算最新季度、同比和 TTM；缺季度时不把更早期间凑进 TTM。"""
    out = QuarterlyMetrics()
    if frame.empty or "revenue" not in frame.columns:
        return out
    work = frame.copy()
    work.index = pd.to_datetime(work.index)
    work = work.sort_index()
    if len(work.index):
        out.reported_latest_period = work.index[-1].strftime("%Y-%m-%d")
    valid_revenue = work[work["revenue"].notna()]
    if valid_revenue.empty:
        return out
    latest = valid_revenue.index[-1]
    row = work.loc[latest]
    out.latest_period = latest.strftime("%Y-%m-%d")
    out.latest_period_incomplete = latest != work.index[-1]
    out.revenue = _number(row, "revenue")
    out.gross_margin = _ratio(_number(row, "gross_profit"), out.revenue)
    out.operating_margin = _ratio(_number(row, "operating_income"), out.revenue)
    out.operating_cash_flow = _number(row, "operating_cash_flow")
    out.capital_expenditure = _number(row, "capital_expenditure")
    out.free_cash_flow = _number(row, "free_cash_flow")
    if out.free_cash_flow is None and out.operating_cash_flow is not None \
            and out.capital_expenditure is not None:
        out.free_cash_flow = out.operating_cash_flow + out.capital_expenditure

    balance = work.loc[:latest]
    for field in ("cash", "total_debt"):
        values = balance[field].dropna() if field in balance.columns else pd.Series(dtype=float)
        setattr(out, field, float(values.iloc[-1]) if not values.empty else None)

    prior_date = _year_ago(work.index, latest)
    if prior_date is not None:
        prior = work.loc[prior_date]
        prior_revenue = _number(prior, "revenue")
        prior_operating = _number(prior, "operating_income")
        out.year_ago_period = prior_date.strftime("%Y-%m-%d")
        if prior_revenue is not None and prior_revenue != 0:
            out.revenue_yoy = out.revenue / prior_revenue - 1
        latest_operating = _number(row, "operating_income")
        if prior_operating is not None and prior_operating > 0 and latest_operating is not None:
            out.operating_income_yoy = latest_operating / prior_operating - 1
        out.gross_margin_year_ago = _ratio(_number(prior, "gross_profit"), prior_revenue)
        out.operating_margin_year_ago = _ratio(prior_operating, prior_revenue)

    dates = _continuous_four(work.index, latest)
    for field in ("revenue", "operating_income", "operating_cash_flow",
                  "capital_expenditure", "free_cash_flow"):
        if not dates or field not in work.columns:
            continue
        values = work.loc[dates, field]
        if values.notna().all():
            setattr(out, "ttm_" + field, float(values.sum()))
    if out.ttm_free_cash_flow is None and out.ttm_operating_cash_flow is not None \
            and out.ttm_capital_expenditure is not None:
        out.ttm_free_cash_flow = out.ttm_operating_cash_flow + out.ttm_capital_expenditure
    return out


def describe_quarterly_change(metrics: QuarterlyMetrics) -> str:
    """只陈述可计算的变化，不生成无来源的原因解释。"""
    if metrics.latest_period is None:
        return "暂无有效季度营收记录。"
    parts = [f"截至 {metrics.latest_period}"]
    if metrics.revenue_yoy is not None:
        parts.append(f"营收同比 {metrics.revenue_yoy:+.1%}")
    if metrics.operating_income_yoy is not None:
        parts.append(f"营业利润同比 {metrics.operating_income_yoy:+.1%}")
    if metrics.gross_margin is not None:
        margin = f"毛利率 {metrics.gross_margin:.1%}"
        if metrics.gross_margin_year_ago is not None:
            margin += f"（同比 {metrics.gross_margin - metrics.gross_margin_year_ago:+.1%}）"
        parts.append(margin)
    if metrics.operating_margin is not None:
        margin = f"营业利润率 {metrics.operating_margin:.1%}"
        if metrics.operating_margin_year_ago is not None:
            margin += f"（同比 {metrics.operating_margin - metrics.operating_margin_year_ago:+.1%}）"
        parts.append(margin)
    return "；".join(parts) + "。"
