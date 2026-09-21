"""AI 基建个股页面的纯计算层：增长/盈利指标、赛道统治力、赛道汇总。不含 UI。

这是一个**观察/研究页面**的计算后端，不产生任何交易信号、不接入任何策略、不进模型组合。

增长指标从 financials 表（年度利润表）算出，口径与 fundamentals 表的 TTM 快照不同：
- 营收 3 年 CAGR：(最新财年营收 / 3年前营收)^(1/3) - 1。年报不足 4 期时按实际期数降级
  （如只有 3 期就算 2 年 CAGR），并在返回值里标注实际用了几年。
  **这一列跨了一整轮半导体周期，不是"AI 时代的增速"**：起点 FY2022 恰是存储/模拟的周期顶，
  MU 的 3 年 CAGR 只有 +6.7% 而最新季度同比是 +345.7%，差 50 倍——因为一个从周期顶量起、
  一个从周期底量起。窗口内出现单年断崖的（业务分拆或周期顶）由 `cagr_break` 标出，页面打备注；
  INTC 那种连年小幅下滑不打标，它的负 CAGR 是真实的结构性掉队。
- 毛利率/净利率：最新财年 gross_profit/net_income / revenue（不用 fundamentals.gross_margins
  的 TTM 口径，避免同一页两个毛利率打架）。

赛道统治力用的是**公司整体市值份额，不是 AI 业务份额**——yfinance 拿不到业务分部数据，
无法量化"AI 纯度"。这个页面回答的是"这条赛道里谁体量大"，不是"谁最纯粹受益于 AI"。
"""

from __future__ import annotations

import math
import json
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
    cagr_break: float | None = None  # CAGR 窗口内最大单年营收跌幅（小数，负值），无断崖为 None


@dataclass
class ReitMetrics:
    """REIT 专用指标；FFO/AFFO 只使用直接披露的季度事实。"""
    dividend_yield: float | None = None
    annual_dividend_per_share: float | None = None
    ev_to_ebitda: float | None = None
    net_debt_to_ebitda: float | None = None
    ttm_ffo: float | None = None
    price_to_ffo: float | None = None
    ffo_payout_ratio: float | None = None
    ttm_affo: float | None = None
    price_to_affo: float | None = None
    affo_payout_ratio: float | None = None
    latest_period: str | None = None


def is_reit_company(raw: dict | None) -> bool:
    """用供应商行业标签识别 REIT，不把所有房地产公司都自动视为 REIT。"""
    if not raw:
        return False
    labels = " ".join(str(raw.get(key) or "") for key in (
        "industry", "industryKey", "industryDisp",
    )).lower()
    return "reit" in labels or "real estate investment trust" in labels


def compute_reit_metrics(fundamental: pd.Series | dict | None,
                         quarterly: pd.DataFrame) -> ReitMetrics:
    """计算 REIT 指标，不用 GAAP 净利润或普通 FCF 伪造 FFO/AFFO。"""
    result = ReitMetrics()
    if fundamental is None:
        return result
    raw_value = fundamental.get("raw_json")
    if isinstance(raw_value, str):
        try:
            raw = json.loads(raw_value) if raw_value else {}
        except (TypeError, ValueError, json.JSONDecodeError):
            raw = {}
    elif isinstance(raw_value, dict):
        raw = raw_value
    else:
        raw = {}

    def positive(value) -> float | None:
        number = pd.to_numeric(value, errors="coerce")
        return float(number) if pd.notna(number) and float(number) > 0 else None

    dividend_yield = positive(fundamental.get("dividend_yield"))
    # yfinance 的 dividendYield 历史上既出现过 0.0259，也出现过 2.59。
    result.dividend_yield = (dividend_yield / 100
                             if dividend_yield is not None and dividend_yield > 1
                             else dividend_yield)
    result.annual_dividend_per_share = positive(
        raw.get("dividendRate") or raw.get("trailingAnnualDividendRate"))
    result.ev_to_ebitda = positive(fundamental.get("ev_to_ebitda"))
    debt, cash, ebitda = (positive(raw.get("totalDebt")), positive(raw.get("totalCash")),
                          positive(raw.get("ebitda")))
    if debt is not None and ebitda is not None:
        result.net_debt_to_ebitda = (debt - (cash or 0.0)) / ebitda

    if quarterly.empty:
        return result
    work = quarterly.copy()
    work.index = pd.to_datetime(work.index)
    work = work.sort_index()
    dates = list(work.index[-4:])
    if len(dates) != 4 or not all(
            70 <= (b - a).days <= 120 for a, b in zip(dates, dates[1:])):
        return result
    statement_currency = quarterly.attrs.get("currency")
    market_currency = raw.get("currency") or quarterly.attrs.get("trading_currency")
    if (not statement_currency or not market_currency
            or str(statement_currency).upper() != str(market_currency).upper()):
        return result
    result.latest_period = dates[-1].strftime("%Y-%m-%d")

    def direct_ttm(metric: str) -> float | None:
        if metric not in work:
            return None
        values = pd.to_numeric(work.loc[dates, metric], errors="coerce")
        total = float(values.sum()) if values.notna().all() else None
        return total if total is not None and total > 0 else None

    result.ttm_ffo = direct_ttm("funds_from_operations")
    result.ttm_affo = direct_ttm("adjusted_funds_from_operations")
    market_cap = positive(fundamental.get("market_cap"))
    shares = positive(raw.get("sharesOutstanding"))
    annual_distribution = (result.annual_dividend_per_share * shares
                           if result.annual_dividend_per_share is not None
                           and shares is not None else None)
    if result.ttm_ffo is not None:
        result.price_to_ffo = market_cap / result.ttm_ffo if market_cap else None
        result.ffo_payout_ratio = (annual_distribution / result.ttm_ffo
                                   if annual_distribution is not None else None)
    if result.ttm_affo is not None:
        result.price_to_affo = market_cap / result.ttm_affo if market_cap else None
        result.affo_payout_ratio = (annual_distribution / result.ttm_affo
                                    if annual_distribution is not None else None)
    return result


def valuation_scenarios(
    revenue: float | None,
    net_debt: float | None,
    shares_outstanding: float | None,
    assumptions: list[dict],
    current_price: float | None = None,
) -> pd.DataFrame:
    """用显式的收入增长、EBITDA 利润率和 EV/EBITDA 生成多情景结果。

    所有金额必须由调用方保证处于同一币种，股数必须与证券口径一致。本函数不猜汇率、
    不回填股本，也不选择所谓“最可能”的单一目标价。
    """
    columns = [
        "scenario", "revenue_growth", "ebitda_margin", "ev_to_ebitda",
        "projected_revenue", "projected_ebitda", "enterprise_value",
        "equity_value", "implied_price", "upside",
    ]

    def finite(value) -> float | None:
        number = pd.to_numeric(value, errors="coerce")
        return float(number) if pd.notna(number) and math.isfinite(float(number)) else None

    revenue_value = finite(revenue)
    debt_value = finite(net_debt)
    shares_value = finite(shares_outstanding)
    price_value = finite(current_price)
    if (revenue_value is None or revenue_value <= 0 or debt_value is None
            or shares_value is None or shares_value <= 0):
        return pd.DataFrame(columns=columns)
    rows = []
    for item in assumptions:
        growth = finite(item.get("revenue_growth"))
        margin = finite(item.get("ebitda_margin"))
        multiple = finite(item.get("ev_to_ebitda"))
        if growth is None or margin is None or multiple is None or margin <= 0 or multiple <= 0:
            continue
        projected_revenue = revenue_value * (1 + growth)
        if projected_revenue <= 0:
            continue
        projected_ebitda = projected_revenue * margin
        enterprise_value = projected_ebitda * multiple
        equity_value = enterprise_value - debt_value
        implied_price = equity_value / shares_value if equity_value > 0 else None
        upside = (implied_price / price_value - 1
                  if implied_price is not None and price_value is not None and price_value > 0
                  else None)
        rows.append({
            "scenario": str(item.get("scenario") or "未命名"),
            "revenue_growth": growth,
            "ebitda_margin": margin,
            "ev_to_ebitda": multiple,
            "projected_revenue": projected_revenue,
            "projected_ebitda": projected_ebitda,
            "enterprise_value": enterprise_value,
            "equity_value": equity_value,
            "implied_price": implied_price,
            "upside": upside,
        })
    return pd.DataFrame(rows, columns=columns)


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

    # 营收最近一年同比。若中间缺了财年，不能把两年变化冒充单年同比。
    latest_rev = float(rev_rows.iloc[-1]["revenue"])
    prev_rev = float(rev_rows.iloc[-2]["revenue"])
    latest_date = pd.Timestamp(rev_rows.iloc[-1]["fiscal_date"])
    prev_date = pd.Timestamp(rev_rows.iloc[-2]["fiscal_date"])
    if prev_rev > 0 and _fiscal_year_gap(prev_date, latest_date) == 1:
        result.revenue_yoy = latest_rev / prev_rev - 1

    # 营收 CAGR：目标 3 个日历年；缺年时按实际 fiscal_date 年数计算。
    # 不能用「有效记录条数 - 1」当年份：2022→2024 只有两条记录，但应是 2 年 CAGR。
    end_date = pd.Timestamp(rev_rows.iloc[-1]["fiscal_date"])
    candidates = []
    for i in range(len(rev_rows) - 1):
        start_date = pd.Timestamp(rev_rows.iloc[i]["fiscal_date"])
        year_gap = _fiscal_year_gap(start_date, end_date)
        if year_gap >= 1:
            candidates.append((abs(year_gap - 3), year_gap, i, start_date))
    if candidates:
        _, actual_years, start_i, _ = min(candidates, key=lambda item: (item[0], -item[1]))
        start_rev = float(rev_rows.iloc[start_i]["revenue"])
        end_rev = float(rev_rows.iloc[-1]["revenue"])
        if start_rev > 0 and end_rev > 0:
            result.revenue_cagr = (end_rev / start_rev) ** (1 / actual_years) - 1
            result.cagr_years = actual_years
            result.cagr_break = _worst_yoy_drop(rev_rows.iloc[start_i:])

    return result


# CAGR 失真阈值：窗口内单年营收跌幅超过这个数就打备注。
# 40% 不是统计出来的，是"正常经营波动到不了、必须是周期顶起量或业务被拆走"的经验线：
# WDC 分拆 SanDisk 后 -66%，MU 撞上 2022 存储周期顶后 -50%，都被抓住；
# INTC 那种连年 -3%~-14% 的真实结构性下滑不会被误标，它的负 CAGR 就该原样呈现。
CAGR_BREAK_THRESHOLD = -0.40


def _worst_yoy_drop(rev_rows: pd.DataFrame) -> float | None:
    """CAGR 窗口内最大的单年营收跌幅；没有超过阈值的断崖则返回 None。"""
    drops = []
    rows = list(rev_rows.itertuples(index=False))
    for prev, curr in zip(rows, rows[1:]):
        prev_date = pd.Timestamp(getattr(prev, "fiscal_date"))
        curr_date = pd.Timestamp(getattr(curr, "fiscal_date"))
        # 只把连续财年间的变化标为单年断崖；跨年缺口另由 cagr_years 暴露。
        if _fiscal_year_gap(prev_date, curr_date) != 1:
            continue
        prev_rev = float(getattr(prev, "revenue"))
        curr_rev = float(getattr(curr, "revenue"))
        if prev_rev > 0:
            drops.append(curr_rev / prev_rev - 1)
    if not drops:
        return None
    worst = min(drops)
    return worst if worst <= CAGR_BREAK_THRESHOLD else None


def _fiscal_year_gap(start: pd.Timestamp, end: pd.Timestamp) -> int:
    """把财年结束日间隔换算成最接近的完整年数。

    财年可能落在不同月份，单看 ``end.year - start.year`` 会把
    2023-12-31→2025-01-01 误判成两年；按天数四舍五入更贴近实际报告期间。
    """
    return max(0, int(round((end - start).days / 365.25)))


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


def compact_amount(value) -> str:
    """把财务金额缩写成易读的 K/M/B/T，缺失值显示破折号。"""
    if value is None or _is_nan(value):
        return "—"
    number = float(value)
    absolute = abs(number)
    for threshold, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if absolute >= threshold:
            scaled = number / threshold
            decimals = 1 if abs(scaled) >= 10 else 2
            return f"{scaled:,.{decimals}f}{suffix}"
    return f"{number:,.0f}"


# ---------------------------------------------------------------------------
# 币种映射与多币种市值换算
# ---------------------------------------------------------------------------

# 交易所后缀 → 币种（兜底推断；优先用 fundamentals.raw_json 里的 currency）
SUFFIX_TO_CURRENCY: dict[str, str] = {
    ".KS": "KRW",
    ".KQ": "KRW",
    ".T": "JPY",
    ".TW": "TWD",
    ".TWO": "TWD",
    ".SZ": "CNY",
    ".SS": "CNY",
    ".HK": "HKD",
}


# 非美标的的中文名。美股代码（NVDA/MU…）本身可读，不做映射；但 `005930.KS`、`300308.SZ`
# 这类纯数字代码完全无法辨认，而 yfinance 只给英文名（"Hon Hai Precision Industry"），
# 对中文语境仍不够直观。只维护这一小张表（非美龙头），成本可忽略。
CN_NAMES: dict[str, str] = {
    "005930.KS": "三星电子",
    "000660.KS": "SK海力士",
    "300308.SZ": "中际旭创",
    "8035.T": "东京电子",
    "6857.T": "爱德万测试",
    "2317.TW": "鸿海精密",
    "TSM": "台积电",
    "ASML": "阿斯麦",
}


def display_name(symbol: str, raw_json: dict | None = None) -> str:
    """标的显示名：优先中文名表 → yfinance 的 shortName/longName → 代码本身。

    返回的是**纯名字**（不含代码），调用方自行决定怎么跟代码组合展示。
    """
    if symbol in CN_NAMES:
        return CN_NAMES[symbol]
    if raw_json and isinstance(raw_json, dict):
        for key in ("shortName", "longName"):
            v = raw_json.get(key)
            # 先 strip 再判空——纯空格的名字等同于没有，不能当有效值返回
            if isinstance(v, str) and v.strip():
                return v.strip()
    return symbol


def infer_currency(symbol: str) -> str:
    """根据交易所后缀推断币种。无后缀或未知后缀默认 USD。"""
    for suffix, ccy in SUFFIX_TO_CURRENCY.items():
        if symbol.endswith(suffix):
            return ccy
    return "USD"


def get_currency_for_symbol(
    symbol: str,
    raw_json: dict | None = None,
) -> str:
    """获取标的的交易币种。

    优先从 fundamentals 的 raw_json 里取 'currency' 字段（零成本、最准确），
    兜底按交易所后缀推断。
    """
    if raw_json and isinstance(raw_json, dict):
        ccy = raw_json.get("currency")
        if ccy and isinstance(ccy, str):
            return ccy.upper()
    return infer_currency(symbol)


def to_usd_market_cap(
    market_caps: dict[str, float | None],
    currencies: dict[str, str],
    fx_rates: dict[str, float],
) -> dict[str, float | None]:
    """本币市值 → 美元市值。

    currencies: symbol → 币种代码（如 'KRW', 'JPY', 'USD'）。
    fx_rates:   币种代码 → 1 USD 兑多少本币（如 KRW=1419 表示 1美元=1419韩元）。

    USD 标的直接返回原值；非 USD 标的做 市值_USD = 市值_本币 / 汇率。
    **缺汇率的返回 None**（不 fallback 成原值——那等于把韩元当美元，是灾难性错误）。
    """
    result: dict[str, float | None] = {}
    for symbol, cap in market_caps.items():
        if cap is None:
            result[symbol] = None
            continue
        ccy = currencies.get(symbol, "USD")
        if ccy == "USD":
            result[symbol] = cap
        else:
            rate = fx_rates.get(ccy)
            if rate is not None and rate > 0:
                result[symbol] = cap / rate
            else:
                # 缺汇率 → None，宁可显示"—"也不能把万亿韩元当万亿美元
                result[symbol] = None
    return result


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


def valuation_warning(forward_pe: float | None, ev_to_ebitda: float | None,
                      price_to_sales: float | None) -> str:
    """标出只适合回查原始数据的极端估值倍数，不擅自缩尾或删除。"""
    flags = []
    for label, value, threshold in (
        ("forward PE", forward_pe, 100.0),
        ("EV/EBITDA", ev_to_ebitda, 100.0),
        ("P/S", price_to_sales, 50.0),
    ):
        if value is not None and not _is_nan(value) and value > threshold:
            flags.append(f"{label}>{threshold:g}")
    return "⚠️ 待核实：" + "、".join(flags) if flags else ""


def valuation_history(fundamentals: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """返回公司的有效估值快照；不跨日期补空值。"""
    columns = ["forward_pe", "ev_to_ebitda", "price_to_sales"]
    if fundamentals.empty or "symbol" not in fundamentals.columns:
        return pd.DataFrame(columns=columns)
    work = fundamentals[fundamentals["symbol"] == symbol].copy()
    if work.empty:
        return pd.DataFrame(columns=columns)
    work["date"] = pd.to_datetime(work["date"])
    work = work.set_index("date").sort_index()
    available = [column for column in columns if column in work.columns]
    out = work[available].apply(pd.to_numeric, errors="coerce")
    # 负分母对应的倍数没有估值含义；保留极端正值供页面打标，不缩尾。
    out = out.where(out > 0)
    return out.dropna(how="all")


def point_in_time_ttm_valuation(
    fundamentals: pd.DataFrame, facts: pd.DataFrame, symbol: str,
) -> pd.DataFrame:
    """按每个基本面观察时点当时可见的季度快照重建 TTM 估值。

    每个观察点只选 ``captured_at`` 不晚于基本面快照的最新一版财报；四个季度
    必须连续且字段完整。财报币种与交易币种不一致时不换算，也不拿今天的数据回填。
    """
    columns = ["captured_at", "report_snapshot_at", "latest_period", "ttm_pe",
               "ttm_ev_ebitda", "ttm_ps"]
    if fundamentals.empty or facts.empty or "symbol" not in fundamentals.columns:
        return pd.DataFrame(columns=columns)
    fund = fundamentals[fundamentals["symbol"] == symbol].copy()
    if fund.empty or "captured_at" not in fund or "captured_at" not in facts:
        return pd.DataFrame(columns=columns)
    fund["_captured"] = pd.to_datetime(fund["captured_at"], utc=True, errors="coerce")
    work = facts.copy()
    work["_captured"] = pd.to_datetime(work["captured_at"], utc=True, errors="coerce")
    fund = fund.dropna(subset=["_captured"]).sort_values("_captured")
    work = work.dropna(subset=["_captured"])
    rows = []
    for _, observation in fund.iterrows():
        visible = work[work["_captured"] <= observation["_captured"]]
        if visible.empty:
            continue
        latest_capture = visible["_captured"].max()
        candidates = visible[visible["_captured"] == latest_capture]
        snapshot_id = candidates["snapshot_id"].iloc[-1]
        snapshot = candidates[candidates["snapshot_id"] == snapshot_id]
        financial_currency = snapshot["currency"].iloc[0]
        trading_currency = snapshot["trading_currency"].iloc[0]
        try:
            raw = json.loads(observation.get("raw_json") or "{}")
        except (TypeError, ValueError, json.JSONDecodeError):
            raw = {}
        market_currency = raw.get("currency") or trading_currency
        if (not financial_currency or not market_currency
                or str(financial_currency).upper() != str(market_currency).upper()):
            continue
        quarterly = snapshot.pivot_table(
            index="period_end", columns="metric", values="value",
            aggfunc="first", dropna=False,
        ).sort_index()
        quarterly.index = pd.to_datetime(quarterly.index)
        if quarterly.empty:
            continue
        dates = list(quarterly.index[-4:])
        if len(dates) != 4 or not all(
                70 <= (b - a).days <= 120 for a, b in zip(dates, dates[1:])):
            continue

        def total(metric: str) -> float | None:
            if metric not in quarterly:
                return None
            values = pd.to_numeric(quarterly.loc[dates, metric], errors="coerce")
            return float(values.sum()) if values.notna().all() else None

        market_cap = pd.to_numeric(observation.get("market_cap"), errors="coerce")
        enterprise_value = pd.to_numeric(raw.get("enterpriseValue"), errors="coerce")
        revenue, net_income, ebitda = total("revenue"), total("net_income"), total("ebitda")
        ttm_pe = (float(market_cap) / net_income
                  if pd.notna(market_cap) and net_income is not None and net_income > 0 else None)
        ttm_ps = (float(market_cap) / revenue
                  if pd.notna(market_cap) and revenue is not None and revenue > 0 else None)
        ttm_ev_ebitda = (float(enterprise_value) / ebitda
                         if pd.notna(enterprise_value) and ebitda is not None and ebitda > 0
                         else None)
        if ttm_pe is None and ttm_ps is None and ttm_ev_ebitda is None:
            continue
        rows.append({
            "date": pd.to_datetime(observation.get("date")),
            "captured_at": observation["captured_at"],
            "report_snapshot_at": snapshot["captured_at"].iloc[0],
            "latest_period": dates[-1].strftime("%Y-%m-%d"),
            "ttm_pe": ttm_pe,
            "ttm_ev_ebitda": ttm_ev_ebitda,
            "ttm_ps": ttm_ps,
        })
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows).set_index("date").sort_index()


def historical_percentile(series: pd.Series) -> dict:
    """当前值在已有快照中的百分位，并明确返回样本范围。"""
    values = pd.to_numeric(series, errors="coerce").dropna()
    values = values[values > 0]
    if values.empty:
        return {"percentile": None, "sample_count": 0, "start": None, "end": None}
    current = float(values.iloc[-1])
    percentile = float((values <= current).mean())
    index = pd.to_datetime(values.index)
    return {
        "percentile": percentile,
        "sample_count": int(len(values)),
        "start": index.min().strftime("%Y-%m-%d"),
        "end": index.max().strftime("%Y-%m-%d"),
    }


def peer_valuation_percentile(frame: pd.DataFrame) -> pd.Series:
    """同一赛道内按可用估值倍数计算便宜分位；倍数越低分越高。"""
    metrics = [column for column in ("forward PE", "EV/EBITDA", "P/S")
               if column in frame.columns]
    if not metrics:
        return pd.Series(index=frame.index, dtype=float)
    ranks = []
    for column in metrics:
        values = pd.to_numeric(frame[column], errors="coerce").where(lambda item: item > 0)
        ranks.append(values.rank(pct=True, ascending=False, method="average"))
    return pd.concat(ranks, axis=1).mean(axis=1, skipna=True)


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
