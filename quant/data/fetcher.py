"""yfinance 日线数据拉取：增量更新 + 失败重试。"""

import logging
import time
import uuid
from datetime import datetime, timedelta, timezone
from urllib.parse import quote

import pandas as pd
import yfinance as yf

from quant.data import store

log = logging.getLogger(__name__)

MAX_RETRIES = 3


def _to_float(value):
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

COLUMN_MAP = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adj_close",
    "Volume": "volume",
}


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=COLUMN_MAP)
    if "adj_close" not in df.columns:
        df["adj_close"] = df["close"]
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df[["open", "high", "low", "close", "adj_close", "volume"]]


def fetch_history(symbol: str, start: str) -> pd.DataFrame:
    """拉取 start 至今的日线，带指数退避重试。失败抛 RuntimeError。

    注意：yfinance 被限流时经常不抛异常而是静默返回空表，因此空表也按
    可重试处理；重试用尽仍为空才返回空表（真退市的代码就是这种表现）。"""
    last_err: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        df = None
        try:
            df = yf.download(symbol, start=start, auto_adjust=False, progress=False)
        except Exception as e:  # noqa: BLE001 - yfinance 抛的异常类型不稳定
            last_err = e
        if df is not None and not df.empty:
            return _normalize(df)
        if attempt < MAX_RETRIES:
            wait = 2**attempt
            log.warning("%s 第 %d 次拉取%s，%ds 后重试", symbol, attempt,
                        f"失败: {last_err}" if last_err else "返回空表（疑似限流）", wait)
            time.sleep(wait)
    if last_err is not None:
        raise RuntimeError(f"{symbol}: 拉取失败（重试 {MAX_RETRIES} 次）: {last_err}")
    return pd.DataFrame()


def update_symbol(conn, symbol: str, history_start: str, full: bool = False) -> int:
    """增量更新单个标的，返回写入行数。

    full=True 时忽略库内进度、从 history_start 全量重拉并覆盖：yfinance 的
    adj_close 以下载日为基准回溯复权，分红后历史行会整体变化，增量拼接会在
    衔接点留下微小错位，建议每季度全量刷新一次。

    增量起点是库内最新日期"本身"而非其后一天：盘中运行会存下当日的半根K线，
    从最新日期重拉可保证下次运行将其覆盖为收盘定稿（REPLACE 幂等）。"""
    latest = None if full else store.latest_price_date(conn, symbol)
    start = latest if latest else history_start
    df = fetch_history(symbol, start)
    if df.empty:
        if latest is None:
            # yfinance 拉取失败时常静默返回空表；首拉/全量拿到空必属异常
            raise RuntimeError(f"{symbol}: 拉取返回空数据（网络受限或代码无效？）")
        return 0
    return store.upsert_prices(conn, symbol, df)


def update_all(conn, symbols: list[str], history_start: str,
               full: bool = False) -> tuple[int, list[str]]:
    """更新全部标的。返回 (总写入行数, 失败标的列表)。"""
    total, failed = 0, []
    for symbol in symbols:
        try:
            n = update_symbol(conn, symbol, history_start, full=full)
            log.info("%s 更新 %d 行", symbol, n)
            total += n
        except Exception as e:  # noqa: BLE001
            log.error("%s 更新失败: %s", symbol, e)
            failed.append(symbol)
    return total, failed


# ---------- 基本面快照 ----------

# yfinance info 键 → 本地列名
_FUNDAMENTALS_MAP = {
    "trailingPE": "trailing_pe",
    "forwardPE": "forward_pe",
    "priceToBook": "price_to_book",
    "priceToSalesTrailing12Months": "price_to_sales",
    "enterpriseToEbitda": "ev_to_ebitda",
    "pegRatio": "peg_ratio",
    "dividendYield": "dividend_yield",
    "trailingEps": "trailing_eps",
    "returnOnEquity": "return_on_equity",
    "profitMargins": "profit_margins",
    "grossMargins": "gross_margins",
    "debtToEquity": "debt_to_equity",
    "marketCap": "market_cap",
    "bookValue": "book_value",
    "beta": "beta",
    # Yahoo 口径：revenueGrowth = 最新季度营收同比（不是年报同比）；
    # earningsGrowth = 最新季度盈利同比。比年报口径新 6~12 个月，见 AI 基建页说明。
    "revenueGrowth": "revenue_growth",
    "earningsGrowth": "earnings_growth",
}


def fetch_fundamentals(symbol: str) -> dict | None:
    """拉取 yfinance ticker.info 全量数据，带指数退避重试。

    返回 {"metrics": {抽取列}, "raw": {完整 info}} 或 None（重试用尽仍失败）。"""
    last_err: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        info = None
        try:
            info = yf.Ticker(symbol).info
        except Exception as e:  # noqa: BLE001 - yfinance 抛的异常类型不稳定
            last_err = e
        if info:
            metrics = {local: info.get(yf_key) for yf_key, local in _FUNDAMENTALS_MAP.items()}
            return {"metrics": metrics, "raw": info}
        if attempt < MAX_RETRIES:
            wait = 2 ** attempt
            log.warning("%s 基本面第 %d 次拉取%s，%ds 后重试", symbol, attempt,
                        f"失败: {last_err}" if last_err else "返回空（疑似限流）", wait)
            time.sleep(wait)
    log.error("%s 基本面拉取失败（重试 %d 次）: %s", symbol, MAX_RETRIES,
              last_err or "info 为空")
    return None


def update_fundamentals(conn, symbols: list[str], as_of_date: str,
                        stale_days: int = 7, report: dict | None = None) -> tuple[int, list[str]]:
    """批量更新基本面快照。每个 symbol 若最近 stale_days 天内已有记录则跳过（自限流）。

    返回 (成功数, 失败列表)。单个失败只记日志，不中断批量。"""
    from datetime import datetime, timedelta, timezone
    cutoff = (datetime.strptime(as_of_date, "%Y-%m-%d") - timedelta(days=stale_days)).strftime("%Y-%m-%d")
    ok_count, failed = 0, []
    for symbol in symbols:
        latest = store.latest_fundamentals_date(conn, symbol)
        if latest and latest >= cutoff:
            log.debug("%s 基本面已是最新（%s），跳过", symbol, latest)
            if report is not None:
                report[symbol] = ("skipped", f"最近快照 {latest}")
            continue
        result = fetch_fundamentals(symbol)
        if result is None:
            log.error("%s 基本面抓取失败，跳过", symbol)
            failed.append(symbol)
            if report is not None:
                report[symbol] = ("failed", "供应商返回空或请求失败")
            continue
        captured_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        store.upsert_fundamentals(conn, symbol, as_of_date, captured_at,
                                  result["metrics"], result["raw"])
        log.info("%s 基本面快照已更新 (date=%s)", symbol, as_of_date)
        ok_count += 1
        if report is not None:
            report[symbol] = ("updated", f"快照 {as_of_date}")
    return ok_count, failed


# ---------- 年度财报（financials 表，供 AI 基建页增长指标用）----------

def fetch_financials(symbol: str) -> pd.DataFrame | None:
    """拉取 yfinance 年度利润表（income_stmt），带指数退避重试。

    返回 DataFrame（行=指标，列=财年结束日）或 None（重试用尽仍失败）。"""
    last_err: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        stmt = None
        try:
            ticker = yf.Ticker(symbol)
            stmt = ticker.income_stmt
        except Exception as e:  # noqa: BLE001 - yfinance 抛的异常类型不稳定
            last_err = e
        if stmt is not None and not stmt.empty:
            fast_info = ticker.fast_info
            trading_currency = fast_info.get("currency") if fast_info is not None else None
            try:
                info = ticker.info or {}
            except Exception:  # noqa: BLE001
                info = {}
            stmt.attrs.update({
                "currency": info.get("financialCurrency") or trading_currency,
                "trading_currency": trading_currency,
                "source": "Yahoo Finance via yfinance",
                "source_url": f"https://finance.yahoo.com/quote/{quote(symbol, safe='')}/financials",
            })
            return stmt
        if attempt < MAX_RETRIES:
            wait = 2 ** attempt
            log.warning("%s 财报第 %d 次拉取%s，%ds 后重试", symbol, attempt,
                        f"失败: {last_err}" if last_err else "返回空（疑似限流）", wait)
            time.sleep(wait)
    log.error("%s 财报拉取失败（重试 %d 次）: %s", symbol, MAX_RETRIES,
              last_err or "income_stmt 为空")
    return None


def update_financials(conn, symbols: list[str],
                      stale_days: int = 30, report: dict | None = None) -> tuple[int, list[str]]:
    """批量更新年度财报。每个 symbol 若最近 stale_days 天内已有拉取记录则跳过（自限流）。

    返回 (成功数, 失败列表)。单个失败只记日志，不中断批量。"""
    from datetime import datetime, timedelta, timezone
    cutoff = (datetime.now(timezone.utc) - timedelta(days=stale_days)).isoformat(timespec="seconds")
    ok_count, failed = 0, []
    for symbol in symbols:
        latest = store.latest_financial_date(conn, symbol)
        if latest and latest >= cutoff:
            log.debug("%s 财报已是最新（captured_at=%s），跳过", symbol, latest)
            if report is not None:
                report[symbol] = ("skipped", f"最近抓取 {latest}")
            continue
        stmt = fetch_financials(symbol)
        if stmt is None:
            log.error("%s 财报抓取失败，跳过", symbol)
            failed.append(symbol)
            if report is not None:
                report[symbol] = ("failed", "供应商返回空或请求失败")
            continue
        captured_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        rows = []
        for col in stmt.columns:
            fiscal_date = col.strftime("%Y-%m-%d") if hasattr(col, 'strftime') else str(col)
            revenue = stmt.at["Total Revenue", col] if "Total Revenue" in stmt.index else None
            gross_profit = stmt.at["Gross Profit", col] if "Gross Profit" in stmt.index else None
            operating_income = stmt.at["Operating Income", col] if "Operating Income" in stmt.index else None
            net_income = stmt.at["Net Income", col] if "Net Income" in stmt.index else None
            # Convert numpy types to Python native for SQLite
            def _to_float(v):
                if v is None or (hasattr(v, '__class__') and v.__class__.__name__ == 'NaTType'):
                    return None
                try:
                    import math
                    f = float(v)
                    return None if math.isnan(f) else f
                except (TypeError, ValueError):
                    return None
            rows.append((
                fiscal_date, _to_float(revenue), _to_float(gross_profit),
                _to_float(operating_income), _to_float(net_income),
                stmt.attrs.get("currency"), stmt.attrs.get("trading_currency"),
                stmt.attrs.get("source"), stmt.attrs.get("source_url"), captured_at,
            ))
        if rows:
            store.upsert_financials(conn, symbol, rows)
            log.info("%s 财报已更新（%d 期）", symbol, len(rows))
            ok_count += 1
            if report is not None:
                report[symbol] = ("updated", f"写入 {len(rows)} 期")
        elif report is not None:
            report[symbol] = ("failed", "财报没有可写入的报告期")
            failed.append(symbol)
    return ok_count, failed


# ---------- 季度三表（阶段 2 样本验证）----------

QUARTERLY_METRICS = {
    "income": {
        "revenue": ("Total Revenue",),
        "gross_profit": ("Gross Profit",),
        "operating_income": ("Operating Income",),
        "net_income": ("Net Income", "Net Income Common Stockholders"),
        "ebitda": ("EBITDA", "Normalized EBITDA"),
        # REIT 专用口径只接受供应商直接报告的 FFO/AFFO；不从普通 FCF 推导。
        "funds_from_operations": ("Funds From Operations",),
        "adjusted_funds_from_operations": ("Adjusted Funds From Operations",),
    },
    "balance": {
        "cash": ("Cash Cash Equivalents And Short Term Investments",
                 "Cash And Cash Equivalents"),
        "total_debt": ("Total Debt",),
    },
    "cashflow": {
        "operating_cash_flow": ("Operating Cash Flow",),
        "capital_expenditure": ("Capital Expenditure",),
        "free_cash_flow": ("Free Cash Flow",),
    },
}


def fetch_quarterly_statements(symbol: str) -> dict | None:
    """拉取 Yahoo 季度三表；至少一张表有数据即返回，其余缺失保持为空。"""
    last_err: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            ticker = yf.Ticker(symbol)
            frames = {
                "income": ticker.quarterly_income_stmt,
                "balance": ticker.quarterly_balance_sheet,
                "cashflow": ticker.quarterly_cashflow,
            }
            if any(df is not None and not df.empty for df in frames.values()):
                fast_info = ticker.fast_info
                trading_currency = (fast_info.get("currency")
                                    if fast_info is not None else None)
                # yfinance 的表格数值使用财报币种，不一定是证券交易币种。
                # 典型例子：TSM ADR 用 USD 交易，但合并财报表格是 TWD。
                try:
                    info = ticker.info or {}
                except Exception:  # noqa: BLE001 - 币种补充失败不应丢掉三表
                    info = {}
                financial_currency = info.get("financialCurrency") or trading_currency
                return {
                    "frames": frames,
                    "currency": financial_currency,
                    "trading_currency": trading_currency,
                }
        except Exception as exc:  # noqa: BLE001
            last_err = exc
        if attempt < MAX_RETRIES:
            wait = 2 ** attempt
            log.warning("%s 季度三表第 %d 次拉取失败，%ds 后重试: %s",
                        symbol, attempt, wait, last_err or "返回空表")
            time.sleep(wait)
    log.error("%s 季度三表拉取失败（重试 %d 次）: %s", symbol, MAX_RETRIES,
              last_err or "三张表均为空")
    return None


def normalize_quarterly_facts(result: dict) -> list[dict]:
    """把 yfinance 的宽表转成长表事实；保留空值，避免后续跨版本补值。"""
    facts = []
    currency = result.get("currency") or "UNKNOWN"
    for statement, metric_map in QUARTERLY_METRICS.items():
        frame = result["frames"].get(statement)
        if frame is None or frame.empty:
            continue
        for metric, aliases in metric_map.items():
            raw_label = next((label for label in aliases if label in frame.index), None)
            if raw_label is None:
                continue
            for period in frame.columns:
                period_end = pd.Timestamp(period).strftime("%Y-%m-%d")
                facts.append({
                    "period_start": None,
                    "period_end": period_end,
                    "statement": statement,
                    "metric": metric,
                    "value": _to_float(frame.at[raw_label, period]),
                    "unit": currency,
                    "raw_label": raw_label,
                })
    return facts


def update_quarterly_financials(conn, symbols: list[str], stale_days: int = 7,
                                report: dict | None = None) -> tuple[int, list[str]]:
    """保存季度三表版本快照；不会覆盖之前抓取到的版本。"""
    cutoff = (datetime.now(timezone.utc) - timedelta(days=stale_days)).isoformat()
    ok_count, failed = 0, []
    for symbol in symbols:
        latest = store.latest_statement_capture(conn, symbol, "quarterly")
        if latest and latest >= cutoff:
            if report is not None:
                report[symbol] = ("skipped", f"最近抓取 {latest}")
            continue
        result = fetch_quarterly_statements(symbol)
        if result is None:
            failed.append(symbol)
            if report is not None:
                report[symbol] = ("failed", "季度三表均为空或请求失败")
            continue
        facts = normalize_quarterly_facts(result)
        if not facts:
            failed.append(symbol)
            if report is not None:
                report[symbol] = ("failed", "未找到已映射的季度字段")
            continue
        captured_at = datetime.now(timezone.utc).isoformat(timespec="microseconds")
        metadata = {
            "snapshot_id": uuid.uuid4().hex,
            "symbol": symbol,
            "frequency": "quarterly",
            "currency": result.get("currency"),
            "trading_currency": result.get("trading_currency"),
            "source": "Yahoo Finance via yfinance",
            "source_url": f"https://finance.yahoo.com/quote/{quote(symbol, safe='')}/financials",
            "published_at": None,
            "captured_at": captured_at,
        }
        count = store.insert_financial_snapshot(conn, metadata, facts)
        ok_count += 1
        if report is not None:
            periods = len({fact["period_end"] for fact in facts})
            report[symbol] = ("updated", f"写入 {periods} 期 / {count} 个事实")
        log.info("%s 季度三表已更新（%d 个事实）", symbol, count)
    return ok_count, failed
