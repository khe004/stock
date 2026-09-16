"""AI 研究数据状态；只依据本地记录，不推断供应商是否有新财报。"""

from datetime import date, datetime, timedelta, timezone

import pandas as pd

from quant.data import store


def research_status(conn, symbols: list[str], today: date | None = None) -> pd.DataFrame:
    today = today or date.today()
    rows = {s: {"代码": s, "行情日期": None, "基本面日期": None,
                "财报抓取时间": None} for s in symbols}
    if not symbols:
        return pd.DataFrame()
    placeholders = ",".join("?" for _ in symbols)
    for table, date_col, label in (
        ("prices", "date", "行情日期"),
        ("fundamentals", "date", "基本面日期"),
        ("financials", "captured_at", "财报抓取时间"),
    ):
        query = f"SELECT symbol, MAX({date_col}) AS last_date FROM {table} WHERE symbol IN ({placeholders}) GROUP BY symbol"
        for row in conn.execute(query, symbols):
            rows[row["symbol"]][label] = row["last_date"]
    updates = store.load_research_updates(conn, symbols)
    for item in updates.itertuples(index=False):
        label = "基本面" if item.data_type == "fundamentals" else "财报"
        rows[item.symbol][f"{label}最近检查"] = item.checked_at
        rows[item.symbol][f"{label}更新结果"] = item.status
        rows[item.symbol][f"{label}说明"] = item.detail
    result = pd.DataFrame([rows[s] for s in symbols])
    price_cutoff = (today - timedelta(days=7)).isoformat()
    fund_cutoff = (today - timedelta(days=7)).isoformat()
    fin_cutoff = (datetime.combine(today, datetime.min.time(), timezone.utc)
                  - timedelta(days=30)).isoformat()
    for label, cutoff in (("行情日期", price_cutoff), ("基本面日期", fund_cutoff),
                          ("财报抓取时间", fin_cutoff)):
        result[label + "状态"] = result[label].apply(
            lambda v: "缺失" if pd.isna(v) else ("过期" if str(v) < cutoff else "正常"))
    return result
