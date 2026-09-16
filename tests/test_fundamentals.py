import json
import pandas as pd
import pytest
from datetime import datetime, timezone

import quant.data.fetcher as fetcher
from quant.data import store
from quant.config import load_config


# ---------- fixture ----------

def make_conn():
    return store.connect(":memory:")


def test_research_symbols_include_sp500_and_ai_pool():
    """研究数据刷新名单覆盖策略候选池之外的 AI 基建公司。"""
    cfg = load_config()
    symbols = set(cfg.research_symbols)
    assert {"NVDA", "AMKR", "005930.KS", "300308.SZ"} <= symbols
    assert set(cfg.ai_infra_symbols) >= {"AMKR", "TSM", "005930.KS"}


FAKE_INFO = {
    "trailingPE": 25.5,
    "forwardPE": 22.1,
    "priceToBook": 3.2,
    "priceToSalesTrailing12Months": 5.0,
    "enterpriseToEbitda": 18.7,
    "pegRatio": 1.3,
    "dividendYield": 0.015,
    "trailingEps": 6.8,
    "returnOnEquity": 0.35,
    "profitMargins": 0.25,
    "grossMargins": 0.45,
    "debtToEquity": 120.0,
    "marketCap": 2_800_000_000_000,
    "bookValue": 22.5,
    "beta": 1.1,
    "shortName": "Apple Inc.",
    "sector": "Technology",
}


# ---------- store 测试 ----------

def test_fundamentals_upsert_and_load():
    """upsert → load 往返：抽取列与 raw_json 均能正确存取。"""
    conn = make_conn()
    metrics = {"trailing_pe": 25.5, "forward_pe": 22.1, "price_to_book": 3.2}
    raw = {"trailingPE": 25.5, "shortName": "Apple Inc."}
    store.upsert_fundamentals(conn, "AAPL", "2026-07-18",
                              "2026-07-18T12:00:00+00:00", metrics, raw)
    df = store.load_fundamentals(conn, symbol="AAPL")
    assert len(df) == 1
    assert df.iloc[0]["trailing_pe"] == 25.5
    assert df.iloc[0]["forward_pe"] == 22.1
    parsed = json.loads(df.iloc[0]["raw_json"])
    assert parsed["shortName"] == "Apple Inc."


def test_fundamentals_upsert_idempotent():
    """同一 (symbol, date) 重复 upsert 应覆盖而非报错。"""
    conn = make_conn()
    m1 = {"trailing_pe": 25.5}
    m2 = {"trailing_pe": 30.0}
    store.upsert_fundamentals(conn, "AAPL", "2026-07-18", "t1", m1, {})
    store.upsert_fundamentals(conn, "AAPL", "2026-07-18", "t2", m2, {})
    df = store.load_fundamentals(conn, symbol="AAPL")
    assert len(df) == 1
    assert df.iloc[0]["trailing_pe"] == 30.0


def test_latest_fundamentals_date():
    conn = make_conn()
    assert store.latest_fundamentals_date(conn, "AAPL") is None
    store.upsert_fundamentals(conn, "AAPL", "2026-07-10", "t", {}, {})
    store.upsert_fundamentals(conn, "AAPL", "2026-07-18", "t", {}, {})
    assert store.latest_fundamentals_date(conn, "AAPL") == "2026-07-18"
    assert store.latest_fundamentals_date(conn, "MSFT") is None


def test_load_fundamentals_filters():
    """symbol 与 start 过滤参数。"""
    conn = make_conn()
    store.upsert_fundamentals(conn, "AAPL", "2026-07-10", "t", {}, {})
    store.upsert_fundamentals(conn, "AAPL", "2026-07-18", "t", {}, {})
    store.upsert_fundamentals(conn, "MSFT", "2026-07-15", "t", {}, {})
    # 全部
    assert len(store.load_fundamentals(conn)) == 3
    # 按 symbol
    assert len(store.load_fundamentals(conn, symbol="AAPL")) == 2
    # 按 start
    assert len(store.load_fundamentals(conn, start="2026-07-15")) == 2


def test_load_latest_fundamentals_keeps_latest_row_nulls():
    """最新快照按整行读取，不能用旧快照的非空字段拼接。"""
    conn = make_conn()
    store.upsert_fundamentals(conn, "AAPL", "2026-07-17", "t1",
                               {"forward_pe": 20.0, "market_cap": 100.0}, {})
    store.upsert_fundamentals(conn, "AAPL", "2026-07-18", "t2",
                               {"forward_pe": None, "market_cap": 110.0}, {})
    latest = store.load_latest_fundamentals(conn)
    assert len(latest) == 1
    row = latest.iloc[0]
    assert row["date"] == "2026-07-18"
    assert pd.isna(row["forward_pe"])
    assert row["market_cap"] == 110.0


# ---------- fetcher 测试 ----------

@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(fetcher.time, "sleep", lambda *_: None)


def test_fetch_fundamentals_extracts_metrics(monkeypatch):
    """正常返回时应正确映射抽取列，且 raw 包含完整 info。"""
    class FakeTicker:
        def __init__(self, symbol):
            self.info = FAKE_INFO

    monkeypatch.setattr(fetcher.yf, "Ticker", FakeTicker)
    result = fetcher.fetch_fundamentals("AAPL")
    assert result is not None
    assert result["metrics"]["trailing_pe"] == 25.5
    assert result["metrics"]["market_cap"] == 2_800_000_000_000
    assert result["metrics"]["beta"] == 1.1
    assert result["raw"]["shortName"] == "Apple Inc."
    assert result["raw"]["sector"] == "Technology"


def test_fetch_fundamentals_missing_keys(monkeypatch):
    """info 中缺少某些键时，对应 metrics 值应为 None（不报错）。"""
    class FakeTicker:
        def __init__(self, symbol):
            self.info = {"shortName": "Test", "trailingPE": 10.0}

    monkeypatch.setattr(fetcher.yf, "Ticker", FakeTicker)
    result = fetcher.fetch_fundamentals("TEST")
    assert result is not None
    assert result["metrics"]["trailing_pe"] == 10.0
    assert result["metrics"]["forward_pe"] is None
    assert result["metrics"]["market_cap"] is None


def test_fetch_fundamentals_empty_info_retries(monkeypatch):
    """info 为空/None 应触发重试。"""
    calls = []

    class FakeTicker:
        def __init__(self, symbol):
            calls.append(symbol)
            self.info = FAKE_INFO if len(calls) >= 3 else {}

    monkeypatch.setattr(fetcher.yf, "Ticker", FakeTicker)
    result = fetcher.fetch_fundamentals("AAPL")
    assert len(calls) == 3, "前两次空 info 应触发重试"
    assert result is not None


def test_fetch_fundamentals_persistent_failure_returns_none(monkeypatch):
    """重试用尽仍失败返回 None（不抛异常）。"""
    class FakeTicker:
        def __init__(self, symbol):
            raise ConnectionError("network error")

    monkeypatch.setattr(fetcher.yf, "Ticker", FakeTicker)
    result = fetcher.fetch_fundamentals("DEAD")
    assert result is None


def test_update_fundamentals_stale_days_skip(monkeypatch):
    """最近 stale_days 天内已有记录的 symbol 应跳过。"""
    conn = make_conn()
    # 预先插入一条 7 天前的记录
    store.upsert_fundamentals(conn, "AAPL", "2026-07-15", "t", {"trailing_pe": 20.0}, {})

    fetch_calls = []

    class FakeTicker:
        def __init__(self, symbol):
            fetch_calls.append(symbol)
            self.info = FAKE_INFO

    monkeypatch.setattr(fetcher.yf, "Ticker", FakeTicker)

    # as_of_date = 2026-07-18，stale_days=7 → cutoff = 2026-07-11
    # AAPL latest=2026-07-15 >= cutoff → 跳过
    # MSFT 无记录 → 抓取
    ok, failed = fetcher.update_fundamentals(conn, ["AAPL", "MSFT"], "2026-07-18", stale_days=7)
    assert ok == 1
    assert failed == []
    assert fetch_calls == ["MSFT"]


def test_update_fundamentals_failure_continues(monkeypatch):
    """单个 symbol 失败不中断批量，记入 failed 列表。"""
    conn = make_conn()
    calls = []

    class FakeTicker:
        def __init__(self, symbol):
            calls.append(symbol)
            if symbol == "BAD":
                raise ConnectionError("fail")
            self.info = FAKE_INFO

    monkeypatch.setattr(fetcher.yf, "Ticker", FakeTicker)
    ok, failed = fetcher.update_fundamentals(conn, ["AAPL", "BAD", "MSFT"], "2026-07-18")
    assert ok == 2
    assert "BAD" in failed
    # AAPL 和 MSFT 都应成功入库
    df = store.load_fundamentals(conn)
    assert set(df["symbol"]) == {"AAPL", "MSFT"}


def test_annual_financials_keep_statement_and_trading_currencies(monkeypatch):
    periods = pd.to_datetime(["2025-12-31"])
    statement = pd.DataFrame(
        [[100.0], [60.0], [30.0], [20.0]],
        index=["Total Revenue", "Gross Profit", "Operating Income", "Net Income"],
        columns=periods,
    )

    class FakeTicker:
        def __init__(self, symbol):
            self.income_stmt = statement.copy()
            self.fast_info = {"currency": "USD"}
            self.info = {"financialCurrency": "TWD"}

    monkeypatch.setattr(fetcher.yf, "Ticker", FakeTicker)
    conn = make_conn()
    ok, failed = fetcher.update_financials(conn, ["TSM"], stale_days=-1)
    assert (ok, failed) == (1, [])
    saved = store.load_financials(conn, "TSM").iloc[0]
    assert saved["currency"] == "TWD"
    assert saved["trading_currency"] == "USD"
    assert saved["source"] == "Yahoo Finance via yfinance"
