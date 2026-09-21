import pandas as pd
import pytest
import run_daily

from quant.analysis.quarterly import compute_quarterly_metrics, describe_quarterly_change
from quant.analysis.quarterly_validation import (
    OFFICIAL_FALLBACK_SOURCE,
    apply_official_quarterly_fallbacks,
    official_baseline_facts,
    validate_quarterly_sample,
)
from quant.config import load_config
from quant.data import fetcher, store


def _metadata(snapshot_id, captured_at):
    return {
        "snapshot_id": snapshot_id, "symbol": "NVDA", "frequency": "quarterly",
        "currency": "TWD", "trading_currency": "USD",
        "source": "test", "source_url": "https://example.test",
        "published_at": None, "captured_at": captured_at,
    }


def _fact(metric, value, period="2026-06-30", statement="income"):
    return {"period_start": None, "period_end": period, "statement": statement,
            "metric": metric, "value": value, "unit": "USD", "raw_label": metric}


def test_financial_snapshots_keep_versions_and_latest_nulls():
    conn = store.connect(":memory:")
    store.insert_financial_snapshot(conn, _metadata("v1", "2026-09-14T00:00:00+00:00"),
                                    [_fact("revenue", 100), _fact("gross_profit", 60)])
    store.insert_financial_snapshot(conn, _metadata("v2", "2026-09-15T00:00:00+00:00"),
                                    [_fact("revenue", None), _fact("gross_profit", 70)])
    latest = store.load_latest_financial_facts(conn, "NVDA")
    assert set(latest["snapshot_id"]) == {"v2"}
    assert latest["currency"].iloc[0] == "TWD"
    assert latest["trading_currency"].iloc[0] == "USD"
    assert pd.isna(latest.loc[latest["metric"] == "revenue", "value"].iloc[0])
    assert conn.execute("SELECT COUNT(*) FROM financial_statement_snapshots").fetchone()[0] == 2


def test_fetch_and_normalize_quarterly_statements(monkeypatch):
    periods = pd.to_datetime(["2026-06-30", "2026-03-31"])
    income = pd.DataFrame([[120, 100], [72, 55], [31, 27], [20, 18]],
                          index=["Total Revenue", "Gross Profit", "Normalized EBITDA",
                                 "Funds From Operations"],
                          columns=periods)
    balance = pd.DataFrame([[40, 35], [20, 22]],
                           index=["Cash And Cash Equivalents", "Total Debt"], columns=periods)
    cashflow = pd.DataFrame([[30, 25], [-10, -8]],
                            index=["Operating Cash Flow", "Capital Expenditure"], columns=periods)

    class FakeTicker:
        def __init__(self, symbol):
            self.quarterly_income_stmt = income
            self.quarterly_balance_sheet = balance
            self.quarterly_cashflow = cashflow
            self.fast_info = {"currency": "USD"}
            self.info = {"financialCurrency": "TWD"}

    monkeypatch.setattr(fetcher.yf, "Ticker", FakeTicker)
    result = fetcher.fetch_quarterly_statements("NVDA")
    facts = fetcher.normalize_quarterly_facts(result)
    assert result["currency"] == "TWD"
    assert result["trading_currency"] == "USD"
    assert any(f["metric"] == "revenue" and f["value"] == 120 for f in facts)
    assert any(f["metric"] == "ebitda" and f["value"] == 31 for f in facts)
    assert any(f["metric"] == "funds_from_operations" and f["value"] == 20
               for f in facts)
    assert any(f["metric"] == "cash" and f["raw_label"] == "Cash And Cash Equivalents"
               for f in facts)


def test_quarterly_metrics_yoy_ttm_and_derived_fcf():
    index = ["2025-03-31", "2025-06-30", "2025-09-30", "2025-12-31", "2026-03-31"]
    frame = pd.DataFrame({
        "revenue": [100, 110, 120, 130, 150],
        "gross_profit": [50, 57, 63, 70, 83],
        "operating_income": [10, 12, 14, 16, 21],
        "operating_cash_flow": [20, 21, 22, 23, 30],
        "capital_expenditure": [-5, -6, -7, -8, -10],
        "cash": [30, 31, 32, 33, 40],
        "total_debt": [20, 20, 19, 19, 18],
    }, index=index)
    metrics = compute_quarterly_metrics(frame)
    assert metrics.latest_period == "2026-03-31"
    assert metrics.revenue_yoy == pytest.approx(0.50)
    assert metrics.free_cash_flow == pytest.approx(20)
    assert metrics.ttm_revenue == pytest.approx(510)
    assert metrics.ttm_free_cash_flow == pytest.approx((21 + 22 + 23 + 30) - (6 + 7 + 8 + 10))
    assert "营收同比 +50.0%" in describe_quarterly_change(metrics)


def test_missing_quarter_does_not_form_ttm():
    frame = pd.DataFrame({"revenue": [100, 110, 130, 150]},
                         index=["2025-03-31", "2025-06-30", "2025-12-31", "2026-03-31"])
    assert compute_quarterly_metrics(frame).ttm_revenue is None


def test_empty_newest_provider_period_is_flagged_not_silently_hidden():
    frame = pd.DataFrame(
        {"revenue": [100.0, None], "operating_income": [20.0, None]},
        index=["2026-03-31", "2026-06-30"],
    )
    metrics = compute_quarterly_metrics(frame)
    assert metrics.reported_latest_period == "2026-06-30"
    assert metrics.latest_period == "2026-03-31"
    assert metrics.latest_period_incomplete is True


def test_quarterly_sample_is_configured():
    assert load_config().quarterly_research_symbols == [
        "NVDA", "MU", "VRT", "MSFT", "DLR", "TSM", "005930.KS", "ARM"]


def test_official_baseline_validation_catches_currency_missing_period_and_value():
    frame = pd.DataFrame(
        {"revenue": [99.0], "net_income": [20.0]}, index=["2026-03-31"])
    frame.attrs["currency"] = "USD"
    baseline = {
        "period_end": "2026-06-30", "currency": "TWD",
        "metrics": {"revenue": 100.0},
    }
    problems = validate_quarterly_sample(frame, baseline)
    assert any("财报币种" in item for item in problems)
    assert any("缺少官方最新报告期" in item for item in problems)

    good = pd.DataFrame({"revenue": [100.1]}, index=["2026-06-30"])
    good.attrs["currency"] = "TWD"
    assert validate_quarterly_sample(good, baseline) == []

    normalized = pd.DataFrame({"revenue": [100.0]}, index=["2026-06-30"])
    normalized.attrs["currency"] = "TWD"
    month_end_baseline = dict(baseline, period_end="2026-06-28",
                              provider_period_end="2026-06-30")
    assert validate_quarterly_sample(normalized, month_end_baseline) == []


def test_official_baseline_facts_only_imports_verified_fields():
    baseline = {
        "period_end": "2026-06-30", "currency": "USD",
        "metrics": {"revenue": 100.0, "operating_cash_flow": 20.0},
    }
    facts = official_baseline_facts(baseline)
    assert {item["metric"] for item in facts} == {"revenue", "operating_cash_flow"}
    assert {item["statement"] for item in facts} == {"income", "cashflow"}
    assert all(item["raw_label"] == "official_verified_baseline" for item in facts)


def test_official_fallback_is_minimal_and_idempotent():
    conn = store.connect(":memory:")
    baseline = {
        "period_end": "2026-06-30", "published_at": "2026-07-23",
        "currency": "USD", "trading_currency": "USD",
        "source_url": "https://example.test/official",
        "metrics": {"revenue": 100.0, "net_income": 20.0},
    }
    first = apply_official_quarterly_fallbacks(
        conn, {"DLR": baseline}, captured_at="2026-09-17T10:00:00+00:00")
    assert first["DLR"][0] == "verified_fallback"
    frame = store.load_latest_quarterly_financials(conn, "DLR")
    assert list(frame.index) == ["2026-06-30"]
    assert set(frame.columns) == {"revenue", "net_income"}
    assert frame.attrs["source"] == OFFICIAL_FALLBACK_SOURCE
    assert frame.attrs["published_at"] == "2026-07-23"
    assert frame.attrs["trading_currency"] == "USD"

    second = apply_official_quarterly_fallbacks(
        conn, {"DLR": baseline}, captured_at="2026-09-17T11:00:00+00:00")
    assert second["DLR"][0] == "verified"
    count = conn.execute(
        "SELECT COUNT(*) FROM financial_statement_snapshots WHERE symbol = 'DLR'"
    ).fetchone()[0]
    assert count == 1


def test_official_fallback_does_not_overwrite_currency_or_value_conflict():
    conn = store.connect(":memory:")
    store.insert_financial_snapshot(
        conn,
        {**_metadata("provider", "2026-09-17T00:00:00+00:00"),
         "symbol": "NVDA", "currency": "USD"},
        [_fact("revenue", 90.0)],
    )
    baseline = {
        "period_end": "2026-06-30", "currency": "USD",
        "source_url": "https://example.test/official",
        "metrics": {"revenue": 100.0},
    }
    result = apply_official_quarterly_fallbacks(conn, {"NVDA": baseline})
    assert result["NVDA"][0] == "validation_failed"
    latest = store.load_latest_quarterly_financials(conn, "NVDA")
    assert latest.loc["2026-06-30", "revenue"] == 90.0
    assert conn.execute(
        "SELECT COUNT(*) FROM financial_statement_snapshots"
    ).fetchone()[0] == 1


def test_latest_statement_metadata_returns_one_source_per_symbol():
    conn = store.connect(":memory:")
    store.insert_financial_snapshot(
        conn, _metadata("old", "2026-09-15T00:00:00+00:00"), [_fact("revenue", 90)])
    store.insert_financial_snapshot(
        conn, _metadata("new", "2026-09-16T00:00:00+00:00"), [_fact("revenue", 100)])
    metadata = store.load_latest_statement_metadata(conn, ["NVDA"])
    assert metadata[["symbol", "snapshot_id"]].to_dict("records") == [
        {"symbol": "NVDA", "snapshot_id": "new"}]


def test_quarterly_update_uses_official_fallback_after_provider_failure(monkeypatch):
    conn = store.connect(":memory:")

    def fake_provider(_conn, symbols, stale_days=7, report=None):
        report.update({symbol: ("failed", "供应商无数据") for symbol in symbols})
        return 0, list(symbols)

    monkeypatch.setattr(run_daily.fetcher, "update_quarterly_financials", fake_provider)
    monkeypatch.setattr(run_daily, "load_official_baselines", lambda _path: {"DLR": {}})
    monkeypatch.setattr(
        run_daily, "apply_official_quarterly_fallbacks",
        lambda _conn, _baselines, symbols: {
            "DLR": ("verified_fallback", "导入官方 2026-06-30 的 2 个已核验字段")},
    )
    report = {}
    ok, failed = run_daily.update_quarterly_research(conn, ["DLR"], report=report)
    assert ok == 1
    assert failed == []
    assert report["DLR"][0] == "verified_fallback"
