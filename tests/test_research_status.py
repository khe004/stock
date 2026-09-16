from datetime import date

from quant.analysis.research_status import research_status
from quant.data import fetcher, store


def test_status_shows_missing_stale_and_last_failure():
    conn = store.connect(":memory:")
    store.upsert_fundamentals(conn, "NVDA", "2026-09-01", "2026-09-01T00:00:00+00:00",
                              {"forward_pe": 20}, {})
    store.record_research_update(conn, "NVDA", "fundamentals", "failed", "供应商返回空")
    status = research_status(conn, ["NVDA", "AMKR"], date(2026, 9, 15)).set_index("代码")
    assert status.at["NVDA", "基本面日期状态"] == "过期"
    assert status.at["NVDA", "基本面更新结果"] == "failed"
    assert status.at["AMKR", "基本面日期状态"] == "缺失"
    assert status.at["AMKR", "财报抓取时间状态"] == "缺失"


def test_force_fundamentals_rechecks_existing_today(monkeypatch):
    conn = store.connect(":memory:")
    store.upsert_fundamentals(conn, "NVDA", "2026-09-15", "t1", {"forward_pe": 20}, {})
    called = []

    def fake_fetch(symbol):
        called.append(symbol)
        return {"metrics": {"forward_pe": 21}, "raw": {}}

    monkeypatch.setattr(fetcher, "fetch_fundamentals", fake_fetch)
    report = {}
    fetcher.update_fundamentals(conn, ["NVDA"], "2026-09-15", stale_days=-1, report=report)
    assert called == ["NVDA"]
    assert report["NVDA"][0] == "updated"
