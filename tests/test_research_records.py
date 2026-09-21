import pytest
import pandas as pd

from quant.analysis.business_evidence import (
    audit_business_evidence, extract_evidence_candidates,
)
from quant.data import store


def test_research_notes_append_versions_without_overwrite():
    conn = store.connect(":memory:")
    first = store.save_research_note(
        conn, "NVDA", "待研究", thesis="需求增长", assumptions="资本开支持续")
    second = store.save_research_note(
        conn, "NVDA", "跟踪", thesis="需求增长但估值偏高", risks="客户集中")
    notes = store.load_research_notes(conn, "NVDA")
    assert list(notes["id"]) == [second, first]
    assert notes.iloc[0]["status"] == "跟踪"
    assert notes.iloc[1]["thesis"] == "需求增长"


def test_business_evidence_requires_source_and_keeps_metadata():
    conn = store.connect(":memory:")
    with pytest.raises(ValueError, match="来源链接"):
        store.save_business_evidence(conn, "NVDA", "数据中心收入", "")
    evidence_id = store.save_business_evidence(
        conn, "NVDA", "数据中心收入", "https://example.test/filing",
        value_text="$89.0B", period="2026Q2", unit="USD",
        published_at="2026-08-26", excerpt="Data Center revenue was...",
        entry_method="人工", verification_status="已核验",
        definition_text="数据中心分部收入", scope="分部", period_basis="单季",
        currency="USD", value_numeric=89.0, value_scale="十亿",
    )
    evidence = store.load_business_evidence(conn, "NVDA")
    assert evidence.iloc[0]["id"] == evidence_id
    assert evidence.iloc[0]["verification_status"] == "已核验"
    assert evidence.iloc[0]["source_url"] == "https://example.test/filing"
    assert evidence.iloc[0]["definition_text"] == "数据中心分部收入"
    assert evidence.iloc[0]["value_numeric"] == pytest.approx(89.0)


def test_business_evidence_consistency_allows_scale_conversion_and_latest_version_only():
    rows = pd.DataFrame([
        {"id": 1, "symbol": "A", "metric_name": "订单积压", "value_numeric": 900,
         "value_scale": "百万", "unit": "金额", "currency": "USD", "period": "2026Q2",
         "period_basis": "期末时点", "scope": "公司合并口径", "definition_text": "已签约未确认收入",
         "verification_status": "已核验"},
        {"id": 2, "symbol": "A", "metric_name": "订单积压", "value_numeric": 1,
         "value_scale": "十亿", "unit": "金额", "currency": "USD", "period": "2026Q2",
         "period_basis": "期末时点", "scope": "公司合并口径", "definition_text": "已签约未确认收入",
         "verification_status": "已核验"},
        {"id": 3, "symbol": "B", "metric_name": "订单积压", "value_numeric": 2,
         "value_scale": "十亿", "unit": "金额", "currency": "USD", "period": "2026Q2",
         "period_basis": "期末时点", "scope": "公司合并口径", "definition_text": "已签约未确认收入",
         "verification_status": "已核验"},
    ])
    result = audit_business_evidence(rows, "订单积压")
    assert result["compatible"] is True
    assert result["company_count"] == 2
    assert result["aggregate"] == pytest.approx(3e9)
    assert set(result["rows"].loc[result["rows"]["symbol"] == "A", "id"]) == {2}


def test_business_evidence_consistency_blocks_definition_period_and_verification_mismatch():
    rows = pd.DataFrame([
        {"id": 1, "symbol": "A", "metric_name": "数据中心收入", "value_numeric": 10,
         "value_scale": "十亿", "unit": "金额", "currency": "USD", "period": "2026Q2",
         "period_basis": "单季", "scope": "分部", "definition_text": "含网络业务",
         "verification_status": "已核验"},
        {"id": 2, "symbol": "B", "metric_name": "数据中心收入", "value_numeric": 12,
         "value_scale": "十亿", "unit": "金额", "currency": "USD", "period": "2026Q1",
         "period_basis": "单季", "scope": "分部", "definition_text": "不含网络业务",
         "verification_status": "待核验"},
    ])
    result = audit_business_evidence(rows, "数据中心收入")
    assert result["compatible"] is False
    assert result["aggregate"] is None
    assert "报告期不一致" in result["issues"]
    assert "指标定义不一致" in result["issues"]
    assert "存在未核验或有争议证据" in result["issues"]


def test_assisted_extraction_returns_traceable_candidates_without_claiming_verification():
    text = (
        "In 2026Q2, Data Center revenue was $89.0 billion, up 20%.\n"
        "Remaining performance obligation (RPO) reached USD 12.5bn in 2026Q2."
    )
    result = extract_evidence_candidates(text)
    assert list(result["metric_name"]) == ["数据中心收入", "订单积压"]
    revenue = result.iloc[0]
    assert revenue["value_numeric"] == pytest.approx(89.0)
    assert revenue["value_scale"] == "十亿"
    assert revenue["currency"] == "USD"
    assert revenue["period"] == "2026Q2"
    assert "Data Center revenue" in revenue["excerpt"]
    assert result["采用"].eq(False).all()


def test_assisted_extraction_skips_bare_numbers_and_flags_missing_period():
    result = extract_evidence_candidates("Power capacity increased to 250 MW across the portfolio.")
    assert len(result) == 1
    assert result.iloc[0]["metric_name"] == "产能"
    assert result.iloc[0]["unit"] == "MW"
    assert "未识别报告期" in result.iloc[0]["warning"]
    assert extract_evidence_candidates("Data center revenue grew 20 percent.").empty


def test_latest_notes_and_changes_since_view_are_deduplicated_queries():
    conn = store.connect(":memory:")
    store.save_research_note(conn, "NVDA", "待研究", thesis="v1")
    store.save_research_note(conn, "NVDA", "跟踪", thesis="v2")
    latest = store.load_latest_research_notes(conn)
    assert len(latest) == 1
    assert latest.iloc[0]["thesis"] == "v2"

    before = store.research_changes_since_view(conn, "NVDA")
    assert [item["类型"] for item in before] == ["研究判断"]
    store.mark_research_viewed(conn, "NVDA", "9999-01-01T00:00:00+00:00")
    assert store.research_changes_since_view(conn, "NVDA") == []


def test_research_events_keep_estimate_confirmation_and_correction_history():
    conn = store.connect(":memory:")
    estimated = store.save_research_event(
        conn, "NVDA", "财报", "2026-11-18", "预计", "2026Q3 财报",
        "https://example.test/calendar")
    confirmed = store.save_research_event(
        conn, "NVDA", "财报", "2026-11-19", "已确认", "2026Q3 财报",
        "https://example.test/ir")
    correction = store.save_research_event(
        conn, "NVDA", "订正", "2026-11-19", "已确认", "财报日期订正",
        "https://example.test/correction", correction_of_id=estimated)
    events = store.load_research_events(conn, "NVDA")
    assert set(events["id"]) == {estimated, confirmed, correction}
    assert events.loc[events["id"] == correction, "correction_of_id"].iloc[0] == estimated


def test_research_event_validation_and_change_feed():
    conn = store.connect(":memory:")
    with pytest.raises(ValueError, match="来源链接"):
        store.save_research_event(conn, "NVDA", "假设到期", "2026-12-01",
                                  "预计", "检查资本开支假设", "")
    with pytest.raises(ValueError, match="引用原事件"):
        store.save_research_event(conn, "NVDA", "订正", "2026-12-01",
                                  "已确认", "订正", "https://example.test", None)
    store.save_research_event(conn, "NVDA", "假设到期", "2026-12-01",
                              "预计", "检查资本开支假设", "https://example.test")
    assert "研究事件" in [item["类型"] for item in store.research_changes_since_view(conn, "NVDA")]
