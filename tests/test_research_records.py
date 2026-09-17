import pytest

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
    )
    evidence = store.load_business_evidence(conn, "NVDA")
    assert evidence.iloc[0]["id"] == evidence_id
    assert evidence.iloc[0]["verification_status"] == "已核验"
    assert evidence.iloc[0]["source_url"] == "https://example.test/filing"


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
