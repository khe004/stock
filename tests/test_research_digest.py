from datetime import date

from quant.analysis.research_digest import build_research_digest
from quant.config import load_config
from quant.data import store


def test_digest_contains_changes_and_upcoming_events_with_stable_hash():
    conn = store.connect(":memory:")
    store.save_research_note(conn, "NVDA", "跟踪", thesis="需求")
    store.save_research_event(
        conn, "NVDA", "财报", "2026-10-01", "已确认", "季度财报",
        "https://example.test/ir")
    body1, digest1 = build_research_digest(conn, date(2026, 9, 18))
    body2, digest2 = build_research_digest(conn, date(2026, 9, 19))
    assert "NVDA（跟踪）：研究判断、研究事件" in body1
    assert "2026-10-01 NVDA｜财报｜已确认｜季度财报" in body1
    assert body1 != body2
    assert digest1 == digest2


def test_digest_empty_after_view_when_no_upcoming_events():
    conn = store.connect(":memory:")
    store.save_research_note(conn, "NVDA", "跟踪")
    store.mark_research_viewed(conn, "NVDA", "9999-01-01T00:00:00+00:00")
    assert build_research_digest(conn, date(2026, 9, 18)) == ("", "")


def test_digest_delivery_is_idempotent_per_channel():
    conn = store.connect(":memory:")
    store.mark_research_digest_delivered(conn, "hash", "telegram")
    store.mark_research_digest_delivered(conn, "hash", "telegram")
    assert store.research_digest_delivered(conn, "hash", "telegram") is True
    assert store.research_digest_delivered(conn, "hash", "email") is False
    count = conn.execute("SELECT COUNT(*) FROM research_digest_deliveries").fetchone()[0]
    assert count == 1


def test_research_digest_defaults_off():
    assert load_config().research_digest_enabled is False
