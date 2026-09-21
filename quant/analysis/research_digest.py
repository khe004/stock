"""AI 公司研究摘要：从关注列表变化与未来事件生成可预览、可去重的文本。"""

import hashlib
from datetime import date, timedelta

from quant.data import store


def build_research_digest(conn, today: date | None = None, event_days: int = 30) -> tuple[str, str]:
    today = today or date.today()
    lines = [f"🔎 AI 基建研究摘要 {today:%Y-%m-%d}"]
    notes = store.load_latest_research_notes(conn)
    changes_found = False
    for note in notes.itertuples(index=False):
        changes = store.research_changes_since_view(conn, note.symbol)
        if changes:
            changes_found = True
            kinds = "、".join(item["类型"] for item in changes)
            lines.append(f"• {note.symbol}（{note.status}）：{kinds}")

    end = (today + timedelta(days=event_days)).isoformat()
    events = store.load_research_events(conn, start=today.isoformat())
    if not events.empty:
        events = events[events["event_date"] <= end]
    if not events.empty:
        lines.append(f"\n📅 未来 {event_days} 天事件")
        for row in events.itertuples(index=False):
            lines.append(
                f"• {row.event_date} {row.symbol}｜{row.event_type}｜"
                f"{row.date_status}｜{row.title}")

    if not changes_found and events.empty:
        return "", ""
    body = "\n".join(lines)
    # 标题日期只用于阅读，不参与去重；同一批变化跨天仍视为同一摘要。
    payload = "\n".join(lines[1:])
    return body, hashlib.sha256(payload.encode("utf-8")).hexdigest()
