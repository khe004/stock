#!/usr/bin/env python3
"""供应商缺少官方最新季度时，导入已人工核验的最小事实快照。

只导入基准文件明确列出的字段，不从旧快照拼值。用于样本验证阶段的来源兜底，
不会把未核验字段伪装为完整三表。
"""

import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quant.analysis.quarterly_validation import (  # noqa: E402
    load_official_baselines,
    official_baseline_facts,
    validate_quarterly_sample,
)
from quant.config import load_config  # noqa: E402
from quant.data import store  # noqa: E402


def main() -> int:
    cfg = load_config()
    conn = store.connect(cfg.db_path)
    baselines = load_official_baselines(ROOT / "quarterly_sample_baselines.yaml")
    imported = 0
    for symbol, baseline in baselines.items():
        frame = store.load_latest_quarterly_financials(conn, symbol)
        problems = validate_quarterly_sample(frame, baseline)
        missing_latest = any("缺少官方最新报告期" in item or "缺失" in item
                             for item in problems)
        if not missing_latest:
            continue
        facts = official_baseline_facts(baseline)
        if not facts:
            continue
        metadata = {
            "snapshot_id": uuid.uuid4().hex,
            "symbol": symbol,
            "frequency": "quarterly",
            "currency": baseline["currency"],
            "trading_currency": frame.attrs.get("trading_currency"),
            "source": "公司官方材料（人工核验基准）",
            "source_url": baseline["source_url"],
            "published_at": baseline.get("published_at"),
            "captured_at": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
        }
        store.insert_financial_snapshot(conn, metadata, facts)
        store.record_research_update(
            conn, symbol, "quarterly", "verified_fallback",
            f"供应商缺期；导入官方 {baseline['period_end']} 的 {len(facts)} 个已核验字段",
        )
        imported += 1
        print(f"IMPORTED {symbol}: {len(facts)} 个官方核验字段")
    print(f"汇总：导入 {imported} 个兜底快照")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
