#!/usr/bin/env python3
"""供应商缺少官方最新季度时，导入已人工核验的最小事实快照。

只导入基准文件明确列出的字段，不从旧快照拼值。用于样本验证阶段的来源兜底，
不会把未核验字段伪装为完整三表。
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quant.analysis.quarterly_validation import (  # noqa: E402
    apply_official_quarterly_fallbacks,
    load_official_baselines,
)
from quant.config import load_config  # noqa: E402
from quant.data import store  # noqa: E402


def main() -> int:
    cfg = load_config()
    conn = store.connect(cfg.db_path)
    baselines = load_official_baselines(ROOT / "quarterly_sample_baselines.yaml")
    results = apply_official_quarterly_fallbacks(conn, baselines)
    failed = 0
    imported = 0
    for symbol, (status, detail) in results.items():
        store.record_research_update(conn, symbol, "quarterly", status, detail)
        print(f"{status.upper():<19} {symbol}: {detail}")
        imported += status == "verified_fallback" and "导入官方" in detail
        failed += status == "validation_failed"
    print(f"汇总：导入 {imported} 个兜底快照，核验失败 {failed} 个")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
