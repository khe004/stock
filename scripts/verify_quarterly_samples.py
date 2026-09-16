#!/usr/bin/env python3
"""核对本地季度快照与已人工确认的 8 家官方基准。"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quant.analysis.quarterly_validation import (  # noqa: E402
    load_official_baselines,
    validate_quarterly_sample,
)
from quant.config import load_config  # noqa: E402
from quant.data import store  # noqa: E402


def main() -> int:
    cfg = load_config()
    conn = store.connect(cfg.db_path)
    baselines = load_official_baselines(ROOT / "quarterly_sample_baselines.yaml")
    failed = 0
    for symbol, baseline in baselines.items():
        frame = store.load_latest_quarterly_financials(conn, symbol)
        problems = validate_quarterly_sample(frame, baseline)
        if problems:
            failed += 1
            print(f"FAIL {symbol}: " + "；".join(problems))
        else:
            print(f"OK   {symbol}: {baseline['period_end']} 已核验字段一致")
    print(f"汇总：{len(baselines) - failed}/{len(baselines)} 通过")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
