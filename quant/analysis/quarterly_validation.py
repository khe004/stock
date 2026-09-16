"""用已人工核验的公司原始材料基准检查供应商季度快照。"""

from pathlib import Path

import pandas as pd
import yaml


STATEMENTS = {
    "cash": "balance", "total_debt": "balance",
    "operating_cash_flow": "cashflow", "capital_expenditure": "cashflow",
    "free_cash_flow": "cashflow",
}


def load_official_baselines(path: Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def validate_quarterly_sample(frame: pd.DataFrame, baseline: dict,
                              relative_tolerance: float = 0.002) -> list[str]:
    """返回不一致说明；空列表表示已列出的核验字段均匹配。"""
    problems: list[str] = []
    expected_period = str(baseline["period_end"])
    provider_period = str(baseline.get("provider_period_end", expected_period))
    expected_currency = baseline.get("currency")
    actual_currency = frame.attrs.get("currency")
    if actual_currency != expected_currency:
        problems.append(f"财报币种 {actual_currency or '缺失'}，预期 {expected_currency}")
    if frame.empty or provider_period not in frame.index.astype(str):
        problems.append(f"缺少官方最新报告期 {expected_period}")
        return problems
    row = frame.loc[provider_period]
    for metric, expected in baseline.get("metrics", {}).items():
        actual = row.get(metric)
        if actual is None or pd.isna(actual):
            problems.append(f"{expected_period} {metric} 缺失")
            continue
        scale = max(abs(float(expected)), 1.0)
        error = abs(float(actual) - float(expected)) / scale
        if error > relative_tolerance:
            problems.append(
                f"{expected_period} {metric}={float(actual):.6g}，"
                f"官方={float(expected):.6g}（偏差 {error:.2%}）"
            )
    return problems


def official_baseline_facts(baseline: dict) -> list[dict]:
    """把人工核验字段转换为可追溯事实；未核验字段保持缺失。"""
    currency = baseline["currency"]
    return [
        {
            "period_start": baseline.get("period_start"),
            "period_end": str(baseline["period_end"]),
            "statement": STATEMENTS.get(metric, "income"),
            "metric": metric,
            "value": float(value),
            "unit": currency,
            "raw_label": "official_verified_baseline",
        }
        for metric, value in baseline.get("metrics", {}).items()
    ]
