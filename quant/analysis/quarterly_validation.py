"""用已人工核验的公司原始材料基准检查供应商季度快照并提供最小兜底。"""

import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

from quant.data import store


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


OFFICIAL_FALLBACK_SOURCE = "公司官方材料（人工核验基准）"


def apply_official_quarterly_fallbacks(
    conn,
    baselines: dict,
    symbols: list[str] | None = None,
    captured_at: str | None = None,
) -> dict[str, tuple[str, str]]:
    """核验最新季度快照，并在供应商缺期/缺字段时写入官方最小快照。

    只导入基准明确列出的字段。数值或币种冲突不会自动覆盖，而是返回
    ``validation_failed`` 交给调用方提示人工复核。最新快照已经通过时不写库，
    因而对同一供应商版本重复运行是幂等的。
    """
    selected = list(dict.fromkeys(symbols or baselines.keys()))
    results: dict[str, tuple[str, str]] = {}
    now = captured_at or datetime.now(timezone.utc).isoformat(timespec="microseconds")
    for symbol in selected:
        baseline = baselines.get(symbol)
        if not baseline:
            results[symbol] = ("not_configured", "没有官方核验基准")
            continue
        frame = store.load_latest_quarterly_financials(conn, symbol)
        problems = validate_quarterly_sample(frame, baseline)
        if not problems:
            source = frame.attrs.get("source") or "未知来源"
            results[symbol] = ("verified", f"官方基准核验通过；当前来源：{source}")
            continue

        eligible = any("缺少官方最新报告期" in item or item.endswith("缺失")
                       for item in problems)
        actual_currency = frame.attrs.get("currency") if not frame.empty else None
        conflicts = [item for item in problems if "偏差" in item]
        # 完全没有上游快照时“币种缺失”是兜底要解决的问题；已有快照却币种
        # 不一致则属于真实冲突，不能用官方最小快照静默覆盖。
        if actual_currency:
            conflicts.extend(item for item in problems if "财报币种" in item)
        if not eligible or conflicts:
            results[symbol] = ("validation_failed", "；".join(problems))
            continue

        facts = official_baseline_facts(baseline)
        if not facts:
            results[symbol] = ("validation_failed", "官方基准没有可导入字段")
            continue
        # 同一份官方材料 + 当前上游快照生成稳定 ID。若最新已经是兜底，上面的
        # validate 已直接返回 verified；若上游产生新版本，允许生成新兜底版本覆盖其缺口。
        upstream_id = frame.attrs.get("snapshot_id") or "no-upstream-snapshot"
        identity = "|".join((symbol, str(baseline["period_end"]),
                             str(baseline["source_url"]), str(upstream_id)))
        snapshot_id = uuid.uuid5(uuid.NAMESPACE_URL, identity).hex
        exists = conn.execute(
            "SELECT 1 FROM financial_statement_snapshots WHERE snapshot_id = ?",
            (snapshot_id,),
        ).fetchone()
        if exists:
            results[symbol] = ("verified_fallback", "官方兜底快照已存在，无需重复导入")
            continue
        metadata = {
            "snapshot_id": snapshot_id,
            "symbol": symbol,
            "frequency": "quarterly",
            "currency": baseline["currency"],
            "trading_currency": baseline.get("trading_currency")
                                or frame.attrs.get("trading_currency"),
            "source": OFFICIAL_FALLBACK_SOURCE,
            "source_url": baseline["source_url"],
            "published_at": baseline.get("published_at"),
            "captured_at": now,
        }
        store.insert_financial_snapshot(conn, metadata, facts)
        results[symbol] = (
            "verified_fallback",
            f"供应商缺期/缺字段；导入官方 {baseline['period_end']} 的 {len(facts)} 个已核验字段",
        )
    return results
