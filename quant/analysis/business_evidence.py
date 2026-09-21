"""业务证据跨公司口径检查；不猜定义、不把文本值强行转成数字。"""

from __future__ import annotations

import re

import pandas as pd


SCALE_MULTIPLIERS = {
    "原值": 1.0,
    "万": 1e4,
    "千": 1e3,
    "百万": 1e6,
    "十亿": 1e9,
    "亿": 1e8,
    "万亿": 1e12,
}

METRIC_KEYWORDS = {
    "数据中心收入": ("数据中心收入", "data center revenue", "datacenter revenue"),
    "订单积压": ("订单积压", "backlog", "remaining performance obligation", "rpo"),
    "资本开支": ("资本开支", "capital expenditure", "capital expenditures", "capex"),
    "产能": ("产能", "capacity"),
    "电力容量": ("电力容量", "power capacity", "megawatt", "gigawatt", " mw", " gw"),
    "AI相关收入": ("ai相关收入", "ai 相关收入", "ai revenue", "artificial intelligence revenue"),
}

_VALUE_RE = re.compile(
    r"(?P<currency>US\$|USD|CNY|RMB|EUR|JPY|KRW|TWD|\$|¥|€)?\s*"
    r"(?P<number>\d[\d,]*(?:\.\d+)?)\s*"
    r"(?P<scale>trillion|billion|million|bn|mn|[TBM](?![Ww])|万亿|十亿|亿|百万|万)?\s*"
    r"(?P<physical>GW|MW|kW|台|座|平方英尺|sq\.?\s*ft\.?)?",
    re.IGNORECASE,
)


def extract_evidence_candidates(text: str) -> pd.DataFrame:
    """从用户粘贴的原文提取候选值；结果必须人工检查后才能入库。"""
    columns = [
        "采用", "metric_name", "value_text", "value_numeric", "value_scale",
        "unit", "currency", "period", "excerpt", "warning",
    ]
    if not text or not text.strip():
        return pd.DataFrame(columns=columns)
    sentences = [item.strip() for item in re.split(r"[\n\r]+|(?<=[。！？!?;；])", text)
                 if item.strip()]
    rows = []
    seen = set()
    currency_map = {"$": "USD", "US$": "USD", "¥": "CNY", "€": "EUR", "RMB": "CNY"}
    scale_map = {
        "t": "万亿", "trillion": "万亿", "b": "十亿", "bn": "十亿",
        "billion": "十亿", "m": "百万", "mn": "百万", "million": "百万",
        "万亿": "万亿", "十亿": "十亿", "亿": "亿", "百万": "百万", "万": "万",
    }
    for sentence in sentences:
        lowered = sentence.casefold()
        metric = next((name for name, keywords in METRIC_KEYWORDS.items()
                       if any(keyword.casefold() in lowered for keyword in keywords)), None)
        if not metric:
            continue
        matches = []
        for match in _VALUE_RE.finditer(sentence):
            token = match.group(0).strip()
            number = float(match.group("number").replace(",", ""))
            has_context = any((match.group("currency"), match.group("scale"),
                               match.group("physical")))
            # 裸年份通常不是披露值；没有单位的普通数字不自动提取。
            if not has_context or (1900 <= number <= 2100 and not match.group("scale")):
                continue
            matches.append((match, token, number))
        if not matches:
            continue
        match, token, number = matches[0]
        raw_currency = match.group("currency") or ""
        currency = currency_map.get(raw_currency, raw_currency.upper())
        physical = (match.group("physical") or "").upper().replace(".", "")
        raw_scale = (match.group("scale") or "").casefold()
        value_scale = scale_map.get(raw_scale, "原值")
        unit = physical or ("金额" if currency else "")
        period_match = re.search(
            r"(?:FY\s*)?20\d{2}\s*Q[1-4]|Q[1-4]\s*(?:FY\s*)?20\d{2}|"
            r"20\d{2}年(?:第?[一二三四1234]季度|上半年|下半年|全年)",
            sentence, re.IGNORECASE,
        )
        period = period_match.group(0).replace(" ", "") if period_match else ""
        identity = (metric, token, sentence)
        if identity in seen:
            continue
        seen.add(identity)
        warnings = []
        if not period:
            warnings.append("未识别报告期")
        if unit == "金额" and not currency:
            warnings.append("未识别币种")
        rows.append({
            "采用": False, "metric_name": metric, "value_text": token,
            "value_numeric": number, "value_scale": value_scale, "unit": unit,
            "currency": currency, "period": period, "excerpt": sentence,
            "warning": "；".join(warnings),
        })
    return pd.DataFrame(rows, columns=columns)


def _normalized(value) -> str:
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).strip()).casefold()


def audit_business_evidence(frame: pd.DataFrame, metric_name: str) -> dict:
    """检查同名指标能否跨公司比较/合计，并返回最新证据行及问题。

    只有定义、范围、报告期、期间口径、单位和币种一致，全部已核验且数值结构化时，
    才返回 aggregate。不同数量级会先换算到原值，因此“百万/十亿”本身不构成冲突。
    """
    empty = {
        "compatible": False, "aggregate": None, "issues": ["没有可检查的证据"],
        "rows": pd.DataFrame(), "company_count": 0,
    }
    if frame.empty or not metric_name.strip() or "metric_name" not in frame:
        return empty
    target = _normalized(metric_name)
    work = frame[frame["metric_name"].map(_normalized) == target].copy()
    if work.empty:
        return empty
    if "id" in work:
        work = work.sort_values("id", ascending=False)
    # 同一公司同一指标只检查最新版本，避免历史修订重复进入合计。
    work = work.drop_duplicates(subset=["symbol"], keep="first")
    issues: list[str] = []
    required = {
        "definition_text": "指标定义", "scope": "统计范围", "period": "报告期",
        "period_basis": "期间口径", "unit": "单位",
    }
    for column, label in required.items():
        if column not in work or work[column].map(_normalized).eq("").any():
            issues.append(f"{label}缺失")
            continue
        variants = work[column].map(_normalized).nunique()
        if variants > 1:
            issues.append(f"{label}不一致")
    if "currency" in work:
        currencies = work["currency"].map(_normalized)
        # 非金额指标可全部留空；只要部分填写或填写值冲突就不能直接合计。
        if currencies.ne("").any() and (currencies.eq("").any() or currencies.nunique() > 1):
            issues.append("币种不一致或部分缺失")
    if "verification_status" not in work or not work["verification_status"].eq("已核验").all():
        issues.append("存在未核验或有争议证据")
    numeric = (pd.to_numeric(work.get("value_numeric"), errors="coerce")
               if "value_numeric" in work else pd.Series(index=work.index, dtype=float))
    scales = (work.get("value_scale", pd.Series("原值", index=work.index))
              .fillna("原值"))
    multipliers = scales.map(SCALE_MULTIPLIERS)
    if numeric.isna().any():
        issues.append("结构化数值缺失")
    if multipliers.isna().any():
        issues.append("数量级无法识别")
    work["normalized_value"] = numeric * multipliers
    if len(work) < 2:
        issues.append("至少需要两家公司")
    compatible = not issues
    aggregate = float(work["normalized_value"].sum()) if compatible else None
    return {
        "compatible": compatible,
        "aggregate": aggregate,
        "issues": issues,
        "rows": work.sort_values("symbol"),
        "company_count": int(len(work)),
    }
