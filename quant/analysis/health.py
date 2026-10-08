"""运行健康自检：流水线自己说清"今天这份日报能不能信"。

起因（2026-10-06）：信号日期被海外标的顶到次日，周一至周四的信号被静默丢弃近三个月，
而日报每天照发"今日无新信号"——"没信号"与"坏了"在推送里长得一模一样。
这里只做纯计算（不读库、不联网），由 run_daily 把结果拼进日报并在标题上示警。

分两级，避免天天喊狼来了：
- ``issues``：今天的信号可能不完整，标题加 ⚠️。
- ``notes``：值得知道但不影响今天的信号（如长期无数据的疑似退市标的），只在正文列出。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pandas as pd

MAX_RUN_LAG_DAYS = 4      # 周末 + 一个假日；信号日期比运行日更旧就说明行情没更新
DORMANT_DAYS = 7          # 选股池里的个股超过这么久没数据视为疑似退市/被并购，降级为 note
DISPLAY_STALE_DAYS = 10   # 纯展示标的（海外市场有长假）的过期容忍


@dataclass
class HealthReport:
    issues: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    summary: str = ""

    @property
    def ok(self) -> bool:
        return not self.issues

    def render(self) -> str:
        lines = ["🩺 运行状态"]
        lines.append(("✅ " if self.ok else "⚠️ ") + self.summary)
        lines += [f"⚠️ {x}" for x in self.issues]
        lines += [f"ℹ️ {x}" for x in self.notes]
        return "\n".join(lines)


def _shown(symbols: list[str], limit: int = 8) -> str:
    head = ", ".join(symbols[:limit])
    return head + (f" 等 {len(symbols)} 个" if len(symbols) > limit else "")


def _lag_days(prices: dict[str, pd.DataFrame], symbol: str, ref: pd.Timestamp) -> int | None:
    """标的最新行情日落后参考日多少个日历天；库里没有该标的返回 None。"""
    if symbol not in prices or prices[symbol].empty:
        return None
    return int((ref - prices[symbol].index.max()).days)


def check_health(prices: dict[str, pd.DataFrame], strategy_symbols: list[str],
                 display_symbols: list[str], as_of: str, today: date,
                 fetch_failed: list[str] | None = None,
                 options: tuple[int, list[str]] | None = None,
                 pool_symbols: set[str] | None = None,
                 check_run_lag: bool = True) -> HealthReport:
    """汇总本次运行的数据健康状况。

    ``strategy_symbols`` 是喂给策略的标的（缺数据 = 信号可能漏），``display_symbols`` 是其余
    只做展示/研究的标的。``options`` 为 (成功数, 失败标的) 或 None（本次未采集）。
    ``pool_symbols`` 是只来自 universe_file 大名单的个股：几百只里隔三差五有退市/并购，停更超过
    DORMANT_DAYS 就降级为 note；watchlist 里手挑的 ETF 停更则永远是 issue。
    ``check_run_lag=False`` 用于 --date 补跑与 --no-fetch：信号日期本来就不是今天。
    """
    report = HealthReport()
    pool_symbols = pool_symbols or set()
    ref = pd.Timestamp(as_of)
    strategy_symbols = list(dict.fromkeys(strategy_symbols))

    if check_run_lag:
        lag = (pd.Timestamp(today) - ref).days
        if lag < 0:
            report.issues.append(f"信号日期 {as_of} 晚于运行日 {today}，当日信号可能匹配不上")
        elif lag > MAX_RUN_LAG_DAYS:
            report.issues.append(f"信号日期 {as_of} 已落后运行日 {lag} 天，行情可能没更新")

    missing, behind, dormant = [], [], []
    for s in strategy_symbols:
        lag = _lag_days(prices, s, ref)
        if lag is None:
            missing.append(s)
        elif lag > DORMANT_DAYS and s in pool_symbols:
            dormant.append(s)
        elif lag > 0:
            behind.append(s)
    if missing:
        report.issues.append(f"策略标的库内无行情：{_shown(missing)}")
    if behind:
        report.issues.append(f"策略标的行情未更新到 {as_of}：{_shown(behind)}，其信号可能缺失")
    if dormant:
        report.notes.append(
            f"选股池个股超过 {DORMANT_DAYS} 天无行情（疑似退市/被并购）：{_shown(dormant)}")

    stale_display = [s for s in dict.fromkeys(display_symbols)
                     if s not in strategy_symbols
                     and (_lag_days(prices, s, ref) or 0) > DISPLAY_STALE_DAYS]
    if stale_display:
        report.notes.append(
            f"展示/研究标的超过 {DISPLAY_STALE_DAYS} 天未更新：{_shown(stale_display)}")

    if fetch_failed:
        report.issues.append(f"行情拉取失败：{_shown(fetch_failed, 20)}，信号可能不完整")

    opt_text = ""
    if options is not None:
        opt_ok, opt_fail = options
        opt_text = f"，期权快照 {opt_ok}/{opt_ok + len(opt_fail)}"
        if opt_fail:
            report.issues.append(f"期权快照失败（当天数据无法补采）：{_shown(opt_fail)}")

    fresh = len(strategy_symbols) - len(missing) - len(behind) - len(dormant)
    report.summary = (f"信号日期 {as_of}，策略标的 {fresh}/{len(strategy_symbols)} 已更新到当日"
                      f"{opt_text}")
    return report
