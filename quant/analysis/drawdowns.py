"""崩盘/回撤分段与避险资产表现：给「崩盘类型 → 最佳避险」面板用（纯计算）。

方法：以基准（默认 SPY）总回报口径（adj_close）识别「水下区段」——从历史新高跌下去、
到收复新高为一段；取最大回撤 ≤ threshold 的段为一次「像样下跌」。对每段【下跌期间】
（峰 → 谷；进行中的段用 峰 → 最新）测各候选避险资产的总回报，谁跌得少/涨得多即当次
最佳避险。

崩盘类型自动判定（让「对号入座」可自动化）：
- 峰→谷 ≤ FLASH_DAYS 天 → 「闪崩」：太快，避险来不及涨，现金/短债靠不跌取胜
- 否则看长债 TLT 在此段的表现：
    TLT 为正 → 「通缩/避险型」：资金抢国债，长债 TLT 有效（2015、2020 COVID 型）
    TLT 为负 → 「通胀/加息型」：加息/通胀，长债 TLT 也崩，需商品/黄金（2022 型）

严重度按 SPY 最大回撤分：熊市 ≤ -20%、准熊 ≤ -15%、回调 其余。
"""

import pandas as pd

# 峰→谷 ≤ 此天数视为「闪崩」（太快，任何避险都来不及走出趋势）
FLASH_DAYS = 25

# 候选避险资产（代码 → 是否为现金基线）：短债 BIL ≈ 持现金的基准线
HEDGE_CANDIDATES = ["TLT", "GLD", "BIL", "DBC", "HYG", "XLP", "XLU", "XLV"]
CASH_PROXY = "BIL"


def find_drawdown_episodes(bench: pd.Series, threshold: float = 0.08) -> list[dict]:
    """在基准总回报序列上找回撤 ≤ threshold 的下跌段（外加末尾进行中的段，不论深浅）。

    返回每段：peak_date / trough_date / recover_date(None=未收复) / end_date(测算截止,
    进行中=最新日) / maxdd / ongoing。
    """
    bench = bench.dropna()
    n = len(bench)
    if n < 2:
        return []
    runmax = bench.cummax()
    dd = bench / runmax - 1
    underwater = (dd < -1e-9).to_numpy()
    idx = bench.index

    episodes: list[dict] = []
    i = 0
    while i < n:
        if underwater[i]:
            start = i - 1 if i > 0 else i          # 峰 = 前一日（仍在历史新高）
            j = i
            while j < n and underwater[j]:
                j += 1
            seg = dd.iloc[i:j]
            trough = i + int(seg.to_numpy().argmin())
            maxdd = float(seg.min())
            ongoing = j >= n                        # 到序列末尾还没收复 = 进行中
            if maxdd <= -threshold or ongoing:
                episodes.append({
                    "peak_date": idx[start],
                    "trough_date": idx[trough],
                    "recover_date": None if ongoing else idx[j],
                    "end_date": idx[-1] if ongoing else idx[trough],
                    "maxdd": maxdd,
                    "ongoing": ongoing,
                })
            i = j
        else:
            i += 1
    return episodes


def episode_returns(prices: pd.DataFrame, peak_date, end_date,
                    candidates=HEDGE_CANDIDATES) -> dict[str, float]:
    """各候选资产在 [peak_date, end_date] 的总回报（缺数据的跳过）。"""
    out: dict[str, float] = {}
    for sym in candidates:
        if sym not in prices.columns:
            continue
        s = prices[sym].loc[peak_date:end_date].dropna()
        if len(s) >= 2 and float(s.iloc[0]) > 0:
            out[sym] = float(s.iloc[-1]) / float(s.iloc[0]) - 1
    return out


def classify_episode(episode: dict, tlt_ret: float | None) -> tuple[str, str]:
    """(类型, 一句人话) —— 按持续天数 + 长债 TLT 表现自动判定崩盘机制。"""
    days = (episode["end_date"] - episode["peak_date"]).days
    if days <= FLASH_DAYS:
        return "闪崩", "下跌太快，避险来不及走趋势，现金/短债靠不跌取胜"
    if tlt_ret is not None and tlt_ret > 0:
        return "通缩/避险型", "资金抢国债避险，长债 TLT 有效"
    return "通胀/加息型", "加息/通胀，长债 TLT 也崩，需商品/黄金对冲"


def defense_spans(signals: list, index: pd.DatetimeIndex,
                  defense_symbols: set[str]) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """从策略信号重建"全仓防守"的连续区段：每日持仓非空、且完全落在 defense_symbols 内
    （例如 canary_mom 的 {IEF, BIL}，不含哨兵 TIP——哨兵只当开关，从不被持有）。

    用于把"策略这段时间在干嘛"跟"大盘这段时间有没有真的暴跌"交叉对比：
    - 防守区段与某次大回撤重叠 → 防住了
    - 防守区段跟任何大回撤都不重叠 → 假信号（白白空仓错过涨幅）
    - 大回撤发生时策略没有防守区段覆盖 → 没防住
    """
    events: dict[str, list] = {}
    for s in signals:
        events.setdefault(s.date, []).append(s)
    held: set[str] = set()
    in_defense = []
    for ts in index:
        d = ts.strftime("%Y-%m-%d")
        for s in events.get(d, []):
            # 当日全部事件处理完才读 in_defense，同日内 sell/buy 顺序不影响最终持仓集合
            if s.direction == "sell":
                held.discard(s.symbol)
            else:
                held.add(s.symbol)
        in_defense.append(bool(held) and held <= defense_symbols)

    spans: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    start = None
    prev = None
    for ts, v in zip(index, in_defense):
        if v and start is None:
            start = ts
        if not v and start is not None:
            spans.append((start, prev))
            start = None
        prev = ts
    if start is not None:
        spans.append((start, index[-1]))
    return spans


def spans_overlap(a_start, a_end, b_start, b_end) -> bool:
    """两个闭区间 [a_start,a_end] 与 [b_start,b_end] 是否有重叠。"""
    return a_start <= b_end and b_start <= a_end


def severity(maxdd: float) -> str:
    """按最大回撤给严重度标签。"""
    if maxdd <= -0.20:
        return "🐻 熊市"
    if maxdd <= -0.15:
        return "🐻 准熊"
    return "📉 回调"
