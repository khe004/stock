"""回撤档位提醒：某标的离历史高点跌到 X% 时在日报里提醒一次，同一轮回撤里每档只提醒一次。

用户的用法是"按计划分批补仓"（2026-10-16）：每档对应一笔补仓，所以档位按**回撤段**去重——
一轮回撤从创新高开始、到再次创新高结束；段内触及过的档位不再提醒（跌到 12% 后反弹到 5% 再跌回 11%
不会重复提醒 -10%），创新高后各档重置，并发一条"本轮回撤结束"。

纯计算：哪些提醒已发过由调用方从库里读出（``sent_keys``）传进来，这里只算"现在该发哪些"。
用 key 而不是"昨天 vs 今天"判断，是为了 Mac 睡眠漏跑一天时提醒不会丢——下次运行照样补发。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Alert:
    key: str
    text: str            # 空串 = 被同日更深一档合并，只记账不展示
    tag: str = ""        # 标题前缀，如「📉 QQQ -20%」


def _episode_ids(close: pd.Series) -> pd.Series:
    """每次收盘创历史新高开启一个新回撤段，返回每天所属段的编号。"""
    return (close >= close.cummax()).cumsum()


def level_history(df: pd.DataFrame, level: float, before: pd.Timestamp | None = None) -> dict:
    """历史上每轮回撤首次触及该档的日子，及之后 1 年的总回报（adj_close）。

    ``before`` 给出时只统计在它之前开始的回撤段（排除当前这一轮）。"""
    close = df["close"].dropna()
    tr = df["adj_close"].reindex(close.index).ffill()
    dd = close / close.cummax() - 1
    eid = _episode_ids(close)
    hits = []
    for _, g in dd.groupby(eid):
        if before is not None and g.index[0] >= before:
            continue
        x = g[g <= -level / 100]
        if len(x):
            hits.append(x.index[0])
    fwd = []
    for d in hits:
        i = tr.index.get_loc(d)
        if i + 252 < len(tr):
            fwd.append(tr.iloc[i + 252] / tr.iloc[i] - 1)
    return {"years": sorted({d.year for d in hits}), "n": len(hits),
            "fwd_1y": np.array(fwd)}


def _history_line(df: pd.DataFrame, level: float, episode_start: pd.Timestamp) -> str:
    h = level_history(df, level, before=episode_start)
    years = h["years"]
    if not years or len(h["fwd_1y"]) == 0:
        return "历史上没有触及过这一档。"
    neg = int((h["fwd_1y"] < 0).sum())
    where = "/".join(map(str, years)) if len(years) <= 8 else f"分布在 {len(years)} 个年份"
    return (f"自 {df.index[0].year} 年以来此前有 {h['n']} 轮回撤触及这一档（{where}），"
            f"首次触及后一年总回报中位 {np.median(h['fwd_1y']):+.0%}，"
            f"{len(h['fwd_1y'])} 次里 {neg} 次一年后仍为负（样本很少，仅作参照）。")


def pending_drawdown_alerts(symbol: str, df: pd.DataFrame, levels: list[float],
                            sent_keys: set[str]) -> list[Alert]:
    """算出现在该发、但还没发过的回撤提醒。

    - 当前回撤段内已触及、尚未发过的档位：合并成一条（跳空一次跌穿两档只说最深那档）。
    - 某个已结束的回撤段发过档位提醒、而结束（创新高）提醒还没发：发一条"本轮回撤结束"。
    key 形如 ``QQQ:dd20:2026-10-06``（段起点 = 高点日）与 ``QQQ:newhigh:2026-10-06``。
    """
    close = df["close"].dropna()
    if len(close) < 2 or not levels:
        return []
    levels = sorted(float(x) for x in levels)
    dd = close / close.cummax() - 1
    eid = _episode_ids(close)
    cur = dd[eid == eid.iloc[-1]]
    peak_date = cur.index[0].strftime("%Y-%m-%d")
    peak = float(close.loc[cur.index[0]])
    deepest = float(cur.min())
    alerts: list[Alert] = []

    # 已结束的回撤段：发过档位提醒、没发过结束提醒
    ended = {k.split(":")[2] for k in sent_keys
             if k.startswith(f"{symbol}:dd") and k.split(":")[2] != peak_date}
    for p in sorted(ended):
        key = f"{symbol}:newhigh:{p}"
        if key in sent_keys:
            continue
        seg = dd[(dd.index >= pd.Timestamp(p)) & (dd.index < cur.index[0])]
        alerts.append(Alert(key, (
            f"✅ {symbol} 收盘 {peak:.2f} 创新高（{peak_date}），{p} 开始的那轮回撤结束"
            f"（最深 {seg.min():+.1%}），各档提醒重置。")))

    reached = [x for x in levels if deepest <= -x / 100]
    new = [x for x in reached if f"{symbol}:dd{x:g}:{peak_date}" not in sent_keys]
    if new:
        top = max(new)
        now = float(close.iloc[-1])
        nxt = [x for x in levels if x > max(reached)]
        first_day = cur[cur <= -top / 100].index[0]
        lines = [
            f"📉 {symbol} 离高点已跌 {-dd.iloc[-1]:.1%}（高点 {peak:.2f} 于 {peak_date}，"
            f"现 {now:.2f}），{first_day:%m-%d} 首次触及 -{top:g}% 档"
            + (f"（同时跨过 {'、'.join(f'-{x:g}%' for x in new if x != top)}）" if len(new) > 1 else "")
            + "。",
            "如果在按档分批补仓，这一档对应一笔；"
            + (f"下一档 -{nxt[0]:g}%。" if nxt else "已是最深一档。"),
            _history_line(df, top, cur.index[0]),
        ]
        text = "\n".join(lines)
        for x in new:
            alerts.append(Alert(f"{symbol}:dd{x:g}:{peak_date}", text if x == top else "",
                                f"📉 {symbol} -{top:g}%" if x == top else ""))
    return alerts


def render_alerts(alerts: list[Alert]) -> str:
    texts = [a.text for a in alerts if a.text]
    return ("🔔 回撤档位提醒\n" + "\n\n".join(texts)) if texts else ""


def subject_tag(alerts: list[Alert]) -> str:
    """标题前缀，如「📉 QQQ -20% 」；只有结束提醒时为空。"""
    return "".join(f"{a.tag} " for a in alerts if a.tag)
