"""回撤档位提醒：每轮回撤每档只提醒一次、跳空合并、漏跑补发、创新高重置。"""

import pandas as pd

from quant.analysis.alerts import pending_drawdown_alerts, render_alerts, subject_tag
from quant.data import store


def _df(closes):
    idx = pd.bdate_range("2026-01-01", periods=len(closes))
    s = pd.Series(closes, index=idx, dtype=float)
    return pd.DataFrame({"close": s, "adj_close": s})


def _replay(closes, levels=(10, 20, 30)):
    """逐日运行，模拟每天发出后记账；返回 {日期序号: [key...]}。"""
    sent: set[str] = set()
    out = {}
    df = _df(closes)
    for i in range(2, len(df) + 1):
        new = pending_drawdown_alerts("QQQ", df.iloc[:i], list(levels), sent)
        if new:
            out[i - 1] = [a.key for a in new]
            sent |= {a.key for a in new}
    return out


def test_each_level_once_per_episode_no_repeat_between_levels():
    # 100 高点 → 89(-11%) → 88(-12%) → 95(-5%) → 89(-11%) → 79(-21%)
    got = _replay([100, 89, 88, 95, 89, 79])
    assert got == {1: ["QQQ:dd10:2026-01-01"], 5: ["QQQ:dd20:2026-01-01"]}


def test_gap_through_two_levels_merges_into_one_message():
    df = _df([100, 75])
    alerts = pending_drawdown_alerts("QQQ", df, [10, 20, 30], set())
    assert {a.key for a in alerts} == {"QQQ:dd10:2026-01-01", "QQQ:dd20:2026-01-01"}
    shown = [a for a in alerts if a.text]
    assert len(shown) == 1 and "-20% 档" in shown[0].text and "-10%" in shown[0].text
    assert subject_tag(alerts) == "📉 QQQ -20% "
    assert render_alerts(alerts).count("📉") == 1


def test_missed_run_still_alerts_next_time():
    # 第 1 天触及 -10% 但没运行；第 2 天反弹到 -8% 时运行，仍要补发
    df = _df([100, 89, 92])
    alerts = pending_drawdown_alerts("QQQ", df, [10, 20], set())
    assert [a.key for a in alerts] == ["QQQ:dd10:2026-01-01"]
    assert "01-02 首次触及" in alerts[0].text


def test_new_high_resets_and_announces_end_once():
    got = _replay([100, 89, 101, 90, 102, 91])
    assert got[1] == ["QQQ:dd10:2026-01-01"]
    assert got[2] == ["QQQ:newhigh:2026-01-01"]
    assert got[3] == ["QQQ:dd10:2026-01-05"]
    assert got[4] == ["QQQ:newhigh:2026-01-05"]
    assert got[5] == ["QQQ:dd10:2026-01-07"]


def test_no_alert_episode_does_not_announce_end():
    assert _replay([100, 95, 101, 97, 103]) == {}


def test_alert_log_roundtrip():
    conn = store.connect(":memory:")
    alerts = pending_drawdown_alerts("QQQ", _df([100, 85]), [10], set())
    store.record_alerts(conn, alerts, "2026-01-02")
    assert store.sent_alert_keys(conn, "QQQ:") == {"QQQ:dd10:2026-01-01"}
    assert store.sent_alert_keys(conn, "SPY:") == set()
    store.record_alerts(conn, alerts, "2026-01-03")  # 幂等
    assert len(store.sent_alert_keys(conn)) == 1


def test_config_drawdown_alert_symbols_get_price_updates():
    from quant.config import load_config
    cfg = load_config()
    for sym, levels in cfg.drawdown_alerts.items():
        assert sym in cfg.update_symbols
        assert levels == sorted(levels) and all(0 < x < 100 for x in levels)
