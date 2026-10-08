from datetime import date

import pandas as pd

from quant.analysis.health import check_health


def _df(last: str) -> pd.DataFrame:
    idx = pd.date_range(end=last, periods=3, freq="D")
    return pd.DataFrame({"close": [1.0, 2.0, 3.0]}, index=idx)


TODAY = date(2026, 10, 7)


def _check(prices, strategy, display=(), **kw):
    return check_health(prices, list(strategy), list(display), "2026-10-07", TODAY, **kw)


def test_all_fresh_is_ok():
    r = _check({"SPY": _df("2026-10-07"), "QQQ": _df("2026-10-07")}, ["SPY", "QQQ"],
               options=(45, []))
    assert r.ok and not r.notes
    assert "策略标的 2/2" in r.summary and "期权快照 45/45" in r.summary
    assert r.render().splitlines()[1].startswith("✅")


def test_signal_date_ahead_of_run_day_is_an_issue():
    # 2026-10-06 那个 bug 的直接症状：信号日期跑到运行日之后
    r = check_health({"SPY": _df("2026-10-08")}, ["SPY"], [], "2026-10-08", TODAY)
    assert not r.ok and "晚于运行日" in r.issues[0]


def test_stale_signal_date_is_an_issue_unless_backfilling():
    prices = {"SPY": _df("2026-09-25")}
    assert not check_health(prices, ["SPY"], [], "2026-09-25", TODAY).ok
    assert check_health(prices, ["SPY"], [], "2026-09-25", TODAY, check_run_lag=False).ok


def test_strategy_symbol_behind_or_missing_is_an_issue():
    r = _check({"SPY": _df("2026-10-07"), "TLT": _df("2026-10-05")}, ["SPY", "TLT", "GLD"])
    text = " ".join(r.issues)
    assert "TLT" in text and "GLD" in text and "策略标的 1/3" in r.summary


def test_long_dormant_symbol_is_a_note_not_an_issue():
    # 疑似退市的标的天天报警只会让人学会忽略警报
    prices = {"SPY": _df("2026-10-07"), "EA": _df("2026-08-10")}
    r = _check(prices, ["SPY", "EA"], pool_symbols={"EA"})
    assert r.ok and "EA" in r.notes[0]
    # 手挑的 watchlist 标的停更不降级
    assert not _check(prices, ["SPY", "EA"]).ok
    # 刚停更几天的个股仍是 issue：可能只是当天没拉到
    fresh = {"SPY": _df("2026-10-07"), "WBD": _df("2026-10-05")}
    assert not _check(fresh, ["SPY", "WBD"], pool_symbols={"WBD"}).ok


def test_display_symbols_tolerate_foreign_holidays():
    prices = {"SPY": _df("2026-10-07"), "300308.SZ": _df("2026-09-30"),
              "BTC-USD": _df("2026-10-08"), "OLD": _df("2026-09-01")}
    r = _check(prices, ["SPY"], ["300308.SZ", "BTC-USD", "OLD", "SPY"])
    assert r.ok and len(r.notes) == 1 and "OLD" in r.notes[0] and "300308" not in r.notes[0]


def test_fetch_and_option_failures_are_issues():
    r = _check({"SPY": _df("2026-10-07")}, ["SPY"], fetch_failed=["XLK"], options=(44, ["NVDA"]))
    text = " ".join(r.issues)
    assert "XLK" in text and "NVDA" in text and "期权快照 44/45" in r.summary
