"""卖 put 研究：合成指数回测（putwrite）+ 期权快照采集/前向结算（option_forward）。"""

import math

import numpy as np
import pandas as pd
import pytest

import quant.data.fetcher as fetcher
from quant.analysis.option_forward import (annotate_quotes, forward_summary, monthly_first,
                                           pick_csp, settle_csp)
from quant.analysis.putwrite import (beta_matched_blend, bs_put, bs_put_delta, cycle_summary,
                                     regression_beta, simulate_putwrite, strike_for_delta)
from quant.data import store


# ---------- Black-Scholes ----------

def test_bs_put_known_value_and_parity():
    # S=K=100, T=1, σ=20%, r=5% → put ≈ 5.5735（教科书值）
    assert bs_put(100, 100, 1.0, 0.2, 0.05) == pytest.approx(5.5735, abs=1e-3)
    # 到期/零波动 → 内在价值
    assert bs_put(90, 100, 0.0, 0.2) == 10
    assert bs_put(110, 100, 0.5, 0.0) == 0


def test_strike_for_delta_roundtrip():
    k = strike_for_delta(100, 30 / 252, 0.2, 0.30, 0.02)
    assert k < 100, "30-delta put 应是虚值"
    assert abs(bs_put_delta(100, k, 30 / 252, 0.2, 0.02)) == pytest.approx(0.30, abs=1e-9)
    with pytest.raises(ValueError):
        strike_for_delta(100, 0.1, 0.2, 1.2)


# ---------- 合成卖 put ----------

def _flat_market(n=130, spot=100.0, vol=0.2):
    idx = pd.bdate_range("2024-01-01", periods=n)
    return pd.Series(spot, index=idx), pd.Series(vol, index=idx)


def test_putwrite_flat_market_keeps_every_premium():
    """横盘：ATM put 每期到期作废，权益 = 本金 × Π(1+权利金率)。"""
    spot, iv = _flat_market()
    eq, cyc = simulate_putwrite(spot, iv, premium_haircut=0.0)
    assert len(cyc) >= 5
    assert (cyc["payoff_pct"] == 0).all() and not cyc["assigned"].any()
    expected = 10_000 * np.prod(1 + cyc["premium_pct"])
    # 最后一张还在场内：期末权益 = 现金 − 未到期期权价值，小于把权利金全算进去
    settled_value = eq.loc[cyc["expiry"].iloc[-1]]
    assert settled_value == pytest.approx(expected, rel=1e-9)
    assert eq.iloc[0] == pytest.approx(10_000)


def test_putwrite_crash_cycle_loss_matches_intrinsic():
    """月中暴跌 20% 并停在那里：当期亏损 = 权利金 − (K−S_T)/K。"""
    spot, iv = _flat_market(n=60)
    first_months = spot.index.to_period("M")
    crash_from = spot.index[(first_months == first_months[0]).sum() + 5]
    spot = spot.where(spot.index < crash_from, 80.0)
    eq, cyc = simulate_putwrite(spot, iv, premium_haircut=0.0)
    hit = cyc.iloc[1]
    assert hit["assigned"]
    assert hit["ret_pct"] == pytest.approx(hit["premium_pct"] - 0.20)
    summ = cycle_summary(cyc)
    assert summ["worst"] == pytest.approx(hit["ret_pct"])
    assert summ["worst_in_premiums"] > 5


def test_putwrite_otm_strike_and_haircut():
    spot, iv = _flat_market()
    eq_atm, cyc_atm = simulate_putwrite(spot, iv, premium_haircut=0.0)
    eq_otm, cyc_otm = simulate_putwrite(spot, iv, target_delta=0.25, premium_haircut=0.0)
    assert (cyc_otm["strike"] < 100).all()
    assert cyc_otm["delta"].abs().round(6).eq(0.25).all()
    assert (cyc_otm["premium_pct"] < cyc_atm["premium_pct"]).all()
    _, cyc_hc = simulate_putwrite(spot, iv, premium_haircut=0.10)
    assert cyc_hc["premium_pct"].iloc[0] == pytest.approx(cyc_atm["premium_pct"].iloc[0] * 0.9)


def test_putwrite_cash_return_accrues():
    spot, iv = _flat_market()
    cash = pd.Series(0.0001, index=spot.index)
    eq0, _ = simulate_putwrite(spot, iv, premium_haircut=0.0)
    eq1, _ = simulate_putwrite(spot, iv, cash_ret=cash, premium_haircut=0.0)
    assert eq1.iloc[-1] > eq0.iloc[-1]


def test_beta_matched_blend_and_beta():
    idx = pd.bdate_range("2024-01-01", periods=300)
    rng = np.random.default_rng(0)
    mkt = pd.Series(rng.normal(0.0004, 0.01, len(idx)), index=idx)
    strat = 0.5 * mkt
    assert regression_beta(strat, mkt, freq=None) == pytest.approx(0.5, abs=1e-9)
    assert regression_beta(strat, mkt) == pytest.approx(0.5, abs=0.01)  # 月度复利近似
    blend = beta_matched_blend(mkt, pd.Series(0.0, index=idx), 0.5)
    assert blend.iloc[0] == pytest.approx(10_000)
    r = blend.pct_change().dropna()
    assert np.allclose(r.to_numpy(), 0.5 * mkt.iloc[1:].to_numpy())


# ---------- 期权快照：采集 ----------

def test_select_expirations_keeps_weekly_fridays_in_window():
    exps = ["2026-10-07", "2026-10-09", "2026-10-12", "2026-10-14", "2026-10-16",
            "2026-11-20", "2026-11-25", "2026-11-26", "2026-12-31", "2027-03-19"]
    # 2026-11-26 是周四且该周没有周五到期（模拟假日周）→ 保留周四
    got = fetcher.select_expirations(exps, "2026-10-06", 7, 60)
    assert got == ["2026-10-16", "2026-11-20", "2026-11-26"]
    # 同一周既有周四又有周五 → 只留周五
    got2 = fetcher.select_expirations(["2026-10-15", "2026-10-16"], "2026-10-06", 7, 60)
    assert got2 == ["2026-10-16"]


def _quotes(exp="2026-11-06"):
    return pd.DataFrame({
        "expiration": [exp, exp], "option_type": ["put", "put"], "strike": [90.0, 95.0],
        "bid": [1.0, 2.0], "ask": [1.2, 2.2], "last": [1.1, 2.1], "volume": [10, np.nan],
        "open_interest": [100, 200], "iv": [0.30, 0.28], "last_trade": [pd.NaT, pd.NaT],
    })


def test_save_option_snapshot_is_idempotent():
    conn = store.connect(":memory:")
    n = store.save_option_snapshot(conn, "2026-10-06", "AAA", 100.0, "2026-10-28",
                                   _quotes(), "2026-10-06T21:00:00+00:00")
    assert n == 2
    assert store.save_option_snapshot(conn, "2026-10-06", "AAA", 101.0, None,
                                      _quotes(), "later") == 0
    q = store.load_option_quotes(conn, "2026-10-06")
    assert len(q) == 2 and q["spot"].iloc[0] == 100.0
    assert q["next_earnings"].iloc[0] == "2026-10-28"
    assert pd.isna(q.loc[q["strike"] == 95.0, "volume"].iloc[0])


def test_update_option_snapshots_uses_db_date_and_skips_existing(monkeypatch):
    conn = store.connect(":memory:")
    idx = pd.bdate_range("2026-10-01", "2026-10-06")
    store.upsert_prices(conn, "AAA", pd.DataFrame(
        {"open": 1.0, "high": 1.0, "low": 1.0, "close": 100.0, "adj_close": 100.0,
         "volume": 1}, index=idx))
    calls = []

    def fake_fetch(symbol, as_of, spot, *a, **kw):
        calls.append((symbol, as_of, spot))
        return _quotes(), None

    monkeypatch.setattr(fetcher, "fetch_option_puts", fake_fetch)
    report: dict = {}
    ok, failed = fetcher.update_option_snapshots(conn, ["AAA", "NOPX"], report=report)
    assert ok == 1 and failed == ["NOPX"]
    assert calls == [("AAA", "2026-10-06", 100.0)]
    ok2, _ = fetcher.update_option_snapshots(conn, ["AAA"], report=report)
    assert ok2 == 0 and report["AAA"][0] == "skipped"
    assert len(calls) == 1


def test_options_research_config_symbols_get_price_updates():
    """结算要用到期日收盘价 → 期权采集的标的必须都在每日行情更新列表里。"""
    from quant.config import load_config
    cfg = load_config()
    opt = cfg.options_research
    assert opt["symbols"]
    missing = [s for s in opt["symbols"] if s not in cfg.update_symbols]
    assert not missing, f"这些期权标的没有日线更新，无法结算: {missing}"
    assert opt["min_dte"] < opt["max_dte"]


# ---------- 期权快照：前向结算 ----------

def _annotated():
    q = pd.DataFrame({
        "snapshot_date": ["2026-10-06"] * 4,
        "symbol": ["AAA"] * 4,
        "expiration": ["2026-11-06", "2026-11-06", "2026-11-06", "2026-12-18"],
        "option_type": ["put"] * 4,
        "strike": [85.0, 90.0, 95.0, 90.0],
        "bid": [0.4, 1.0, 2.2, 2.0],
        "iv": [0.35, 0.32, 0.30, 0.32],
        "spot": [100.0] * 4,
        "next_earnings": ["2026-10-28"] * 4,
    })
    return annotate_quotes(q)


def test_annotate_quotes_metrics():
    a = _annotated()
    row = a[(a["strike"] == 90.0) & (a["expiration"] == "2026-11-06")].iloc[0]
    assert row["dte"] == 31
    assert row["buffer"] == pytest.approx(0.10)
    assert row["prem_yield"] == pytest.approx(1 / 90)
    assert row["ann_yield"] == pytest.approx(1 / 90 * 365 / 31)
    assert -0.5 < row["delta"] < 0
    assert row["earnings_in_window"]


def test_pick_and_settle_csp():
    a = _annotated()
    picks = pick_csp(a, target_dte=30, target_delta=0.25)
    assert len(picks) == 1
    p = picks.iloc[0]
    assert p["expiration"] == "2026-11-06", "应选离 30 天最近的到期日"
    deltas = a[a["expiration"] == "2026-11-06"]["delta"].abs()
    assert abs(abs(p["delta"]) - 0.25) == pytest.approx((deltas - 0.25).abs().min())

    idx = pd.bdate_range("2026-10-06", "2026-11-06")
    closes = {"AAA": pd.Series(np.linspace(100, p["strike"] - 5, len(idx)), index=idx)}
    s = settle_csp(picks, closes).iloc[0]
    assert s["status"] == "settled" and s["assigned"]
    assert s["ret"] == pytest.approx(p["bid"] / p["strike"] - 5 / p["strike"])

    short = {"AAA": closes["AAA"].iloc[:5]}
    assert settle_csp(picks, short).iloc[0]["status"] == "open"


def test_forward_summary_groups_and_monthly_first():
    settled = pd.DataFrame({
        "snapshot_date": ["2026-10-06", "2026-10-07", "2026-11-02", "2026-11-02"],
        "symbol": ["SPY", "SPY", "AAA", "SPY"],
        "status": ["settled"] * 4,
        "prem_yield": [0.01, 0.01, 0.02, 0.01],
        "ret": [0.01, 0.01, -0.08, 0.01],
        "assigned": [False, False, True, False],
    })
    m = monthly_first(settled)
    assert list(m["snapshot_date"]) == ["2026-10-06", "2026-11-02", "2026-11-02"]
    summ = forward_summary(settled, index_symbols={"SPY"}).set_index("分组")
    assert summ.loc["全部", "笔数"] == 4
    assert summ.loc["个股", "最差单笔"] == pytest.approx(-0.08)
    assert summ.loc["个股", "最差≈几笔权利金"] == pytest.approx(4.0)
    assert math.isnan(summ.loc["指数 ETF", "最差≈几笔权利金"])
