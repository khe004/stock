import numpy as np
import pandas as pd
import pytest

from quant.analysis.market import range_position, sector_breadth, yield_curve_spread
from quant.analysis.scoring import signal_forward_returns, summarize_scores
from quant.strategies.base import BUY, SELL, Signal


def make_df(closes, start="2024-01-01"):
    closes = np.asarray(closes, dtype=float)
    idx = pd.bdate_range(start, periods=len(closes))
    return pd.DataFrame({
        "open": closes, "high": closes, "low": closes,
        "close": closes, "adj_close": closes, "volume": 1000,
    }, index=idx)


def test_range_position_bounds():
    prices = pd.Series(np.linspace(100, 150, 60))
    assert range_position(prices, window=60) == pytest.approx(1.0)
    prices_down = pd.Series(np.linspace(150, 100, 60))
    assert range_position(prices_down, window=60) == pytest.approx(0.0)
    mid = pd.Series([100.0, 200.0, 150.0])
    assert range_position(mid, window=3) == pytest.approx(0.5)


def test_range_position_insufficient_or_flat():
    assert range_position(pd.Series([100.0]), window=252) is None
    assert range_position(pd.Series([100.0, 100.0, 100.0]), window=3) is None


def test_sector_breadth():
    up = pd.Series(np.linspace(100, 150, 250))     # 上升趋势，站上均线
    down = pd.Series(np.linspace(150, 100, 250))    # 下降趋势，跌破均线
    short = pd.Series(np.linspace(100, 110, 50))    # 数据不足 200 日，应被跳过
    result = sector_breadth({"UP": up, "DOWN": down, "SHORT": short}, ma=200)
    assert result == {"above": 1, "total": 2}


def test_yield_curve_spread():
    long_y = pd.Series([4.5, 4.6])
    short_y = pd.Series([5.0, 5.1])
    assert yield_curve_spread(long_y, short_y) == pytest.approx(4.6 - 5.1)
    assert yield_curve_spread(pd.Series(dtype=float), short_y) is None


def sig(date, symbol, direction, strategy="s", price=100.0):
    return Signal(date=date, symbol=symbol, strategy=strategy, direction=direction,
                  price=price, strength=0.5, reason="test reason")


def test_forward_returns_buy_and_sell_sign():
    df = make_df(np.linspace(100, 130, 40))  # 单调上涨
    d = [ts.strftime("%Y-%m-%d") for ts in df.index]
    signals = [sig(d[0], "TEST", BUY), sig(d[0], "TEST", SELL)]
    fwd = signal_forward_returns(signals, {"TEST": df}, horizons=(5, 20))
    buy_row = fwd[fwd["direction"] == BUY].iloc[0]
    sell_row = fwd[fwd["direction"] == SELL].iloc[0]
    assert buy_row["ret_5"] > 0, "上涨行情里 buy 应为正收益"
    assert sell_row["ret_5"] < 0, "上涨行情里 sell 应为负收益（卖早了）"
    assert buy_row["ret_5"] == pytest.approx(-sell_row["ret_5"])


def test_forward_returns_pending_when_insufficient_future():
    df = make_df(np.linspace(100, 110, 10))
    d = [ts.strftime("%Y-%m-%d") for ts in df.index]
    signals = [sig(d[-1], "TEST", BUY)]  # 最后一天发信号，未来数据不够
    fwd = signal_forward_returns(signals, {"TEST": df}, horizons=(5, 20, 60))
    row = fwd.iloc[0]
    assert row["ret_now"] == pytest.approx(0.0)
    assert pd.isna(row["ret_5"])
    assert pd.isna(row["ret_20"])


def test_forward_returns_trade_symbol_map():
    # 信号标的是 ^VIX，但应映射到 SPY 计算真实收益
    vix = make_df([20.0] * 10)
    spy = make_df(np.linspace(100, 120, 10))
    d = [ts.strftime("%Y-%m-%d") for ts in vix.index]
    signals = [sig(d[0], "^VIX", BUY, strategy="vix_regime")]
    fwd = signal_forward_returns(
        signals, {"^VIX": vix, "SPY": spy},
        horizons=(5,), trade_symbol_map={"vix_regime": "SPY"},
    )
    assert len(fwd) == 1
    assert fwd.iloc[0]["trade_symbol"] == "SPY"
    assert fwd.iloc[0]["ret_5"] > 0


def test_forward_returns_skips_signal_outside_price_range():
    df = make_df([100.0] * 5, start="2024-06-01")
    signals = [sig("2020-01-01", "TEST", BUY)]  # 信号日早于行情范围
    fwd = signal_forward_returns(signals, {"TEST": df})
    assert fwd.empty


def test_summarize_scores_groups_and_flags_low_sample():
    df = pd.DataFrame([
        {"strategy": "s", "direction": BUY, "ret_5": 0.02, "ret_20": 0.05, "ret_60": None},
        {"strategy": "s", "direction": BUY, "ret_5": -0.01, "ret_20": 0.03, "ret_60": None},
        {"strategy": "s", "direction": SELL, "ret_5": 0.01, "ret_20": None, "ret_60": None},
    ])
    summary = summarize_scores(df, horizons=(5, 20, 60), min_samples=2)
    buy_row = summary[summary["direction"] == BUY].iloc[0]
    assert buy_row["n"] == 2
    assert buy_row["n_5"] == 2
    assert buy_row["mean_5"] == pytest.approx(0.005)
    assert buy_row["win_5"] == pytest.approx(0.5)
    assert buy_row["n_60"] == 0
    assert buy_row["mean_60"] is None
    assert not buy_row["low_sample"]
    sell_row = summary[summary["direction"] == SELL].iloc[0]
    assert sell_row["low_sample"], "样本数 1 < min_samples 2 应标记样本不足"


def test_summarize_scores_empty():
    assert summarize_scores(pd.DataFrame()).empty


# ── screening 市场筛选 ──────────────────────────────────────────

from quant.analysis.screening import compute_strength, market_regime  # noqa: E402


def test_compute_strength_ranks_strongest_first():
    """强势标的（持续上涨、临近高点、站上均线）综合分应最高、排在最前。"""
    n = 300
    strong = make_df(np.linspace(100, 300, n))          # 持续大涨
    weak = make_df(np.linspace(200, 100, n))            # 持续下跌
    flat = make_df(100 + np.zeros(n) + np.linspace(0, 5, n))  # 基本走平微涨
    df = compute_strength({"STRONG": strong, "WEAK": weak, "FLAT": flat})
    assert list(df.index)[0] == "STRONG", "最强标的应排第一"
    assert df.index[-1] == "WEAK", "最弱标的应垫底"
    # 综合分在 [0,1]
    assert (df["composite"] >= 0).all() and (df["composite"] <= 1).all()
    # 强势标的：动量为正、站上均线
    assert df.loc["STRONG", "mom"] > 0
    assert bool(df.loc["STRONG", "above_ma"])
    assert not bool(df.loc["WEAK", "above_ma"])


def test_compute_strength_skips_short_history():
    """历史不足 lookback+1 的标的被跳过。"""
    short = make_df(np.linspace(100, 120, 50))   # 只有 50 天
    long = make_df(np.linspace(100, 200, 300))
    df = compute_strength({"SHORT": short, "LONG": long})
    assert "SHORT" not in df.index
    assert "LONG" in df.index


def test_compute_strength_empty():
    assert compute_strength({}).empty


def test_compute_strength_value_factor():
    """提供基本面时综合分融入价值分（动量半+价值半）：便宜股价值分更高。"""
    n = 300
    # 两只动量相近（都温和上涨），但 A 便宜（低PE=高盈利收益率）、B 贵
    a_px = make_df(np.linspace(100, 130, n))
    b_px = make_df(np.linspace(100, 130, n))
    fund = pd.DataFrame(
        {"trailing_pe": [8.0, 40.0]}, index=["A", "B"]  # A 便宜、B 贵
    )
    df = compute_strength({"A": a_px, "B": b_px}, fundamentals=fund)
    assert "value_score" in df.columns and "pe" in df.columns
    assert df.loc["A", "earn_yield"] == pytest.approx(1 / 8.0)
    # A 更便宜 → 价值分更高 → 综合分更高 → 排在 B 前
    assert df.loc["A", "value_score"] > df.loc["B", "value_score"]
    assert list(df.index)[0] == "A"


def test_compute_strength_dual_value_metrics():
    """价值分融合 forward盈利收益率 + EV/EBITDA：一次性收益压低的假便宜 PE 被 EV/EBITDA 制衡。"""
    n = 300
    px = {s: make_df(np.linspace(100, 130, n)) for s in ["FAKE", "REAL", "MID"]}
    # FAKE：一次性收益灌低 PE（PE 5 假便宜），但 EV/EBITDA 40 真贵
    # REAL：PE 12、EV/EBITDA 8，两口径都扎实便宜
    fund = pd.DataFrame(
        {"forward_pe": [5.0, 12.0, 20.0], "ev_to_ebitda": [40.0, 8.0, 15.0]},
        index=["FAKE", "REAL", "MID"],
    )
    df = compute_strength(px, fundamentals=fund)
    assert "ev_ebitda" in df.columns and "ebitda_yield" in df.columns
    # REAL 综合两口径最便宜 → 价值分最高；FAKE 的假便宜被 EV/EBITDA 拉回，未能凭低PE夺魁
    assert df.loc["REAL", "value_score"] > df.loc["FAKE", "value_score"]
    assert df.loc["FAKE", "value_score"] < 1.0


def test_compute_strength_ebitda_only_when_no_pe():
    """无 PE 但有 EV/EBITDA（如亏损但经营正常）时，价值分仍可由 EV/EBITDA 单独给出。"""
    n = 300
    px = {s: make_df(np.linspace(100, 130, n)) for s in ["A", "B"]}
    fund = pd.DataFrame(
        {"forward_pe": [None, None], "ev_to_ebitda": [8.0, 20.0]},
        index=["A", "B"],
    )
    df = compute_strength(px, fundamentals=fund)
    assert df["value_score"].notna().all()  # 靠 EV/EBITDA 也能给价值分
    assert df.loc["A", "value_score"] > df.loc["B", "value_score"]  # A 的 EV/EBITDA 更低=更便宜


def test_compute_strength_prefers_forward_pe():
    """默认用 forward PE：成长股 trailing 畸高但 forward 便宜时，按 forward 判价值。"""
    n = 300
    px = {s: make_df(np.linspace(100, 130, n)) for s in ["GROWTH", "VALUE"]}
    fund = pd.DataFrame(
        {"trailing_pe": [100.0, 20.0], "forward_pe": [10.0, 25.0]},
        index=["GROWTH", "VALUE"],
    )
    df = compute_strength(px, fundamentals=fund)
    # GROWTH 用 forward=10（而非 trailing=100）→ 盈利收益率更高、价值分更高
    assert df.loc["GROWTH", "pe"] == pytest.approx(10.0)
    assert df.loc["GROWTH", "earn_yield"] > df.loc["VALUE", "earn_yield"]


def test_compute_strength_forward_fallback_to_trailing():
    """forward 缺失/非正时回退 trailing。"""
    n = 300
    px = {s: make_df(np.linspace(100, 130, n)) for s in ["A", "B"]}
    fund = pd.DataFrame(
        {"trailing_pe": [12.0, 20.0], "forward_pe": [None, -5.0]},  # A 无 forward、B 负 forward
        index=["A", "B"],
    )
    df = compute_strength(px, fundamentals=fund)
    assert df.loc["A", "pe"] == pytest.approx(12.0)  # 回退 trailing
    assert df.loc["B", "pe"] == pytest.approx(20.0)  # 负 forward → 回退 trailing


def test_compute_strength_value_sector_neutral():
    """价值分行业内中性化：高PE行业里的便宜股，价值分应高于低PE行业里的贵股。"""
    n = 300
    px = {s: make_df(np.linspace(100, 130, n)) for s in ["TECH_A", "TECH_B", "BANK_A", "BANK_B"]}
    # 科技行业 PE 结构性高（30/50），银行结构性低（8/12）
    fund = pd.DataFrame(
        {"trailing_pe": [30.0, 50.0, 8.0, 12.0]},
        index=["TECH_A", "TECH_B", "BANK_A", "BANK_B"],
    )
    sectors = {"TECH_A": "科技", "TECH_B": "科技", "BANK_A": "银行", "BANK_B": "银行"}
    df = compute_strength(px, fundamentals=fund, sectors=sectors)
    # 行业内：TECH_A（30<50）是科技里便宜的 → 价值分 = 组内高分（1.0）
    #        TECH_B 是科技里贵的 → 组内低分
    assert df.loc["TECH_A", "value_score"] > df.loc["TECH_B", "value_score"]
    assert df.loc["BANK_A", "value_score"] > df.loc["BANK_B", "value_score"]
    # 关键：PE=30 的科技便宜股，价值分不输给 PE=12 的银行贵股（横截面会反过来）
    assert df.loc["TECH_A", "value_score"] >= df.loc["BANK_B", "value_score"]


def test_compute_strength_no_pe_falls_back_to_momentum():
    """负盈利/无 P/E 的标的价值分缺失，综合分退回只用动量分（不倒扣为0）。"""
    n = 300
    strong = make_df(np.linspace(100, 300, n))   # 强动量、但无 PE
    weak = make_df(np.linspace(150, 120, n))     # 弱动量、有便宜 PE
    fund = pd.DataFrame({"trailing_pe": [None, 8.0]}, index=["STRONG", "WEAK"])
    df = compute_strength({"STRONG": strong, "WEAK": weak}, fundamentals=fund)
    # STRONG 无 PE：composite == trend_score（未被价值缺失拖到 0）
    assert pd.isna(df.loc["STRONG", "earn_yield"])
    assert df.loc["STRONG", "composite"] == pytest.approx(df.loc["STRONG", "trend_score"])


def test_compute_strength_reit_pe_leg_excluded():
    """REIT（房地产）PE 腿整段剔除：即便给了 PE，也不进 earn_yield/value_score，
    只靠 EV/EBITDA 定价值——折旧压低 GAAP 利润让 PE 结构性失真（如 ARE 曾为负PE）。"""
    n = 300
    px = {s: make_df(np.linspace(100, 130, n)) for s in ["REIT_A", "REIT_B", "TECH_A"]}
    fund = pd.DataFrame(
        # REIT_A 的 PE 看着比 REIT_B 便宜很多（8 vs 40），但 EV/EBITDA 反过来（贵）
        {"trailing_pe": [8.0, 40.0, 15.0], "ev_to_ebitda": [30.0, 10.0, 12.0]},
        index=["REIT_A", "REIT_B", "TECH_A"],
    )
    sectors = {"REIT_A": "房地产", "REIT_B": "房地产", "TECH_A": "科技"}
    df = compute_strength(px, fundamentals=fund, sectors=sectors)
    # REIT 的 pe/earn_yield 被整段剔除（NaN），非 REIT 的科技股不受影响
    assert pd.isna(df.loc["REIT_A", "pe"]) and pd.isna(df.loc["REIT_A", "earn_yield"])
    assert pd.isna(df.loc["REIT_B", "pe"]) and pd.isna(df.loc["REIT_B", "earn_yield"])
    assert df.loc["TECH_A", "pe"] == pytest.approx(15.0)
    # value_score 完全由 EV/EBITDA 决定：REIT_B（EV/EBITDA 10，更便宜）应高于 REIT_A（30）
    # 若 PE 腿没被剔除，PE 更"便宜"的 REIT_A 会拉高其价值分，结论会反过来
    assert df.loc["REIT_B", "value_score"] > df.loc["REIT_A", "value_score"]


def test_compute_strength_reit_pe_kept_when_no_sectors():
    """未提供 sectors 时无法识别 REIT，PE 腿正常参与（不误伤，只是没有行业信息可用）。"""
    n = 300
    px = {s: make_df(np.linspace(100, 130, n)) for s in ["A", "B"]}
    fund = pd.DataFrame({"trailing_pe": [8.0, 40.0]}, index=["A", "B"])
    df = compute_strength(px, fundamentals=fund)  # 不传 sectors
    assert df.loc["A", "pe"] == pytest.approx(8.0)


def test_market_regime_detects_trend():
    up = make_df(np.linspace(100, 200, 300))
    down = make_df(np.linspace(200, 100, 300))
    assert market_regime(up)["risk_on"] is True
    assert market_regime(up)["dist"] > 0
    assert market_regime(down)["risk_on"] is False
    # 数据不足返回 None
    assert market_regime(make_df(np.linspace(100, 110, 50)))["risk_on"] is None


# ── drawdowns 避险手册 ─────────────────────────────────────────
def test_find_drawdown_episodes_detects_dip_and_ongoing():
    from quant.analysis.drawdowns import find_drawdown_episodes
    # 涨到 120（新高）→ 跌到 96（-20%）→ 收复到 130（新高）→ 再跌到 123（-5%，进行中）
    up = list(np.linspace(100, 120, 40))
    down = list(np.linspace(120, 96, 30))
    recover = list(np.linspace(96, 130, 40))
    dip2 = list(np.linspace(130, 123, 15))
    s = pd.Series(up + down + recover + dip2,
                  index=pd.bdate_range("2020-01-01", periods=125))
    eps = find_drawdown_episodes(s, threshold=0.08)
    assert len(eps) == 2
    closed = [e for e in eps if not e["ongoing"]]
    ongoing = [e for e in eps if e["ongoing"]]
    assert len(closed) == 1 and len(ongoing) == 1
    assert closed[0]["maxdd"] < -0.19               # 约 -20%
    assert closed[0]["recover_date"] is not None
    # 进行中的段即便没到阈值也保留（用于实时"对号入座"）
    assert ongoing[0]["recover_date"] is None
    assert ongoing[0]["maxdd"] > -0.08


def test_classify_episode_flash_vs_deflation_vs_inflation():
    from quant.analysis.drawdowns import classify_episode
    fast = {"peak_date": pd.Timestamp("2020-01-01"),
            "end_date": pd.Timestamp("2020-01-15")}     # 14 天 → 闪崩
    slow = {"peak_date": pd.Timestamp("2020-01-01"),
            "end_date": pd.Timestamp("2020-06-01")}     # 慢跌
    assert classify_episode(fast, 0.10)[0] == "闪崩"
    assert classify_episode(slow, 0.10)[0] == "通缩/避险型"   # TLT 正
    assert classify_episode(slow, -0.20)[0] == "通胀/加息型"  # TLT 负
    assert classify_episode(slow, None)[0] == "通胀/加息型"   # TLT 缺→保守归通胀型


def test_episode_returns_total_return_window():
    from quant.analysis.drawdowns import episode_returns
    idx = pd.bdate_range("2022-01-03", periods=20)
    prices = pd.DataFrame({
        "TLT": np.linspace(100, 90, 20),    # 跌 10%
        "DBC": np.linspace(100, 120, 20),   # 涨 20%
    }, index=idx)
    r = episode_returns(prices, idx[0], idx[-1], ["TLT", "DBC"])
    assert r["TLT"] < 0 < r["DBC"]
    assert abs(r["DBC"] - 0.20) < 1e-6


# ── 稳健性检验（调仓日 timing luck / 分段 / 池子等权）────────────────────────

def test_month_anchors_offset_picks_nth_trading_day():
    from quant.strategies.base import month_anchors
    idx = pd.bdate_range("2024-01-01", "2024-04-30")
    assert list(month_anchors(idx, 0)[:2]) == [pd.Timestamp("2024-01-01"),
                                               pd.Timestamp("2024-02-01")]
    # offset=5 → 每月第 6 个交易日
    assert list(month_anchors(idx, 5)[:2]) == [pd.Timestamp("2024-01-08"),
                                               pd.Timestamp("2024-02-08")]
    assert len(month_anchors(idx, 0)) == len(month_anchors(idx, 5)) == 4
    assert len(month_anchors(pd.DatetimeIndex([]), 3)) == 0


def test_month_anchors_skips_months_too_short():
    from quant.strategies.base import month_anchors
    # 一月只有 3 个交易日 → offset=5 时该月无锚点
    idx = pd.DatetimeIndex(list(pd.bdate_range("2024-01-01", periods=3))
                           + list(pd.bdate_range("2024-02-01", periods=10)))
    assert len(month_anchors(idx, 0)) == 2
    assert len(month_anchors(idx, 5)) == 1


def test_rebalance_offset_changes_signal_dates_but_default_is_month_first():
    from quant import strategies
    from quant.strategies.base import month_anchors
    rng = np.random.default_rng(0)
    prices = {}
    for i, sym in enumerate(["A", "B", "C", "D"]):
        walk = 100 * np.exp(np.cumsum(rng.normal(0.0004 * (i + 1), 0.01, 700)))
        prices[sym] = make_df(walk, start="2022-01-03")
    params = {"lookback_days": 252, "skip_days": 21, "top_n": 2}
    idx = prices["A"].index
    base = strategies.build("momentum", params).generate(prices)
    assert base, "基准信号不应为空"
    first_days = {d.strftime("%Y-%m-%d") for d in month_anchors(idx, 0)}
    assert {s.date for s in base} <= first_days      # 默认 = 月首日，行为未变

    shifted = strategies.build("momentum", {**params, "rebalance_offset": 10}).generate(prices)
    tenth = {d.strftime("%Y-%m-%d") for d in month_anchors(idx, 10)}
    assert {s.date for s in shifted} <= tenth
    assert {s.date for s in shifted} != {s.date for s in base}


def test_equal_weight_equity_subset_excludes_defensive_leg():
    from quant.analysis.robustness import defensive_symbols, equal_weight_equity
    prices = {"A": make_df(np.linspace(100, 200, 60)),      # 翻倍
              "TLT": make_df(np.linspace(100, 50, 60))}     # 腰斩
    params = {"safe_assets": ["TLT"], "cash_asset": "BIL"}
    assert defensive_symbols(params) == {"TLT", "BIL"}
    pool = {s for s in prices if s not in defensive_symbols(params)}
    only_a = equal_weight_equity(prices, 10_000.0, pool)
    both = equal_weight_equity(prices, 10_000.0)
    assert only_a.iloc[-1] > both.iloc[-1]                  # 排除避险腿后基准更高
    assert equal_weight_equity({}, 10_000.0) is None


def test_split_windows_covers_history_without_gap():
    from quant.analysis.robustness import split_windows
    idx = pd.bdate_range("2015-01-01", "2026-01-01")
    w = split_windows(idx, 2)
    assert len(w) == 2
    assert w[0][1] == idx[0].strftime("%Y-%m-%d")
    assert w[-1][2] == idx[-1].strftime("%Y-%m-%d")
    assert w[0][2] == w[1][1]                               # 前段末 = 后段初，无缺口
    assert split_windows(pd.DatetimeIndex([]), 2) == []


def test_timing_luck_sweep_reports_spread_and_tranche():
    from quant.analysis.robustness import timing_luck_sweep
    rng = np.random.default_rng(7)
    prices = {}
    for i, sym in enumerate(["A", "B", "C", "D"]):
        walk = 100 * np.exp(np.cumsum(rng.normal(0.0003 * (i + 1), 0.012, 900)))
        prices[sym] = make_df(walk, start="2021-01-04")
    out = timing_luck_sweep("momentum",
                            {"lookback_days": 252, "skip_days": 21, "top_n": 2},
                            prices, 10_000.0, 5.0)
    assert len(out["per_offset"]) == 4
    assert out["per_offset"][0]["offset"] == 0
    assert out["spread_cagr_bps"] >= 0
    # 错峰组合应落在各单口径之间（不牺牲收益、纯降方差）
    cagrs = [r["cagr"] for r in out["per_offset"]]
    assert min(cagrs) - 1e-9 <= out["tranched"]["cagr"] <= max(cagrs) + 1e-9
    assert out["current_is_best"] == (cagrs[0] == max(cagrs))


def test_config_model_portfolio_is_explicit_and_consistent():
    """推荐配方必须显式配在 config；策略成分须是启用中且会推送的策略，
    买入持有成分（hold_assets）是 ETF 代码、不必是策略。"""
    from quant.config import load_config
    cfg = load_config()
    mp = cfg.model_portfolio
    assert mp, "config 应显式配置 model_portfolio"
    assert len(set(mp)) == len(mp), "推荐配方有重复成分"
    assert len(mp) >= 2, "组合至少要 2 个成分才谈得上分散"

    hold = set(cfg.model_portfolio_hold_assets)
    strat_names = [s for s in mp if s not in hold]         # 策略成分 = 全部减去买入持有
    params = dict(cfg.enabled_strategies())
    enabled = set(params)
    # 策略成分必须是启用中的策略，且会推送信号（推荐却不推是自相矛盾）
    assert set(strat_names) <= enabled, f"推荐配方含未启用策略：{set(strat_names) - enabled}"
    silent = [s for s in strat_names if not params[s].get("notify", True)]
    assert not silent, f"推荐配方策略成分未开启推送：{silent}"
    # 买入持有成分不应同时是策略名（避免语义歧义）
    assert not (hold & enabled), f"hold_assets 与策略名冲突：{hold & enabled}"


def test_levered_returns_charges_financing_and_scales_drawdown():
    from quant.analysis.robustness import leverage_to_target_vol, levered_returns
    idx = pd.bdate_range("2022-01-03", periods=500)
    rng = np.random.default_rng(11)
    r = pd.Series(rng.normal(0.0004, 0.010, 500), index=idx)
    cash = pd.Series(0.02 / 252, index=idx)            # 年化 2% 的无风险利率

    # k=1 时与原序列完全一致（不收融资费）
    assert np.allclose(levered_returns(r, 1.0, cash, 0.005), r)
    # 融资价差越高，收益越低
    a = (1 + levered_returns(r, 2.0, cash, 0.0)).prod()
    b = (1 + levered_returns(r, 2.0, cash, 0.02)).prod()
    assert a > b
    # 杠杆放大波动与回撤
    lev = levered_returns(r, 2.0, cash, 0.005)
    assert lev.std() > 1.9 * r.std()
    def dd(x):
        eq = (1 + x).cumprod()
        return float((eq / eq.cummax() - 1).min())
    assert dd(lev) < dd(r)
    # 目标波动率反解
    k = leverage_to_target_vol(r, float(r.std() * np.sqrt(252)) * 1.5)
    assert abs(k - 1.5) < 1e-9
    assert leverage_to_target_vol(pd.Series([0.0] * 10), 0.2) != leverage_to_target_vol(
        pd.Series([0.0] * 10), 0.2)   # 零波动 → NaN


def test_defense_spans_reconstructs_contiguous_periods():
    from quant.analysis.drawdowns import defense_spans
    from quant.strategies.base import BUY, SELL, Signal
    idx = pd.bdate_range("2022-01-03", periods=40)

    def sig(date, symbol, direction):
        return Signal(date=date, symbol=symbol, strategy="canary_mom",
                     direction=direction, price=100.0, strength=0.5, reason="test")

    d = [ts.strftime("%Y-%m-%d") for ts in idx]
    signals = [
        sig(d[5], "IEF", BUY),          # 第5天起进入防守
        sig(d[15], "IEF", SELL),
        sig(d[15], "SPY", BUY),         # 第15天起切回进攻（不算防守）
        sig(d[25], "SPY", SELL),
        sig(d[25], "BIL", BUY),         # 第25天起再次防守，直到序列末尾
    ]
    spans = defense_spans(signals, idx, {"IEF", "BIL"})
    assert spans == [(idx[5], idx[14]), (idx[25], idx[-1])]


def test_defense_spans_empty_when_never_in_defense():
    from quant.analysis.drawdowns import defense_spans
    from quant.strategies.base import BUY, Signal
    idx = pd.bdate_range("2022-01-03", periods=20)
    sig = Signal(date=idx[2].strftime("%Y-%m-%d"), symbol="SPY", strategy="x",
                direction=BUY, price=100.0, strength=0.5, reason="t")
    assert defense_spans([sig], idx, {"IEF", "BIL"}) == []


def test_spans_overlap():
    from quant.analysis.drawdowns import spans_overlap
    a0, a1 = pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-31")
    assert spans_overlap(a0, a1, pd.Timestamp("2020-01-15"), pd.Timestamp("2020-02-15"))  # 部分重叠
    assert spans_overlap(a0, a1, pd.Timestamp("2019-12-01"), pd.Timestamp("2020-01-05"))  # 部分重叠
    assert spans_overlap(a0, a1, a0, a1)                                                  # 完全重合
    assert not spans_overlap(a0, a1, pd.Timestamp("2020-02-01"), pd.Timestamp("2020-02-28"))  # 无重叠


# ── AI 基建计算层 ──


def _fin_df(rows):
    """构造 financials DataFrame 用于测试。
    rows = [(fiscal_date, revenue, gross_profit, operating_income, net_income), ...]
    如果 tuple 只有 4 个元素（省略 operating_income），自动补 None。"""
    expanded = []
    for r in rows:
        if len(r) == 4:
            # (fiscal_date, revenue, gross_profit, net_income) → 插入 operating_income=None
            expanded.append((r[0], r[1], r[2], None, r[3]))
        else:
            expanded.append(r)
    return pd.DataFrame(expanded, columns=["fiscal_date", "revenue", "gross_profit",
                                            "operating_income", "net_income"])


class TestAiInfraGrowth:
    """AI 基建增长指标测试。"""

    def test_cagr_3year_normal(self):
        """4 期年报 → 正常 3 年 CAGR。"""
        from quant.analysis.ai_infra import compute_growth_metrics
        # 营收从 100 增长到 800，3 年 CAGR = (800/100)^(1/3) - 1 = 1.0（+100%）
        df = _fin_df([
            ("2022-01-28", 100, 70, 10),
            ("2023-01-28", 200, 140, 20),
            ("2024-01-28", 400, 280, 40),
            ("2025-01-28", 800, 560, 80),
        ])
        m = compute_growth_metrics(df, "TEST")
        assert m.cagr_years == 3
        assert m.revenue_cagr == pytest.approx(1.0, abs=0.001)  # +100%
        assert m.revenue_yoy == pytest.approx(1.0, abs=0.001)   # 400→800 = +100%
        assert m.gross_margin == pytest.approx(0.7, abs=0.001)   # 560/800
        assert m.net_margin == pytest.approx(0.1, abs=0.001)     # 80/800

    def test_cagr_downgrade_to_2year(self):
        """只有 3 期年报 → 降级为 2 年 CAGR，并标注 cagr_years=2。"""
        from quant.analysis.ai_infra import compute_growth_metrics
        # SNDK 场景：分拆上市只有 3 期
        df = _fin_df([
            ("2023-06-30", 100, 50, 10),
            ("2024-06-30", 200, 100, 20),
            ("2025-06-30", 400, 200, 40),
        ])
        m = compute_growth_metrics(df, "SNDK")
        assert m.cagr_years == 2  # 标注为 2 年，不是 3 年
        assert m.revenue_cagr == pytest.approx(1.0, abs=0.001)  # (400/100)^(1/2)-1 = 1.0

    def test_single_period_no_growth(self):
        """只有 1 期年报 → 无法算增长，CAGR 和 YoY 都为 None。"""
        from quant.analysis.ai_infra import compute_growth_metrics
        df = _fin_df([("2025-01-28", 1000, 700, 100)])
        m = compute_growth_metrics(df, "X")
        assert m.revenue_cagr is None
        assert m.cagr_years is None
        assert m.revenue_yoy is None
        # 毛利率和净利率仍可算
        assert m.gross_margin == pytest.approx(0.7, abs=0.001)
        assert m.net_margin == pytest.approx(0.1, abs=0.001)

    def test_empty_data(self):
        """空 DataFrame → 全部 None。"""
        from quant.analysis.ai_infra import compute_growth_metrics
        df = pd.DataFrame(columns=["fiscal_date", "revenue", "gross_profit",
                                     "operating_income", "net_income"])
        m = compute_growth_metrics(df, "EMPTY")
        assert m.revenue_cagr is None
        assert m.gross_margin is None
        assert m.net_margin is None

    def test_zero_revenue_returns_none(self):
        """营收为 0 → CAGR/YoY 返回 None，不用 0 填充。"""
        from quant.analysis.ai_infra import compute_growth_metrics
        df = _fin_df([
            ("2023-01-28", 0, 0, 0),
            ("2024-01-28", 100, 70, 10),
            ("2025-01-28", 200, 140, 20),
        ])
        m = compute_growth_metrics(df, "ZERO")
        # 第一期营收为 0 被排除，只有 2 期有效 → 1 年 CAGR
        assert m.cagr_years == 1
        assert m.revenue_cagr == pytest.approx(1.0, abs=0.001)  # 100→200

    def test_negative_revenue_returns_none(self):
        """营收为负 → 该期从 CAGR 计算中排除。"""
        from quant.analysis.ai_infra import compute_growth_metrics
        df = _fin_df([
            ("2023-01-28", -50, 0, 0),
            ("2024-01-28", 100, 70, 10),
            ("2025-01-28", 200, 140, 20),
        ])
        m = compute_growth_metrics(df, "NEG")
        # 负营收期被排除，只有 2 期正营收 → 1 年 CAGR
        assert m.cagr_years == 1

    def test_missing_revenue_returns_none(self):
        """营收缺失 → CAGR 为 None。"""
        from quant.analysis.ai_infra import compute_growth_metrics
        df = _fin_df([
            ("2023-01-28", None, 50, 10),
            ("2024-01-28", None, 70, 20),
        ])
        m = compute_growth_metrics(df, "MISS")
        assert m.revenue_cagr is None
        assert m.revenue_yoy is None

    def test_gross_margin_none_when_revenue_zero(self):
        """营收为 0 时毛利率应为 None。"""
        from quant.analysis.ai_infra import compute_growth_metrics
        df = _fin_df([("2025-01-28", 0, 0, 0)])
        m = compute_growth_metrics(df, "X")
        assert m.gross_margin is None
        assert m.net_margin is None


class TestAiInfraMarketShare:
    """赛道统治力（市值份额）测试。"""

    def test_share_sums_to_one(self):
        """赛道内市值份额加总为 1。"""
        from quant.analysis.ai_infra import compute_lane_market_share
        caps = {"A": 1000.0, "B": 2000.0, "C": 3000.0}
        shares = compute_lane_market_share(["A", "B", "C"], caps)
        total = sum(v for v in shares.values() if v is not None)
        assert total == pytest.approx(1.0)
        assert shares["A"] == pytest.approx(1/6)
        assert shares["B"] == pytest.approx(2/6)
        assert shares["C"] == pytest.approx(3/6)

    def test_missing_cap_excluded_from_denominator(self):
        """市值缺失的成分从分母剔除，份额仍加总为 1（有效成分间）。"""
        from quant.analysis.ai_infra import compute_lane_market_share
        caps = {"A": 1000.0, "B": 2000.0, "C": None}
        shares = compute_lane_market_share(["A", "B", "C"], caps)
        assert shares["C"] is None
        valid_total = sum(v for v in shares.values() if v is not None)
        assert valid_total == pytest.approx(1.0)
        assert shares["A"] == pytest.approx(1/3)

    def test_all_missing_caps(self):
        """全部市值缺失 → 全部份额为 None。"""
        from quant.analysis.ai_infra import compute_lane_market_share
        caps = {"A": None, "B": None}
        shares = compute_lane_market_share(["A", "B"], caps)
        assert shares["A"] is None
        assert shares["B"] is None

    def test_multi_lane_no_error(self):
        """同一只股属于多个赛道时不重复计数 / 不报错。"""
        from quant.analysis.ai_infra import compute_lane_market_share
        caps = {"AVGO": 5000.0, "NVDA": 10000.0, "ANET": 2000.0, "CSCO": 3000.0}
        # AVGO 在两个赛道
        share_compute = compute_lane_market_share(["NVDA", "AVGO"], caps)
        share_network = compute_lane_market_share(["ANET", "CSCO", "AVGO"], caps)
        # 各赛道独立计算，不互相干扰
        assert share_compute["AVGO"] == pytest.approx(5000 / 15000)
        assert share_network["AVGO"] == pytest.approx(5000 / 10000)


class TestAiInfraLaneSummary:
    """赛道汇总测试。"""

    def test_weighted_return(self):
        """市值加权近 1 年涨幅。"""
        from quant.analysis.ai_infra import GrowthMetrics, compute_lane_summary
        caps = {"A": 1000.0, "B": 3000.0}
        returns = {"A": 0.10, "B": 0.30}
        growth = {
            "A": GrowthMetrics("A", 0.5, 3, 0.10, 0.7, 0.1),
            "B": GrowthMetrics("B", 0.3, 3, 0.30, 0.6, 0.2),
        }
        s = compute_lane_summary("test", ["A", "B"], caps, growth, returns)
        expected = (1000 * 0.10 + 3000 * 0.30) / 4000  # = 0.25
        assert s["近1年涨幅(市值加权)"] == pytest.approx(expected)
        assert s["成分数"] == 2
        assert s["合计市值"] == pytest.approx(4000.0)

    def test_weighted_momentum(self):
        """赛道 12-1 动量同样市值加权，与近1年涨幅同口径。"""
        from quant.analysis.ai_infra import GrowthMetrics, compute_lane_summary
        caps = {"A": 1000.0, "B": 3000.0}
        returns = {"A": 0.10, "B": 0.30}
        mom = {"A": 0.20, "B": 0.60}
        growth = {"A": GrowthMetrics("A", None, None, None, None, None),
                  "B": GrowthMetrics("B", None, None, None, None, None)}
        s = compute_lane_summary("test", ["A", "B"], caps, growth, returns, mom)
        expected = (1000 * 0.20 + 3000 * 0.60) / 4000  # = 0.50
        assert s["12-1动量(市值加权)"] == pytest.approx(expected)

    def test_multi_period_returns_weighted_and_optional(self):
        """3/5 年涨幅同样市值加权；不传时不输出对应列（向后兼容）。"""
        from quant.analysis.ai_infra import GrowthMetrics, compute_lane_summary
        caps = {"A": 1000.0, "B": 3000.0}
        growth = {"A": GrowthMetrics("A", None, None, None, None, None),
                  "B": GrowthMetrics("B", None, None, None, None, None)}
        s = compute_lane_summary("t", ["A", "B"], caps, growth, {"A": 0.1, "B": 0.2},
                                 returns_3y={"A": 0.5, "B": 1.5},
                                 returns_5y={"A": 1.0, "B": 3.0})
        assert s["近3年涨幅(市值加权)"] == pytest.approx((1000 * 0.5 + 3000 * 1.5) / 4000)
        assert s["近5年涨幅(市值加权)"] == pytest.approx((1000 * 1.0 + 3000 * 3.0) / 4000)
        # 不传则不输出
        s2 = compute_lane_summary("t", ["A", "B"], caps, growth, {"A": 0.1, "B": 0.2})
        assert "近3年涨幅(市值加权)" not in s2 and "近5年涨幅(市值加权)" not in s2

    def test_short_history_excluded_from_that_period_only(self):
        """次新股在长周期列上缺数据时，其市值只从该列分母剔除，不影响短周期列。"""
        from quant.analysis.ai_infra import GrowthMetrics, compute_lane_summary
        caps = {"OLD": 1000.0, "NEW": 9000.0}   # 次新股市值很大，若按 0 计会毁掉读数
        growth = {"OLD": GrowthMetrics("OLD", None, None, None, None, None),
                  "NEW": GrowthMetrics("NEW", None, None, None, None, None)}
        s = compute_lane_summary(
            "t", ["OLD", "NEW"], caps, growth,
            {"OLD": 0.10, "NEW": 0.90},          # 1 年：两者都有
            returns_5y={"OLD": 2.0, "NEW": None},  # 5 年：次新股无数据
        )
        # 1 年列两者都参与
        assert s["近1年涨幅(市值加权)"] == pytest.approx((1000 * 0.10 + 9000 * 0.90) / 10000)
        # 5 年列只由 OLD 决定，不是被 NEW 的 0 拉低成 0.2
        assert s["近5年涨幅(市值加权)"] == pytest.approx(2.0)

    def test_momentum_column_omitted_when_not_provided(self):
        """不传 momentum 时不输出该列（向后兼容，既有调用方不受影响）。"""
        from quant.analysis.ai_infra import GrowthMetrics, compute_lane_summary
        caps = {"A": 1000.0}
        growth = {"A": GrowthMetrics("A", None, None, 0.1, None, None)}
        s = compute_lane_summary("test", ["A"], caps, growth, {"A": 0.1})
        assert "12-1动量(市值加权)" not in s

    def test_weighted_momentum_skips_missing_values(self):
        """某成分动量缺失时，其市值不计入分母（不能当 0 拉低整条赛道）。"""
        from quant.analysis.ai_infra import GrowthMetrics, compute_lane_summary
        caps = {"A": 1000.0, "B": 3000.0}
        mom = {"A": 0.20, "B": None}          # B 缺动量（如次新股窗口不足）
        growth = {"A": GrowthMetrics("A", None, None, None, None, None),
                  "B": GrowthMetrics("B", None, None, None, None, None)}
        s = compute_lane_summary("test", ["A", "B"], caps, growth, {}, mom)
        assert s["12-1动量(市值加权)"] == pytest.approx(0.20)  # 只由 A 决定，不是 0.05

    def test_median_growth(self):
        """营收增长中位数用的是 YoY。"""
        from quant.analysis.ai_infra import GrowthMetrics, compute_lane_summary
        caps = {"A": 1000.0, "B": 1000.0, "C": 1000.0}
        growth = {
            "A": GrowthMetrics("A", None, None, 0.10, None, None),
            "B": GrowthMetrics("B", None, None, 0.50, None, None),
            "C": GrowthMetrics("C", None, None, 1.00, None, None),
        }
        s = compute_lane_summary("test", ["A", "B", "C"], caps, growth, {})
        assert s["营收增长中位数"] == pytest.approx(0.50)  # 中位数
