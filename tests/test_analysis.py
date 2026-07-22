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


def test_market_regime_detects_trend():
    up = make_df(np.linspace(100, 200, 300))
    down = make_df(np.linspace(200, 100, 300))
    assert market_regime(up)["risk_on"] is True
    assert market_regime(up)["dist"] > 0
    assert market_regime(down)["risk_on"] is False
    # 数据不足返回 None
    assert market_regime(make_df(np.linspace(100, 110, 50)))["risk_on"] is None
