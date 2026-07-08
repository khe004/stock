import numpy as np
import pandas as pd
import pytest

from quant.strategies.base import BUY, SELL
from quant.strategies.momentum import Momentum
from quant.strategies.rsi_reversal import RsiReversal
from quant.strategies.sma_cross import SmaCross


def make_df(closes) -> pd.DataFrame:
    closes = np.asarray(closes, dtype=float)
    idx = pd.bdate_range("2024-01-01", periods=len(closes))
    return pd.DataFrame({
        "open": closes, "high": closes * 1.01, "low": closes * 0.99,
        "close": closes, "adj_close": closes, "volume": 1_000_000,
    }, index=idx)


def test_sma_cross_golden_cross_after_uptrend():
    down = np.linspace(120, 80, 60)
    up = np.linspace(80, 140, 60)
    df = make_df(np.concatenate([down, up]))
    signals = SmaCross(fast=5, slow=20).generate({"TEST": df})
    buys = [s for s in signals if s.direction == BUY]
    assert buys, "上涨段应出现金叉买入信号"
    turning_point = df.index[60]
    assert all(pd.Timestamp(s.date) > turning_point for s in buys)
    assert all(0 < s.strength <= 1 for s in signals)
    assert "均线" in buys[0].reason


def test_sma_cross_death_cross_after_downtrend():
    up = np.linspace(80, 140, 60)
    down = np.linspace(140, 80, 60)
    df = make_df(np.concatenate([up, down]))
    signals = SmaCross(fast=5, slow=20).generate({"TEST": df})
    assert any(s.direction == SELL for s in signals), "下跌段应出现死叉卖出信号"


def test_sma_cross_too_short_history():
    df = make_df(np.linspace(100, 110, 10))
    assert SmaCross(fast=5, slow=20).generate({"TEST": df}) == []


def test_rsi_buy_after_oversold_bounce():
    flat = [100.0] * 30
    drop = [100 * (0.98**i) for i in range(1, 11)]
    bounce = [drop[-1] * (1.03**i) for i in range(1, 9)]
    df = make_df(flat + drop + bounce)
    signals = RsiReversal(period=14).generate({"TEST": df})
    buys = [s for s in signals if s.direction == BUY]
    assert buys, "超卖反弹应出现买入信号"
    bounce_start = df.index[40]
    assert pd.Timestamp(buys[0].date) > bounce_start


def test_rsi_sell_after_overbought_pullback():
    flat = [100.0] * 30
    rally = [100 * (1.02**i) for i in range(1, 16)]
    pullback = [rally[-1] * (0.97**i) for i in range(1, 6)]
    df = make_df(flat + rally + pullback)
    signals = RsiReversal(period=14).generate({"TEST": df})
    assert any(s.direction == SELL for s in signals), "超买回落应出现卖出信号"


@pytest.fixture
def rotation_prices():
    n = 100
    steady = 100 * 1.005 ** np.arange(n)          # A: 全程缓涨
    late = np.concatenate([                        # B: 前 70 天横盘，后 30 天加速
        np.full(70, 100.0), 100 * 1.02 ** np.arange(1, 31),
    ])
    return {
        "A": make_df(steady),
        "B": make_df(late),
        "C": make_df(np.full(n, 100.0)),
        "D": make_df(np.full(n, 100.0)),
    }


def test_momentum_rotation(rotation_prices):
    signals = Momentum(lookback_days=20, top_n=1).generate(rotation_prices)
    b_buys = [s for s in signals if s.symbol == "B" and s.direction == BUY]
    a_sells = [s for s in signals if s.symbol == "A" and s.direction == SELL]
    assert b_buys, "B 加速后应进入动量榜首，产生买入信号"
    assert pd.Timestamp(b_buys[0].date) > rotation_prices["B"].index[70]
    assert a_sells, "A 被 B 挤出榜首后应产生卖出信号"


def test_momentum_needs_enough_symbols(rotation_prices):
    two = {k: rotation_prices[k] for k in ["A", "B"]}
    assert Momentum(lookback_days=20, top_n=3).generate(two) == []
