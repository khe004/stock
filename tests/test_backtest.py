import numpy as np
import pandas as pd
import pytest

from quant.backtest.engine import run_backtest
from quant.strategies.base import BUY, SELL, Signal


def make_df(closes) -> pd.DataFrame:
    closes = np.asarray(closes, dtype=float)
    idx = pd.bdate_range("2024-01-01", periods=len(closes))
    return pd.DataFrame({
        "open": closes, "high": closes, "low": closes,
        "close": closes, "adj_close": closes, "volume": 1000,
    }, index=idx)


def sig(date, direction, price):
    return Signal(date=date, symbol="TEST", strategy="s", direction=direction,
                  price=price, strength=0.5, reason="test")


def test_single_winning_trade():
    df = make_df(np.linspace(100, 110, 11))  # 100 → 110
    d = [ts.strftime("%Y-%m-%d") for ts in df.index]
    result = run_backtest(df, [sig(d[0], BUY, 100), sig(d[10], SELL, 110)], "TEST", "s")
    assert result.total_return == pytest.approx(0.10, abs=1e-9)
    assert result.num_trades == 1
    assert result.win_rate == 1.0
    assert result.max_drawdown == pytest.approx(0.0, abs=1e-9)
    assert result.equity.iloc[-1] == pytest.approx(11_000)


def test_losing_trade_and_drawdown():
    df = make_df([100, 100, 90, 80, 80])
    d = [ts.strftime("%Y-%m-%d") for ts in df.index]
    result = run_backtest(df, [sig(d[1], BUY, 100), sig(d[3], SELL, 80)], "TEST", "s")
    assert result.total_return == pytest.approx(-0.20)
    assert result.max_drawdown == pytest.approx(-0.20)
    assert result.win_rate == 0.0
    assert result.num_trades == 1


def test_no_signals_flat_equity():
    df = make_df([100, 120, 80, 100])
    result = run_backtest(df, [], "TEST", "s")
    assert result.total_return == 0.0
    assert result.num_trades == 0
    assert (result.equity == 10_000).all()


def test_duplicate_buy_ignored_and_open_position_marked_to_market():
    df = make_df([100, 100, 105, 110])
    d = [ts.strftime("%Y-%m-%d") for ts in df.index]
    # 第二个 buy 应被忽略（已持仓），最后未平仓按市值计
    result = run_backtest(df, [sig(d[0], BUY, 100), sig(d[2], BUY, 105)], "TEST", "s")
    assert result.num_trades == 0  # 未平仓不算完成交易
    assert result.equity.iloc[-1] == pytest.approx(11_000)


def test_other_symbol_signals_filtered_out():
    df = make_df([100, 110])
    d = [ts.strftime("%Y-%m-%d") for ts in df.index]
    other = Signal(date=d[0], symbol="OTHER", strategy="s", direction=BUY,
                   price=100, strength=0.5, reason="x")
    result = run_backtest(df, [other], "TEST", "s")
    assert result.total_return == 0.0
