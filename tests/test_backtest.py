import numpy as np
import pandas as pd
import pytest

from quant.backtest.engine import (
    dca_equity,
    equity_metrics,
    hold_equity,
    run_backtest,
    run_portfolio_backtest,
    run_smart_dca_backtest,
)
from quant.strategies.base import BUY, SELL, Signal


def make_df(closes, adj=None, start="2024-01-01") -> pd.DataFrame:
    closes = np.asarray(closes, dtype=float)
    idx = pd.bdate_range(start, periods=len(closes))
    return pd.DataFrame({
        "open": closes, "high": closes, "low": closes,
        "close": closes,
        "adj_close": closes if adj is None else np.asarray(adj, dtype=float),
        "volume": 1000,
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
    assert result.trades[0]["entry_date"] == d[0]
    assert result.trades[0]["exit_date"] == d[10]
    assert result.open_position is None


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
    assert result.open_position == {"entry_date": d[0], "entry": 100.0}


def test_other_symbol_signals_filtered_out():
    df = make_df([100, 110])
    d = [ts.strftime("%Y-%m-%d") for ts in df.index]
    other = Signal(date=d[0], symbol="OTHER", strategy="s", direction=BUY,
                   price=100, strength=0.5, reason="x")
    result = run_backtest(df, [other], "TEST", "s")
    assert result.total_return == 0.0


def test_backtest_uses_adj_close():
    # close 横盘、adj_close 上涨（分红再投资），收益应按 adj 口径
    df = make_df([100] * 11, adj=np.linspace(100, 110, 11))
    d = [ts.strftime("%Y-%m-%d") for ts in df.index]
    result = run_backtest(df, [sig(d[0], BUY, 100), sig(d[10], SELL, 100)], "TEST", "s")
    assert result.total_return == pytest.approx(0.10, abs=1e-9)


def test_cost_bps_reduces_return():
    df = make_df(np.linspace(100, 110, 11))
    d = [ts.strftime("%Y-%m-%d") for ts in df.index]
    signals = [sig(d[0], BUY, 100), sig(d[10], SELL, 110)]
    gross = run_backtest(df, signals, "TEST", "s")
    net = run_backtest(df, signals, "TEST", "s", cost_bps=100)  # 单边 1%
    expected = 1.10 * (1 - 0.01) ** 2 - 1
    assert net.total_return == pytest.approx(expected, rel=1e-9)
    assert net.total_return < gross.total_return


def test_equity_metrics_known_series():
    equity = pd.Series([100.0, 110, 99, 108.9],
                       index=pd.bdate_range("2024-01-01", periods=4))
    m = equity_metrics(equity)
    assert m["total_return"] == pytest.approx(0.089)
    assert m["max_drawdown"] == pytest.approx(-0.10)
    assert m["volatility"] > 0
    assert m["calmar"] == pytest.approx(m["cagr"] / 0.10)


def test_hold_and_dca_benchmarks():
    px = make_df(np.linspace(100, 120, 42))["adj_close"]
    hold = hold_equity(px, 10_000)
    assert hold.iloc[-1] == pytest.approx(12_000)
    dca = dca_equity(px, 10_000)
    assert dca.iloc[0] == pytest.approx(10_000)
    # 上涨行情中定投终值应低于长持（后买的份额更贵）
    assert 10_000 < dca.iloc[-1] < hold.iloc[-1]
    # 成本使两者终值都变低
    assert hold_equity(px, 10_000, cost_bps=100).iloc[-1] < hold.iloc[-1]
    assert dca_equity(px, 10_000, cost_bps=100).iloc[-1] < dca.iloc[-1]


def psig(date, symbol, direction, price=100.0):
    return Signal(date=date, symbol=symbol, strategy="p", direction=direction,
                  price=price, strength=0.5, reason="test")


def test_portfolio_swap_sell_funds_same_day_buy():
    # A 横盘、B 从换仓日起翻倍；day0 买 A，day5 卖 A 同日买 B
    a = make_df([100.0] * 20)
    b = make_df([50.0] * 5 + list(np.linspace(50, 100, 15)))
    d = [ts.strftime("%Y-%m-%d") for ts in a.index]
    signals = [psig(d[0], "A", BUY), psig(d[5], "A", SELL), psig(d[5], "B", BUY)]
    result = run_portfolio_backtest({"A": a, "B": b}, signals, "p")
    # 卖 A 所得当日全部买入 B：期末 = 10000 * (100/50)
    assert result.equity.iloc[5] == pytest.approx(10_000)   # 换仓日资金守恒
    assert result.equity.iloc[-1] == pytest.approx(20_000)
    assert result.num_trades == 1
    assert result.trades[0]["symbol"] == "A"
    assert result.open_positions[0]["symbol"] == "B"


def test_portfolio_initial_buys_split_cash():
    a = make_df([100.0] * 10)
    b = make_df([200.0] * 10)
    d = [ts.strftime("%Y-%m-%d") for ts in a.index]
    signals = [psig(d[0], "A", BUY), psig(d[0], "B", BUY)]
    result = run_portfolio_backtest({"A": a, "B": b}, signals, "p")
    assert result.equity.iloc[-1] == pytest.approx(10_000)
    assert len(result.open_positions) == 2


def test_smart_dca_pauses_and_topups():
    # 3 个月横盘（正常定投）→ 急跌触发死叉（暂停）→ 反弹金叉（补投）
    flat = [100.0] * 63
    crash = list(np.linspace(100, 60, 42))
    recover = list(np.linspace(60, 110, 63))
    df = make_df(flat + crash + recover)
    result = run_smart_dca_backtest(df, fast=5, slow=20)
    assert result.skipped_months > 0, "死叉期应暂停定投"
    assert result.topup_dates, "金叉后应补投"
    assert result.paused_spans, "应记录暂停区段"
    # 资金守恒：期末权益为正且全部资金已入账
    assert result.equity.iloc[-1] > 0
