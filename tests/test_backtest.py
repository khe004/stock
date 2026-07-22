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
    vol_scaled_equity,
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
    assert result.trades[0]["profit"] == pytest.approx(1_000)
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
    assert result.trades[0]["profit"] == pytest.approx(0.0)
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


# ── vol_scaled_equity 测试 ──────────────────────────────────────

def test_vol_scaled_look_ahead_safe():
    """前视安全：某日权重只依赖之前的波动——构造分段波动序列，
    高波动段之后的权重应下降。"""
    # 中等波动 100 天 → 高波动 100 天
    rng = np.random.default_rng(42)
    med_vol = 100 + np.cumsum(rng.normal(0, 0.01, 100))   # ~1% 日波动（年化 ~16%）
    high_vol = med_vol[-1] + np.cumsum(rng.normal(0, 0.04, 100))  # ~4% 日波动（年化 ~63%）
    prices = np.concatenate([med_vol, high_vol])
    idx = pd.bdate_range("2023-01-01", periods=len(prices))
    equity = pd.Series(prices, index=idx)

    # cap 设很大确保不截断，纯测波动率驱动
    _, weights = vol_scaled_equity(equity, vol_window=20, target_vol=0.15, cap=100.0)

    # 窗口 20 天、shift(1) → 前 20 个位置权重为 0（无波动率估计）
    assert (weights.iloc[:20] == 0).all(), "窗口不足期权重应为 0"
    # 高波动段开始后（第 100 天之后），权重应下降
    late_med_vol_w = float(weights.iloc[90])   # 中等波动段尾部
    in_high_vol_w = float(weights.iloc[140])   # 高波动段已充分进入窗口
    assert in_high_vol_w < late_med_vol_w, "高波动段后权重应降低"


def test_vol_scaled_cap_enforced():
    """cap=1.0 时权重不超过 1.0。"""
    # 非常低的波动率 → target_vol / realized 会很大，但应被 cap 截断
    prices = np.linspace(100, 101, 200)  # 极低波动的线性上涨
    idx = pd.bdate_range("2023-01-01", periods=200)
    equity = pd.Series(prices, index=idx)

    _, weights = vol_scaled_equity(equity, target_vol=0.15, cap=1.0)
    assert (weights <= 1.0 + 1e-10).all(), "权重不应超过 cap=1.0"
    # 低波动时权重应等于 cap（被截断）
    active = weights[weights > 0]
    assert float(active.min()) == pytest.approx(1.0), "低波动时权重应等于 cap"


def test_vol_scaled_reduces_volatility():
    """缩放后已实现波动率 ≤ 原始（在有波动变化的合成序列上）。"""
    rng = np.random.default_rng(123)
    # 混合波动：低→高→低
    segment1 = 100 + np.cumsum(rng.normal(0, 0.005, 100))
    segment2 = segment1[-1] + np.cumsum(rng.normal(0, 0.03, 100))
    segment3 = segment2[-1] + np.cumsum(rng.normal(0, 0.005, 100))
    prices = np.concatenate([segment1, segment2, segment3])
    idx = pd.bdate_range("2022-01-01", periods=len(prices))
    equity = pd.Series(prices, index=idx)

    scaled, _ = vol_scaled_equity(equity, target_vol=0.15, vol_window=20)
    orig_vol = float(equity.pct_change().dropna().std())
    scaled_vol = float(scaled.pct_change().dropna().std())
    assert scaled_vol <= orig_vol * 1.01, (
        f"缩放后波动率 {scaled_vol:.4f} 不应高于原始 {orig_vol:.4f}"
    )


def test_vol_scaled_nan_window_no_error():
    """窗口不足（NaN）时收益为 0、不报错。"""
    # 很短的序列，短于 vol_window
    prices = np.linspace(100, 110, 30)
    idx = pd.bdate_range("2024-01-01", periods=30)
    equity = pd.Series(prices, index=idx)

    scaled, weights = vol_scaled_equity(equity, vol_window=63)  # 窗口 > 数据长度
    assert len(scaled) == 30
    assert not scaled.isna().any(), "不应有 NaN"
    # 所有权重为 0（窗口不足 + shift(1)）
    assert (weights == 0).all()
    # 缩放后权益曲线应为平线（收益为 0）
    assert scaled.iloc[-1] == pytest.approx(10_000.0)
