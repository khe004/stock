"""策略相关性模块的纯函数测试——合成价格+信号，不联网、不依赖 streamlit。"""

import numpy as np
import pandas as pd
import pytest

from quant.analysis.correlation import (
    combined_portfolio,
    correlation_matrix,
    strategy_return_series,
)
from quant.strategies.base import BUY, SELL, Signal


def make_df(closes, start="2024-01-01"):
    """合成日线 DataFrame，与 test_analysis.py 同款。"""
    closes = np.asarray(closes, dtype=float)
    idx = pd.bdate_range(start, periods=len(closes))
    return pd.DataFrame({
        "open": closes, "high": closes, "low": closes,
        "close": closes, "adj_close": closes, "volume": 1000,
    }, index=idx)


def sig(date, symbol, direction, strategy="sma_cross", price=100.0):
    return Signal(date=date, symbol=symbol, strategy=strategy, direction=direction,
                  price=price, strength=0.5, reason="test")


# ──────────────────────────────────────────────────
# strategy_return_series
# ──────────────────────────────────────────────────

def test_strategy_return_series_single_strategy():
    """单标的策略（sma_cross）应产出日收益率序列。"""
    closes = np.linspace(100, 130, 60)
    df = make_df(closes)
    dates = [ts.strftime("%Y-%m-%d") for ts in df.index]
    signals = [sig(dates[5], "A", BUY), sig(dates[30], "A", SELL)]
    prices = {"A": df}

    result = strategy_return_series(
        prices=prices,
        strategy_signals={"sma_cross": signals},
        strategy_params={"sma_cross": {}},
        strategy_symbols={"sma_cross": ["A"]},
        cost_bps=0.0,
    )
    assert "sma_cross" in result.columns
    assert len(result) == len(df)
    # 买入前和卖出后的日收益率应为 0（持现金）
    assert result["sma_cross"].iloc[0] == 0.0


def test_strategy_return_series_multi_symbol_averaging():
    """单标的策略跨多个标的时，应等权平均日收益率。"""
    # 标的 A 稳步上涨，标的 B 稳步下跌
    up = np.linspace(100, 150, 40)
    down = np.linspace(100, 50, 40)
    df_a = make_df(up)
    df_b = make_df(down)
    dates = [ts.strftime("%Y-%m-%d") for ts in df_a.index]
    # 两个标的都全程持仓
    signals = [
        sig(dates[0], "A", BUY), sig(dates[0], "B", BUY),
    ]
    prices = {"A": df_a, "B": df_b}

    result = strategy_return_series(
        prices=prices,
        strategy_signals={"sma_cross": signals},
        strategy_params={"sma_cross": {}},
        strategy_symbols={"sma_cross": ["A", "B"]},
        cost_bps=0.0,
    )
    assert "sma_cross" in result.columns
    assert len(result) > 0


def test_strategy_return_series_empty_when_no_prices():
    """价格数据为空时应返回空 DataFrame。"""
    result = strategy_return_series(
        prices={},
        strategy_signals={"sma_cross": []},
        strategy_params={"sma_cross": {}},
        strategy_symbols={"sma_cross": ["MISSING"]},
        cost_bps=0.0,
    )
    assert result.empty


def test_strategy_return_series_smart_dca():
    """smart_dca 策略应走专用定投回测路径。"""
    closes = np.linspace(100, 130, 120)
    df = make_df(closes)
    prices = {"SPY": df}

    result = strategy_return_series(
        prices=prices,
        strategy_signals={"smart_dca": []},
        strategy_params={"smart_dca": {"symbol": "SPY", "fast": 20, "slow": 60}},
        strategy_symbols={"smart_dca": ["SPY"]},
        cost_bps=5.0,
    )
    assert "smart_dca" in result.columns
    assert len(result) == len(df)


def test_strategy_return_series_vix_regime():
    """vix_regime 应映射到 trade_symbol 后按单标的处理。"""
    vix_closes = [20.0] * 60
    spy_closes = np.linspace(100, 130, 60)
    df_vix = make_df(vix_closes)
    df_spy = make_df(spy_closes)
    dates = [ts.strftime("%Y-%m-%d") for ts in df_vix.index]
    signals = [sig(dates[5], "^VIX", BUY, strategy="vix_regime")]
    prices = {"^VIX": df_vix, "SPY": df_spy}

    result = strategy_return_series(
        prices=prices,
        strategy_signals={"vix_regime": signals},
        strategy_params={"vix_regime": {"trade_symbol": "SPY"}},
        strategy_symbols={"vix_regime": ["^VIX"]},
        cost_bps=0.0,
        trade_symbol_map={"vix_regime": "SPY"},
    )
    assert "vix_regime" in result.columns


# ──────────────────────────────────────────────────
# correlation_matrix
# ──────────────────────────────────────────────────

def test_correlation_matrix_shape_and_diagonal():
    """相关矩阵应为 N×N、对角线为 1、对称。"""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "A": np.random.randn(n),
        "B": np.random.randn(n),
        "C": np.random.randn(n),
    })
    corr = correlation_matrix(df)
    assert corr.shape == (3, 3)
    # 对角线为 1
    for col in corr.columns:
        assert corr.loc[col, col] == pytest.approx(1.0)
    # 对称
    assert corr.loc["A", "B"] == pytest.approx(corr.loc["B", "A"])
    assert corr.loc["A", "C"] == pytest.approx(corr.loc["C", "A"])


def test_correlation_identical_series():
    """两条完全相同的序列相关系数应为 1。"""
    np.random.seed(123)
    data = np.random.randn(50)
    df = pd.DataFrame({"X": data, "Y": data})
    corr = correlation_matrix(df)
    assert corr.loc["X", "Y"] == pytest.approx(1.0)


def test_correlation_negatively_correlated():
    """完全负相关序列的相关系数应为 -1。"""
    data = np.arange(50, dtype=float)
    df = pd.DataFrame({"A": data, "B": -data})
    corr = correlation_matrix(df)
    assert corr.loc["A", "B"] == pytest.approx(-1.0)


def test_correlation_matrix_single_column():
    """只有一列策略时仍应返回 1×1 矩阵。"""
    df = pd.DataFrame({"only": np.random.randn(20)})
    corr = correlation_matrix(df)
    assert corr.shape == (1, 1)
    assert corr.iloc[0, 0] == pytest.approx(1.0)


def test_correlation_matrix_empty():
    """空 DataFrame 应返回空。"""
    corr = correlation_matrix(pd.DataFrame())
    assert corr.empty


# ──────────────────────────────────────────────────
# combined_portfolio
# ──────────────────────────────────────────────────

def test_combined_portfolio_equity_and_metrics():
    """等权组合应正确合成权益曲线，指标非空。"""
    np.random.seed(0)
    n = 200
    df = pd.DataFrame({
        "A": np.random.randn(n) * 0.01,
        "B": np.random.randn(n) * 0.01,
    }, index=pd.bdate_range("2024-01-01", periods=n))

    equity, metrics = combined_portfolio(df, initial_value=10_000.0)
    assert len(equity) == n
    assert "cagr" in metrics
    assert "sharpe" in metrics
    assert "max_drawdown" in metrics
    assert "volatility" in metrics
    assert float(equity.iloc[0]) == pytest.approx(10_000.0 * (1 + df.iloc[0].mean()))


def test_combined_portfolio_single_strategy():
    """只有一个策略时，等权组合就是它本身。"""
    np.random.seed(1)
    rets = np.random.randn(100) * 0.01
    df = pd.DataFrame({"only": rets}, index=pd.bdate_range("2024-01-01", periods=100))

    equity, metrics = combined_portfolio(df)
    expected = 10_000.0 * (1 + pd.Series(rets)).cumprod()
    assert equity.iloc[-1] == pytest.approx(expected.iloc[-1])


def test_combined_portfolio_empty():
    """空输入应返回空曲线和空指标。"""
    equity, metrics = combined_portfolio(pd.DataFrame())
    assert equity.empty
    assert metrics == {}


def test_combined_portfolio_diversification_effect():
    """不完全相关的策略组合后波动应低于各单策略均值（分散效果）。"""
    np.random.seed(42)
    n = 500
    # 两个不相关的策略
    df = pd.DataFrame({
        "A": np.random.randn(n) * 0.02,
        "B": np.random.randn(n) * 0.02,
    }, index=pd.bdate_range("2024-01-01", periods=n))

    _, combo_metrics = combined_portfolio(df)
    vol_a = float(df["A"].std()) * (252 ** 0.5)
    vol_b = float(df["B"].std()) * (252 ** 0.5)
    avg_vol = (vol_a + vol_b) / 2
    # 不相关策略的组合波动应低于单策略平均波动
    assert combo_metrics["volatility"] < avg_vol
