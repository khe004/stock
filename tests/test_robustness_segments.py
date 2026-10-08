import pandas as pd
import pytest

from quant.analysis import robustness
from quant.strategies.base import BUY, Signal


def _prices(values):
    index = pd.bdate_range("2026-01-01", periods=len(values))
    return pd.DataFrame({"close": values, "adj_close": values}, index=index)


def test_segment_inherits_position_before_window(monkeypatch):
    prices = {"A": _prices([100, 110, 120, 130, 140, 150, 160])}
    first = prices["A"].index[0].strftime("%Y-%m-%d")
    signal = Signal(first, "A", "demo", BUY, 100.0, 1.0, "test")
    monkeypatch.setattr(robustness, "_signals", lambda *args: [signal])
    report = robustness.robustness_report("demo", {}, prices, 10_000, 0,
                                          offsets=(0,), n_windows=2)
    assert report["windows"][2]["strategy"]["total_return"] > 0


def test_equal_weight_buy_and_hold_charges_initial_cost():
    prices = {"A": _prices([100, 200, 200]), "B": _prices([100, 100, 200])}
    equity = robustness.equal_weight_equity(prices, 10_000, cost_bps=10)
    assert equity.iloc[0] == pytest.approx(9_990)
    assert equity.iloc[-1] == pytest.approx(19_980)


def test_equal_weight_holds_cash_until_late_listing():
    a = _prices([100, 110, 120, 130])
    b = _prices([50, 100]).set_axis(a.index[-2:])
    equity = robustness.equal_weight_equity({"A": a, "B": b}, 10_000)
    assert equity.iloc[0] == pytest.approx(10_000)
    assert equity.iloc[-1] == pytest.approx(16_500)


def _px(values, start="2024-01-01"):
    idx = pd.bdate_range(start, periods=len(values))
    return pd.DataFrame({"close": values, "adj_close": values}, index=idx)


def test_pool_equal_weight_holds_monthly_without_daily_rebalance():
    from quant.analysis.robustness import pool_equal_weight_equity
    # A 翻倍、B 不动：买入持有 = +50%；每日再平衡的"收益均值"口径会给出别的数
    a = _px([100, 110, 121, 150, 200])
    b = _px([100, 100, 100, 100, 100])
    eq = pool_equal_weight_equity({"A": a, "B": b}, {a.index[0]: ["A", "B"]}, 1000.0)
    assert eq.iloc[-1] == pytest.approx(1500.0)


def test_pool_equal_weight_new_pool_earns_from_next_day_and_pays_cost():
    from quant.analysis.robustness import pool_equal_weight_equity
    a = _px([100, 100, 100, 100])
    b = _px([100, 100, 200, 200])          # 第 3 天翻倍
    idx = a.index
    # 第 3 天才换进 B：当天的翻倍不该算给新池
    eq = pool_equal_weight_equity({"A": a, "B": b}, {idx[0]: ["A"], idx[2]: ["B"]}, 1000.0)
    assert eq.iloc[-1] == pytest.approx(1000.0)
    # 成本：建仓换手 1000、换池换手约 2000，各按 10bp 扣
    costed = pool_equal_weight_equity({"A": a, "B": b}, {idx[0]: ["A"], idx[2]: ["B"]},
                                      1000.0, cost_bps=10)
    assert costed.iloc[0] == pytest.approx(999.0)
    assert costed.iloc[-1] == pytest.approx(999.0 - 2 * 999.0 * 0.001, rel=1e-4)
