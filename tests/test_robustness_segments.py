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
