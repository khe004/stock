import pytest

from quant.analysis.holdings import holdings_report, look_through_weights, replay_holdings
from quant.strategies.base import BUY, SELL, Signal


def sig(date, symbol, direction, strategy="momentum"):
    return Signal(date=date, symbol=symbol, strategy=strategy, direction=direction,
                  price=1.0, strength=0.5, reason=f"{symbol} {direction}")


SIGS = [
    sig("2026-08-03", "XLK", BUY), sig("2026-08-03", "XLE", BUY), sig("2026-08-03", "XLV", BUY),
    sig("2026-09-01", "XLI", BUY), sig("2026-09-01", "XLV", SELL),
    sig("2026-10-01", "XLV", BUY), sig("2026-10-01", "XLI", SELL),
]


def test_replay_tracks_rotation_and_last_change():
    h = replay_holdings("momentum", SIGS)
    assert set(h.symbols) == {"XLK", "XLE", "XLV"} and h.last_change == "2026-10-01"
    assert h.since["XLK"] == "2026-08-03" and h.since["XLV"] == "2026-10-01"


def test_replay_as_of_ignores_later_signals_and_month_mid_ranking():
    # 持仓只随信号变：9 月中旬排名怎么换位都不影响，直到 10-01 的信号
    h = replay_holdings("momentum", SIGS, as_of="2026-09-20")
    assert set(h.symbols) == {"XLK", "XLE", "XLI"} and h.last_change == "2026-09-01"


def test_same_day_sell_then_buy_of_same_symbol_keeps_it():
    sigs = [sig("2026-01-02", "TLT", BUY, "dual_momentum"),
            sig("2026-02-02", "TLT", BUY, "dual_momentum"), sig("2026-02-02", "TLT", SELL, "dual_momentum")]
    assert replay_holdings("dual_momentum", sigs).symbols == ["TLT"]


def test_all_sold_is_cash_and_dca_is_not_replayed():
    sigs = [sig("2026-01-02", "QQQ", BUY, "aggressive_mom"), sig("2026-03-02", "QQQ", SELL, "aggressive_mom")]
    cash = replay_holdings("aggressive_mom", sigs)
    assert cash.symbols == [] and "空仓" in cash.describe()
    dca = replay_holdings("smart_dca", [sig("2026-10-01", "SPY", BUY, "smart_dca")])
    assert not dca.replayable and "2026-10-01" in dca.describe()


def test_look_through_weights_sum_to_one():
    hs = [replay_holdings("momentum", SIGS),
          replay_holdings("smart_dca", [sig("2026-10-01", "SPY", BUY, "smart_dca")])]
    w = look_through_weights(hs, ["DBMF"])
    assert sum(w.values()) == pytest.approx(1.0)
    assert w["DBMF"] == pytest.approx(1 / 3) and w["XLK"] == pytest.approx(1 / 9)


def test_holdings_report_lists_only_configured_strategies():
    text = holdings_report({"momentum": SIGS}, ["momentum", "canary_mom"], ["DBMF"])
    assert "momentum: " in text and "canary_mom" not in text and "DBMF: 买入持有" in text
    assert holdings_report({}, [], []) == ""
