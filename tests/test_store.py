import pandas as pd

from quant.data import store
from quant.strategies.base import BUY, Signal


def make_conn():
    return store.connect(":memory:")


def sig(**kw):
    base = dict(date="2026-07-07", symbol="SPY", strategy="sma_cross",
                direction=BUY, price=500.0, strength=0.5, reason="test")
    base.update(kw)
    return Signal(**base)


def test_prices_roundtrip_and_incremental():
    conn = make_conn()
    idx = pd.bdate_range("2026-01-01", periods=3)
    df = pd.DataFrame({
        "open": [1.0, 2, 3], "high": [1.0, 2, 3], "low": [1.0, 2, 3],
        "close": [1.0, 2, 3], "adj_close": [1.0, 2, 3], "volume": [10, 20, 30],
    }, index=idx)
    assert store.upsert_prices(conn, "SPY", df) == 3
    assert store.latest_price_date(conn, "SPY") == idx[-1].strftime("%Y-%m-%d")
    loaded = store.load_prices(conn, "SPY")
    assert len(loaded) == 3
    assert loaded["close"].tolist() == [1.0, 2.0, 3.0]
    # 重复写入覆盖而非报错
    assert store.upsert_prices(conn, "SPY", df) == 3
    assert len(store.load_prices(conn, "SPY")) == 3


def test_signal_dedup_and_notify_flow():
    conn = make_conn()
    assert store.insert_signals(conn, [sig(), sig()]) == 1          # 同批去重
    assert store.insert_signals(conn, [sig()]) == 0                 # 跨批幂等
    assert store.insert_signals(conn, [sig(symbol="QQQ")]) == 1

    pending = store.unnotified_signals(conn)
    assert len(pending) == 2
    store.mark_notified(conn, [r["id"] for r in pending])
    assert store.unnotified_signals(conn) == []

    df = store.load_signals(conn, symbol="SPY")
    assert len(df) == 1
    assert df.iloc[0]["strategy"] == "sma_cross"
