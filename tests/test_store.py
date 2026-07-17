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


def test_migrate_legacy_db(tmp_path):
    """模拟旧版库：prices 缺 adj_close、signals 缺 notified_at/created_at。
    connect() 应自愈：prices 补列保数据，signals 重建并保留备份。"""
    import sqlite3
    db = tmp_path / "legacy.db"
    raw = sqlite3.connect(db)
    raw.executescript("""
        CREATE TABLE prices (
            symbol TEXT NOT NULL, date TEXT NOT NULL,
            open REAL, high REAL, low REAL, close REAL, volume INTEGER,
            PRIMARY KEY (symbol, date)
        );
        INSERT INTO prices VALUES ('SPY', '2026-07-01', 1, 1, 1, 1.5, 100);
        CREATE TABLE signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT, symbol TEXT, strategy TEXT, direction TEXT,
            price REAL, reason TEXT
        );
        INSERT INTO signals (date, symbol, strategy, direction, price, reason)
        VALUES ('2026-07-01', 'SPY', 'old', 'buy', 1.0, 'legacy row');
    """)
    raw.commit()
    raw.close()

    conn = store.connect(db)
    # prices：补了 adj_close 列，旧数据仍在
    df = store.load_prices(conn, "SPY")
    assert len(df) == 1
    assert df.iloc[0]["close"] == 1.5
    assert "adj_close" in df.columns
    # signals：重建为新结构，可正常走通知流程
    assert store.insert_signals(conn, [sig()]) == 1
    assert len(store.unnotified_signals(conn)) == 1
    # 旧信号保留在备份表
    legacy = conn.execute("SELECT reason FROM signals_legacy").fetchall()
    assert legacy[0]["reason"] == "legacy row"
    # 再次 connect 幂等，不再触发迁移
    conn2 = store.connect(db)
    assert len(store.unnotified_signals(conn2)) == 1
