import numpy as np
import pandas as pd
import pytest

import quant.data.fetcher as fetcher


def make_yf_df(n=3):
    idx = pd.bdate_range("2026-07-01", periods=n)
    return pd.DataFrame({
        "Open": [1.0] * n, "High": [1.0] * n, "Low": [1.0] * n,
        "Close": [1.0] * n, "Adj Close": [1.0] * n, "Volume": [10] * n,
    }, index=idx)


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(fetcher.time, "sleep", lambda *_: None)


def test_empty_response_is_retried(monkeypatch):
    """yfinance 限流时静默返回空表，应重试而非立即当作无数据。"""
    calls = []

    def flaky_download(symbol, **kw):
        calls.append(symbol)
        return pd.DataFrame() if len(calls) < 3 else make_yf_df()

    monkeypatch.setattr(fetcher.yf, "download", flaky_download)
    df = fetcher.fetch_history("SPY", "2026-07-01")
    assert len(calls) == 3, "前两次空表应触发重试"
    assert len(df) == 3
    assert list(df.columns) == ["open", "high", "low", "close", "adj_close", "volume"]


def test_persistent_empty_returns_empty(monkeypatch):
    """重试用尽仍为空（真退市代码的表现）→ 返回空表不抛异常。"""
    monkeypatch.setattr(fetcher.yf, "download", lambda *a, **kw: pd.DataFrame())
    df = fetcher.fetch_history("DEAD", "2026-07-01")
    assert df.empty


def test_persistent_exception_raises(monkeypatch):
    def boom(*a, **kw):
        raise ConnectionError("rate limited")

    monkeypatch.setattr(fetcher.yf, "download", boom)
    with pytest.raises(RuntimeError, match="rate limited"):
        fetcher.fetch_history("SPY", "2026-07-01")


def test_exception_then_success(monkeypatch):
    calls = []

    def flaky(*a, **kw):
        calls.append(1)
        if len(calls) == 1:
            raise ConnectionError("transient")
        return make_yf_df()

    monkeypatch.setattr(fetcher.yf, "download", flaky)
    df = fetcher.fetch_history("SPY", "2026-07-01")
    assert len(df) == 3


def _norm(dates, close, adj):
    return pd.DataFrame({"open": close, "high": close, "low": close, "close": close,
                         "adj_close": adj, "volume": [10] * len(close)},
                        index=pd.to_datetime(dates))


@pytest.fixture
def conn(tmp_path):
    from quant.data import store
    c = store.connect(tmp_path / "t.db")
    yield c
    c.close()


def _seed(conn):
    from quant.data import store
    store.upsert_prices(conn, "BIL", _norm(["2026-07-01", "2026-07-02"], [91.4, 91.44], [91.4, 91.44]))


def test_new_dividend_triggers_full_refetch(monkeypatch, conn):
    from quant.data import store
    _seed(conn)
    calls = []

    def fake(symbol, start):
        calls.append(start)
        if start == "2026-07-02":     # 增量：重叠日复权价已因 7 月分红整体下移 0.4%
            return _norm(["2026-07-02", "2026-07-03"], [91.44, 91.46], [91.07, 91.46])
        return _norm(["2026-07-01", "2026-07-02", "2026-07-03"],
                     [91.4, 91.44, 91.46], [91.03, 91.07, 91.46])

    monkeypatch.setattr(fetcher, "fetch_history", fake)
    fetcher.update_symbol(conn, "BIL", "2007-01-01")
    assert calls == ["2026-07-02", "2007-01-01"]
    adj = store.load_prices(conn, "BIL")["adj_close"]
    assert adj.iloc[0] == pytest.approx(91.03)       # 旧行已按新口径重写


def test_unchanged_adjustment_stays_incremental(monkeypatch, conn):
    _seed(conn)
    calls = []

    def fake(symbol, start):
        calls.append(start)
        return _norm(["2026-07-02", "2026-07-03"], [91.44, 91.46], [91.44 * (1 + 1e-7), 91.46])

    monkeypatch.setattr(fetcher, "fetch_history", fake)
    fetcher.update_symbol(conn, "BIL", "2007-01-01")
    assert calls == ["2026-07-02"]


def test_needs_full_refresh_detects_split_and_ignores_bad_values():
    stored = pd.Series({"close": 400.0, "adj_close": 400.0})
    assert "拆股" in fetcher.needs_full_refresh(stored, pd.Series({"close": 100.0, "adj_close": 100.0}))
    assert fetcher.needs_full_refresh(stored, pd.Series({"close": None, "adj_close": 1.0})) is None
    assert fetcher.needs_full_refresh(stored, pd.Series({"close": 401.0, "adj_close": 401.0})) is None
