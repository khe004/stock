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
