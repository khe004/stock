"""Streamlit 复盘面板：streamlit run quant/web/app.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from quant import strategies
from quant.backtest.engine import run_backtest
from quant.config import load_config
from quant.data import store
from quant.strategies.base import BUY

st.set_page_config(page_title="个人量化信号", page_icon="📈", layout="wide")

cfg = load_config()
conn = store.connect(cfg.db_path)

page = st.sidebar.radio("页面", ["信号历史", "K线与信号", "回测"])
strategy_names = [name for name, _ in cfg.enabled_strategies()]


def signal_markers(fig, sigs: pd.DataFrame):
    buys = sigs[sigs["direction"] == BUY]
    sells = sigs[sigs["direction"] != BUY]
    if not buys.empty:
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(buys["date"]), y=buys["price"], mode="markers", name="买入",
            marker=dict(symbol="triangle-up", size=12, color="#2ca02c"),
            hovertext=buys["reason"],
        ))
    if not sells.empty:
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(sells["date"]), y=sells["price"], mode="markers", name="卖出",
            marker=dict(symbol="triangle-down", size=12, color="#d62728"),
            hovertext=sells["reason"],
        ))


if page == "信号历史":
    st.title("信号历史")
    col1, col2 = st.columns(2)
    f_strategy = col1.selectbox("策略", ["全部"] + strategy_names)
    f_symbol = col2.selectbox("标的", ["全部"] + cfg.all_symbols)
    df = store.load_signals(
        conn,
        strategy=None if f_strategy == "全部" else f_strategy,
        symbol=None if f_symbol == "全部" else f_symbol,
    )
    st.caption(f"共 {len(df)} 条信号")
    st.dataframe(df, width="stretch", hide_index=True)

elif page == "K线与信号":
    st.title("K线与信号")
    symbol = st.selectbox("标的", cfg.all_symbols)
    prices = store.load_prices(conn, symbol)
    if prices.empty:
        st.warning("库内没有该标的行情，先运行 python run_daily.py 拉取数据")
    else:
        fig = go.Figure(go.Candlestick(
            x=prices.index, open=prices["open"], high=prices["high"],
            low=prices["low"], close=prices["close"], name=symbol,
        ))
        for window, color in ((20, "#1f77b4"), (60, "#ff7f0e")):
            fig.add_trace(go.Scatter(
                x=prices.index, y=prices["close"].rolling(window).mean(),
                mode="lines", name=f"MA{window}", line=dict(width=1, color=color),
            ))
        signal_markers(fig, store.load_signals(conn, symbol=symbol))
        fig.update_layout(height=600, xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, width="stretch")

else:  # 回测
    st.title("回测")
    col1, col2 = st.columns(2)
    strategy_name = col1.selectbox("策略", strategy_names)
    params = dict(cfg.enabled_strategies())[strategy_name]
    group_symbols = cfg.symbols_for(params.get("groups", []))
    symbol = col2.selectbox("标的", group_symbols)

    prices = {s: store.load_prices(conn, s) for s in group_symbols}
    prices = {s: df for s, df in prices.items() if not df.empty}
    if symbol not in prices:
        st.warning("库内没有该标的行情，先运行 python run_daily.py 拉取数据")
    else:
        strat = strategies.build(strategy_name, params)
        sigs = strat.generate(prices)
        result = run_backtest(prices[symbol], sigs, symbol, strategy_name)

        cols = st.columns(6)
        for col, (k, v) in zip(cols, result.metrics().items()):
            col.metric(k, v)

        eq = go.Figure(go.Scatter(x=result.equity.index, y=result.equity, mode="lines", name="权益"))
        eq.update_layout(height=400, title=f"{symbol} · {strategy_name} 权益曲线（{result.start} ~ {result.end}）")
        st.plotly_chart(eq, width="stretch")

        if result.trades:
            st.subheader("交易明细")
            st.dataframe(pd.DataFrame(result.trades), width="stretch", hide_index=True)
