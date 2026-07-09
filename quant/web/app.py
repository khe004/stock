"""Streamlit 复盘面板：streamlit run quant/web/app.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from quant import strategies
from quant.backtest.engine import TRADING_DAYS, run_backtest
from quant.config import load_config
from quant.data import store
from quant.strategies.base import BUY
from quant.strategies.rsi_reversal import wilder_rsi

st.set_page_config(page_title="个人量化信号", page_icon="📈", layout="wide")

cfg = load_config()
conn = store.connect(cfg.db_path)
strategy_params = dict(cfg.enabled_strategies())
strategy_names = list(strategy_params)

BUY_BG, SELL_BG = "#e8f5e9", "#ffebee"
BUY_FG, SELL_FG = "#1b5e20", "#b71c1c"
BUY_COLOR, SELL_COLOR = "#2ca02c", "#d62728"

RANGE_OPTIONS = {"近3月": 63, "近6月": 126, "近1年": 252, "近3年": 756, "全部": None}

MOM_LOOKBACK = strategy_params.get("momentum", {}).get("lookback_days", 63)
MOM_TOP_N = strategy_params.get("momentum", {}).get("top_n", 3)
RSI_PERIOD = strategy_params.get("rsi_reversal", {}).get("period", 14)
RSI_OVERSOLD = strategy_params.get("rsi_reversal", {}).get("oversold", 30)
RSI_OVERBOUGHT = strategy_params.get("rsi_reversal", {}).get("overbought", 70)


def add_signal_markers(fig, sigs: pd.DataFrame, row: int | None = None):
    buys = sigs[sigs["direction"] == BUY]
    sells = sigs[sigs["direction"] != BUY]
    kw = {"row": row, "col": 1} if row else {}
    if not buys.empty:
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(buys["date"]), y=buys["price"], mode="markers", name="买入",
            marker=dict(symbol="triangle-up", size=12, color=BUY_COLOR),
            hovertext=buys["reason"],
        ), **kw)
    if not sells.empty:
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(sells["date"]), y=sells["price"], mode="markers", name="卖出",
            marker=dict(symbol="triangle-down", size=12, color=SELL_COLOR),
            hovertext=sells["reason"],
        ), **kw)


def group_closes(group_key: str) -> pd.DataFrame:
    frames = {}
    for s in cfg.watchlist.get(group_key, []):
        df = store.load_prices(conn, s)
        if not df.empty:
            frames[s] = df["close"]
    return pd.DataFrame(frames)


def render_signal_history():
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
    if df.empty:
        st.info("暂无信号")
        return

    def row_style(row):
        bg = BUY_BG if row["direction"] == BUY else SELL_BG
        return [f"background-color: {bg}"] * len(row)

    def direction_style(v):
        fg = BUY_FG if v == BUY else SELL_FG
        return f"color: {fg}; font-weight: bold"

    styler = (df.style.apply(row_style, axis=1)
                .map(direction_style, subset=["direction"])
                .format({"price": "{:.2f}", "strength": "{:.2f}"}))
    st.dataframe(styler, width="stretch", hide_index=True)


def render_kline():
    st.title("K线与信号")
    symbol = st.selectbox("标的", cfg.all_symbols)
    prices = store.load_prices(conn, symbol)
    if prices.empty:
        st.warning("库内没有该标的行情，先运行 python run_daily.py 拉取数据")
    else:
        close = prices["close"]
        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True,
            row_heights=[0.6, 0.2, 0.2], vertical_spacing=0.03,
        )
        fig.add_trace(go.Candlestick(
            x=prices.index, open=prices["open"], high=prices["high"],
            low=prices["low"], close=close, name=symbol,
        ), row=1, col=1)
        for window, color in ((20, "#1f77b4"), (60, "#ff7f0e")):
            fig.add_trace(go.Scatter(
                x=prices.index, y=close.rolling(window).mean(),
                mode="lines", name=f"MA{window}", line=dict(width=1, color=color),
            ), row=1, col=1)
        add_signal_markers(fig, store.load_signals(conn, symbol=symbol), row=1)

        fig.add_trace(go.Scatter(
            x=prices.index, y=wilder_rsi(close, RSI_PERIOD),
            mode="lines", name=f"RSI({RSI_PERIOD})", line=dict(width=1, color="#9467bd"),
        ), row=2, col=1)
        fig.add_hline(y=RSI_OVERBOUGHT, line_dash="dot", line_color=SELL_COLOR, row=2, col=1)
        fig.add_hline(y=RSI_OVERSOLD, line_dash="dot", line_color=BUY_COLOR, row=2, col=1)
        fig.add_hrect(y0=0, y1=RSI_OVERSOLD, fillcolor=BUY_COLOR, opacity=0.07, line_width=0, row=2, col=1)
        fig.add_hrect(y0=RSI_OVERBOUGHT, y1=100, fillcolor=SELL_COLOR, opacity=0.07, line_width=0, row=2, col=1)

        fig.add_trace(go.Scatter(
            x=prices.index, y=close.pct_change(MOM_LOOKBACK, fill_method=None),
            mode="lines", name=f"动量(近{MOM_LOOKBACK}日)", line=dict(width=1, color="#8c564b"),
        ), row=3, col=1)
        fig.add_hline(y=0, line_dash="dot", line_color="#888", row=3, col=1)

        fig.update_layout(height=800, xaxis_rangeslider_visible=False,
                          legend=dict(orientation="h", yanchor="bottom", y=1.01))
        fig.update_yaxes(title_text="价格", row=1, col=1)
        fig.update_yaxes(title_text=f"RSI({RSI_PERIOD})", range=[0, 100], row=2, col=1)
        fig.update_yaxes(title_text="动量", tickformat=".0%", row=3, col=1)
        st.plotly_chart(fig, width="stretch")

    st.subheader("分组对比（归一化，区间起点 = 100）")
    range_label = st.radio("区间", list(RANGE_OPTIONS), index=2, horizontal=True)
    days = RANGE_OPTIONS[range_label]
    for group_key, title in (("broad", "大盘 ETF"), ("sectors", "行业 ETF"), ("assets", "资产类 ETF")):
        closes = group_closes(group_key).dropna(how="all")
        if closes.empty:
            continue
        if days:
            closes = closes.iloc[-days:]
        base = closes.apply(lambda c: c.loc[c.first_valid_index()] if c.first_valid_index() is not None else pd.NA)
        normed = closes.div(base).mul(100).dropna(axis=1, how="all")
        gfig = go.Figure()
        for s in normed.columns:
            gfig.add_trace(go.Scatter(x=normed.index, y=normed[s], mode="lines", name=s, line=dict(width=1.5)))
        gfig.add_hline(y=100, line_dash="dot", line_color="#888")
        gfig.update_layout(title=f"{title}（{range_label}）", height=400, hovermode="x unified")
        st.plotly_chart(gfig, width="stretch")


def render_momentum_rank():
    st.title("动量排名（行业 ETF）")
    closes = group_closes("sectors")
    rets = closes.pct_change(MOM_LOOKBACK, fill_method=None).dropna(how="all")
    if rets.empty:
        st.warning("行情数据不足（需要至少 63 个交易日），先运行 python run_daily.py 拉取数据")
        return

    latest = rets.iloc[-1].dropna().sort_values(ascending=False)
    as_of = rets.index[-1].strftime("%Y-%m-%d")
    st.caption(f"截至 {as_of}，按近 {MOM_LOOKBACK} 个交易日收益排名；前 {MOM_TOP_N} 名为轮动持有对象。"
               f"第 {MOM_TOP_N} 名与第 {MOM_TOP_N + 1} 名收益接近时，进出信号可能是排名噪音。")

    table = pd.DataFrame({
        "排名": range(1, len(latest) + 1),
        "标的": latest.index,
        f"近{MOM_LOOKBACK}日收益": latest.values,
        "状态": ["✅ 前3" if i < MOM_TOP_N else "" for i in range(len(latest))],
    })

    def top_style(row):
        bg = BUY_BG if row["排名"] <= MOM_TOP_N else ""
        return [f"background-color: {bg}"] * len(row)

    styler = (table.style.apply(top_style, axis=1)
                   .format({f"近{MOM_LOOKBACK}日收益": "{:+.1%}"}))
    st.dataframe(styler, width="stretch", hide_index=True)

    st.subheader("排名走势（近 120 个交易日）")
    st.caption("排名 1 在最上方；在虚线（前3分界）附近反复穿越的板块，其买卖信号可信度低。")
    ranks = rets.rank(axis=1, ascending=False).iloc[-120:]
    rfig = go.Figure()
    for s in ranks.columns:
        rfig.add_trace(go.Scatter(x=ranks.index, y=ranks[s], mode="lines", name=s, line=dict(width=1.5)))
    rfig.add_hline(y=MOM_TOP_N + 0.5, line_dash="dash", line_color="#888",
                   annotation_text=f"前{MOM_TOP_N}分界")
    rfig.update_yaxes(autorange="reversed", dtick=1, title_text="排名")
    rfig.update_layout(height=500, hovermode="x unified")
    st.plotly_chart(rfig, width="stretch")


def render_backtest():
    st.title("回测")
    col1, col2 = st.columns(2)
    strategy_name = col1.selectbox("策略", strategy_names)
    params = strategy_params[strategy_name]
    group_symbols = cfg.symbols_for(params.get("groups", []))
    symbol = col2.selectbox("标的", group_symbols)

    prices = {s: store.load_prices(conn, s) for s in group_symbols}
    prices = {s: df for s, df in prices.items() if not df.empty}
    if symbol not in prices:
        st.warning("库内没有该标的行情，先运行 python run_daily.py 拉取数据")
        return

    df_full = prices[symbol]
    min_d, max_d = df_full.index[0].date(), df_full.index[-1].date()
    start, end = st.slider("时间区间", min_value=min_d, max_value=max_d, value=(min_d, max_d))
    window = df_full.loc[str(start):str(end)]
    if len(window) < 2:
        st.warning("选中区间数据不足")
        return
    st.caption("信号在全量历史上生成（指标不受区间影响）；持仓从区间内第一个买入信号开始，所有指标只统计区间内的表现。")

    strat = strategies.build(strategy_name, params)
    sigs = strat.generate(prices)
    result = run_backtest(window, sigs, symbol, strategy_name)

    cols = st.columns(6)
    for col, (k, v) in zip(cols, result.metrics().items()):
        col.metric(k, v)

    # 不折腾基准一：区间首日买入、一直长持
    close = window["close"]
    initial_cash = float(result.equity.iloc[0])
    bh_total = float(close.iloc[-1] / close.iloc[0]) - 1
    bh_cagr = (1 + bh_total) ** (TRADING_DAYS / len(window)) - 1

    # 不折腾基准二：按月定投——同一笔初始资金按月份等分，
    # 每月第一个交易日买入一份，未投入部分按现金（无息）计
    month_firsts = set(close.groupby([close.index.year, close.index.month]).head(1).index)
    per_month = initial_cash / len(month_firsts)
    dca_cash, dca_shares, dca_values = initial_cash, 0.0, []
    for ts, price in close.items():
        if ts in month_firsts:
            dca_shares += per_month / float(price)
            dca_cash -= per_month
        dca_values.append(dca_cash + dca_shares * float(price))
    dca = pd.Series(dca_values, index=close.index)
    dca_total = float(dca.iloc[-1]) / initial_cash - 1
    dca_cagr = (1 + dca_total) ** (TRADING_DAYS / len(window)) - 1

    st.markdown("**不折腾基准对比**（长持=区间首日全仓买入；定投=同一笔资金按月等分、每月首个交易日买入）")
    bh_cols = st.columns(6)
    bh_cols[0].metric("长持收益", f"{bh_total:+.1%}")
    bh_cols[1].metric("长持年化", f"{bh_cagr:+.1%}")
    bh_cols[2].metric("定投收益", f"{dca_total:+.1%}")
    bh_cols[3].metric("定投年化", f"{dca_cagr:+.1%}")
    for col, (label, base) in zip(bh_cols[4:], (("长持", bh_total), ("定投", dca_total))):
        excess = result.total_return - base
        col.metric(f"策略 vs {label}", f"{excess:+.1%}",
                   delta=f"{'跑赢' if excess > 0 else '跑输'}{label}",
                   delta_color="normal" if excess > 0 else "inverse")

    eq = go.Figure(go.Scatter(x=result.equity.index, y=result.equity, mode="lines", name="策略权益"))
    eq.add_trace(go.Scatter(
        x=window.index, y=initial_cash * close / close.iloc[0],
        mode="lines", name="长持基准", line=dict(dash="dash", color="#888"),
    ))
    eq.add_trace(go.Scatter(
        x=dca.index, y=dca,
        mode="lines", name="定投基准", line=dict(dash="dot", color="#bc8f5f"),
    ))
    entries = [(t["entry_date"], t["entry"]) for t in result.trades]
    if result.open_position:
        entries.append((result.open_position["entry_date"], result.open_position["entry"]))
    exits = [(t["exit_date"], t["exit"]) for t in result.trades]
    for points, name, sym_shape, color in (
        (entries, "买入", "triangle-up", BUY_COLOR),
        (exits, "卖出", "triangle-down", SELL_COLOR),
    ):
        dates = [pd.Timestamp(d) for d, _ in points if pd.Timestamp(d) in result.equity.index]
        if dates:
            eq.add_trace(go.Scatter(
                x=dates, y=result.equity.loc[dates], mode="markers", name=name,
                marker=dict(symbol=sym_shape, size=12, color=color),
            ))
    eq.update_layout(height=400, title=f"{symbol} · {strategy_name} 权益曲线（{result.start} ~ {result.end}）")
    st.plotly_chart(eq, width="stretch")

    if result.trades:
        st.subheader("交易明细")
        trades = pd.DataFrame(result.trades)[["entry_date", "exit_date", "entry", "exit", "pnl_pct"]]
        trades.columns = ["买入日", "卖出日", "买入价", "卖出价", "收益"]

        def pnl_style(v):
            return f"color: {BUY_FG}" if v > 0 else f"color: {SELL_FG}"

        styler = (trades.style.map(pnl_style, subset=["收益"])
                        .format({"买入价": "{:.2f}", "卖出价": "{:.2f}", "收益": "{:+.2%}"}))
        st.dataframe(styler, width="stretch", hide_index=True)
    if result.open_position:
        st.caption(f"区间末仍持仓：{result.open_position['entry_date']} 以 ${result.open_position['entry']:.2f} 买入，未平仓部分按区间末市值计入指标。")


def render_strategy_docs():
    st.title("策略说明")
    sma = strategy_params.get("sma_cross", {"fast": 20, "slow": 60})
    st.markdown(f"""
## 三个策略如何配合

**双均线管大方向**（该在场内还是场外）→ **动量管配置**（钱放哪个板块）→ **RSI 管时机**（回调到哪天动手）。
同一天出现矛盾信号时以大方向为准：大盘死叉之下的逆势买入信号，轻仓或忽略。

---

## 1. sma_cross 双均线（趋势跟踪）

**直觉**：{sma["fast"]} 日均线是"最近一个月的平均成本"，{sma["slow"]} 日均线是"最近一个季度的平均成本"。
短期成本升到长期成本之上，说明近期买入者整体在赚钱、趋势向上。

**规则**：{sma["fast"]} 日线上穿 {sma["slow"]} 日线（金叉）→ 买入；下穿（死叉）→ 卖出。强度按快线近 5 日斜率：拐头越急越强。

**何时灵**：单边大趋势。少赚顶底各一段，换取绝不错过大趋势、绝不深套。

**何时坑**：横盘震荡市，均线反复交叉，假信号多且每次小亏（胜率低、靠大赢单撑收益是它的正常特征）。

**作用范围**：大盘、主题、资产类 ETF。

---

## 2. momentum 动量轮动（相对强弱）

**直觉**：资金分板块轮动，过去 3 个月强的板块未来几周大概率继续强（动量效应，学术上验证最充分的市场异象之一）。

**规则**：{MOM_TOP_N} 名开外的板块**新进入**近 {MOM_LOOKBACK} 个交易日收益前 {MOM_TOP_N} 名 → 买入；跌出前 {MOM_TOP_N} 名 → 卖出。完整用法是组合式的：始终持有排名前 {MOM_TOP_N} 的板块。

**何时灵**：板块分化明显的行情（如 AI 行情中科技/半导体持续霸榜）。

**何时坑**：动量崩溃（大跌后的 V 型反转期追强追在山顶）；排名在第 {MOM_TOP_N}、{MOM_TOP_N + 1} 名之间反复横跳的边缘板块（对照「动量排名」页人工过滤）。

**作用范围**：11 只行业 ETF。

---

## 3. rsi_reversal RSI 反转（均值回归）

**直觉**：跌得又急又久时短期恐慌往往过度。关键是**不接飞刀**：不是 RSI 低就买，而是等它从超卖区**回升穿越**才发信号——恐慌见底、开始回暖的那一天。

**规则**：RSI({RSI_PERIOD}) 从 {RSI_OVERSOLD} 之下回升穿过 {RSI_OVERSOLD} → 买入（超卖越深强度越高）；从 {RSI_OVERBOUGHT} 之上回落穿过 {RSI_OVERBOUGHT} → 卖出。

**何时灵**：牛市或震荡市里的急跌回调，正好补双均线"震荡市难受"的短板。

**何时坑**：持续阴跌的熊市，每次弱反弹都给买入信号。熊市里（大盘死叉之下）它的买入信号要打折看待。

**作用范围**：全部 21 只 ETF。

---

*参数在 `config.yaml` 中修改，本页数值实时读取当前配置。提醒：不要为了回测曲线好看精调参数——那是过拟合；当前默认值是学术与实务中最常用的取值。*
""")


PAGES = {
    "信号历史": render_signal_history,
    "K线与信号": render_kline,
    "动量排名": render_momentum_rank,
    "回测": render_backtest,
    "策略说明": render_strategy_docs,
}

page = st.sidebar.radio("页面", list(PAGES))
PAGES[page]()
