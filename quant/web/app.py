"""Streamlit 复盘面板：streamlit run quant/web/app.py"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml
from plotly.subplots import make_subplots

from quant import strategies
from quant.analysis.correlation import (
    combined_portfolio,
    correlation_matrix,
    strategy_return_series,
)
from quant.analysis.market import range_position, sector_breadth, yield_curve_spread
from quant.analysis.scoring import DEFAULT_HORIZONS, signal_forward_returns, summarize_scores
from quant.analysis.screening import compute_strength, market_regime
from quant.backtest.engine import (
    dca_equity,
    equity_metrics,
    hold_equity,
    run_backtest,
    run_portfolio_backtest,
    run_smart_dca_backtest,
    vol_scaled_equity,
)
from quant.config import ROOT, load_config
from quant.data import store
from quant.strategies.base import BUY, Signal, price_series
from quant.strategies.rsi_reversal import wilder_rsi

st.set_page_config(page_title="个人投资平台", page_icon="📈", layout="wide",
                   initial_sidebar_state="collapsed")

cfg = load_config()
conn = store.connect(cfg.db_path)
strategy_params = dict(cfg.enabled_strategies())
strategy_names = list(strategy_params)

def _dark_theme() -> bool:
    try:
        return st.context.theme.type == "dark"
    except Exception:   # 旧版 Streamlit 没有 st.context.theme
        return False


_DARK = _dark_theme()
# 行背景用半透明色（明暗主题下都保持文字对比度）；前景色按主题选深浅
BUY_BG, SELL_BG = "rgba(46, 125, 50, 0.25)", "rgba(198, 40, 40, 0.25)"
BUY_FG, SELL_FG = ("#81c784", "#ef9a9a") if _DARK else ("#1b5e20", "#b71c1c")
BUY_COLOR, SELL_COLOR = "#2ca02c", "#d62728"


def signed_color(v) -> str:
    """正数用买入色、负数用卖出色，供 pandas Styler 的 map 使用。"""
    if pd.isna(v):
        return ""
    return f"color: {BUY_FG}" if v > 0 else (f"color: {SELL_FG}" if v < 0 else "")

RANGE_OPTIONS = {"近3月": 63, "近6月": 126, "近1年": 252, "近3年": 756, "全部": None}

MOM_LOOKBACK = strategy_params.get("momentum", {}).get("lookback_days", 252)
MOM_SKIP = strategy_params.get("momentum", {}).get("skip_days", 21)
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


def group_closes(group_key: str, adjusted: bool = False) -> pd.DataFrame:
    """adjusted=True 时用 adj_close（总回报口径），用于收益/动量比较。"""
    frames = {}
    for s in cfg.watchlist.get(group_key, []):
        df = store.load_prices(conn, s)
        if not df.empty:
            frames[s] = price_series(df) if adjusted else df["close"]
    return pd.DataFrame(frames)


MACRO_NAMES = {
    "^GSPC": "标普500", "^IXIC": "纳斯达克综合", "^DJI": "道琼斯工业", "^RUT": "罗素2000",
    "^VIX": "VIX恐慌指数", "^TNX": "10年期美债收益率",
    "DX-Y.NYB": "美元指数", "GC=F": "黄金期货", "CL=F": "原油WTI", "BTC-USD": "比特币",
    "TLT": "TLT长债", "QQQ": "QQQ纳指100",
}
MACRO_ROW1 = ["^GSPC", "^IXIC", "^DJI", "^RUT", "^VIX", "^TNX"]
MACRO_ROW2 = ["DX-Y.NYB", "GC=F", "CL=F", "BTC-USD", "TLT", "QQQ"]


def _macro_tile(col, symbol: str, df: pd.DataFrame | None):
    name = MACRO_NAMES.get(symbol, symbol)
    with col.container(border=True):
        if df is None or df.empty or len(df) < 2:
            st.metric(name, "无数据")
            return
        close = df["close"]
        last, prev = float(close.iloc[-1]), float(close.iloc[-2])
        chg = last / prev - 1
        is_yield = symbol in ("^TNX", "^IRX")
        st.metric(name, f"{last:,.2f}{'%' if is_yield else ''}", f"{chg:+.2%}")
        pos = range_position(close)
        if pos is not None:
            st.progress(min(1.0, max(0.0, pos)), text=f"52周区间 {pos:.0%}")


def _data_date_caption(prices: dict[str, pd.DataFrame]):
    """用美股指数的最新交易日作为"数据日期"锚点（比特币周末也交易，会误导）。
    与今天的自然日差超过 4 天（可跨长周末）时标红提示可能没跑最新。"""
    from datetime import date

    index_syms = ["^GSPC", "^IXIC", "^DJI", "^RUT"]
    dates = [prices[s].index[-1] for s in index_syms
             if s in prices and not prices[s].empty]
    if not dates:
        return
    data_date = max(dates).date()
    gap = (date.today() - data_date).days
    if gap <= 4:
        st.caption(f"📅 数据日期：**{data_date:%Y-%m-%d}**（美股最新交易日）")
    else:
        st.caption(f"⚠️ 数据日期：**{data_date:%Y-%m-%d}**，距今 {gap} 天——"
                   f"可能没跑最新，运行 `run_daily.py` 或 `scripts/run_now.command` 更新。")


def render_market_overview():
    st.title("市场概览")
    symbols = list(dict.fromkeys(MACRO_ROW1 + MACRO_ROW2))  # TLT/QQQ 已在 broad/assets 组，一并加载
    prices = {s: store.load_prices(conn, s) for s in symbols}
    if all(df.empty for df in prices.values()):
        st.warning("库内没有宏观行情，先运行 python run_daily.py 拉取数据")
        return

    _data_date_caption(prices)

    for row in (MACRO_ROW1, MACRO_ROW2):
        cols = st.columns(len(row))
        for col, sym in zip(cols, row):
            _macro_tile(col, sym, prices.get(sym))

    st.subheader("市场情绪")
    spy = store.load_prices(conn, "SPY")
    vix = prices.get("^VIX")
    vix3m = store.load_prices(conn, "^VIX3M")
    irx = store.load_prices(conn, "^IRX")
    sector_closes = {s: df["close"] for s, df in
                     ((s, store.load_prices(conn, s)) for s in cfg.watchlist.get("sectors", []))
                     if not df.empty}

    lights = []

    if not spy.empty and len(spy) >= 200:
        close = price_series(spy)
        ma200 = float(close.rolling(200).mean().iloc[-1])
        dev = float(close.iloc[-1]) / ma200 - 1
        ok = dev >= 0
        lights.append(("大盘趋势", "🟢" if ok else "🔴",
                       f"SPY {'高于' if ok else '低于'} 200日均线 {abs(dev):.1%}"))
    else:
        lights.append(("大盘趋势", "⚪", "数据不足"))

    if vix is not None and not vix.empty:
        v = float(vix["close"].iloc[-1])
        if v >= 30:
            icon, label = "🔴", "恐慌"
        elif v <= 15:
            icon, label = "🟡", "自满"
        else:
            icon, label = "🟢", "中性"
        note = ""
        if vix3m is not None and not vix3m.empty:
            spread = yield_curve_spread(vix["close"], vix3m["close"])
            if spread is not None and spread >= 0:
                note = "，期限结构倒挂"
        lights.append(("恐慌温度", icon, f"VIX {v:.1f}（{label}）{note}"))
    else:
        lights.append(("恐慌温度", "⚪", "数据不足"))

    breadth = sector_breadth(sector_closes)
    if breadth["total"] > 0:
        icon = "🟢" if breadth["above"] >= 8 else ("🟡" if breadth["above"] >= 4 else "🔴")
        lights.append(("行业宽度", icon,
                       f"{breadth['above']}/{breadth['total']} 只行业ETF站上200日均线"))
    else:
        lights.append(("行业宽度", "⚪", "数据不足"))

    tnx = prices.get("^TNX")
    if tnx is not None and not tnx.empty and not irx.empty:
        spread = yield_curve_spread(tnx["close"], irx["close"])
        icon = "🔴" if spread is not None and spread < 0 else "🟢"
        lights.append(("收益率曲线", icon,
                       f"10年-3月利差 {spread:+.2f}pp" if spread is not None else "数据不足"))
    else:
        lights.append(("收益率曲线", "⚪", "数据不足"))

    cols = st.columns(4)
    for col, (name, icon, detail) in zip(cols, lights):
        with col.container(border=True):
            st.markdown(f"##### {icon} {name}")
            st.caption(detail)
    st.caption("情绪红绿灯仅作环境参考，不直接构成交易信号；具体规则见「策略说明」页 vix_regime 与 "
               "stock_momentum 章节。")


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

        adj = price_series(prices)
        mom_12_1 = adj.shift(MOM_SKIP) / adj.shift(MOM_LOOKBACK) - 1
        fig.add_trace(go.Scatter(
            x=prices.index, y=mom_12_1,
            mode="lines", name="12-1 动量", line=dict(width=1, color="#8c564b"),
        ), row=3, col=1)
        fig.add_hline(y=0, line_dash="dot", line_color="#888", row=3, col=1)

        fig.update_layout(height=800, xaxis_rangeslider_visible=False,
                          legend=dict(orientation="h", yanchor="bottom", y=1.01))
        fig.update_yaxes(title_text="价格", row=1, col=1)
        fig.update_yaxes(title_text=f"RSI({RSI_PERIOD})", range=[0, 100], row=2, col=1)
        fig.update_yaxes(title_text="12-1 动量", tickformat=".0%", row=3, col=1)
        st.plotly_chart(fig, width="stretch")

    st.subheader("分组对比（归一化总回报，区间起点 = 100，含分红）")
    range_label = st.radio("区间", list(RANGE_OPTIONS), index=2, horizontal=True)
    days = RANGE_OPTIONS[range_label]
    for group_key, title in (("broad", "大盘 ETF"), ("sectors", "行业 ETF"), ("assets", "资产类 ETF")):
        closes = group_closes(group_key, adjusted=True).dropna(how="all")
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
    st.title("动量排名（行业 ETF · 12-1 月度动量）")
    closes = group_closes("sectors", adjusted=True)  # 总回报口径，与 momentum 策略一致
    # 12-1 动量口径：shift(skip) / shift(lookback) - 1，与 momentum 策略计算一致
    mom = closes.shift(MOM_SKIP) / closes.shift(MOM_LOOKBACK) - 1
    mom = mom.dropna(how="all")
    if mom.empty:
        st.warning(f"行情数据不足（需要至少 {MOM_LOOKBACK} 个交易日），先运行 python run_daily.py 拉取数据")
        return

    latest = mom.iloc[-1].dropna().sort_values(ascending=False)
    as_of = mom.index[-1].strftime("%Y-%m-%d")
    st.caption(f"截至 {as_of}，按 12-1 动量排名（近{MOM_LOOKBACK}日收益、跳过最近{MOM_SKIP}日）；"
               f"每月首个交易日调仓，前 {MOM_TOP_N} 名为轮动持有对象。"
               f"第 {MOM_TOP_N} 名与第 {MOM_TOP_N + 1} 名动量接近时，进出信号可能是排名噪音。")

    table = pd.DataFrame({
        "排名": range(1, len(latest) + 1),
        "标的": latest.index,
        "12-1 动量": latest.values,
        "状态": ["✅ 前3" if i < MOM_TOP_N else "" for i in range(len(latest))],
    })

    def top_style(row):
        bg = BUY_BG if row["排名"] <= MOM_TOP_N else ""
        return [f"background-color: {bg}"] * len(row)

    styler = (table.style.apply(top_style, axis=1)
                   .format({"12-1 动量": "{:+.1%}"}))
    st.dataframe(styler, width="stretch", hide_index=True)

    st.subheader("排名走势（近 120 个交易日）")
    st.caption("排名 1 在最上方；在虚线（前3分界）附近反复穿越的板块，其买卖信号可信度低。"
               "月度调仓后换手已大幅降低，但排名走势仍可辅助判断信号质量。")
    ranks = mom.rank(axis=1, ascending=False).iloc[-120:]
    rfig = go.Figure()
    for s in ranks.columns:
        rfig.add_trace(go.Scatter(x=ranks.index, y=ranks[s], mode="lines", name=s, line=dict(width=1.5)))
    rfig.add_hline(y=MOM_TOP_N + 0.5, line_dash="dash", line_color="#888",
                   annotation_text=f"前{MOM_TOP_N}分界")
    rfig.update_yaxes(autorange="reversed", dtick=1, title_text="排名")
    rfig.update_layout(height=500, hovermode="x unified")
    st.plotly_chart(rfig, width="stretch")


def _all_strategy_signals() -> tuple[list[Signal], dict[str, str]]:
    """在全量历史上为全部启用策略重算信号（与回测同一套 generate 逻辑），
    以及需要映射到实际可交易标的的策略表（当前只有 vix_regime）。"""
    all_signals: list[Signal] = []
    trade_map: dict[str, str] = {}
    for name, params in cfg.enabled_strategies():
        group_symbols = cfg.symbols_for(params.get("groups", []))
        if params.get("universe_file"):
            group_symbols += [s for s in cfg.universe_symbols(params["universe_file"])
                              if s not in group_symbols]
        gp = {s: store.load_prices(conn, s) for s in group_symbols}
        gp = {s: df for s, df in gp.items() if not df.empty}
        if not gp:
            continue
        strat = strategies.build(name, params)
        all_signals.extend(strat.generate(gp))
        if name == "vix_regime":
            trade_map[name] = params.get("trade_symbol", "SPY")
    return all_signals, trade_map


def render_strategy_scoring():
    st.title("策略评分")
    st.caption("统计口径：用策略在全量历史上重新生成的信号（与回测同一套逻辑）计算"
               "信号发出后 5/20/60 个交易日的表现——只看单条信号本身，不涉及仓位与资金曲线。"
               "buy 信号以上涨为正、sell 信号以下跌为正，已按方向调整符号，可直接跨方向比较正负。")

    all_signals, trade_map = _all_strategy_signals()
    if not all_signals:
        st.warning("暂无信号，先运行 python run_daily.py 拉取数据")
        return

    needed_symbols = {s.symbol for s in all_signals} | set(trade_map.values())
    all_prices = {s: store.load_prices(conn, s) for s in needed_symbols}
    all_prices = {s: df for s, df in all_prices.items() if not df.empty}

    fwd = signal_forward_returns(all_signals, all_prices, trade_symbol_map=trade_map)
    if fwd.empty:
        st.warning("信号发生日期与库内行情范围不匹配，暂时算不出前瞻收益")
        return
    summary = summarize_scores(fwd)

    st.subheader("汇总记分卡")
    show = pd.DataFrame({
        "策略": summary["strategy"],
        "方向": summary["direction"].map({"buy": "买入", "sell": "卖出"}),
        "信号数": summary["n"],
    })
    for h in DEFAULT_HORIZONS:
        show[f"{h}日均收益"] = summary[f"mean_{h}"]
        show[f"{h}日胜率"] = summary[f"win_{h}"]
    show = show.sort_values(["策略", "方向"]).reset_index(drop=True)

    fmt = {f"{h}日均收益": (lambda v: "" if pd.isna(v) else f"{v:+.1%}") for h in DEFAULT_HORIZONS}
    fmt.update({f"{h}日胜率": (lambda v: "" if pd.isna(v) else f"{v:.0%}") for h in DEFAULT_HORIZONS})
    styler = (show.style
              .map(signed_color, subset=[f"{h}日均收益" for h in DEFAULT_HORIZONS])
              .format(fmt))
    st.dataframe(styler, width="stretch", hide_index=True)

    low_sample = summary[summary["low_sample"]]
    if not low_sample.empty:
        names = "、".join(f"{r.strategy}({'买入' if r.direction == BUY else '卖出'})"
                         for r in low_sample.itertuples())
        st.caption(f"⚠️ 样本不足（信号数 < 10），统计意义弱，仅供参考：{names}")

    st.subheader("信号明细")
    st.caption("最近 20 条信号的逐条追踪：这是每条信号的真实成绩单，比回测更贴近实际使用体验"
               "（回测假设机械执行整套策略，这里只看单条信号本身）。未到期的周期显示'待定'。")
    pick = st.selectbox("策略", sorted(fwd["strategy"].unique()), key="scoring_detail_strategy")
    detail = fwd[fwd["strategy"] == pick].sort_values("date", ascending=False).head(20).copy()
    detail["direction"] = detail["direction"].map({"buy": "买入", "sell": "卖出"})
    cols = ["date", "symbol", "direction", "signal_price", "price_now",
            "ret_now", "ret_5", "ret_20", "ret_60"]
    names = ["日期", "标的", "方向", "信号价", "现价", "至今收益", "5日收益", "20日收益", "60日收益"]
    detail = detail[cols]
    detail.columns = names

    def fmt_ret(v):
        return "待定" if pd.isna(v) else f"{v:+.1%}"

    ret_cols = ["至今收益", "5日收益", "20日收益", "60日收益"]
    styler2 = (detail.style.map(signed_color, subset=ret_cols)
               .format({"信号价": "{:.2f}", "现价": "{:.2f}",
                        **{c: fmt_ret for c in ret_cols}}))
    st.dataframe(styler2, width="stretch", hide_index=True)


def render_correlation():
    """策略相关性 / 组合诊断页面：展示策略间 Pearson 相关矩阵、自动解读冗余/分散对、
    等权组合与各单策略的风险收益对比。

    取数逻辑复用 _all_strategy_signals()，回测逻辑委托给 correlation.py 纯函数。
    """
    import plotly.figure_factory as ff

    st.title("策略相关性 / 组合诊断")

    # ── 方法论说明 ──────────────────────────────────
    with st.expander("本页在算什么 / 怎么解读", expanded=False):
        st.markdown(
            "**方法论**\n\n"
            "把每个启用策略化简成一条**日收益率序列**，然后算策略间的 Pearson 相关系数矩阵：\n\n"
            "| 策略类型 | 如何化简 |\n"
            "|---------|--------|\n"
            "| 组合轮动（momentum / dual_momentum / stock_momentum / low_vol） "
            "| 用 `run_portfolio_backtest` 跑出权益曲线，再 `pct_change()` 转日收益率 |\n"
            "| smart_dca "
            "| 用 `run_smart_dca_backtest` 跑出权益曲线 |\n"
            "| 单标的（sma_cross / rsi_reversal） "
            "| 对该策略交易的**每个标的**各跑 `run_backtest`，取各标的权益曲线日收益率的"
            "**等权平均**作为该策略的收益序列 |\n"
            "| vix_regime "
            "| 映射到实际可交易标的（默认 SPY）后按单标的方式处理 |\n\n"
            "所有回测使用统一的单边成本（config `cost_bps`），价格为复权价（总回报口径）。\n\n"
            "**如何解读**\n\n"
            "- **高相关（> 0.6）= 冗余**：两个策略在大部分时间涨跌一致，"
            "叠加使用 = 给同一个因子加杠杆，分散不了风险\n"
            "- **低相关（0 ~ 0.3）= 有分散效果**：涨跌关联弱，组合波动低于单策略\n"
            "- **负相关（< 0）= 真正分散**：一赚一亏的对冲效果最强，"
            "但实际中长期负相关很少见\n\n"
            "本项目的策略多为**动量家族**（momentum / dual_momentum / stock_momentum "
            "都基于\"强者恒强\"），预期它们之间高度相关——叠加运行并不能带来真正的分散。"
        )
        st.warning(
            "**口径局限的诚实说明**\n\n"
            "策略空仓持现金时当日收益 = 0，这段\"共同不动\"的时间会被 Pearson 全序列口径计入，"
            "导致相关系数被稀释（偏低）。换言之，**仅看策略都在场内的日子，实际相关性可能更高**。"
            "解读时需知晓此局限——本页展示的是保守估计。"
        )

    # ── 日期区间选择 ──────────────────────────────────
    range_label = st.radio("回测区间", list(RANGE_OPTIONS), index=3, horizontal=True,
                           key="corr_range")
    range_days = RANGE_OPTIONS[range_label]

    # ── 取数：复用 _all_strategy_signals 逻辑 ──────────
    with st.spinner("正在为各策略生成信号并回测收益序列……"):
        all_signals, trade_map = _all_strategy_signals()
        has_smart_dca = any(n == "smart_dca" for n, _ in cfg.enabled_strategies())
        if not all_signals and not has_smart_dca:
            st.warning("暂无策略信号，先运行 python run_daily.py 拉取数据")
            return

        # 按策略分组信号
        signals_by_strategy: dict[str, list[Signal]] = {}
        for s in all_signals:
            signals_by_strategy.setdefault(s.strategy, []).append(s)

        # 构建每个策略的标的列表和参数
        strat_params: dict[str, dict] = {}
        strat_symbols: dict[str, list[str]] = {}
        for name, params in cfg.enabled_strategies():
            strat_params[name] = params
            group_symbols = cfg.symbols_for(params.get("groups", []))
            if params.get("universe_file"):
                group_symbols += [s for s in cfg.universe_symbols(params["universe_file"])
                                  if s not in group_symbols]
            strat_symbols[name] = group_symbols
            # smart_dca 无信号但仍需处理
            if name == "smart_dca" and name not in signals_by_strategy:
                signals_by_strategy[name] = []

        # 加载所需价格
        needed: set[str] = set()
        for syms in strat_symbols.values():
            needed.update(syms)
        needed.update(trade_map.values())
        prices = {s: store.load_prices(conn, s) for s in needed}
        prices = {s: df for s, df in prices.items() if not df.empty}

        # 区间截取
        if range_days is not None:
            prices = {s: df.iloc[-range_days:] for s, df in prices.items()
                      if len(df) >= 2}

        returns_df = strategy_return_series(
            prices, signals_by_strategy, strat_params, strat_symbols,
            cfg.cost_bps, trade_map,
        )

    if returns_df.empty or returns_df.shape[1] < 2:
        st.warning("需要至少 2 个策略才能计算相关性。当前成功构建收益序列的策略不足。")
        if not returns_df.empty:
            st.info("仅有策略：" + ", ".join(returns_df.columns))
        return

    # ── 按共同交易日对齐 ──────────────────────────────
    aligned = returns_df.dropna()
    if aligned.empty or len(aligned) < 2:
        st.warning("各策略的共同交易日不足，无法计算相关性。")
        return
    st.caption(
        "共 " + str(len(aligned)) + " 个共同交易日 · "
        + str(returns_df.shape[1]) + " 个策略 · "
        "区间 " + aligned.index[0].strftime("%Y-%m-%d") + " ~ "
        + aligned.index[-1].strftime("%Y-%m-%d")
    )

    # ── 1. 相关性热力图 ────────────────────────────────
    st.subheader("策略相关性热力图")
    corr = correlation_matrix(aligned)

    z = corr.values.tolist()
    labels = list(corr.columns)
    annotations = [["{:.2f}".format(corr.iloc[i, j]) for j in range(len(labels))]
                   for i in range(len(labels))]
    heatmap = ff.create_annotated_heatmap(
        z=z, x=labels, y=labels,
        annotation_text=annotations,
        colorscale="RdBu_r", showscale=True,
        zmin=-1, zmax=1,
    )
    heatmap.update_layout(
        height=max(400, 80 * len(labels)),
        xaxis=dict(side="bottom"),
        margin=dict(l=20, r=20, t=30, b=20),
    )
    font_color = "#fff" if _DARK else "#000"
    for ann in heatmap.layout.annotations:
        ann.font = dict(color=font_color, size=13)
    st.plotly_chart(heatmap, width="stretch")

    # ── 2. 自动解读 ────────────────────────────────────
    st.subheader("自动解读")
    pairs: list[tuple[str, str, float]] = []
    n_strats = len(labels)
    for i in range(n_strats):
        for j in range(i + 1, n_strats):
            pairs.append((labels[i], labels[j], float(corr.iloc[i, j])))
    pairs.sort(key=lambda x: x[2], reverse=True)

    avg_corr = sum(p[2] for p in pairs) / len(pairs) if pairs else 0.0
    if avg_corr > 0.5:
        delta_text = "偏高，分散不足"
        delta_clr = "inverse"
    elif avg_corr > 0.3:
        delta_text = "中等"
        delta_clr = "normal"
    else:
        delta_text = "较低，分散尚可"
        delta_clr = "normal"
    st.metric("策略间平均相关系数", "{:.2f}".format(avg_corr),
              delta=delta_text, delta_color=delta_clr)

    col_r, col_d = st.columns(2)
    with col_r:
        st.markdown("**冗余 / 重复（相关系数最高的对）**")
        redundant = [p for p in pairs if p[2] > 0.5]
        if redundant:
            for a, b, c in redundant[:5]:
                icon = "🔴" if c > 0.7 else "🟠"
                st.markdown("- **{}** ↔ **{}**：`{:.2f}` {}".format(a, b, c, icon))
        else:
            st.info("没有相关系数 > 0.5 的策略对")

    with col_d:
        st.markdown("**真分散（相关系数最低 / 负相关的对）**")
        diversified = sorted(pairs, key=lambda x: x[2])[:5]
        for a, b, c in diversified:
            icon = "🟢" if c < 0.3 else "🟡"
            st.markdown("- **{}** ↔ **{}**：`{:.2f}` {}".format(a, b, c, icon))

    # ── 3. 等权组合 vs 各单策略 ───────────────────────
    st.subheader("等权组合 vs 各单策略")
    st.caption(
        "等权组合 = 把资金等分到所有策略，每天组合收益是各策略日收益率的算术平均。"
        "如果策略间高度相关，组合的波动和回撤与单策略差不多（分散无效）；"
        "反之，组合应该更平稳（波动更低、回撤更小）。"
    )

    combo_equity, combo_metrics = combined_portfolio(aligned)
    if combo_equity.empty:
        st.warning("无法构建等权组合")
        return

    from quant.backtest.engine import equity_metrics as _eq_metrics
    compare_rows: dict[str, dict] = {"等权组合": combo_metrics}
    for col_name in aligned.columns:
        single_eq = INITIAL_CASH * (1 + aligned[col_name]).cumprod()
        compare_rows[col_name] = _eq_metrics(single_eq, INITIAL_CASH)

    compare = pd.DataFrame(compare_rows).T
    compare = compare.rename(columns=RISK_COLS)

    def highlight_best(col):
        best = col.min() if col.name == "年化波动" else col.max()
        return ["background-color: {}; font-weight: bold".format(BUY_BG)
                if v == best else "" for v in col]

    def combo_row_style(row):
        if row.name == "等权组合":
            return ["background-color: rgba(33, 150, 243, 0.15); font-weight: bold"] * len(row)
        return [""] * len(row)

    styler = (compare.style
              .apply(highlight_best, axis=0)
              .apply(combo_row_style, axis=1)
              .format({"总收益": "{:+.1%}", "年化收益": "{:+.1%}", "最大回撤": "{:.1%}",
                       "年化波动": "{:.1%}", "夏普": "{:.2f}", "Calmar": "{:.2f}"}))
    st.dataframe(styler, width="stretch")

    # 分散效果小结
    combo_vol = combo_metrics.get("volatility", 0)
    avg_single_vol = sum(
        compare_rows[c].get("volatility", 0) for c in aligned.columns
    ) / len(aligned.columns)
    if avg_single_vol > 0:
        reduction = 1 - combo_vol / avg_single_vol
        if reduction > 0.1:
            st.success("等权组合波动率比单策略平均低 {:.0%}——分散有一定效果。".format(reduction))
        elif reduction > 0:
            st.info(
                "等权组合波动率仅比单策略平均低 {:.0%}——分散效果有限，".format(reduction)
                + "说明策略间高度相关。"
            )
        else:
            st.warning("等权组合波动率反而高于单策略平均——可能是策略数太少或正好同涨同跌。")


PORTFOLIO_STRATEGIES = {"momentum", "dual_momentum", "stock_momentum", "low_vol"}
INITIAL_CASH = 10_000.0

RISK_COLS = {
    "total_return": "总收益", "cagr": "年化收益", "max_drawdown": "最大回撤",
    "volatility": "年化波动", "sharpe": "夏普", "calmar": "Calmar",
}


def date_window(df: pd.DataFrame, key: str) -> pd.DataFrame | None:
    min_d, max_d = df.index[0].date(), df.index[-1].date()
    start, end = st.slider("时间区间", min_value=min_d, max_value=max_d,
                           value=(min_d, max_d), key=key)
    window = df.loc[str(start):str(end)]
    if len(window) < 2:
        st.warning("选中区间数据不足")
        return None
    return window


def metric_cards(metrics: dict):
    cols = st.columns(6)
    for col, (k, v) in zip(cols, metrics.items()):
        col.metric(k, v)


def excess_chips(strategy_total: float, benchmarks: dict[str, float]):
    cols = st.columns(6)
    for col, (label, base) in zip(cols, benchmarks.items()):
        excess = strategy_total - base
        col.metric(f"策略 vs {label}", f"{excess:+.1%}",
                   delta=f"{'跑赢' if excess > 0 else '跑输'}{label}",
                   delta_color="normal" if excess > 0 else "inverse")


def risk_table(rows: dict[str, pd.Series]):
    """rows: 名称 -> 权益曲线（同一本金起步）。逐列高亮最优值。"""
    table = pd.DataFrame({name: equity_metrics(eq, INITIAL_CASH) for name, eq in rows.items()}).T
    table = table.rename(columns=RISK_COLS)

    def highlight(col):
        best = col.min() if col.name == "年化波动" else col.max()
        return [f"background-color: {BUY_BG}; font-weight: bold" if v == best else "" for v in col]

    styler = (table.style.apply(highlight, axis=0)
              .format({"总收益": "{:+.1%}", "年化收益": "{:+.1%}", "最大回撤": "{:.1%}",
                       "年化波动": "{:.1%}", "夏普": "{:.2f}", "Calmar": "{:.2f}"}))
    st.markdown("**风险收益对比**（绿色=该列最优；Calmar=年化收益÷最大回撤，回撤小、收益稳才高）")
    st.dataframe(styler, width="stretch")


def trades_table(trades: list[dict], with_symbol: bool = False):
    if not trades:
        return
    st.subheader("交易明细")
    cols = (["symbol"] if with_symbol else []) + \
        ["entry_date", "exit_date", "entry", "exit", "pnl_pct", "profit"]
    names = (["标的"] if with_symbol else []) + ["买入日", "卖出日", "买入价", "卖出价", "收益", "利润($)"]
    df = pd.DataFrame(trades)[cols]
    df.columns = names

    styler = (df.style.map(signed_color, subset=["收益", "利润($)"])
              .format({"买入价": "{:.2f}", "卖出价": "{:.2f}",
                       "收益": "{:+.2%}", "利润($)": "{:+,.0f}"}))
    st.dataframe(styler, width="stretch", hide_index=True)

    # 利润集中度：收益依赖少数几笔"彩票"的程度
    profits = pd.Series([t["profit"] for t in trades])
    total = float(profits.sum())
    if len(profits) >= 5 and total > 0:
        k = min(10, len(profits))
        top_idx = profits.nlargest(k).index
        top = float(profits.loc[top_idx].sum())
        text = (f"**利润集中度**：盈利最大的 {k} 笔合计 ${top:,.0f}，"
                f"为总净利 ${total:,.0f} 的 {top / total:.0%}"
                f"（可超过 100%，因为亏损单会抵消）。")
        if with_symbol:
            counts: dict[str, int] = {}
            for i in top_idx:
                sym = trades[i]["symbol"]
                counts[sym] = counts.get(sym, 0) + 1
            breakdown = "、".join(
                f"{sym}×{n}" if n > 1 else sym
                for sym, n in sorted(counts.items(), key=lambda kv: -kv[1]))
            text += (f" 这 {k} 笔的标的分布：{breakdown}——"
                     f"标的越分散说明因子越广谱，若被一两只票刷屏则收益依赖个别彩票。")
        else:
            text += " 占比越高，收益越依赖少数几笔行情，策略的可复制性越弱。"
        st.caption(text)
    elif total <= 0:
        st.caption(f"区间内已平仓交易合计净亏损 ${total:,.0f}。")


def equity_markers(fig, equity: pd.Series, entries: list[str], exits: list[str],
                   entry_texts: list[str] | None = None, exit_texts: list[str] | None = None):
    for dates_raw, texts, name, shape, color in (
        (entries, entry_texts, "买入", "triangle-up", BUY_COLOR),
        (exits, exit_texts, "卖出", "triangle-down", SELL_COLOR),
    ):
        pairs = [(pd.Timestamp(d), (texts[i] if texts else ""))
                 for i, d in enumerate(dates_raw) if pd.Timestamp(d) in equity.index]
        if pairs:
            dates = [p[0] for p in pairs]
            fig.add_trace(go.Scatter(
                x=dates, y=equity.loc[dates], mode="markers", name=name,
                marker=dict(symbol=shape, size=12, color=color),
                hovertext=[p[1] for p in pairs],
            ))


def _render_single_bt(strategy_name: str, params: dict):
    group_symbols = cfg.symbols_for(params.get("groups", []))
    symbol = st.selectbox("标的", group_symbols)
    prices = {s: store.load_prices(conn, s) for s in group_symbols}
    prices = {s: df for s, df in prices.items() if not df.empty}
    if symbol not in prices:
        st.warning("库内没有该标的行情，先运行 python run_daily.py 拉取数据")
        return
    window = date_window(prices[symbol], key=f"single_{strategy_name}")
    if window is None:
        return
    st.caption(f"信号在全量历史上生成（指标不受区间影响）；持仓从区间内第一个买入信号开始。"
               f"价格为复权价（含分红），单边成本 {cfg.cost_bps:.0f}bp。同为期初一次性投入，基准只对比长持。")

    strat = strategies.build(strategy_name, params)
    sigs = strat.generate(prices)
    result = run_backtest(window, sigs, symbol, strategy_name, INITIAL_CASH, cfg.cost_bps)
    metric_cards(result.metrics())

    px = price_series(window)
    hold = hold_equity(px, INITIAL_CASH, cfg.cost_bps)
    excess_chips(result.total_return, {
        "长持": float(hold.iloc[-1]) / INITIAL_CASH - 1,
    })
    risk_table({"策略": result.equity, "长持": hold})

    eq = go.Figure(go.Scatter(x=result.equity.index, y=result.equity, mode="lines", name="策略权益"))
    eq.add_trace(go.Scatter(x=hold.index, y=hold, mode="lines", name="长持基准",
                            line=dict(dash="dash", color="#888")))
    entries = [t["entry_date"] for t in result.trades]
    if result.open_position:
        entries.append(result.open_position["entry_date"])
    equity_markers(eq, result.equity, entries, [t["exit_date"] for t in result.trades])
    eq.update_layout(height=400, title=f"{symbol} · {strategy_name} 权益曲线（{result.start} ~ {result.end}）")
    st.plotly_chart(eq, width="stretch")

    trades_table(result.trades)
    if result.open_position:
        st.caption(f"区间末仍持仓：{result.open_position['entry_date']} 以 ${result.open_position['entry']:.2f} 买入，未平仓部分按区间末市值计入指标。")


def pool_equal_weight_equity(prices: dict[str, pd.DataFrame],
                             pools: dict[pd.Timestamp, list[str]],
                             initial_cash: float) -> pd.Series | None:
    """池子等权基准：每月重建的流动性池内等权持有（月度再平衡，不计成本）。
    与策略共享同一候选超集，幸存者偏差在对比中近似抵消。"""
    if not pools:
        return None
    adj = pd.DataFrame({s: price_series(df) for s, df in prices.items()}).sort_index()
    rets = adj.pct_change(fill_method=None)
    pool_dates = sorted(pools)
    current: list[str] = []
    i = 0
    values = []
    for ts in rets.index:
        while i < len(pool_dates) and pool_dates[i] <= ts:
            current = [s for s in pools[pool_dates[i]] if s in rets.columns]
            i += 1
        if current:
            r = rets.loc[ts, current].dropna()
            values.append(float(r.mean()) if not r.empty else 0.0)
        else:
            values.append(0.0)
    equity = initial_cash * (1 + pd.Series(values, index=rets.index)).cumprod()
    return equity.rename("pool_ew")


def equal_weight_equity(prices: dict[str, pd.DataFrame],
                        initial_cash: float) -> pd.Series | None:
    """等权基准：宇宙内全部标的每日等权持有（日度再平衡，不计成本）。
    去掉"事后挑中赢家"的偏差——比起拿 XLK 长持（十五年里恰好封神的板块）当基准，
    板块等权才是判断"轮动/择时有没有加信息"的公平对照。"""
    adj = pd.DataFrame({s: price_series(df) for s, df in prices.items()}).sort_index()
    if adj.empty:
        return None
    daily = adj.pct_change(fill_method=None).mean(axis=1)  # 每日等权（有数据的标的均值）
    return (initial_cash * (1 + daily.fillna(0.0)).cumprod()).rename("equal_weight")


def _render_portfolio_bt(strategy_name: str, params: dict):
    universe = cfg.symbols_for(params.get("groups", []))
    if params.get("universe_file"):
        universe += [s for s in cfg.universe_symbols(params["universe_file"])
                     if s not in universe]
    prices = {s: store.load_prices(conn, s) for s in universe}
    prices = {s: df for s, df in prices.items() if not df.empty}
    if not prices:
        st.warning("库内没有行情数据，先运行 python run_daily.py 拉取数据")
        return
    bench_symbol = "SPY" if "SPY" in prices else next(iter(prices))
    bench_window = date_window(prices[bench_symbol], key=f"pf_{strategy_name}")
    if bench_window is None:
        return
    start_str = bench_window.index[0].strftime("%Y-%m-%d")
    end_str = bench_window.index[-1].strftime("%Y-%m-%d")
    st.caption(f"组合轮动模式：资金始终在场内换仓，区间起点按区间之前的信号还原应有持仓。"
               f"价格为复权价（含分红），单边成本 {cfg.cost_bps:.0f}bp。"
               f"基准为一次性长持对照（{bench_symbol}，宇宙含 QQQ 时另加 QQQ 长持；"
               f"资金时间敞口一致才可比；定投基准只在智能定投模式提供）。")

    if params.get("universe_file"):
        excluded = st.multiselect(
            "剔除标的（敏感性检验：删掉大赢家看超额是否塌掉，池子等权基准同步剔除）",
            options=sorted(cfg.universe_symbols(params["universe_file"])),
            default=params.get("exclude", []),
        )
        if excluded:
            params = {**params, "exclude": excluded}
            prices = {s: df for s, df in prices.items()
                      if s not in set(excluded) or s in cfg.symbols_for(params.get("groups", []))}

    strat = strategies.build(strategy_name, params)
    sigs = strat.generate(prices)

    # 区间前信号推出起点持仓，在区间首日合成买入
    held: dict[str, None] = {}
    for s in sorted((x for x in sigs if x.date < start_str), key=lambda x: x.date):
        if s.direction == BUY:
            held.setdefault(s.symbol)
        else:
            held.pop(s.symbol, None)
    synth = [Signal(date=start_str, symbol=sym, strategy=strategy_name, direction=BUY,
                    price=0.0, strength=0.5, reason="区间起点已持有（承接区间前信号）")
             for sym in held]
    in_window = [s for s in sigs if start_str <= s.date <= end_str]

    window_prices = {s: df.loc[start_str:end_str] for s, df in prices.items()}
    window_prices = {s: df for s, df in window_prices.items() if not df.empty}
    result = run_portfolio_backtest(window_prices, synth + in_window, strategy_name,
                                    INITIAL_CASH, cfg.cost_bps)
    metric_cards(result.metrics())

    px = price_series(bench_window)
    hold = hold_equity(px, INITIAL_CASH, cfg.cost_bps)
    strategy_total = equity_metrics(result.equity, INITIAL_CASH)["total_return"]

    benchmarks: dict[str, pd.Series] = {f"{bench_symbol}长持": hold}
    # QQQ 长持基准：宇宙含 QQQ 且非主基准时补上。如 dual_momentum 在 SPY/QQQ/TLT 间
    # 来回切，QQQ 2015-2024 极强，需看策略是否真跑赢 QQQ 长持——暴露出来更诚实。
    if "QQQ" in prices and bench_symbol != "QQQ":
        qqq = prices["QQQ"].loc[start_str:end_str]
        if not qqq.empty:
            benchmarks["QQQ长持"] = hold_equity(price_series(qqq), INITIAL_CASH, cfg.cost_bps)
    # 纯板块策略（如 momentum）：加板块等权基准。用户观察"板块策略全跑输 XLK 长持"，
    # 但 XLK 是事后赢家；板块等权才是去掉幸存者偏差、判断轮动有没有加信息的公平对照。
    if params.get("groups") == ["sectors"]:
        ew = equal_weight_equity(window_prices, INITIAL_CASH)
        if ew is not None:
            benchmarks["板块等权"] = ew
    if strategy_name == "stock_momentum":
        # 池子等权：与策略共享同一候选超集，是判断"排名有没有加信息"的最干净对照
        pools = strat.monthly_pools(
            {s: df.loc[start_str:end_str] for s, df in prices.items() if not df.loc[start_str:end_str].empty})
        pool_ew = pool_equal_weight_equity(window_prices, pools, INITIAL_CASH)
        if pool_ew is not None:
            benchmarks["池子等权"] = pool_ew

    excess_chips(strategy_total, {
        name: float(eq_.iloc[-1]) / INITIAL_CASH - 1 for name, eq_ in benchmarks.items()
    })
    risk_rows: dict[str, pd.Series] = {"策略组合": result.equity}

    # ── 可选：波动率缩放（风险管理） ──────────────────────────
    vol_scale_on = st.checkbox(
        "波动率缩放（风险管理：按近期波动倒数减仓，仅减仓不加杠杆）",
        value=False, key=f"vol_scale_{strategy_name}",
    )
    vs_equity: pd.Series | None = None
    vs_weights: pd.Series | None = None
    if vol_scale_on:
        vc1, vc2 = st.columns(2)
        vs_target = vc1.number_input(
            "目标年化波动率", min_value=0.05, max_value=0.50, value=0.15,
            step=0.01, format="%.2f", key=f"vs_target_{strategy_name}",
        )
        vs_window = vc2.number_input(
            "波动率回看窗口（交易日）", min_value=20, max_value=252, value=63,
            step=1, key=f"vs_window_{strategy_name}",
        )
        vs_equity, vs_weights = vol_scaled_equity(
            result.equity, target_vol=vs_target, vol_window=vs_window,
            cap=1.0, initial_cash=INITIAL_CASH,
        )
        risk_rows["策略(波动缩放)"] = vs_equity
        avg_w = float(vs_weights.mean())
        st.caption(
            f"📉 **波动率缩放**：按近 {vs_window} 日已实现波动率的倒数减仓（目标波动率 {vs_target:.0%}），"
            f"仅减仓不加杠杆（cap=1.0，实测加杠杆有害）。"
            f"平均仓位 **{avg_w:.0%}**。\n\n"
            f"定位：降低最大回撤与最差单日（回撤缩减器），代价是收益略降、夏普基本不变——"
            f"**不是夏普放大器**。仅用于分析，不改动实盘信号。"
        )

    risk_rows.update(benchmarks)
    risk_table(risk_rows)

    eq = go.Figure(go.Scatter(x=result.equity.index, y=result.equity, mode="lines", name="策略组合"))
    if vs_equity is not None:
        eq.add_trace(go.Scatter(x=vs_equity.index, y=vs_equity, mode="lines", name="策略(波动缩放)",
                                line=dict(width=2, color="#ab47bc")))
    for i, (name, series) in enumerate(benchmarks.items()):
        eq.add_trace(go.Scatter(x=series.index, y=series, mode="lines", name=name,
                                line=dict(dash=("dash", "dot", "dashdot")[i % 3], color=("#888", "#bc8f5f", "#6a9fb5")[i % 3])))
    entries = [t["entry_date"] for t in result.trades] + [p["entry_date"] for p in result.open_positions]
    entry_texts = [t["symbol"] for t in result.trades] + [p["symbol"] for p in result.open_positions]
    equity_markers(eq, result.equity, entries, [t["exit_date"] for t in result.trades],
                   entry_texts, [t["symbol"] for t in result.trades])
    eq.update_layout(height=400, title=f"{strategy_name} 组合权益曲线（{result.start} ~ {result.end}）")
    st.plotly_chart(eq, width="stretch")

    trades_table(result.trades, with_symbol=True)
    if result.open_positions:
        names = ", ".join(f"{p['symbol']}（{p['entry_date']} 买入）" for p in result.open_positions)
        st.caption(f"区间末持仓：{names}，按区间末市值计入指标。")


def _render_smart_dca_bt(params: dict):
    symbol = params.get("symbol", "SPY")
    df_full = store.load_prices(conn, symbol)
    if df_full.empty:
        st.warning("库内没有该标的行情，先运行 python run_daily.py 拉取数据")
        return
    window = date_window(df_full, key="smart_dca")
    if window is None:
        return
    st.caption(f"智能定投模式（{symbol}）：每月首个交易日定投一份；死叉期暂停积攒，金叉恢复当日一次性补投。"
               f"对照组为同一笔资金的纯定投（投入节奏一致，可比）与长持。"
               f"价格为复权价（含分红），单边成本 {cfg.cost_bps:.0f}bp。")

    fast, slow = params.get("fast", 20), params.get("slow", 60)
    result = run_smart_dca_backtest(window, fast, slow, INITIAL_CASH, cfg.cost_bps)
    metric_cards(result.metrics())

    px = price_series(window)
    hold = hold_equity(px, INITIAL_CASH, cfg.cost_bps)
    dca = dca_equity(px, INITIAL_CASH, cfg.cost_bps)
    smart_total = equity_metrics(result.equity, INITIAL_CASH)["total_return"]
    excess_chips(smart_total, {
        "纯定投": float(dca.iloc[-1]) / INITIAL_CASH - 1,
        "长持": float(hold.iloc[-1]) / INITIAL_CASH - 1,
    })
    risk_table({"智能定投": result.equity, "纯定投": dca, "长持": hold})

    eq = go.Figure(go.Scatter(x=result.equity.index, y=result.equity, mode="lines", name="智能定投"))
    eq.add_trace(go.Scatter(x=dca.index, y=dca, mode="lines", name="纯定投",
                            line=dict(dash="dot", color="#bc8f5f")))
    eq.add_trace(go.Scatter(x=hold.index, y=hold, mode="lines", name="长持",
                            line=dict(dash="dash", color="#888")))
    for span_start, span_end in result.paused_spans:
        eq.add_vrect(x0=span_start, x1=span_end, fillcolor=SELL_COLOR, opacity=0.06, line_width=0)
    topups = [pd.Timestamp(d) for d in result.topup_dates if pd.Timestamp(d) in result.equity.index]
    if topups:
        eq.add_trace(go.Scatter(
            x=topups, y=result.equity.loc[topups], mode="markers", name="金叉补投",
            marker=dict(symbol="star", size=14, color=BUY_COLOR),
        ))
    eq.update_layout(height=400,
                     title=f"{symbol} 智能定投 vs 纯定投（{result.start} ~ {result.end}，红色底纹=暂停定投区段）")
    st.plotly_chart(eq, width="stretch")


def _render_vix_bt(params: dict):
    from dataclasses import replace

    trade_symbol = params.get("trade_symbol", "SPY")
    vix_symbols = [params.get("vix", "^VIX"), params.get("vix3m", "^VIX3M")]
    vix_prices = {s: store.load_prices(conn, s) for s in vix_symbols}
    vix_prices = {s: df for s, df in vix_prices.items() if not df.empty}
    df_trade = store.load_prices(conn, trade_symbol)
    if df_trade.empty or params.get("vix", "^VIX") not in vix_prices:
        st.warning("库内缺少 VIX 或交易标的行情，先运行 python run_daily.py 拉取数据")
        return
    window = date_window(df_trade, key="vix_regime")
    if window is None:
        return
    st.caption(f"VIX 提醒本身不可交易；此处把每条提醒当作 {trade_symbol} 的买卖执行"
               f"（sell=清仓、buy=回补），检验 VIX 择时是否创造价值。"
               f"价格为复权价（含分红），单边成本 {cfg.cost_bps:.0f}bp。")

    strat = strategies.build("vix_regime", params)
    sigs = [replace(s, symbol=trade_symbol) for s in strat.generate(vix_prices)]
    # 起始持仓：区间开始时若无风险预警在身，视为持仓（先合成一笔买入）
    result = run_backtest(window, sigs, trade_symbol, "vix_regime", INITIAL_CASH, cfg.cost_bps)
    metric_cards(result.metrics())

    px = price_series(window)
    hold = hold_equity(px, INITIAL_CASH, cfg.cost_bps)
    excess_chips(result.total_return, {
        f"{trade_symbol}长持": float(hold.iloc[-1]) / INITIAL_CASH - 1,
    })
    risk_table({"VIX择时": result.equity, f"{trade_symbol}长持": hold})

    eq = go.Figure(go.Scatter(x=result.equity.index, y=result.equity, mode="lines", name="VIX择时"))
    eq.add_trace(go.Scatter(x=hold.index, y=hold, mode="lines", name=f"{trade_symbol}长持",
                            line=dict(dash="dash", color="#888")))
    entries = [t["entry_date"] for t in result.trades]
    if result.open_position:
        entries.append(result.open_position["entry_date"])
    equity_markers(eq, result.equity, entries, [t["exit_date"] for t in result.trades])
    eq.update_layout(height=400,
                     title=f"{trade_symbol} · VIX 择时权益曲线（{result.start} ~ {result.end}）")
    st.plotly_chart(eq, width="stretch")
    trades_table(result.trades)
    st.caption("注意：首个信号之前策略持币观望，若区间开头是长牛会显著跑输长持——"
               "重点看有恐慌事件的区间（如 2020、2022）里回撤是否更小。")


def render_backtest():
    st.title("回测")
    strategy_name = st.selectbox("策略", strategy_names)
    params = strategy_params[strategy_name]
    if strategy_name == "stock_momentum":
        st.warning(
            "⚠️ 仅观察策略：历史超额几乎全部来自 NVDA 单只标的，剔除后跑不赢“池子等权”基准。"
            "回测曲线不代表可复制的 alpha，勿据此实盘。"
        )
    if strategy_name in PORTFOLIO_STRATEGIES:
        _render_portfolio_bt(strategy_name, params)
    elif strategy_name == "smart_dca":
        _render_smart_dca_bt(params)
    elif strategy_name == "vix_regime":
        _render_vix_bt(params)
    else:
        _render_single_bt(strategy_name, params)


def render_strategy_docs():
    st.title("策略说明")
    sma = strategy_params.get("sma_cross", {"fast": 20, "slow": 60})
    sdca = strategy_params.get("smart_dca", {"symbol": "SPY", "fast": 20, "slow": 60})
    dm = strategy_params.get("dual_momentum",
                             {"lookback_days": 252, "risk_assets": ["SPY", "QQQ"], "safe_asset": "TLT"})
    vr = strategy_params.get("vix_regime",
                             {"panic": 30, "complacency": 15, "trade_symbol": "SPY"})
    sm = strategy_params.get("stock_momentum", {
        "universe_file": "universe_sp500.yaml", "pool_size": 100, "liquidity_window": 20,
        "lookback_days": 252, "skip_days": 21, "top_n": 6, "max_per_sector": 2,
        "regime_symbol": "SPY", "regime_ma": 200, "safe_asset": "TLT",
    })
    st.markdown(f"""
## 策略如何配合

**双均线管大方向**（该在场内还是场外）→ **动量管配置**（钱放哪个板块）→ **RSI 管时机**（回调到哪天动手）。
同一天出现矛盾信号时以大方向为准：大盘死叉之下的逆势买入信号，轻仓或忽略。
**智能定投**和**双动量**是独立的完整打法（自带仓位规则），直接以"跑赢定投"为目标，可作为主力策略单独执行。

---

## 1. sma_cross 双均线（趋势跟踪）

**直觉**：{sma["fast"]} 日均线是"最近一个月的平均成本"，{sma["slow"]} 日均线是"最近一个季度的平均成本"。
短期成本升到长期成本之上，说明近期买入者整体在赚钱、趋势向上。

**规则**：{sma["fast"]} 日线上穿 {sma["slow"]} 日线（金叉）→ 买入；下穿（死叉）→ 卖出。强度按快线近 5 日斜率：拐头越急越强。

**何时灵**：单边大趋势。少赚顶底各一段，换取绝不错过大趋势、绝不深套。

**何时坑**：横盘震荡市，均线反复交叉，假信号多且每次小亏（胜率低、靠大赢单撑收益是它的正常特征）。

**作用范围**：大盘、主题、资产类 ETF。

---

## 2. momentum 行业 12-1 月度动量轮动（相对强弱）

**直觉**：资金分板块轮动，过去 12 个月强的板块未来一个月大概率继续强（动量效应，学术上验证最充分的市场异象之一）。
跳过最近 1 个月是为了避开短期反转——大涨之后的板块短期常回调（"买在山顶"），
去掉这段噪音后的动量信号更干净。经典学术文献（Jegadeesh & Titman 1993）即用 12-1 口径。

**规则**：每月首个交易日——
1. 计算各行业 ETF 的 12-1 动量（近 {MOM_LOOKBACK} 个交易日收益，跳过最近 {MOM_SKIP} 日）；
2. 横截面按动量降序排名，取前 {MOM_TOP_N} 名纳入轮动组合；
3. 上月持有但本月跌出前 {MOM_TOP_N} → 卖出（先卖后买）；新进入 → 买入。

**旧版为何改造**：旧版用 63 日回看 + 每日进出，实测总收益 +89% 惨败于板块等权 +251%。
原因是短周期 + 日度调仓 = 把波动性龙头反复甩出去（whipsaw），频繁小亏侵蚀收益。
改为 252/skip21 月度后，交易次数从 932→84，总收益 +264%、回撤 -31.6%、Calmar 0.38，
跑赢板块等权且风险调整收益显著提升。

**何时灵**：板块分化明显的行情（如 AI 行情中科技/半导体持续霸榜），月度拿得住，不被日内噪音甩出。

**何时坑**：动量崩溃（大跌后的 V 型反转期追强追在山顶，月度频率下反应仍偏慢）；
板块收益高度聚集于少数行情主导时期，其余时间可能跑平或微跑输等权。

**作用范围**：11 只行业 ETF。

---

## 3. rsi_reversal RSI 反转（均值回归）

**直觉**：跌得又急又久时短期恐慌往往过度。关键是**不接飞刀**：不是 RSI 低就买，而是等它从超卖区**回升穿越**才发信号——恐慌见底、开始回暖的那一天。

**规则**：RSI({RSI_PERIOD}) 从 {RSI_OVERSOLD} 之下回升穿过 {RSI_OVERSOLD} → 买入（超卖越深强度越高）；从 {RSI_OVERBOUGHT} 之上回落穿过 {RSI_OVERBOUGHT} → 卖出。

**何时灵**：牛市或震荡市里的急跌回调，正好补双均线"震荡市难受"的短板。

**何时坑**：持续阴跌的熊市，每次弱反弹都给买入信号。熊市里（大盘死叉之下）它的买入信号要打折看待。

**作用范围**：全部 21 只 ETF。

---

## 4. smart_dca 智能定投（定投 + 趋势开关）

**直觉**：定投的弱点是熊市里持续接飞刀。给定投装一个趋势开关：趋势向上正常投，
趋势向下把钱攒着，趋势恢复时把攒的钱一次性投在相对低位。不追求跑赢牛市，追求熊市少挨打。

**规则**：每月首个交易日为定投日。MA{sdca["fast"]} ≥ MA{sdca["slow"]}（复权价）→ 正常定投一份；
死叉期暂停，份额累积；金叉恢复当日一次性补投全部累积款。信号每月至多一条，就是你的定投提醒。

**何时灵**：有像样熊市的区间（2022 类）；暂停避开下跌主段，补投买在恢复初期。

**何时坑**：单边慢牛里和纯定投几乎没差别（开关很少触发）；V 型急跌快速反转时，
暂停错过的底部比补投买回的更便宜，会小幅跑输纯定投。

**作用范围**：{sdca["symbol"]}（config 可改）。

---

## 5. dual_momentum 双动量 GEM（绝对动量 + 相对动量）

**直觉**：相对动量选最强的风险资产，绝对动量决定要不要在场——过去 12 个月连绝对收益都是负的，
说明整体是熊市，切到避险资产等风暴过去。经典 Gary Antonacci GEM 打法，牛市跟上、熊市少亏。

**规则**：每月首个交易日，比较 {", ".join(dm["risk_assets"])} 近 {dm["lookback_days"]} 日总回报（复权价）：
最强者为正 → 持有它；为负 → 切换到 {dm["safe_asset"]}。目标变化才换仓，每月至多一次。

**何时灵**：趋势分明的大级别行情，尤其是漫长熊市（2000、2008 型），避险腿的价值全在这里。

**何时坑**：两点必须知道。一是**震荡年的鞭打**：动量在正负之间反复横跳，来回换仓两头挨耳光；
二是**{dm["safe_asset"]} 的久期风险**：TLT 是 20 年长债，2022 年加息导致股债双杀，
它不但没避险反而放大回撤——如果更在意这种情形，可把 `safe_asset` 换成短债 BIL（近似现金）。
另外它的收益大头是票息，回测必须用复权价（本平台已是）。

**作用范围**：风险腿 {", ".join(dm["risk_assets"])}，避险腿 {dm["safe_asset"]}（均可在 config 修改）。

---

## 6. vix_regime VIX 情绪提醒（期权市场的信息浓缩）

**直觉**：VIX 是标普 500 期权隐含波动率指数，反映期权市场为"保险"支付的价格。
恐慌时保险贵（VIX 高），自满时保险便宜（VIX 低）；而 VIX 超过三个月期 VIX3M（期限倒挂）
意味着市场对"眼前"的恐惧超过对"未来"的恐惧——历史上是可靠性较高的风险预警。

**规则**（提醒信号，不直接对应交易）：
- VIX 上穿 {vr["panic"]:.0f} → ⚠️ 进入恐慌区，控制仓位
- VIX 回落穿 {vr["panic"]:.0f} → ✅ 恐慌消退，历史上常是分批回补窗口
- VIX 跌破 {vr["complacency"]:.0f} → ⚠️ 自满区，防范突发回调
- VIX ≥ VIX3M（倒挂）→ ⚠️ 风险预警；倒挂解除 → ✅ 预警撤除

**何时灵**：急跌/危机前后（2020.2 倒挂先于崩盘主段出现）；给其他策略的信号做交叉验证。

**何时坑**：VIX 高不代表马上跌完——恐慌区里它可以继续冲到 80；自满区可以持续数年
（2017 全年 VIX < 15 且市场一路涨）。它是"环境判断"，不是精确择时器。
回测页把提醒映射到 {vr["trade_symbol"]} 执行只是检验手段，实际建议当作仓位调节参考。

---

## 7. stock_momentum 个股横截面动量（选股版动量轮动）

**直觉**：指数按市值加权——市值是"过去涨出来的结果"；动量按近期强弱加权——押"强者恒强"。
横截面动量（Jegadeesh & Titman 1993）是实证金融里被验证最充分的异象之一。

**规则**：每月首个交易日三步走——
1. **动态池**：候选超集（{sm["universe_file"]}，约 500 只）按近 {sm["liquidity_window"]} 日平均成交额取前 {sm["pool_size"]} 名。
   池子只用当时的数据重建（point-in-time），新贵在变得足够大、足够流动时被规则自动接纳；
2. **选股**：池内按 12-1 动量（近 {sm["lookback_days"]} 日收益、跳过最近 {sm["skip_days"]} 日避开短期反转）
   排名，取前 {sm["top_n"]} 只，单行业最多 {sm["max_per_sector"]} 只；
3. **风控**：{sm["regime_symbol"]} 跌破 {sm["regime_ma"]} 日均线 → 全部清仓切 {sm["safe_asset"]}，防动量崩溃。

**何时灵**：趋势分明、板块轮动清晰的行情；能在主升浪早中段抓住 NVDA 式的大动量股。

**何时坑**：三个都要记住。
一是**幸存者偏差**：候选超集是今天的成分快照，中途退市的输家缺席，**绝对收益虚高**
（量级约每年 1-2 个点）——所以回测页给了"池子等权"基准，它与策略共享同一偏差，
**跑赢池子等权才说明动量排名真的加了信息**，这是本策略回测唯一该信的对比；
二是**动量崩溃**：V 型反转月纯动量能亏 20-30%，regime 过滤只能缓解不能免疫；
三是**个股波动**：{sm["top_n"]} 只集中持仓的回撤和波动显著高于指数，看 Calmar 别只看总收益。

**作用范围**：动态流动性池（候选超集见 `{sm["universe_file"]}`，建议每半年手工更新）。
""")
    st.warning(
        "⚠️ 仅观察，不建议实盘。"
        "敏感性检验结论：剔除 NVDA 单只标的即让 2015–2020 收益跑输 SPY/QQQ，"
        "且剔除后连“池子等权”基准都跑不赢——历史超额几乎全部来自 NVDA 的集中暴露，"
        "12-1 排名本身未加信息（等权反而更强）。请勿凭回测曲线给该策略分配真实资金。"
    )
    lv = strategy_params.get("low_vol", {"lookback_days": 90, "top_n": 3})
    st.markdown(f"""
---

## 8. low_vol 低波动因子（首个非动量分散因子）

**低波动异象是什么**：传统金融理论认为高风险应有高回报（CAPM），但大量实证研究
（Baker, Bradley & Wurgler 2011; Ang et al. 2006）发现低波动资产的长期**风险调整后收益反而更好**。
原因包括：投资者的彩票偏好（高估高波动股的暴涨概率）、杠杆约束（机构不能杠杆买低波动所以它被低估）、
以及基准追踪导致基金经理系统性忽视低波动标的。

**本策略怎么算**：每月首个交易日——
1. 计算各标的近 {lv["lookback_days"]} 个交易日的**已实现波动率**（日收益率标准差 × √252 年化），
   收益率用总回报口径（复权价，含分红再投资）；
2. 横截面排名，波动率最低的前 {lv["top_n"]} 只纳入持仓组合；
3. 新进入最低波动前 {lv["top_n"]} → 买入；跌出 → 卖出。先卖后买，月度调仓。

**参数含义**：
- `lookback_days`（当前 {lv["lookback_days"]}）：波动率回看窗口天数。越长越稳定（减少换手），
  越短越灵敏（快速反映近期波动变化）。90 日约 4 个月，是常用的中等窗口。
- `top_n`（当前 {lv["top_n"]}）：持有波动最低的前 N 只。与 momentum 同为 3，方便对比。

**它的定位**：本平台首个**非动量因子**。现有策略多为动量家族
（momentum / dual_momentum / stock_momentum 都基于"强者恒强"），平均相关系数约 0.51，
叠加使用分散效果有限。低波动因子的选择标准与动量正交——动量选"涨得猛的"，低波动选"波动小的"。
但**实测发现**：若只在 sectors（行业 ETF）里选，low_vol 与 momentum 相关仍约 0.52——因为二者
都是长多股票、共享市场 beta，"换了个理由拿股票"而已。**把宇宙扩到含 TLT 长债/GLD 黄金后，
low_vol 能选到非股票资产，与 momentum 相关降至约 0.26**，才真正成为分散来源。
可到「策略相关性」页验证实际相关系数。

**何时灵**：市场风格轮动偏防御、股票普遍回调而债金稳守的行情（如 2022 年科技股暴跌）。

**何时坑**：单边大牛市中低波动资产严重跑输高动量板块（如 2023-2024 AI 行情中 XLP/XLU/TLT
远逊 XLK/SMH），持有体验差。另外标的间波动率差异有时很小，排名可能因微小差异频繁换手。

**作用范围**：行业 ETF（sectors）+ 资产类（assets：TLT 长债 / GLD 黄金 / IBIT），
低波动天然会偏向债金——这正是它分散股票动量的关键。
""")
    st.markdown("""
---

## 🔧 波动率缩放（回测页可选开关）

回测页的组合策略（momentum / dual_momentum 等）提供可选的「波动率缩放」复选框。

**原理**：按近期已实现波动率的倒数调整仓位——波动越高，仓位越低（只减仓不加杠杆，cap=1.0）。

**实测效果**（实验验证）：
- ✅ **降低最大回撤与最差单日**（如 momentum 回撤 -31.6%→-21.1%、最差单日 -10.6%→-6.1%）
- ⚠️ **收益略降**（高波时减仓会错过部分反弹）
- ⚠️ **夏普基本不变**（波动率和收益同比例缩小，比率不变）

**定位**：它是【回撤/尾部缩减器】，不是夏普放大器。适合关注回撤控制的场景。
加杠杆（cap>1）实测反而有害——波动放大抵消收益、回撤恶化——所以只做减仓方向。

**仅用于分析**：这是回测页的可选分析工具，不改动实盘信号生成。

---

*参数在 `config.yaml` 中修改，本页数值实时读取当前配置。回测统一使用复权价（含分红）
与 `backtest.cost_bps` 单边成本。提醒：不要为了回测曲线好看精调参数——那是过拟合；
当前默认值是学术与实务中最常用的取值。*
""")


SECTOR_NAMES = {
    "XLK": "科技", "XLV": "医疗", "XLF": "金融", "XLY": "可选消费",
    "XLP": "必需消费", "XLE": "能源", "XLI": "工业", "XLB": "材料",
    "XLU": "公用事业", "XLRE": "房地产", "XLC": "通信",
}


def _stock_sector_map() -> dict[str, str]:
    """从 universe_sp500.yaml 建 个股代码 -> GICS 行业 映射（供强弱榜展示）。"""
    try:
        with open(ROOT / "universe_sp500.yaml", encoding="utf-8") as f:
            grouped = yaml.safe_load(f)
    except (OSError, yaml.YAMLError):
        return {}
    return {sym: sector for sector, syms in grouped.items() for sym in syms}


def _render_strength_table(df: pd.DataFrame, label_map: dict | None = None):
    """渲染强弱表：12-1动量/52周位置/距200MA/综合分，明暗主题下前景色随正负切换。"""
    d = df.copy()
    d.insert(0, "标的", [
        f"{s}｜{label_map[s]}" if label_map and s in label_map else s for s in d.index
    ])
    cols = ["标的"] + (["行业"] if "行业" in d.columns else []) + \
        ["mom", "pos_52w", "dist_ma", "composite"]
    show = d[cols].rename(columns={
        "mom": "12-1动量", "pos_52w": "52周位置", "dist_ma": "距200MA", "composite": "综合分",
    })
    styler = (show.style
              .map(signed_color, subset=["12-1动量", "距200MA"])
              .format({"12-1动量": "{:+.1%}", "52周位置": "{:.0%}",
                       "距200MA": "{:+.1%}", "综合分": "{:.0%}"}, na_rep="—"))
    st.dataframe(styler, width="stretch", hide_index=True)


def render_market_screen():
    """市场筛选：大盘趋势灯 + 板块强弱 + 个股强弱榜 + 透明综合分。"""
    st.title("市场筛选")
    st.caption("当前强弱【快照】（非回测）：综合分 = 12-1动量 / 52周位置 / 距200日均线 三维"
               "横截面百分位排名的等权平均，刻意简单、不做权重优化（避免过拟合）。"
               "个股宇宙为今日成分快照含幸存者偏差；12-1 动量买的是已涨完的强者，"
               "短期有反转/买在山顶风险，仅作强弱参考，不构成交易建议。")

    # ── 大盘趋势灯 ──
    spy = store.load_prices(conn, "SPY")
    reg = market_regime(spy) if not spy.empty else {"risk_on": None, "dist": None}
    sect_prices = {s: store.load_prices(conn, s) for s in cfg.symbols_for(["sectors"])}
    sect_prices = {s: df for s, df in sect_prices.items() if not df.empty}
    breadth = sector_breadth({s: df["close"] for s, df in sect_prices.items()})

    c1, c2 = st.columns(2)
    if reg["risk_on"] is None:
        c1.metric("大盘趋势（SPY vs 200日线）", "数据不足")
    else:
        c1.metric("大盘趋势（SPY vs 200日线）",
                  "🟢 站上 risk-on" if reg["risk_on"] else "🔴 跌破 risk-off",
                  delta=f"{reg['dist']:+.1%} 偏离均线",
                  delta_color="normal" if reg["risk_on"] else "inverse")
    if breadth["total"]:
        c2.metric("板块宽度（站上200日线）",
                  f"{breadth['above']}/{breadth['total']} 个板块",
                  delta="偏强" if breadth["above"] * 2 >= breadth["total"] else "偏弱",
                  delta_color="normal" if breadth["above"] * 2 >= breadth["total"] else "inverse")

    # ── 排序依据（同时作用于板块表与个股榜）──
    SORT_OPTIONS = {"综合分": "composite", "12-1动量": "mom",
                    "52周位置": "pos_52w", "距200日均线": "dist_ma"}
    sort_label = st.selectbox(
        "排序依据", list(SORT_OPTIONS), index=0, key="screen_sort",
        help="综合分=三维等权融合；选单一维度可看纯榜单（如按 12-1动量 看动量冠军，"
             "哪怕它已从高点回落）",
    )
    sort_col = SORT_OPTIONS[sort_label]

    # ── 板块强弱 ──
    st.subheader("板块强弱")
    sect_str = compute_strength(sect_prices)
    if sect_str.empty:
        st.warning("板块行情不足，先运行 python run_daily.py 拉取数据")
    else:
        sect_sorted = sect_str.dropna(subset=[sort_col]).sort_values(sort_col, ascending=False)
        _render_strength_table(sect_sorted, label_map=SECTOR_NAMES)

    # ── 个股强弱榜 ──
    st.subheader("个股强弱榜（S&P500 候选池）")
    with st.spinner("加载个股行情并计算强弱…"):
        stock_syms = cfg.universe_symbols("universe_sp500.yaml")
        stock_prices = {s: store.load_prices(conn, s) for s in stock_syms}
        stock_prices = {s: df for s, df in stock_prices.items() if not df.empty}
        stock_str = compute_strength(stock_prices)
    if stock_str.empty:
        st.warning("个股行情不足（需要至少约 1 年数据）")
        return
    sec_map = _stock_sector_map()
    stock_str["行业"] = [sec_map.get(s, "") for s in stock_str.index]
    st.caption(f"共 {len(stock_str)} 只个股参与排名（历史足够者），截至各自最新交易日。"
               f"当前按【{sort_label}】排序。")
    ranked = stock_str.dropna(subset=[sort_col]).sort_values(sort_col, ascending=False)
    n = st.slider("每侧显示数量", 5, 30, 15, key="screen_n")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"**🟢 最强（{sort_label}居前）**")
        _render_strength_table(ranked.head(n))
    with col_b:
        st.markdown(f"**🔴 最弱（{sort_label}垫底）**")
        _render_strength_table(ranked.tail(n).iloc[::-1])


PAGES = {
    "📊 市场概览": render_market_overview,
    "📡 信号历史": render_signal_history,
    "🕯️ K线与信号": render_kline,
    "🏆 动量排名": render_momentum_rank,
    "🔍 市场筛选": render_market_screen,
    "🎯 策略评分": render_strategy_scoring,
    "🔗 策略相关性": render_correlation,
    "🧪 回测": render_backtest,
    "📖 策略说明": render_strategy_docs,
}

page = st.pills("页面导航", list(PAGES), default=next(iter(PAGES)),
                required=True, label_visibility="collapsed")
st.divider()
PAGES[page]()
