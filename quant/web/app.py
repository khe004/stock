"""Streamlit 复盘面板：streamlit run quant/web/app.py"""

import json
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
    average_pairwise_corr,
    combined_portfolio,
    correlation_matrix,
    strategy_return_series,
    suggest_low_corr_set,
)
from quant.analysis.market import ETF_NAMES, etf_label, range_position, sector_breadth, yield_curve_spread
from quant.analysis.robustness import (defensive_symbols, equal_weight_equity,
                                      leverage_to_target_vol, levered_returns,
                                      robustness_report, split_windows)
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
        "板块": [etf_label(s) for s in latest.index],
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

    # ── 进攻档成长 ETF 池的 12-1 动量排名（独立一份，不与板块混）──
    st.divider()
    st.subheader("进攻档成长 ETF · 12-1 月度动量")
    agg_p = strategy_params.get("aggressive_mom", {})
    agg_lb, agg_sk = agg_p.get("lookback_days", 252), agg_p.get("skip_days", 21)
    agg_tn = agg_p.get("top_n", 1)
    agg_safe = set(agg_p.get("safe_assets", ["TLT"]))
    agg_closes = group_closes("aggressive_growth", adjusted=True)
    agg_mom = (agg_closes.shift(agg_sk) / agg_closes.shift(agg_lb) - 1).dropna(how="all") \
        if not agg_closes.empty else pd.DataFrame()
    if agg_mom.empty:
        st.warning(f"进攻档 ETF 行情不足（需要至少 {agg_lb} 个交易日）")
        return
    agg_latest = agg_mom.iloc[-1].dropna().sort_values(ascending=False)
    as_of_agg = agg_mom.index[-1].strftime("%Y-%m-%d")
    st.caption(f"截至 {as_of_agg}，进攻档在成长 ETF 里持有 12-1 动量最强的前 {agg_tn} 只；"
               f"现金感知——需动量为正才买，成长全负则切正动量避险，避险也负则切现金等价 BIL 吃短债利率。")
    growth_rows = [(s, v) for s, v in agg_latest.items() if s not in agg_safe]
    gtable = pd.DataFrame({
        "排名": range(1, len(growth_rows) + 1),
        "成长ETF": [etf_label(s) for s, _ in growth_rows],
        "12-1 动量": [v for _, v in growth_rows],
        "状态": ["🟢 进攻持有" if (i < agg_tn and v > 0) else ("⚠️ 动量为负" if v <= 0 else "")
                 for i, (_, v) in enumerate(growth_rows)],
    })

    def agg_style(row):
        bg = BUY_BG if (row["排名"] <= agg_tn and row["12-1 动量"] > 0) else ""
        return [f"background-color: {bg}"] * len(row)

    st.dataframe(gtable.style.apply(agg_style, axis=1).format({"12-1 动量": "{:+.1%}"}),
                 width="stretch", hide_index=True)
    safe_rows = [(s, v) for s, v in agg_latest.items() if s in agg_safe]
    if safe_rows:
        sname, sval = safe_rows[0]
        state = "🟢 可避险" if sval > 0 else "🔴 动量为负 → 持现金"
        st.caption(f"避险资产 {sname}：12-1 动量 {sval:+.1%}（{state}）——"
                   f"成长全负时仅在避险动量为正才切入，否则持现金。")


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
            "| 组合轮动（momentum / dual_momentum / stock_momentum / low_vol / cross_asset_mom） "
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

        # 加载所需价格（全量历史，不在这里截断——见下方说明）
        needed: set[str] = set()
        for syms in strat_symbols.values():
            needed.update(syms)
        needed.update(trade_map.values())
        prices = {s: store.load_prices(conn, s) for s in needed}
        prices = {s: df for s, df in prices.items() if not df.empty}

        # 在全量历史上回测出完整日收益率序列，区间截取放在【之后】对 returns_df 做，
        # 不能对 prices 做：run_portfolio_backtest 只按传入价格的日期范围推进，
        # 窗口开始前的信号会被直接丢弃、且不会补开「区间起点已持有」的仓位——
        # 若先截断 prices 再生成/回测，低换手策略（如 aggressive_mom/dual_momentum）
        # 会在窗口开头出现一段虚假的空仓期（曾实测 3 年窗口下 aggressive_mom 有
        # ~13 个月完全平线 0% 收益，而它其实一直持有仓位，只是那笔交易发生在窗口外）。
        # 对已经算出的日收益率序列做窗口截取则没有这个问题：每天的收益本就基于
        # 全历史下连续正确的持仓，截哪段都不会引入冷启动偏差。
        returns_df = strategy_return_series(
            prices, signals_by_strategy, strat_params, strat_symbols,
            cfg.cost_bps, trade_map,
        )
        if range_days is not None:
            returns_df = returns_df.tail(range_days)

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

    _render_model_portfolio(aligned, corr)


def _render_leverage(combo_daily: pd.Series, combo_m: dict) -> None:
    """杠杆分析：把组合的高夏普换成收益，能不能追上 QQQ。**仅分析用，不改实盘信号。**"""
    with st.expander("⚖️ 杠杆分析（把夏普换成收益，追 QQQ 的绝对收益）", expanded=False):
        st.caption(
            "本平台的胜利一直集中在**风险轴**（夏普/回撤），绝对收益却输 QQQ。"
            "「高夏普低收益」的教科书解法是加杠杆——但只有在夏普**确实**更高时才成立"
            "（AQR《Risk Parity: Why We Lever》）。这里诚实地检验换不换得动：融资成本按 "
            "**BIL 日收益 + 你设的价差**扣，每日再平衡杠杆天然含波动率拖累，回撤直接看数字。"
        )
        bil_df = store.load_prices(conn, "BIL")
        if bil_df.empty:
            st.warning("库内没有 BIL 行情，无法建模融资成本")
            return
        cash = price_series(bil_df).pct_change(fill_method=None).reindex(
            combo_daily.index).fillna(0.0)

        c1, c2 = st.columns(2)
        spread_bp = c1.slider("融资价差（bp/年，BIL 之上）", 0, 300, 50, 25,
                              help="期货基差约 25~50bp；资本效率 ETF 费率约 100bp；零售融资券 150bp+")
        target = c2.radio("杠杆到谁的波动率", ["SPY（保守）", "QQQ（激进）", "不加杠杆"],
                          index=0, horizontal=True, key="lev_target")

        rows: dict[str, dict] = {"组合（1.0x）": combo_m}
        bench_m: dict[str, dict] = {}
        for b in ("SPY", "QQQ"):
            bdf = store.load_prices(conn, b)
            if bdf.empty:
                continue
            br = price_series(bdf).pct_change(fill_method=None).reindex(
                combo_daily.index).fillna(0.0)
            bench_m[b] = equity_metrics(INITIAL_CASH * (1 + br).cumprod(), INITIAL_CASH)
            rows[f"{b} 长持"] = bench_m[b]

        k = 1.0
        if target != "不加杠杆" and bench_m:
            tgt = bench_m["SPY" if target.startswith("SPY") else "QQQ"]["volatility"]
            k = leverage_to_target_vol(combo_daily, tgt)
            for kk in sorted({round(k, 2), 1.25, 1.5, 2.0}):
                lev = levered_returns(combo_daily, kk, cash, spread_bp / 10_000)
                label = f"杠杆 {kk:.2f}x" + ("（= 目标波动）" if abs(kk - k) < 1e-9 else "")
                rows[label] = equity_metrics(INITIAL_CASH * (1 + lev).cumprod(), INITIAL_CASH)

        st.dataframe(
            pd.DataFrame(rows).T.rename(columns=RISK_COLS).style.format(
                {"总收益": "{:+.1%}", "年化收益": "{:+.1%}", "最大回撤": "{:.1%}",
                 "年化波动": "{:.1%}", "夏普": "{:.2f}", "Calmar": "{:.2f}"}),
            width="stretch")

        if target != "不加杠杆" and k == k:
            st.caption(
                f"到目标波动需要 **{k:.2f}x**。注意夏普会**下降**（融资成本 + 波动率拖累），"
                "杠杆买的是收益不是效率。\n\n"
                "**⚠️ 这不是一个可以直接照做的方案，四个坎**：\n"
                "1. **要杠杆我们的组合只能靠融资券/期货，得开保证金账户。** 用 TQQQ/QLD 这类杠杆 ETF "
                "能绕开保证金，但它们杠杆的是 **QQQ 的 beta 不是我们组合的 alpha**——实测（2026-07-29）"
                "严格更差：同样 +720% 收益下回撤 -37% vs 直接杠杆的 -30%，因为把分散掉的 QQQ 风险又掺"
                "回来了、且每日重置拖累巨大（3x 一年吃 21pt）。「杠杆低波的组合、别杠杆高波的 QQQ」。\n"
                "2. **常数杠杆要每日再平衡**，与本平台「月频低换手」的定位冲突；按月或按带宽再平衡"
                "会产生这里没建模的路径差异。\n"
                "3. **保证金追缴没建模**。回撤最深处被迫减仓，实际结果会比表里差得多。\n"
                "4. **整件事押在夏普估计上**，而 11 年样本的夏普误差棒很大，且组合的夏普优势"
                "有相当一部分是分散的数学结果 + 配方的 in-sample 选择。真实夏普若低一档，优势就没了。"
            )


def _render_model_portfolio(aligned: pd.DataFrame, corr: pd.DataFrame) -> None:
    """模型组合（Model Portfolio）：自选几个低相关策略等权，对比单策略与长持基准。

    定位见 analysis/correlation.py 的 suggest_low_corr_set 文档：这是对
    specification risk 与"策略有时效性"的实操回答——不是找那个永远有效的策略，
    而是同时持有几个决策方式不同、相互低相关的。
    """
    st.subheader("🧺 模型组合（Model Portfolio）")
    st.caption(
        "上面那张「等权组合」把**所有**启用策略都算了进去，包含 sma_cross / rsi_reversal / "
        "vix_regime / stock_momentum 这些**仅观察不建仓**的——那是分散度诊断，不是可执行组合。\n\n"
        "这里自己挑几个真会持有的策略等权组合。**为什么要组合而不是选一个最好的**："
        "单跑一个策略等于押注「这一套设定恰好对」（specification risk），而策略是有时效性的、"
        "事前又分不清哪个在当季——Allocate Smartly 跟踪 90+ 个 TAA 策略，其用户组合 78% 含多个策略、"
        "平均 3.8 个。挑选标准是**决策方式不同 + 相互低相关**，买的是「过程分散」而非只有「资产分散」。\n\n"
        "📅 本节沿用页面顶部的**回测区间**选择（默认近 3 年）——想看全历史请切到「全部」。\n\n"
        "💡 **几个实测过的配方**（全历史口径，2026-07-28）：`canary_mom` 与 `cross_asset_mom` "
        "相关 0.76（共享宇宙），所以是**替代**不是补位。把 cross_asset 换成 canary 后夏普 0.96→1.01、"
        "回撤 -20.6%→-18.0%，且**两个半段的夏普都是全场第一**；`canary+aggressive+low_vol` 三件套"
        "相关最低(0.38)、夏普最高(1.03)；`canary+aggressive` 双腿杠铃 +454%/回撤-21.2%/Calmar0.75，"
        "**每一项都优于 SPY 长持**，但只有 2 个成分 = specification risk 最高。"
        "注意 canary_mom 目前是 notify:false 仅观察，尚无样本外记录。"
    )

    # 买入持有型成分（如管理期货 DBMF）：不是策略、没有信号，作为独立收益腿并进来。
    # 只在本小节的局部 frame `ext` 里生效，不碰页面全局 aligned（否则 DBMF 的短历史
    # 会 dropna 掉整页 2019 之前的数据，污染上面的热力图/诊断表——这是之前修过的坑）。
    ext = aligned.copy()
    for sym in cfg.model_portfolio_hold_assets:
        hdf = store.load_prices(conn, sym)
        if not hdf.empty:
            ext[sym] = price_series(hdf).pct_change(fill_method=None).reindex(ext.index)
    # 组合层用的相关矩阵：含 DBMF 时按"共同窗口"（DBMF 有数据起）重算，其余情况与全局一致
    ext_corr = ext.dropna().corr() if len(ext.columns) > len(aligned.columns) else corr

    enabled = dict(cfg.enabled_strategies())
    # 默认成分 = config 里显式配置的推荐配方（strategies + hold_assets）；
    # 没配或成分不在当前收益序列里时，回落到"会推送的实盘候选"
    # （notify:false 的仅观察策略与已定性"不作实盘仓位"的 stock_momentum 除外）
    live = [c for c in aligned.columns
            if enabled.get(c, {}).get("notify", True) and c != "stock_momentum"]
    recommended = [c for c in cfg.model_portfolio if c in ext.columns]
    default = recommended or live or list(ext.columns)
    if "mp_pick" not in st.session_state:
        st.session_state["mp_pick"] = default

    c1, c2 = st.columns([3, 1])
    with c2:
        n_sug = st.number_input("推荐个数", 2, max(2, len(ext.columns)), 4, key="mp_n")
        if st.button("推荐低相关组合", key="mp_suggest"):
            # 只从实盘候选/推荐成分里挑——推荐里混进 vix_regime 这类仅观察策略是误导
            st.session_state["mp_pick"] = suggest_low_corr_set(ext_corr, int(n_sug), default)
    label = ("组合成分（默认=config 的 model_portfolio 推荐配方）" if recommended
             else "组合成分（默认=会推送的实盘候选，已排除仅观察策略）")
    with c1:
        picked = st.multiselect(label, options=list(ext.columns), key="mp_pick")
    if len(picked) < 2:
        st.info("至少选 2 个策略才能构成组合。")
        return

    # dropna over 选中列：含 DBMF 这类短历史成分时，窗口自动收窄到共同区间；
    # 不含时保持全历史。所有下游（组合、逐成分、分段）都基于这个共同窗口，口径一致。
    sub = ext[picked].dropna()
    combo_eq, combo_m = combined_portfolio(sub)
    if combo_eq.empty:
        st.warning("无法构建组合")
        return

    hold_picked = [c for c in picked if c in cfg.model_portfolio_hold_assets]
    if hold_picked and not sub.empty and len(sub) < len(aligned):
        st.caption(
            f"⚠️ 含买入持有成分 {('、'.join(hold_picked))}（历史较短），组合窗口已收窄到"
            f"**共同区间 {sub.index[0]:%Y-%m-%d} ~ {sub.index[-1]:%Y-%m-%d}**（{len(sub)} 天）——"
            "下面所有数字都是这段公共窗口内算的，不是全历史，跨配方比较时注意口径。"
        )

    rows: dict[str, dict] = {"🧺 模型组合": combo_m}
    for c in picked:
        rows[c] = equity_metrics(INITIAL_CASH * (1 + sub[c]).cumprod(), INITIAL_CASH)
    for b in ("SPY", "QQQ"):
        bdf = store.load_prices(conn, b)
        if bdf.empty:
            continue
        bret = price_series(bdf).pct_change(fill_method=None).reindex(sub.index).fillna(0.0)
        rows[f"{b} 长持"] = equity_metrics(INITIAL_CASH * (1 + bret).cumprod(), INITIAL_CASH)

    table = pd.DataFrame(rows).T.rename(columns=RISK_COLS)

    def _combo_style(row):
        return (["background-color: rgba(33, 150, 243, 0.15); font-weight: bold"] * len(row)
                if row.name == "🧺 模型组合" else [""] * len(row))

    st.dataframe(
        table.style.apply(_combo_style, axis=1).format(
            {"总收益": "{:+.1%}", "年化收益": "{:+.1%}", "最大回撤": "{:.1%}",
             "年化波动": "{:.1%}", "夏普": "{:.2f}", "Calmar": "{:.2f}"}),
        width="stretch")

    avg_c = average_pairwise_corr(ext_corr, picked)
    worst_dd = min(rows[c]["max_drawdown"] for c in picked)     # 回撤是负数，min = 最深
    st.caption(
        f"成分平均两两相关 **{avg_c:.2f}**（越低越好；>0.5 说明成分在做同一件事，组合是假分散）。"
        f"组合回撤 {combo_m['max_drawdown']:.1%}，最差成分回撤 {worst_dd:.1%}——"
        "**组合的价值主要在这里：把「最差那个」的痛苦削掉，而不是把最好那个的收益抬上去。**\n\n"
        "⚠️ 两条诚实提醒：\n"
        "1. **组合夏普高于每个成分，很大一部分是分散的数学结果，不是 alpha。** "
        "把不完全相关的序列平均，波动降得比收益快，夏普自然上去——这是真实可得的好处，"
        "但别把它读成「我们找到了更好的策略」。\n"
        "2. **策略成分沿用各自默认的月首日调仓**，而 dual_momentum / cross_asset_mom 的月首日恰好是"
        "四个调仓日里最好的那个（见回测页稳健性检验），所以组合数字也继承了这份 timing luck。"
        "「推荐低相关组合」也是拿相关矩阵挑的，属于轻度 in-sample 选择。"
        "（买入持有成分如 DBMF 无调仓，不涉及 timing luck，但样本历史更短——见上方窗口提示。）"
    )

    # ── 分段：组合的名次稳不稳 ─────────────────────────
    st.markdown("**分段检验：组合的名次稳不稳？**")
    wins = split_windows(sub.index, 2)
    seg_rows: dict[str, dict] = {}
    combo_ranks = []
    for label, s, e in wins:
        seg = sub.loc[s:e]
        if len(seg) < 20:
            continue
        seg_combo, seg_m = combined_portfolio(seg)
        if seg_combo.empty:
            continue
        seg_rows[f"{label}｜🧺 模型组合"] = seg_m
        cagrs = {}
        for c in picked:
            m = equity_metrics(INITIAL_CASH * (1 + seg[c]).cumprod(), INITIAL_CASH)
            seg_rows[f"{label}｜{c}"] = m
            cagrs[c] = m["cagr"]
        rank = 1 + sum(1 for v in cagrs.values() if v > seg_m["cagr"])
        combo_ranks.append((label, rank, len(picked) + 1))
    if seg_rows:
        st.dataframe(
            pd.DataFrame(seg_rows).T.rename(columns=RISK_COLS).style.format(
                {"总收益": "{:+.1%}", "年化收益": "{:+.1%}", "最大回撤": "{:.1%}",
                 "年化波动": "{:.1%}", "夏普": "{:.2f}", "Calmar": "{:.2f}"}),
            width="stretch")
        rank_txt = "、".join(f"{lab} 第 {r}/{n} 名" for lab, r, n in combo_ranks)
        st.caption(
            f"组合在各段的收益名次：{rank_txt}。**组合几乎永远不会是收益第一名**——"
            "它的目标不是当冠军，是**不当垫底**：每段都有策略跑输，但事前你不知道是哪个。\n\n"
            "读法与回测页的稳健性检验一致：看区间与最坏那一段，不看最大值；"
            "也不要因为某个成分在某段跑输就把它踢掉——好因子连输很多年是常态"
            "（价值输了 13 年），事后按近期表现换策略本身就是一种过拟合。"
        )

    _render_leverage(sub.mean(axis=1), combo_m)

    eqfig = go.Figure(go.Scatter(x=combo_eq.index, y=combo_eq, mode="lines",
                                 name="🧺 模型组合", line=dict(width=3, color="#2196f3")))
    for c in picked:
        eqfig.add_trace(go.Scatter(x=sub.index, y=INITIAL_CASH * (1 + sub[c]).cumprod(),
                                   mode="lines", name=c, line=dict(width=1)))
    eqfig.update_layout(height=420, title="模型组合 vs 各成分策略权益曲线")
    st.plotly_chart(eqfig, width="stretch")


PORTFOLIO_STRATEGIES = {"momentum", "dual_momentum", "stock_momentum", "low_vol",
                        "cross_asset_mom", "aggressive_mom", "canary_mom"}
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
    table.insert(0, "对象", table.index)  # 名称作为带表头的首列，避免索引列过窄显示不全

    def highlight(col):
        if col.name == "对象":
            return [""] * len(col)
        best = col.min() if col.name == "年化波动" else col.max()
        return [f"background-color: {BUY_BG}; font-weight: bold" if v == best else "" for v in col]

    styler = (table.style.apply(highlight, axis=0)
              .format({"总收益": "{:+.1%}", "年化收益": "{:+.1%}", "最大回撤": "{:.1%}",
                       "年化波动": "{:.1%}", "夏普": "{:.2f}", "Calmar": "{:.2f}"}))
    st.markdown("**风险收益对比**（绿色=该列最优；Calmar=年化收益÷最大回撤，回撤小、收益稳才高）")
    st.dataframe(styler, width="stretch", hide_index=True,
                 column_config={"对象": st.column_config.Column(width="medium")})


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


def _fmt_metrics_row(m: dict) -> dict:
    return {"总收益": m["total_return"], "年化": m["cagr"], "最大回撤": m["max_drawdown"],
            "夏普": m["sharpe"], "Calmar": m["calmar"]}


_METRIC_FMT = {"总收益": "{:+.0%}", "年化": "{:+.1%}", "最大回撤": "{:.1%}",
               "夏普": "{:.2f}", "Calmar": "{:.2f}"}


def _render_robustness(strategy_name: str, params: dict,
                       prices: dict[str, pd.DataFrame]) -> None:
    """稳健性检验面板：调仓日 timing luck 散布 + 分段 vs 池子等权。

    刻意【不出通过/不通过】，只出期望区间——见 quant/analysis/robustness.py 文档头：
    11 年月频数据的独立观测只有 2-3 个，不可能回答"有没有 alpha"，只能回答"数字有没有虚高"。
    """
    with st.expander("🧭 稳健性检验（调仓日 timing luck + 分段 + 池子等权）", expanded=False):
        st.caption(
            "回答的是「我报出来的数字有没有虚高」，**不是**「这个策略有没有 alpha」——"
            "11 年月频数据按 regime 高度自相关，独立观测只有 2-3 个，没有统计功效去判后者。\n\n"
            "三个问题：**①调仓日** 这个数字是不是恰好挑中了最好的调仓日（全平台默认锚在"
            "每月首个交易日，是个从未被检验的隐含选择）；**②分段** 是不是靠单一 regime 撑起来的；"
            "**③池子等权** 是不是宇宙本身好、而非选择能力。\n\n"
            "⚠️ 用**全部历史**计算，不受上方区间选择影响；要跑 4 遍策略，需要几秒。"
        )
        if len(prices) > 50:
            st.warning(f"本策略宇宙有 {len(prices)} 只标的，跑 4 遍可能要一分钟以上，请耐心等待。")
        key = f"robust_{strategy_name}"
        if st.button("跑稳健性检验", key=f"btn_{key}"):
            with st.spinner("正在按 4 个调仓日与各分段重跑…"):
                st.session_state[key] = robustness_report(
                    strategy_name, params, prices, INITIAL_CASH, cfg.cost_bps)
        rep = st.session_state.get(key)
        if rep is None:
            return

        tl = rep["timing_luck"]
        st.markdown("**① 调仓日散布（timing luck）**")
        rows = {r["label"]: _fmt_metrics_row(r) for r in tl["per_offset"]}
        rows["4-tranche 错峰组合"] = _fmt_metrics_row(tl["tranched"])
        df_tl = pd.DataFrame(rows).T
        st.dataframe(df_tl.style.format(_METRIC_FMT), width="stretch")
        verdict = ("现用的月首日恰好是四个调仓日里**最好**的一个——报数字时该用错峰口径，"
                   "否则虚高" if tl["current_is_best"] else
                   "现用的月首日是四个里**最差**的一个——现有文档数字偏保守，不是虚高"
                   if tl["current_is_worst"] else
                   "现用的月首日居中，没有明显挑到好日子")
        st.caption(
            f"年化收益跨度 **{tl['spread_cagr_bps']:.0f}bp**"
            f"（文献典型值约 100bp，Newfound / Hoffstein-Faber-Braun）。{verdict}。\n\n"
            "错峰组合 = 资金等分 4 份、各按不同调仓日独立跑再求和（tranching），"
            "是「去掉调仓日运气」的公平估计；文献称其不牺牲收益、纯降方差。"
            "**散布大不等于策略差**——要看错峰组合是否仍站得住。"
        )

        st.markdown("**② 分段 vs ③ 池子等权（公平基准）**")
        recs = {}
        for w in rep["windows"]:
            recs[f"{w['label']}｜策略(错峰)"] = _fmt_metrics_row(w["strategy"])
            if w["pool_ew"]:
                recs[f"{w['label']}｜★池子等权"] = _fmt_metrics_row(w["pool_ew"])
        st.dataframe(pd.DataFrame(recs).T.style.format(_METRIC_FMT), width="stretch")
        pool_txt = "、".join(rep["pool_symbols"])
        def_txt = ("；已排除避险/现金腿 " + "、".join(rep["defensive_symbols"])
                   if rep["defensive_symbols"] else "")
        rng = rep["excess_range"]
        st.caption(
            f"★池子等权 = 与策略共享候选名单的等权持有（{pool_txt}{def_txt}），"
            "是判断「选择有没有加信息」的公平对照。**别拿 SPY/QQQ/XLK 这类事后赢家当基准。**"
            + (f"\n\n**期望区间：分段超额年化 {rng[0]:+.1%} ~ {rng[1]:+.1%}。**"
               "点估计取区间内的中间水平而不是最大值；最坏那一段是你真的要扛的东西——"
               "策略有时效性，好因子也会连输很多年（价值输了 13 年、动量 2009 崩过），"
               "所以这里刻意不给「通过/不通过」。事前分不清哪个策略在当季，"
               "实操解药是同时持有几个低相关策略，而不是找那个永远有效的。"
               if rng else "")
            + "\n\n同时看夏普和 Calmar：集中型策略常见「赢 Calmar 输夏普」，"
              "那是集中度换来的回撤形状而非风险调整效率，只报有利的那个就是虚高。"
        )


def _render_universe_pool(strategy_name: str, params: dict, universe: list[str]) -> None:
    """标的池一览：按角色（候选/进攻、哨兵、防守、避险、现金）分组显示。

    不同策略的候选池差异很大（板块11只 vs 跨资产9类 vs 个股485只动态池 vs 进攻档8只），
    之前只能从下面的交易明细反推，加这行省得每次都要数交易记录。

    角色划分复用 defensive_symbols()（robustness.py 已在用同一套口径排除公平基准），
    但 dual_momentum 例外——它的 `groups` 为了给市场概览等页面提供数据，混进了
    risk_assets 之外从不参与策略逻辑的标的（IWM/DIA/GLD/IBIT），所以显式用
    risk_assets 参数而非"universe 减防守类"，否则会把没用到的标的也算进候选池。
    """
    if strategy_name == "stock_momentum":
        uf = params.get("universe_file", "")
        n_super = len(cfg.universe_symbols(uf)) if uf else len(universe)
        st.caption(
            f"🎯 标的池：**动态流动性池**（非固定名单，point-in-time）——候选超集 {n_super} 只"
            f"（`{uf}`），每月按近 {params.get('liquidity_window', 20)} 日平均成交额取前 "
            f"{params.get('pool_size', 100)} 名，持有动量最强的 {params.get('top_n', 6)} 只"
            f"（单行业上限 {params.get('max_per_sector', 2)} 只），"
            f"大盘破 {params.get('regime_ma', 200)} 日均线整体切避险 "
            f"{etf_label(params.get('safe_asset', 'TLT'))}。"
        )
        return

    defensive = defensive_symbols(params)
    if params.get("risk_assets"):
        offense = [s for s in params["risk_assets"] if s in universe]
    else:
        offense = [s for s in universe if s not in defensive]

    parts = [f"🎯 标的池（共 {len(universe)} 只）"]
    if offense:
        parts.append(f"**候选/进攻 {len(offense)} 只**：" + "、".join(etf_label(s) for s in offense))
    canary = [s for s in (params.get("canary_assets") or []) if s in universe]
    if canary:
        parts.append(f"**哨兵 {len(canary)} 只**（只判断风险开关，不参与持仓）：" +
                     "、".join(etf_label(s) for s in canary))
    defense = [s for s in (params.get("defense_assets") or []) if s in universe]
    if defense:
        parts.append(f"**防守腿 {len(defense)} 只**：" + "、".join(etf_label(s) for s in defense))
    safe_raw = params.get("safe_assets") or ([params["safe_asset"]] if params.get("safe_asset") else [])
    safe = [s for s in safe_raw if s in universe and s not in canary and s not in defense]
    if safe:
        parts.append(f"**避险腿 {len(safe)} 只**：" + "、".join(etf_label(s) for s in safe))
    cash = params.get("cash_asset")
    if cash and cash in universe and cash not in canary and cash not in defense and cash not in safe:
        parts.append(f"**现金等价**：{etf_label(cash)}")
    st.caption("；".join(parts) + "。")


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
               f"基准：SPY长持（风险标杆）+ QQQ长持（增长标杆）固定对照，部分策略另有"
               f"专属公平基准（板块等权/池子等权/等权全资产）。理想=收益/年化/夏普优于"
               f"QQQ、回撤/Calmar 优于 SPY。")
    _render_universe_pool(strategy_name, params, universe)

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
    in_window = [s for s in sigs if start_str <= s.date <= end_str]
    # 防御：区间首日已有真实买入信号的标的不再合成买入，否则同标的当天被买两次，
    # 组合引擎把现金对半分且第二次覆盖第一次，导致起始权益凭空减半。
    bought_on_start = {s.symbol for s in in_window
                       if s.date == start_str and s.direction == BUY}
    synth = [Signal(date=start_str, symbol=sym, strategy=strategy_name, direction=BUY,
                    price=0.0, strength=0.5, reason="区间起点已持有（承接区间前信号）")
             for sym in held if sym not in bought_on_start]

    window_prices = {s: df.loc[start_str:end_str] for s, df in prices.items()}
    window_prices = {s: df for s, df in window_prices.items() if not df.empty}
    result = run_portfolio_backtest(window_prices, synth + in_window, strategy_name,
                                    INITIAL_CASH, cfg.cost_bps)
    metric_cards(result.metrics())

    strategy_total = equity_metrics(result.equity, INITIAL_CASH)["total_return"]

    # 通用基准：SPY长持（风险标杆）+ QQQ长持（增长标杆）——对所有策略固定显示，
    # 无论策略宇宙是否含它们（从库内单独加载）。理想：收益/年化/夏普优于 QQQ、
    # 回撤/Calmar 优于 SPY。
    benchmarks: dict[str, pd.Series] = {}
    for _b in ("SPY", "QQQ"):
        _bdf = store.load_prices(conn, _b)
        if not _bdf.empty:
            _bw = _bdf.loc[start_str:end_str]
            if not _bw.empty:
                benchmarks[f"{_b}长持"] = hold_equity(price_series(_bw), INITIAL_CASH, cfg.cost_bps)
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
    if strategy_name == "cross_asset_mom":
        # 等权全资产：宇宙内全部标的等权持有，判断"跨资产动量轮动有没有加信息"的公平基准
        ew = equal_weight_equity(window_prices, INITIAL_CASH)
        if ew is not None:
            benchmarks["等权全资产"] = ew

    excess_chips(strategy_total, {
        name: float(eq_.iloc[-1]) / INITIAL_CASH - 1 for name, eq_ in benchmarks.items()
    })
    st.caption(
        "⚠️ 以上是**单一调仓日口径**（默认月首日），未做 timing luck 修正——"
        "月首日可能恰好是四个调仓日里最好或最差的一个，"
        "「跑赢/跑输」的判断请以下方**🧭 稳健性检验**的错峰口径与分段对比为准。"
    )
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

    _render_robustness(strategy_name, params, prices)

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


STRATEGY_CATEGORIES = {
    "研究中": ["cross_asset_mom", "aggressive_mom", "momentum", "dual_momentum", "low_vol",
             "canary_mom"],
    "定投": ["smart_dca"],
    "仅观察": ["stock_momentum", "sma_cross", "rsi_reversal", "vix_regime"],
}
STRATEGY_CATEGORY_NOTE = {
    "研究中": "在风险或收益上有可取之处、仍在打磨（稳健档 cross_asset_mom、进攻档 "
              "aggressive_mom、板块 12-1 momentum、GEM dual_momentum、低波动 low_vol、"
              "哨兵动量 canary_mom——已进入推荐模型组合，但尚无样本外记录）。",
    "定投": "定期定额、非择时——单独一类（口径以后可再细化）。",
    "仅观察": "早期测试或已验证无可复制的独立 alpha，仅作观察对照，勿据此实盘。",
}


def render_backtest():
    st.title("回测")
    cat = st.radio("策略分类", list(STRATEGY_CATEGORIES), horizontal=True, key="bt_category")
    st.caption(STRATEGY_CATEGORY_NOTE.get(cat, ""))
    options = [s for s in STRATEGY_CATEGORIES[cat] if s in strategy_params]
    # 兜底：未归类的启用策略并入"研究中"，避免遗漏
    if cat == "研究中":
        categorized = {s for lst in STRATEGY_CATEGORIES.values() for s in lst}
        options += [s for s in strategy_names if s not in categorized]
    if not options:
        st.info("该分类下暂无启用的策略")
        return
    strategy_name = st.selectbox("策略", options)
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
最强者为正 → 持有它；为负 → 切换到 {dm["safe_asset"]}（现金感知：{dm["safe_asset"]} 自己动量也为负则
不硬拿，退到现金等价 **{dm.get("cash_asset", "BIL")}** 吃短债利率）。目标变化才换仓，每月至多一次。

**何时灵**：趋势分明的大级别行情，尤其是漫长熊市（2000、2008 型），避险腿的价值全在这里。

**何时坑**：两点必须知道。一是**震荡年的鞭打**：动量在正负之间反复横跳，来回换仓两头挨耳光；
二是**{dm["safe_asset"]} 的久期风险**：TLT 是 20 年长债，2022 年加息导致股债双杀，
它不但没避险反而放大回撤——本策略已用「现金感知」化解：TLT 动量也为负时切现金等价
{dm.get("cash_asset", "BIL")}（1-3月短债，近零波动、零久期）吃无风险利率，而非硬抱下跌的 TLT
或持 0% 现金。三步演进：无条件切 TLT +367% → 现金感知 +403% → 现金感知+BIL吃利息 **+419%**
（夏普 0.85、2022 仅 -13%）。它的收益大头是票息，回测必须用复权价（本平台已是）。

⚠️ **上面的 +419% 偏乐观（调仓日 timing luck，2026-07-27 检验）**：月首日恰好是四个调仓锚点
（每月第 1/6/11/16 个交易日）里最好的一个，另外三天只有 +319%~+330%，年化跨度 209bp。
去掉这份运气的公平估计 = 4 个错峰 tranche 等权组合 **+347% / 年化 13.9% / 夏普 0.82 / 回撤 -28.7%**。
策略本身站得住（错峰后夏普仍接近最优，回撤明显好于最差单日的 -37.1%），
但绝对收益该按 **+347%** 而非 +419% 来预期。

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
    cam = strategy_params.get("cross_asset_mom", {
        "lookback_days": 252, "skip_days": 21, "top_n": 3, "abs_momentum": True,
    })
    st.markdown(f"""
---

## 9. cross_asset_mom 跨资产动量（稳健档；曾被认为跑赢等权基准，已于 2026-07-27 撤回）

**跨资产动量为什么*可能*有效**：动量因子的信息量取决于宇宙内资产间的**离散度和低相关性**。
板块 ETF 宇宙（11 只 SPDR）彼此相关约 0.7，日涨日跌方向大致相同，横截面排名里
大部分信息是共同的市场 beta，真正的"谁比谁强"信号被噪音淹没——所以板块动量和个股动量
很难稳定跑赢等权基准。跨资产宇宙（股/债/金/商品/国际/REITs）间相关低至 0.0-0.5，
资产类别间的涨跌分化真实且持续，动量排名才包含可操作的信息。

**宇宙**：SPY/QQQ/VEA/VWO/TLT/HYG/GLD/DBC/XLRE（9 类资产）。

**规则**：每月首个交易日——
1. 计算各资产 12-1 动量（近 {cam["lookback_days"]} 日收益，跳过最近 {cam["skip_days"]} 日）；
2. 横截面排名，取动量最高的前 {cam["top_n"]} 名；
3. **绝对动量开关**（abs_momentum={'开' if cam.get('abs_momentum', True) else '关'}）：
   只买入动量为正的 picks——动量为负的位置空着 = 持现金。
   极端时全部资产动量转负 = 全持现金，自动减仓。

**绝对动量开关的作用**：参考 dual_momentum 的思想——过去 12 个月连绝对收益都是负的，
说明该资产处于趋势下行，不值得持有。开关在**熊市**（如 2008/2020 初）能显著降低回撤；
但在 2015-2026 这段美股长牛中，开关反而拖累收益（和 dual_momentum 的 TLT 避险腿同理），
因为偶尔动量短暂转负时被迫出场、再进场买在更高价。它是**熊市 regime 的保险**，
牛市里保费为负。

**诚实结论（2015-2026 实测）**：
- **策略**（top3 + 绝对动量，月首日调仓）：总收益 +234%，夏普 0.80，回撤 -21.5%，Calmar 0.51
- **等权全资产基准**：总收益 +171%，夏普 **0.81**，Calmar 0.39——**按夏普只是打平**
- **SPY 长持**：总收益更高——2015-2026 美股独大，任何分散到非美/债/商品的策略都会
  拖累绝对收益

**⚠️ 2026-07-27 重要撤回：本策略原被称为"首个干净跑赢公平基准的动量策略"，该说法不成立。**

两个原因：

1. **旧口径写串了**。此前文档写"基准 +172%/0.39"并与策略的"夏普 0.80"并列，容易读成
   0.80 vs 0.39 的碾压——但那个 0.39 是基准的 **Calmar**，基准夏普其实是 0.81。
2. **调仓日 timing luck**（Newfound / Hoffstein-Faber-Braun）。把调仓锚点从每月第 1 个
   交易日错开到第 6/11/16 个交易日，本策略年化收益跨度高达 **569bp**（+5.3%~+11.0%），
   全平台最脆弱；而现用的月首日恰好是四个日期里**最好**的那个（第 16 日只剩 +82%、
   夏普 0.41）。去掉这份运气的公平估计 = 4 个错峰 tranche 等权组合：
   **+162% / 夏普 0.66 / Calmar 0.33，三项全输**给等权基准的 +171% / 0.81 / 0.39。

walk-forward 其实早有暗示：2015-2020 段月首日口径就已输给基准（+57%/夏普0.61 vs
+63%/0.76），只有 2021-2026 略胜（+71% vs +67%，夏普仍输 0.71 vs 0.88）。

**所以：本平台至今没有任何一个动量策略被证明"干净跑赢自己的公平基准"。**
板块动量跑不赢板块等权，个股动量的超额全靠 NVDA 一只票，跨资产动量则死在调仓日运气上。
本策略保留为稳健档观察对象（其回撤控制和绝对动量的熊市保险仍有独立价值），
但不要再用"跑赢公平基准"给它背书。

**方法论**：timing luck 与 walk-forward 是**两根正交的稳健性轴**——前者问"换个调仓日
还成立吗"，后者问"换个时间段还成立吗"。本策略是两根都过不了、但只看单一口径完全
看不出来的典型。以后任何月频策略的结论，两根轴都要过。

**何时灵**：资产间趋势分化明显时（如 2022 年商品/黄金上涨而股债双杀），动量能抓住对的资产。

**何时坑**：V 型急反转（动量追在旧趋势末端），以及美股一枝独秀的长牛（分散=拖累）。

**作用范围**：9 类跨资产宇宙（`cross_asset_mom_universe` 组）。
""")
    st.markdown("""
---

## 10. aggressive_mom 进攻档（成长集中动量 + 现金感知避险）

**定位**：本平台的**进攻档**——目标是**够 QQQ 的增长**同时把回撤控住，与稳健档
`cross_asset_mom`（分散、低波动）形成"进攻/稳健"两档。

**为什么这样设计**：想跑赢 QQQ 的收益，分散策略结构上做不到（QQQ 本身就是集中成长）——
只能**比 QQQ 更集中**。所以进攻档在成长/科技 ETF（QQQ/XLK/SMH/IGV/XLY/XBI）里
**集中持有动量最强的 1 只**（top1），用集中度换增长。

**现金感知避险（关键）**：成长动量转负时切入避险（TLT），但——**避险资产自己也在跌就不拿**。
旧版硬切 TLT 时，2022 年股债双杀 TLT 也崩，导致 -45% 回撤；改成"TLT 动量也为负就不拿"后，
2022 躲开、回撤降到 -34%。这条"不持下跌的东西（包括避险）"是把回撤压下来的关键。
连避险也负时的退路不再是 0% 现金，而是**现金等价 BIL**（1-3月短债，近零波动、零久期）
吃无风险利率——top1 全进全出，该防御的整段就是 100% BIL 吃利息。

**诚实结论（2015-2026 实测 + walk-forward）**：
- **进攻档**（top1 + 现金感知避险 + BIL 吃利息）：总收益 **+737%**，年化 20.2%，回撤 **-34.3%**，夏普 0.81，Calmar **0.59**
- **调仓日 timing luck 检验最稳的策略**：4-tranche 错峰口径 **+717% / 年化 20.0% / 回撤 -33.1% /
  夏普 0.81 / Calmar 0.60**，与月首日口径几乎一致 → **优势不靠调仓日运气**
  （对比 `cross_asset_mom` 跨度 569bp、结论被推翻）。以下比较一律用错峰口径。

**⚠️ 2026-07-27 补测：与公平基准（池子等权）的对比**

按平台标准（判断"排名有没有加信息"的可信对比是**与策略共享候选名单的等权持有**），
**QQQ 不是公平基准**——它是这十年的事后赢家。成长池等权 = 6 只进攻腿 ETF 每日等权。

| 全历史 2015-2026 | 总收益 | 年化 | 回撤 | 夏普 | Calmar |
|---|---|---|---|---|---|
| **策略（错峰口径）** | **+717%** | +20.0% | -33.1% | 0.81 | **0.60** |
| ★ 成长池等权（公平基准） | +606% | +18.5% | -37.4% | 0.85 | 0.49 |
| 全宇宙等权（含 TLT） | +374% | +14.4% | -32.1% | 0.87 | 0.45 |
| QQQ 长持 | +621% | +18.7% | -35.1% | **0.89** | 0.53 |

全历史赢池子等权的收益/回撤/Calmar，**但夏普输**（0.81 vs 0.85）；而且**四个口径里策略
夏普最低**（0.81 < 0.85 < 0.87 < 0.89）。这是**集中押注的标准签名**——买到的是更高的绝对
回报和更好的回撤形状，不是更高的风险调整效率。

**walk-forward 分裂：超额只在后半段**

| 2015-2020 | 总收益 | 夏普 | Calmar | 回撤 |
|---|---|---|---|---|
| 策略（错峰） | +231% | 0.96 | 0.67 | -33.1% |
| ★ 成长池等权 | **+232%** | **1.01** | **0.71** | **-31.1%** |

| 2021-2026 | 总收益 | 夏普 | Calmar | 回撤 |
|---|---|---|---|---|
| 策略（错峰） | **+138%** | 0.68 | **0.52** | **-32.8%** |
| ★ 成长池等权 | +114% | 0.69 | 0.39 | -37.4% |

前半段六年：收益打平、夏普与 Calmar 都输、回撤还更差——**排名一点信息都没加**。
后半段才赢，回撤优势也只在后半段。与下面 4 段拆解的结论互相印证。

**所以口径必须这么说**：
- ✅ 跑赢 QQQ 收益、Calmar 更高 —— **真的，但 QQQ 不是公平基准**
- ⚠️ 跑赢自己池子的等权 —— **只在 2021-2026 成立，2015-2020 平局偏输**
- ❌ 夏普跑赢任一基准 —— **从未**，全历史是四个口径里最低的

→ **"12-1 排名加了信息"在本策略上未被证明。** 它站得住的是【集中成长押注 + 有效的
避险腿 + 不吃调仓日运气】，不是选股能力。

**诚实边界**：本质是**集中的成长/科技押注**（吃了这段科技牛市）——成长/科技失宠的 regime
会明显落后 QQQ；现金感知帮大忙主要因 2022 是**罕见的股债双杀**（多数熊市债券会涨，届时
切 TLT 也够）；单标的集中 = 波动和回撤都比稳健档 `cross_asset_mom` 大。**进攻档就是睁着眼
承担更多风险去够 QQQ，不是免费午餐。**

**⚠️ 超额是 regime 依赖的（4 段拆分实证）**：把 2015-2026 切成 4 段（每段约 3 年）后，
进攻档"跑赢 QQQ"**绝大部分来自 2024-2026 那一段**（AI/半导体集中爆发，SMH 封神：进攻档
+136% vs QQQ +75%）。而在 2015-2017、2018-2020 两段基本**与 QQQ 打平**，在带 2022 崩盘的
2021-2023 段**反而落后 QQQ**（+18% vs +35%）。所以它不是"稳定跑赢 QQQ"，而是**在对的
regime 里跑赢、错的 regime 里落后**——这才是集中进攻档的真实画像。

**结构性局限**：ETF 池是固定的（成长/科技那几只）。若下一个热门主题**不在池子里**（如某个
新兴板块的 ETF 未纳入），进攻档就轮不进去、只能持池内"最不差"的，从而跑输 QQQ——因为
QQQ 作为宽基会自动纳入新赢家。这是任何**固定小宇宙**动量策略对宽指的天生劣势。

**作用范围**：成长 ETF + TLT 避险 + BIL 现金等价（`aggressive_growth` + `cash` 组）。
""")
    st.markdown("""
---

## 11. canary_mom 哨兵动量（把「何时防守」与「持什么」拆开）

**来源与先验**：Keller & Keuning 的 VAA(2017) → DAA(2018) → BAA(2022) → HAA(2023) 系列，
SSRN 公开论文 + Allocate Smartly 第三方复现。**这一点很重要**——它是外部验证过的架构，
不是我们自己在数据里翻出来的，所以 walk-forward 的结果该配一个强先验来读。

**它想修的毛病**：现有 `dual_momentum` / `cross_asset_mom` / `aggressive_mom` 的防守都是
**自指的**——每个资产用它自己的绝对动量决定要不要持有。问题反复出现：要等我持有的资产
自己跌到动量转负才减仓，往往已经跌了一段；虚晃一枪后又得在更高价买回来。
哨兵架构把决策拆成两层：一个**与持仓无关**的小资产集合当预警器，只管「现在该不该冒险」。

**两层用不同速度的动量，这是刻意的**：
- **选择层（慢）**：12-1 月度动量，与平台其它策略一致（我们测过短窗口选择在前半段崩掉）。
- **触发层（快）**：13612W —— 1/3/6/12 个月收益各自年化后取平均。年化系数让最近一个月
  主导得分，反应远快于 12-1。**预警器要不要快、和选择器要不要快，是两个独立问题**；
  我们过去一直用同一把尺子量这两件事。

**规则**（每月一次）：哨兵 TIP 的 13612W 转负 → 风险关闭，持防守腿（IEF/BIL 里动量高的那只）；
哨兵为正 → 按 12-1 动量持进攻池前 3（宇宙与 `cross_asset_mom` 完全相同，便于 A/B）。

**实测（2026-07-27，全部走稳健性三根轴，策略数字均为去 timing luck 的错峰口径）**：

| 全历史 | 总收益 | 年化 | 回撤 | 波动 | 夏普 | Calmar |
|---|---|---|---|---|---|---|
| **策略（错峰）** | +208% | 10.2% | **-19.8%** | 10.7% | **0.97** | 0.52 |
| ★ 进攻池等权（公平基准） | +171% | 9.0% | -23.0% | 11.4% | 0.81 | 0.39 |

- **walk-forward 两段都赢三项**：2015-2020 超额 +1.1% / 夏普 +0.21 / 回撤好 3.3pt；
  2020-2026 超额 +0.7% / 夏普 +0.07 / 回撤好 7.5pt。
- **这是本平台第一个在 walk-forward 两段都跑赢自己公平基准的策略。**（板块动量输板块等权、
  个股动量超额全在 NVDA、`cross_asset_mom` 两段超额均为负、`aggressive_mom` 只赢后半段。）
- 夏普 0.97 也高于 SPY 长持(0.81) 与 QQQ 长持(0.89)。timing luck 跨度 194bp（中等）。

**A/B 支持了「哨兵能替代自指开关」这个假设**：

| 防守方式 | 全历史超额年化 | 夏普差 | 两段是否都为正 |
|---|---|---|---|
| **哨兵单保险（本策略）** | **+1.2%** | **+0.15** | ✅ 都正 |
| 哨兵 + 自指（HAA 原版双保险） | -0.8% | -0.08 | ❌ |
| 仅自指（现有 cross_asset_mom） | +0.4% | -0.07 | ❌ 两段均为负 |
| 完全不防守 | +0.5% | -0.07 | ❌ 两段均为负 |

**⚠️ 诚实边界（必须和上面的数字一起讲）**：
- **机制故事没有论文那么漂亮。** 实际是 21 段风险关闭、其中 20 段只持续 1 个月——更像
  「频繁的短暂减仓」而不是「预判大崩盘」。而且它**没躲开 2020 年 3 月的新冠崩盘**
  （2020-04 才转防守，已在底部之后）。别把它讲成危机预警器。
- **关掉自指过滤是我们在自己数据上选的**，偏离了 HAA 的公开规格（原版是双保险）。
  「哨兵架构」有强先验，「去掉自指过滤」没有——这一半是 in-sample 选择。
- 我们只有 2015 年起的数据，**没有 2008 级别的熊市**；哨兵架构的卖点恰恰是大级别下跌的
  保护，样本内几乎没机会体现。
- 超额主要来自**风险端**（波动、回撤），收益端优势只有 +1.2%/年。
- 绝对收益仍**大幅跑输 SPY 长持**（+208% vs +335%）——分散到非美/债/商品在这段美股独大的
  历史里必然拖累绝对收益。

→ **2026-07-28 起 `notify: true`**：它已进入 config 的推荐模型组合（换掉了 `cross_asset_mom`——
两者相关 0.76 属替代关系，换后组合夏普 0.96→1.01、回撤 -20.6%→-18.0%，且两个半段的夏普
都是全场第一）。但本平台只发信号不下单，而它**至今没有任何样本外记录**——收到信号 ≠ 该照做。

**作用范围**：跨资产进攻池 + 哨兵 TIP + 防守腿 IEF/BIL
（`cross_asset_mom_universe` + `canary` + `defense` + `cash` 组）。
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


SECTOR_NAMES = ETF_NAMES  # 中文名映射统一在 analysis/market.py（面板与邮件共用）


def _stock_sector_map() -> dict[str, str]:
    """从 universe_sp500.yaml 建 个股代码 -> GICS 行业 映射（供强弱榜展示）。"""
    try:
        with open(ROOT / "universe_sp500.yaml", encoding="utf-8") as f:
            grouped = yaml.safe_load(f)
    except (OSError, yaml.YAMLError):
        return {}
    return {sym: sector for sector, syms in grouped.items() for sym in syms}


def _render_strength_table(df: pd.DataFrame, label_map: dict | None = None):
    """渲染强弱表：12-1动量/52周位置/距200MA/(P/E)/行业内价值分位/综合分，前景色随正负切换。
    含 pe 列（个股）时展示 P/E；含 value_score 时展示行业内价值分位（板块无基本面则不显示这两列）。"""
    d = df.copy()
    d.insert(0, "标的", [
        f"{s}｜{label_map[s]}" if label_map and s in label_map else s for s in d.index
    ])
    cols = ["标的"] + (["行业"] if "行业" in d.columns else []) + ["mom", "pos_52w", "dist_ma"]
    for c in ("pe", "ev_ebitda"):
        if c in d.columns:
            cols.append(c)
    if "value_score" in d.columns:
        cols.append("value_score")
    cols.append("composite")
    show = d[cols].rename(columns={
        "mom": "12-1动量", "pos_52w": "52周位置", "dist_ma": "距200MA",
        "pe": "远期P/E", "ev_ebitda": "EV/EBITDA",
        "value_score": "行业内价值分位", "composite": "综合分",
    })
    fmt = {"12-1动量": "{:+.1%}", "52周位置": "{:.0%}", "距200MA": "{:+.1%}", "综合分": "{:.0%}"}
    if "行业内价值分位" in show.columns:
        fmt["行业内价值分位"] = "{:.0%}"
    if "远期P/E" in show.columns:
        fmt["远期P/E"] = "{:.1f}"
    if "EV/EBITDA" in show.columns:
        fmt["EV/EBITDA"] = "{:.1f}"
    styler = (show.style
              .map(signed_color, subset=[c for c in ["12-1动量", "距200MA"] if c in show.columns])
              .format(fmt, na_rep="—"))
    st.dataframe(styler, width="stretch", hide_index=True)


def _sort_strength(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """按 col 降序排；缺失该列（如板块表无 earn_yield）时回落到综合分。"""
    c = col if col in df.columns else "composite"
    return df.dropna(subset=[c]).sort_values(c, ascending=False)


def render_market_screen():
    """市场筛选：大盘趋势灯 + 板块强弱 + 个股强弱榜 + 动量+价值多因子综合分。"""
    st.title("市场筛选")
    st.caption("当前强弱【快照】（非回测）：**综合分 = 动量半 + 价值半**——"
               "动量分 = 12-1动量 / 52周位置 / 距200日均线 三维横截面百分位均值；"
               "价值分 = forward盈利收益率 + EV/EBITDA收益率 双口径的【行业内】百分位均值"
               "（仅个股）——forward 与 EV/EBITDA 都剔除一次性投资收益（如 Alphabet $99B 股权"
               "收益灌高 GAAP EPS、压低 trailing PE 的假便宜）；行业内中性化消除科技高PE/"
               "银行低PE的结构性偏差，否则'价值'沦为行业押注。"
               "动量买贵的赢家、价值买便宜的，50/50 融合是刻意折中；不做权重优化（避免过拟合）。"
               "REIT（房地产）折旧压低 GAAP 利润让 PE 结构性失真，PE 腿整段剔除、"
               "只用 EV/EBITDA（定义上排除折旧摊销）定价值。"
               "价值用当前基本面快照（非 point-in-time 历史）；个股宇宙含幸存者偏差；"
               "12-1 动量有短期反转/买在山顶风险。仅作强弱参考，不构成交易建议。")

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
    SORT_OPTIONS = {"综合分（动量+价值）": "composite", "12-1动量": "mom",
                    "52周位置": "pos_52w", "距200日均线": "dist_ma",
                    "价值分（行业内·仅个股）": "value_score"}
    sort_label = st.selectbox(
        "排序依据", list(SORT_OPTIONS), index=0, key="screen_sort",
        help="综合分=动量半+价值半；选单一维度可看纯榜单（如 12-1动量 看动量冠军哪怕已回落，"
             "价值分 看行业内最便宜的）。价值分=forward盈利收益率+EV/EBITDA收益率的行业内百分位。"
             "板块无基本面，选价值维度时板块表回落到综合分。",
    )
    sort_col = SORT_OPTIONS[sort_label]

    # ── 板块强弱 ──
    st.subheader("板块强弱")
    sect_str = compute_strength(sect_prices)  # 板块无个股基本面，综合分为纯动量分
    if sect_str.empty:
        st.warning("板块行情不足，先运行 python run_daily.py 拉取数据")
    else:
        _render_strength_table(_sort_strength(sect_str, sort_col), label_map=SECTOR_NAMES)

    # ── 个股强弱榜 ──
    st.subheader("个股强弱榜（S&P500 候选池）")
    sec_map = _stock_sector_map()
    with st.spinner("加载个股行情与基本面并计算强弱…"):
        stock_syms = cfg.universe_symbols("universe_sp500.yaml")
        stock_prices = {s: store.load_prices(conn, s) for s in stock_syms}
        stock_prices = {s: df for s, df in stock_prices.items() if not df.empty}
        fdf = store.load_fundamentals(conn)
        latest_fund = (fdf.sort_values("date").groupby("symbol").last()
                       if not fdf.empty else None)
        # 价值分行业内中性化：传入行业映射，消除科技高PE/银行低PE的结构性偏差
        stock_str = compute_strength(stock_prices, fundamentals=latest_fund, sectors=sec_map)
    if stock_str.empty:
        st.warning("个股行情不足（需要至少约 1 年数据）")
        return
    stock_str["行业"] = [sec_map.get(s, "") for s in stock_str.index]
    has_val = "value_score" in stock_str.columns
    n_pe = int(stock_str["pe"].notna().sum()) if "pe" in stock_str.columns else 0
    fund_date = str(latest_fund["date"].iloc[0]) if latest_fund is not None and not latest_fund.empty else "无"
    st.caption(f"共 {len(stock_str)} 只个股参与排名，截至各自最新交易日。当前按【{sort_label}】排序。"
               + (f"综合分=动量半+价值半；价值用 {fund_date} 快照的 forward盈利收益率+EV/EBITDA 双口径"
                  f"（{n_pe} 只有有效远期PE；两口径全缺者价值分空缺、综合分退回只用动量）。价值分已按"
                  f"行业内百分位中性化——「行业内价值分位」列即所在行业内的便宜程度，不是全市场比。" if has_val
                  else "（基本面表暂无数据，综合分为纯动量；跑 run_daily 记录基本面后生效。）"))

    sector_options = ["全部板块"] + sorted({s for s in stock_str["行业"] if s})
    sector_filter = st.selectbox("按板块筛选", sector_options, index=0, key="screen_sector")
    filtered = (stock_str if sector_filter == "全部板块"
                else stock_str[stock_str["行业"] == sector_filter])
    if filtered.empty:
        st.warning(f"「{sector_filter}」板块无符合条件的个股")
        return

    ranked = _sort_strength(filtered, sort_col)
    n = st.slider("每侧显示数量", 5, 30, 15, key="screen_n")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"**🟢 最强（{sort_label}居前）**")
        _render_strength_table(ranked.head(n))
    with col_b:
        st.markdown(f"**🔴 最弱（{sort_label}垫底）**")
        _render_strength_table(ranked.tail(n).iloc[::-1])


def render_drawdown_playbook():
    from collections import Counter

    from quant.analysis.drawdowns import (
        HEDGE_CANDIDATES,
        classify_episode,
        episode_returns,
        find_drawdown_episodes,
        severity,
    )

    st.subheader("🛡️ 崩盘类型 → 最佳避险")
    st.caption("以基准总回报识别历史下跌段，测每段【下跌期间】各避险资产的总回报——看清"
               "没有万能避险：通缩型崩盘靠长债 TLT，通胀型崩盘 TLT 反崩要靠商品/黄金，闪崩靠现金。"
               "口径：adj_close 总回报，峰→谷（进行中的段用峰→最新）；BIL 短债≈持现金基线。")

    c1, c2 = st.columns([1, 2])
    with c1:
        bench_sym = st.selectbox("基准", ["SPY", "QQQ"], index=0)
    with c2:
        thr = st.slider("下跌段阈值（SPY 最大回撤 ≥）", 5, 25, 8, step=1,
                        format="%d%%", key="dd_thr") / 100

    syms = [bench_sym] + [s for s in HEDGE_CANDIDATES if s != bench_sym]
    frames = {}
    for s in syms:
        df = store.load_prices(conn, s)
        if not df.empty:
            frames[s] = price_series(df)
    px = pd.DataFrame(frames).sort_index()
    if bench_sym not in px.columns:
        st.warning(f"{bench_sym} 行情缺失，无法分析")
        return
    bench = px[bench_sym].dropna()
    cands = [s for s in HEDGE_CANDIDATES if s in px.columns]
    episodes = find_drawdown_episodes(bench, threshold=thr)
    st.caption(f"数据区间 {bench.index[0]:%Y-%m-%d} ~ {bench.index[-1]:%Y-%m-%d}")

    # ── 当前进行中的回撤（对号入座）──
    ongoing = [e for e in episodes if e["ongoing"]]
    if ongoing:
        e = ongoing[0]
        rets = episode_returns(px, e["peak_date"], e["end_date"], cands)
        typ, note = classify_episode(e, rets.get("TLT"))
        days = (e["end_date"] - e["peak_date"]).days
        st.error(f"🔴 **当前进行中的回撤** ｜ 从 {e['peak_date']:%Y-%m-%d} 高点至今 {days} 天，"
                 f"{bench_sym} {e['maxdd']:+.1%}（{severity(e['maxdd'])}）")
        if rets:
            best = max(rets, key=rets.get)
            ranked = sorted(rets.items(), key=lambda x: -x[1])[:4]
            for col, (sym, r) in zip(st.columns(len(ranked)), ranked):
                col.metric(etf_label(sym), f"{r:+.1%}", "最抗跌" if sym == best else None)
            st.info(f"**自动判定：{typ}** —— {note}。当前最抗跌：**{etf_label(best)} {rets[best]:+.1%}**。")
    else:
        st.success(f"✅ {bench_sym} 目前在历史新高附近，无进行中回撤。")

    # ── 历史下跌段 × 避险资产矩阵 ──
    closed = [e for e in episodes if not e["ongoing"]]
    if not closed:
        st.info("当前阈值下没有已结束的下跌段，调低阈值试试。")
        return
    st.markdown(f"**历史下跌段 × 避险资产**（下跌期间总回报，共 {len(closed)} 段 · 绿底=当次最佳）")

    wins: Counter = Counter()
    rows = []
    for e in closed:
        rets = episode_returns(px, e["peak_date"], e["end_date"], cands)
        typ, _ = classify_episode(e, rets.get("TLT"))
        best = max(rets, key=rets.get) if rets else None
        if best:
            wins[best] += 1
        row = {
            "时段": f"{e['peak_date']:%Y-%m}→{e['trough_date']:%m-%d}",
            "天数": (e["trough_date"] - e["peak_date"]).days,
            "严重度": severity(e["maxdd"]),
            f"{bench_sym}回撤": e["maxdd"],
            "类型": typ,
        }
        for s in cands:
            row[etf_label(s)] = rets.get(s)
        row["最佳避险"] = f"{etf_label(best)} {rets[best]:+.1%}" if best else "—"
        rows.append(row)
    table = pd.DataFrame(rows)
    hedge_cols = [etf_label(s) for s in cands]
    pct_cols = [f"{bench_sym}回撤"] + hedge_cols

    def hi_best(r):
        styles = [""] * len(r)
        vals = {c: r[c] for c in hedge_cols if pd.notna(r[c])}
        if vals:
            bestc = max(vals, key=vals.get)
            styles[list(r.index).index(bestc)] = "font-weight:700; background-color: rgba(46,125,50,0.22)"
        return styles

    styler = (table.style
              .apply(hi_best, axis=1)
              .map(signed_color, subset=hedge_cols)
              .format({c: "{:+.1%}" for c in pct_cols}, na_rep="—"))
    st.dataframe(styler, width="stretch", hide_index=True)

    win_str = "、".join(f"{etf_label(s)} {c}次" for s, c in wins.most_common())
    st.markdown(f"**各资产夺冠次数**：{win_str}")
    st.markdown("""
**对号入座定律（没有万能避险）**
- 🟦 **通缩/避险型**（2015、2020 COVID 型）→ **TLT 长债封神**（+14% 级）：利率下行、资金抢国债。
- 🟥 **通胀/加息型**（2022 型）→ **TLT 是灾难**（2022 −29%，比股市还惨）→ 换 **商品 DBC / 黄金 GLD**。
- ⚡ **闪崩**（10–20 天）→ 谁都来不及走趋势，**现金/短债 BIL 靠不跌取胜**。
- 🟡 **黄金 GLD** 是全天候：赢面最广、从不致命；**TLT** 是双峰高 β 对冲（通缩最猛、通胀最毒）。

这正是防御策略「现金感知」的实证依据——不押单一避险，而是让 dual_momentum / aggressive_mom
按崩盘类型自动躲开错误的避险腿（TLT 动量转负就退到 BIL 吃短债利率）。
""")

    _render_strategy_vs_drawdowns(px, bench, episodes, bench_sym, thr)


def _render_strategy_vs_drawdowns(px: pd.DataFrame, bench: pd.Series, episodes: list[dict],
                                  bench_sym: str, thr: float) -> None:
    """策略实测：把哨兵/避险类策略的真实信号跟上面识别出的大回撤段交叉对比——
    哪些回撤防住了、减少多少；哪些没防住；策略触发避险的区段里哪些没对应到真实回撤（假信号），
    错过了多少涨幅。回答用户的具体问题：不是靠机制故事，是把信号和实测回撤直接对表。
    """
    from quant.analysis.drawdowns import defense_spans, spans_overlap

    # 只列有「明确避险腿/现金等价」概念的策略——其它策略（如 cross_asset_mom）
    # 防守是"空槽持现金"而非切到某个具体标的，跟这里"held ⊆ 避险集合"的判定口径对不上
    role_map = {
        "canary_mom": ("哨兵动量", lambda p: set(p.get("defense_assets", []))
                      | ({p["cash_asset"]} if p.get("cash_asset") else set())),
        "dual_momentum": ("双动量", lambda p: ({p["safe_asset"]} if p.get("safe_asset") else set())
                          | ({p["cash_asset"]} if p.get("cash_asset") else set())),
        "aggressive_mom": ("进攻档", lambda p: set(p.get("safe_assets", []))
                           | ({p["cash_asset"]} if p.get("cash_asset") else set())),
    }
    available = [k for k in role_map if k in strategy_params]
    if not available:
        return

    st.markdown("---")
    st.subheader("🐤 策略实测：防住了吗？")
    st.caption(
        "把策略的真实信号（月首日调仓口径，与推送一致）跟上面识别出的大回撤段直接对表：这次回撤"
        "策略有没有在避险、实际少亏/多赚多少；以及策略触发避险的区段里，有多少次根本没对应到"
        "真实回撤（假信号），白白错过了多少涨幅。「避险」定义为该区段持仓完全落在避险腿/现金等价内"
        "（不含哨兵资产本身——哨兵只判断开关，从不被持有）。"
    )
    strat_name = st.selectbox("策略", available,
                              format_func=lambda k: f"{k}（{role_map[k][0]}）", key="dd_strat")
    params = strategy_params[strat_name]
    defense_syms = role_map[strat_name][1](params)
    if not defense_syms:
        st.info("该策略没有配置明确的避险腿/现金等价，无法交叉对比")
        return

    universe = cfg.symbols_for(params.get("groups", []))
    prices_s = {s: store.load_prices(conn, s) for s in universe}
    prices_s = {s: d for s, d in prices_s.items() if not d.empty}
    if not prices_s:
        st.warning("库内没有该策略所需行情")
        return
    sigs = strategies.build(strat_name, params).generate(prices_s)
    result = run_portfolio_backtest(prices_s, sigs, strat_name, INITIAL_CASH, cfg.cost_bps)
    eq = result.equity
    bench_aligned = bench.reindex(eq.index).ffill()
    spans = defense_spans(sigs, eq.index, defense_syms)

    # ── 表1：大回撤段 × 策略实际表现 ──────────────────────
    rows1 = []
    for e in episodes:
        seg = eq.loc[e["peak_date"]:e["end_date"]]
        if len(seg) < 2:
            continue
        strat_ret = float(seg.iloc[-1] / seg.iloc[0] - 1)
        overlapped = any(spans_overlap(e["peak_date"], e["end_date"], s0, s1) for s0, s1 in spans)
        if overlapped and strat_ret > e["maxdd"]:
            verdict = "✅ 避险机制生效"
        elif strat_ret > e["maxdd"]:
            verdict = "🟡 分散躲过（非避险机制，选中了别的抗跌标的）"
        else:
            verdict = "❌ 没防住"
        rows1.append({
            "回撤段": f"{e['peak_date']:%Y-%m}→{e['trough_date']:%m-%d}",
            f"{bench_sym}回撤": e["maxdd"],
            "策略同期收益": strat_ret,
            "减少的回撤": strat_ret - e["maxdd"],
            "避险区间覆盖": "是" if overlapped else "否",
            "结论": verdict,
        })
    if rows1:
        t1 = pd.DataFrame(rows1)
        st.markdown(f"**大回撤段 × {strat_name} 实际表现**")
        st.dataframe(
            t1.style.map(signed_color, subset=["减少的回撤"])
                    .format({f"{bench_sym}回撤": "{:+.1%}", "策略同期收益": "{:+.1%}",
                             "减少的回撤": "{:+.1%}"}),
            width="stretch", hide_index=True)

    # ── 表2：避险区段 × 是否对应真实回撤（假信号检测） ──────
    rows2 = []
    for s0, s1 in spans:
        seg = eq.loc[s0:s1]
        bseg = bench_aligned.loc[s0:s1]
        if len(seg) < 2 or len(bseg) < 2:
            continue
        strat_ret = float(seg.iloc[-1] / seg.iloc[0] - 1)
        bench_ret = float(bseg.iloc[-1] / bseg.iloc[0] - 1)
        overlapped = any(spans_overlap(s0, s1, e["peak_date"], e["end_date"]) for e in episodes)
        rows2.append({
            "避险区段": f"{s0:%Y-%m-%d}~{s1:%Y-%m-%d}",
            "天数": (s1 - s0).days,
            "策略同期收益": strat_ret,
            f"{bench_sym}同期涨跌": bench_ret,
            "错过涨幅": bench_ret - strat_ret,
            "对应真实回撤": "是" if overlapped else "否",
        })
    if rows2:
        t2 = pd.DataFrame(rows2)
        false_mask = t2["对应真实回撤"] == "否"
        n_false = int(false_mask.sum())
        costly = t2.loc[false_mask, "错过涨幅"].clip(lower=0)
        st.markdown(f"**避险区段 × 是否对应真实回撤**（共 {len(rows2)} 段避险，**{n_false} 段**没对应到"
                    f"本页设定的 {thr:.0%} 回撤阈值，其中 **{int((costly > 0).sum())} 段**确实错过了涨幅，"
                    f"累计错过约 **{costly.sum():+.1%}**）")
        st.dataframe(
            t2.style.map(signed_color, subset=["错过涨幅"])
                    .format({"策略同期收益": "{:+.1%}", f"{bench_sym}同期涨跌": "{:+.1%}",
                             "错过涨幅": "{:+.1%}"}),
            width="stretch", hide_index=True)
        st.caption("「对应真实回撤=否」≠ 一定亏——如果那段时间大盘本来就没怎么涨（横盘/微跌），"
                  "「错过涨幅」会是负数或接近 0，代表避险没花什么代价，只是分类上没匹配到本页设定的"
                  f"回撤阈值（{thr:.0%}）而已，调低上面的滑杆阈值能看到更多被计入「真实回撤」的段。")


# AI 基建赛道概览里的百分比列（显示时 ×100，配 printf 格式；见 render_ai_infra 内注释）
_AI_PCT_COLS = ["近1年涨幅(市值加权)", "近3年涨幅(市值加权)", "近5年涨幅(市值加权)",
                "12-1动量(市值加权)", "营收增长中位数"]


def render_ai_infra():
    """🤖 AI 基建：按赛道细分的 AI 基建个股观察页面。

    这是一个观察/研究页面，不产生任何交易信号、不接入任何策略、不进模型组合。
    """
    from quant.analysis.ai_infra import (
        compute_growth_metrics,
        compute_lane_market_share,
        compute_lane_summary,
        display_name,
        get_currency_for_symbol,
        to_usd_market_cap,
    )
    from quant.strategies.selectors import momentum_return

    st.title("🤖 AI 基建")

    # ── 顶部说明 ──
    st.caption("这是一个**观察/研究页面**，不产生任何交易信号、不接入任何策略、不进模型组合。")
    st.info(
        "⚠️ **市值份额 ≠ AI 业务份额**：下方「统治力」列是**公司整体市值**在赛道内的占比，"
        "不是 AI 业务占营收的比例。MSFT 在「云与超大规模」赛道市值占比很高，但它的 AI 基建"
        "业务占自身营收的比例远低于 NVDA。yfinance 拿不到业务分部数据，我们**无法量化"
        "「AI 纯度」**。这个页面回答的是「这条赛道里谁体量大」，不是「谁最纯粹受益于 AI」。\n\n"
        "**增长指标有两种口径，别混用**：「营收同比(季)」是 Yahoo `revenueGrowth` = **最新季度**"
        "同比，最新但单季波动大、易被低基数放大（如 MU 存储周期反转后 +346%）；"
        "「营收CAGR」「毛利率」「净利率」来自**年度利润表**，更平滑但**滞后 6~12 个月**"
        "（最新财年早已结束，MU 的年报同比只有 +49%）。赛道概览的「营收增长中位数」用季度口径，"
        "与股价涨幅的时效对齐。\n\n"
        "🌏 **含非美龙头**（三星/SK海力士/中际旭创/东京电子/爱德万/鸿海等）："
        "它们的**市值已按最新汇率换算成美元**才参与统治力与合计市值计算（否则把万亿韩元与"
        "十亿美元相加是垃圾数字）；但**涨幅与 12-1 动量是本币计价收益**，与美元收益差一个"
        "汇率变动——「币种」列标出了各标的的计价货币。比率类指标（PE/毛利率/净利率/营收同比）"
        "无量纲，不涉及换算。"
        "赛道近 1 年涨幅用**市值加权**（不是等权——等权会让一只小盘股的暴涨绑架整条赛道的读数）；"
        "赛道营收增长用**中位数**（不用均值，避免 NVDA +100% 这种极值绑架）。",
        icon="ℹ️",
    )

    # ── 加载赛道定义 ──
    try:
        with open(ROOT / "universe_ai_infra.yaml", encoding="utf-8") as f:
            lanes: dict[str, list[str]] = yaml.safe_load(f)
    except (OSError, yaml.YAMLError) as e:
        st.error(f"无法加载 universe_ai_infra.yaml: {e}")
        return

    # 去重获取全部标的
    all_syms_set: dict[str, None] = {}
    for syms in lanes.values():
        for s in syms:
            all_syms_set.setdefault(s)
    all_syms = list(all_syms_set)

    # ── 加载数据 ──
    with st.spinner("加载行情、基本面和财报数据…"):
        # 行情
        all_prices = {s: store.load_prices(conn, s) for s in all_syms}
        all_prices = {s: df for s, df in all_prices.items() if not df.empty}

        if not all_prices:
            st.warning("库内没有 AI 基建标的行情数据，先运行 `python run_daily.py` 拉取数据。")
            return

        # 基本面（市值用）
        fdf = store.load_fundamentals(conn)
        latest_fund = (fdf.sort_values("date").groupby("symbol").last()
                       if not fdf.empty else pd.DataFrame())

        # 市值字典（**本币**，下面统一换算成美元）
        market_caps_local: dict[str, float | None] = {}
        for s in all_syms:
            if not latest_fund.empty and s in latest_fund.index:
                mc = latest_fund.at[s, "market_cap"] if "market_cap" in latest_fund.columns else None
                market_caps_local[s] = float(mc) if mc is not None and pd.notna(mc) else None
            else:
                market_caps_local[s] = None

        # 币种与显示名：都来自 fundamentals.raw_json，只解析一次
        currencies: dict[str, str] = {}
        names: dict[str, str] = {}
        for s in all_syms:
            raw = None
            if not latest_fund.empty and s in latest_fund.index and "raw_json" in latest_fund.columns:
                rj = latest_fund.at[s, "raw_json"]
                if isinstance(rj, str) and rj:
                    try:
                        raw = json.loads(rj)
                    except (ValueError, TypeError):
                        raw = None
            currencies[s] = get_currency_for_symbol(s, raw)
            names[s] = display_name(s, raw)

        # 汇率：prices 表里的 `XXX=X`（1 USD 兑多少本币），取最新收盘
        fx_rates: dict[str, float] = {}
        for ccy in sorted({c for c in currencies.values() if c != "USD"}):
            fx_df = store.load_prices(conn, f"{ccy}=X")
            if not fx_df.empty:
                v = fx_df["close"].dropna()
                if not v.empty and float(v.iloc[-1]) > 0:
                    fx_rates[ccy] = float(v.iloc[-1])

        # **统一换算成美元**——不换算就把万亿韩元和十亿美元加在一起求份额，那是垃圾数字。
        # 缺汇率的标的市值返回 None（宁可显示"—"，也不能把本币当美元混进分母）。
        market_caps = to_usd_market_cap(market_caps_local, currencies, fx_rates)

        # 财报
        fin_df = store.load_financials(conn)

        # 增长指标
        growth_metrics = {}
        for s in all_syms:
            if not fin_df.empty:
                sym_fin = fin_df[fin_df["symbol"] == s]
            else:
                sym_fin = pd.DataFrame()
            growth_metrics[s] = compute_growth_metrics(sym_fin, s)

        # 近 1/3/5 年涨幅（adj_close 总回报口径）。历史不足一律 None——**不退化成
        # "上市以来涨幅"**：次新股（SNDK 仅 371 天）的 1.5 年 +4000% 若冒充"5年涨幅"，
        # 会把整条赛道的加权读数彻底污染。缺失项在市值加权时连同其市值一并剔除。
        def _trailing(adj: pd.Series, bars: int) -> float | None:
            a = adj.dropna()
            if len(a) < bars:
                return None
            return float(a.iloc[-1] / a.iloc[-bars] - 1)

        returns_1y: dict[str, float | None] = {}
        returns_3y: dict[str, float | None] = {}
        returns_5y: dict[str, float | None] = {}
        for s in all_syms:
            adj = price_series(all_prices[s]).dropna() if s in all_prices else pd.Series(dtype=float)
            returns_1y[s] = _trailing(adj, 252)
            returns_3y[s] = _trailing(adj, 756)
            returns_5y[s] = _trailing(adj, 1260)

        # 营收同比：用 Yahoo 的 revenueGrowth = **最新季度**同比。年报同比滞后 6~12 个月
        # （最新财年早已结束，MU 那种滞后 340 天的会把 +346% 的季度增长显示成 +49%）。
        rev_growth_q: dict[str, float | None] = {}
        for s in all_syms:
            v = (latest_fund.at[s, "revenue_growth"]
                 if s in latest_fund.index and "revenue_growth" in latest_fund.columns
                 else None)
            rev_growth_q[s] = float(v) if v is not None and pd.notna(v) else None

        # 12-1 动量（复用 selectors.momentum_return）
        mom_dict: dict[str, float | None] = {}
        adj_all = pd.DataFrame({s: price_series(df) for s, df in all_prices.items()}).sort_index()
        if not adj_all.empty:
            mom_df = momentum_return(adj_all, lookback=252, skip=21)
            for s in all_syms:
                if s in mom_df.columns:
                    v = mom_df[s].iloc[-1] if not mom_df[s].dropna().empty else None
                    mom_dict[s] = float(v) if v is not None and pd.notna(v) else None
                else:
                    mom_dict[s] = None

        # 价值分位（复用 screening.compute_strength 的行业内中性化逻辑）
        sp500_map = _stock_sector_map()
        sp500_syms_set = set(sp500_map.keys())
        # 只对 S&P500 内的标的算价值分位
        sp500_prices = {s: all_prices[s] for s in all_syms if s in sp500_syms_set and s in all_prices}
        value_pctile: dict[str, float | None] = {}
        if sp500_prices and not latest_fund.empty:
            strength = compute_strength(sp500_prices, fundamentals=latest_fund, sectors=sp500_map)
            for s in all_syms:
                if s in strength.index and "value_score" in strength.columns:
                    v = strength.at[s, "value_score"]
                    value_pctile[s] = float(v) if pd.notna(v) else None
                else:
                    value_pctile[s] = None
        else:
            for s in all_syms:
                value_pctile[s] = None

    # ── 赛道概览表 ──
    st.subheader("赛道概览")
    st.caption("赛道涨幅与 12-1 动量均为市值加权（非等权）；营收增长=成分中位数（非均值）。"
               "「近1/3/5年涨幅」是已经走完的行情，「12-1动量」跳过最近 1 个月、反映当前趋势强弱——"
               "两者背离时（如涨幅高但动量已回落）说明该赛道行情可能正在退潮。\n\n"
               "⚠️ 上市/分拆晚于窗口的成分（ARM、GEV、SNDK、CEG 等）在该期涨幅上无数据，"
               "其市值一并从该列加权分母中剔除（不按 0 计）——所以**同一行的 1/3/5 年涨幅"
               "未必基于同一批成分**，跨列比较时注意这点。")
    lane_summaries = []
    for lane_name, lane_syms in lanes.items():
        summary = compute_lane_summary(lane_name, lane_syms, market_caps,
                                        growth_metrics, returns_1y,
                                        momentum=mom_dict,
                                        returns_3y=returns_3y,
                                        returns_5y=returns_5y,
                                        revenue_growth_q=rev_growth_q)
        lane_summaries.append(summary)

    overview_df = pd.DataFrame(lane_summaries)
    if not overview_df.empty:
        # 保留数值类型交给 column_config 做显示格式化——**不要**先 apply 成字符串，
        # 否则点表头是按字典序排（"+1102.1%" 会排在 "+192.9%" 前面，因为第二位 '1'<'9'；
        # 市值列 "$477B" 与 "$9.10T" 混排更是毫无意义）。
        # 百分比列显示时 ×100 配 "%+.1f%%"：仍是数值（排序正确），且保住正负号与 1 位小数
        # （内建的 format="percent" 会丢掉 "+" 号并强制 2 位小数）。
        display_ov = overview_df.copy()
        for c in _AI_PCT_COLS:
            if c in display_ov.columns:
                display_ov[c] = display_ov[c] * 100
        st.dataframe(
            display_ov, width="stretch", hide_index=True,
            column_config={
                # compact 不支持 $ 前缀，把单位写进列名
                "合计市值": st.column_config.NumberColumn("合计市值($)", format="compact"),
                **{c: st.column_config.NumberColumn(c, format="%+.1f%%")
                   for c in _AI_PCT_COLS if c in display_ov.columns},
            },
        )

    # ── 赛道明细 ──
    st.subheader("赛道明细")
    lane_names = list(lanes.keys())
    selected_lane = st.selectbox("选择赛道", lane_names, key="ai_infra_lane")

    if selected_lane and selected_lane in lanes:
        lane_syms = lanes[selected_lane]
        shares = compute_lane_market_share(lane_syms, market_caps)

        # 构建明细表
        detail_rows = []
        for s in lane_syms:
            gm = growth_metrics.get(s)
            mc = market_caps.get(s)
            in_sp500 = s in sp500_syms_set

            # 数值列一律保留原始数值（缺失=None），格式化交给 column_config，
            # 否则点表头按字典序排。CAGR 的"几年"标注单独拆一列，既保住信息又不毁排序。
            vp = value_pctile.get(s)
            # forward PE：取自基本面最新快照（TTM 快照口径，与本页增长指标的财年口径不同）。
            # 负值（亏损公司的 forward PE）置空——负 PE 在估值上无意义，留着会污染排序。
            fpe = None
            if s in latest_fund.index and "forward_pe" in latest_fund.columns:
                _v = latest_fund.at[s, "forward_pe"]
                if pd.notna(_v) and float(_v) > 0:
                    fpe = float(_v)
            detail_rows.append({
                "代码": s,
                "名称": names.get(s, s),
                "币种": currencies.get(s, "USD"),
                "市值份额": shares.get(s),
                "市值": mc,
                "营收CAGR": gm.revenue_cagr if gm else None,
                "CAGR年数": (f"{gm.cagr_years}年"
                             if gm and gm.cagr_years is not None else "—"),
                # 窗口内有单年营收断崖 → CAGR 起点不可比（业务分拆或周期顶），标出来。
                # 不改 CAGR 数值本身：删了会连 INTC 那种真实下滑一起丢掉
                "CAGR备注": (f"⚠️ 期间{gm.cagr_break * 100:+.0f}%"
                             if gm and gm.cagr_break is not None else ""),
                "营收同比(季)": rev_growth_q.get(s),
                "毛利率": gm.gross_margin if gm else None,
                "净利率": gm.net_margin if gm else None,
                "12-1动量": mom_dict.get(s),
                "forward PE": fpe,
                "价值分位": vp,
                # 价值分位为空时区分"池外无此指标"与"池内但缺数据"——数值列放不下这个
                # 说明，单独一列标注，保证价值分位列仍可按数值排序
                "价值分位备注": ("" if vp is not None
                                 else ("池外" if not in_sp500 else "缺数据")),
                "S&P500": "✅" if in_sp500 else "❌",
            })

        detail_df = pd.DataFrame(detail_rows)

        # 缺失市值的成分数
        n_missing_cap = sum(1 for s in lane_syms if market_caps.get(s) is None)
        if n_missing_cap:
            st.caption(f"⚠️ 本赛道有 {n_missing_cap} 只标的市值缺失，已从统治力分母中剔除。")

        # 同概览表：百分比列 ×100 保数值排序 + 保住正负号；市值单位写进列名
        display_detail = detail_df.copy()
        signed_cols = ["营收CAGR", "营收同比(季)", "12-1动量"]   # 可能为负，带 +/- 号
        plain_cols = ["市值份额", "毛利率", "净利率", "价值分位"]  # 恒非负，不需要 + 号
        for c in signed_cols + plain_cols:
            if c in display_detail.columns:
                display_detail[c] = display_detail[c] * 100
        st.dataframe(
            display_detail, width="stretch", hide_index=True,
            column_config={
                "市值": st.column_config.NumberColumn("市值($)", format="compact"),
                # forward PE 是倍数不是百分比，单独格式化；越低越便宜
                "forward PE": st.column_config.NumberColumn("forward PE", format="%.1f"),
                **{c: st.column_config.NumberColumn(c, format="%+.1f%%")
                   for c in signed_cols if c in display_detail.columns},
                **{c: st.column_config.NumberColumn(c, format="%.1f%%")
                   for c in plain_cols if c in display_detail.columns},
            },
        )

        # CAGR 断崖说明：只有本赛道真有标的被标记时才出现，避免刷屏
        broken = [r["代码"] for r in detail_rows if r["CAGR备注"]]
        if broken:
            st.caption(
                f"⚠️ {', '.join(broken)} 的营收 CAGR 窗口内出现单年断崖（跌幅 >40%），"
                f"**起点不可比、CAGR 失真**：要么业务被分拆剥离（如 WDC 2025 年拆出 SanDisk，"
                f"营收腰斩不是经营萎缩），要么起点撞上周期顶（MU 的 3 年 CAGR 只有 +6.7%，"
                f"而最新季度同比 +345.7%）。看 AI 拉动强度请用「营收同比(季)」列。"
            )

        # 池外标的说明
        pool_external = [s for s in lane_syms if s not in sp500_syms_set]
        if pool_external:
            st.caption(
                f"💡 池外标的（{', '.join(pool_external)}）不在 S&P500 候选池中，"
                f"无行业内价值分位——价值分位为空、备注列标「池外」。这是池外标的的已知代价，"
                f"不影响其他指标的准确性。"
            )


PAGES = {
    "📊 市场概览": render_market_overview,
    "📡 信号历史": render_signal_history,
    "🕯️ K线与信号": render_kline,
    "🏆 动量排名": render_momentum_rank,
    "🔍 市场筛选": render_market_screen,
    "🤖 AI 基建": render_ai_infra,
    "🛡️ 避险手册": render_drawdown_playbook,
    "🎯 策略评分": render_strategy_scoring,
    "🔗 策略相关性": render_correlation,
    "🧪 回测": render_backtest,
    "📖 策略说明": render_strategy_docs,
}

page = st.pills("页面导航", list(PAGES), default=next(iter(PAGES)),
                required=True, label_visibility="collapsed")
st.divider()
PAGES[page]()
