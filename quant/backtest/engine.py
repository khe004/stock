"""回测引擎：单标的多头、组合轮动、智能定投三种模拟。

统一口径：价格用 adj_close（总回报，含分红再投资）；cost_bps 为单边交易成本
（万分之一为 1bp），买入少得份额、卖出少得现金。
"""

import math
from dataclasses import dataclass, field

import pandas as pd

from quant.strategies.base import BUY, SELL, Signal, price_series

TRADING_DAYS = 252


def equity_metrics(equity: pd.Series, initial_value: float | None = None) -> dict:
    """从权益曲线计算风险收益指标，策略与各基准复用同一口径。
    initial_value 为投入本金：首日若有建仓成本，权益首值会低于本金，
    收益必须以本金为分母，否则成本被算没。"""
    initial = float(initial_value) if initial_value else float(equity.iloc[0])
    final = float(equity.iloc[-1])
    total_return = final / initial - 1
    n = len(equity)
    cagr = (final / initial) ** (TRADING_DAYS / n) - 1 if final > 0 and n > 0 else -1.0
    max_drawdown = float((equity / equity.cummax() - 1).min())
    ret = equity.pct_change().dropna()
    if len(ret) > 1 and ret.std() > 0:
        volatility = float(ret.std() * math.sqrt(TRADING_DAYS))
        sharpe = float(ret.mean() / ret.std() * math.sqrt(TRADING_DAYS))
    else:
        volatility, sharpe = 0.0, 0.0
    calmar = cagr / abs(max_drawdown) if max_drawdown < 0 else 0.0
    return {
        "total_return": total_return,
        "cagr": cagr,
        "max_drawdown": max_drawdown,
        "volatility": volatility,
        "sharpe": sharpe,
        "calmar": calmar,
    }


def hold_equity(px: pd.Series, initial_cash: float = 10_000.0, cost_bps: float = 0.0) -> pd.Series:
    """长持基准：首日一次性买入（扣一次买入成本）。"""
    shares = initial_cash * (1 - cost_bps / 1e4) / float(px.iloc[0])
    return pd.Series(shares * px.to_numpy(dtype=float), index=px.index, name="hold")


def dca_equity(px: pd.Series, initial_cash: float = 10_000.0, cost_bps: float = 0.0) -> pd.Series:
    """纯定投基准：同一笔资金按月份等分，每月首个交易日买入一份（每笔扣买入成本）。"""
    month_firsts = set(px.groupby([px.index.year, px.index.month]).head(1).index)
    per = initial_cash / len(month_firsts)
    bp = cost_bps / 1e4
    cash, shares, values = initial_cash, 0.0, []
    for ts, price in px.items():
        if ts in month_firsts:
            shares += per * (1 - bp) / float(price)
            cash -= per
        values.append(cash + shares * float(price))
    return pd.Series(values, index=px.index, name="dca")


def vol_scaled_equity(
    equity: pd.Series,
    target_vol: float = 0.15,
    vol_window: int = 63,
    cap: float = 1.0,
    initial_cash: float = 10_000.0,
) -> tuple[pd.Series, pd.Series]:
    """波动率缩放权益曲线：按近期已实现波动率的倒数调整仓位，高波降仓、低波满仓。

    用途定位（实验验证）：
    - 它是【回撤/尾部缩减器】——降低最大回撤与最差单日（如 momentum 回撤
      -31.6%→-21.1%、最差单日 -10.6%→-6.1%），代价是收益略降、夏普基本打平。
    - **不是**夏普放大器：缩放后年化波动率降低，夏普比率通常不变或微降。
    - 仅用于分析（回测页可选开关），不改动实盘信号生成。

    前视安全：
    - realized = rolling(vol_window).std().shift(1) × √252
    - t 日仓位权重只用截至 t-1 的已实现波动率，不含当日信息。

    参数含义：
    - target_vol：目标年化波动率，决定平均仓位水平。0.15 → 当已实现波动率
      恰好 15% 时满仓（w=1），波动更高时减仓。
    - vol_window：回看窗口（交易日），默认 63（约 3 个月）。
    - cap：仓位上限，默认 1.0（只减仓不加杠杆）。实测 cap>1 加杠杆反而有害
      （波动放大抵消收益，回撤恶化），所以默认限制为不超过满仓。
    - initial_cash：起始金额，用于从缩放后日收益还原权益曲线。

    返回：
    - (scaled_equity, weights)：缩放后权益曲线与每日仓位权重序列。
      窗口不足期（前 vol_window 日）权重为 0、收益为 0（持现金）。
    """
    r = equity.pct_change().fillna(0.0)
    # 近期已实现波动率（年化），shift(1) 保证前视安全
    realized = r.rolling(vol_window).std().shift(1) * math.sqrt(TRADING_DAYS)
    # 目标仓位权重：波动越高→仓位越低；clip 到 [0, cap]
    w = (target_vol / realized).clip(lower=0.0, upper=cap).fillna(0.0)
    # 缩放后日收益 & 还原权益曲线
    scaled_r = w * r
    scaled_equity = initial_cash * (1 + scaled_r).cumprod()
    scaled_equity.name = "vol_scaled"
    w.name = "vol_weight"
    return scaled_equity, w


@dataclass
class BacktestResult:
    symbol: str
    strategy: str
    start: str
    end: str
    total_return: float
    cagr: float
    max_drawdown: float
    sharpe: float
    win_rate: float
    num_trades: int
    equity: pd.Series = field(repr=False)
    trades: list[dict] = field(repr=False)
    open_position: dict | None = field(default=None, repr=False)

    def metrics(self) -> dict:
        return {
            "总收益": f"{self.total_return:+.1%}",
            "年化收益": f"{self.cagr:+.1%}",
            "最大回撤": f"{self.max_drawdown:.1%}",
            "夏普比率": f"{self.sharpe:.2f}",
            "胜率": f"{self.win_rate:.0%}",
            "交易次数": self.num_trades,
        }


def run_backtest(
    df: pd.DataFrame,
    signals: list[Signal],
    symbol: str,
    strategy: str,
    initial_cash: float = 10_000.0,
    cost_bps: float = 0.0,
) -> BacktestResult:
    """单标的多头回测：buy 全仓买入，sell 全部卖出。"""
    sig_by_date: dict[str, str] = {}
    for s in sorted(signals, key=lambda x: x.date):
        if s.symbol == symbol and s.strategy == strategy:
            sig_by_date[s.date] = s.direction

    px = price_series(df)
    bp = cost_bps / 1e4
    cash = initial_cash
    shares = 0.0
    invested = 0.0
    entry_price = 0.0
    entry_date = ""
    trades: list[dict] = []
    equity_values = []

    for ts, price in px.items():
        price = float(price)
        direction = sig_by_date.get(ts.strftime("%Y-%m-%d"))
        if direction == BUY and shares == 0 and price > 0:
            invested = cash
            shares = cash * (1 - bp) / price
            cash = 0.0
            entry_price = price
            entry_date = ts.strftime("%Y-%m-%d")
        elif direction == SELL and shares > 0:
            cash = shares * price * (1 - bp)
            trades.append({
                "entry_date": entry_date, "exit_date": ts.strftime("%Y-%m-%d"),
                "entry": entry_price, "exit": price,
                "pnl_pct": cash / invested - 1,
                "profit": cash - invested,
            })
            shares = 0.0
        equity_values.append(cash + shares * price)

    open_position = {"entry_date": entry_date, "entry": entry_price} if shares > 0 else None

    equity = pd.Series(equity_values, index=px.index, name="equity")
    if equity.empty:
        raise ValueError("行情数据为空，无法回测")

    m = equity_metrics(equity, initial_cash)
    wins = sum(1 for t in trades if t["pnl_pct"] > 0)

    return BacktestResult(
        symbol=symbol,
        strategy=strategy,
        start=px.index[0].strftime("%Y-%m-%d"),
        end=px.index[-1].strftime("%Y-%m-%d"),
        total_return=m["total_return"],
        cagr=m["cagr"],
        max_drawdown=m["max_drawdown"],
        sharpe=m["sharpe"],
        win_rate=wins / len(trades) if trades else 0.0,
        num_trades=len(trades),
        equity=equity,
        trades=trades,
        open_position=open_position,
    )


@dataclass
class PortfolioBacktestResult:
    strategy: str
    start: str
    end: str
    win_rate: float
    num_trades: int
    equity: pd.Series = field(repr=False)
    trades: list[dict] = field(repr=False)          # symbol/entry_date/exit_date/entry/exit/pnl_pct
    open_positions: list[dict] = field(default_factory=list, repr=False)
    initial_cash: float = 10_000.0

    def metrics(self) -> dict:
        m = equity_metrics(self.equity, self.initial_cash)
        return {
            "总收益": f"{m['total_return']:+.1%}",
            "年化收益": f"{m['cagr']:+.1%}",
            "最大回撤": f"{m['max_drawdown']:.1%}",
            "夏普比率": f"{m['sharpe']:.2f}",
            "胜率": f"{self.win_rate:.0%}",
            "换仓次数": self.num_trades,
        }


def run_portfolio_backtest(
    prices: dict[str, pd.DataFrame],
    signals: list[Signal],
    strategy: str,
    initial_cash: float = 10_000.0,
    cost_bps: float = 0.0,
) -> PortfolioBacktestResult:
    """组合轮动回测：同日先处理全部 sell（清仓入现金、扣卖出成本），再把现金
    等分买入所有新增标的（扣买入成本）。一卖一买即自然换仓，资金不出场。"""
    closes = pd.DataFrame({s: price_series(df) for s, df in prices.items()}).sort_index().ffill()
    if closes.empty:
        raise ValueError("行情数据为空，无法回测")

    events: dict[str, list[Signal]] = {}
    for s in sorted(signals, key=lambda x: x.date):
        if s.strategy == strategy and s.symbol in closes.columns:
            events.setdefault(s.date, []).append(s)

    bp = cost_bps / 1e4
    cash = initial_cash
    holdings: dict[str, dict] = {}   # symbol -> {shares, entry, entry_date, invested}
    trades: list[dict] = []
    equity_values = []

    for ts in closes.index:
        d = ts.strftime("%Y-%m-%d")
        todays = events.get(d, [])
        # 先卖后买：卖出所得当日即可用于买入新标的
        for sig in todays:
            if sig.direction != SELL or sig.symbol not in holdings:
                continue
            price = closes.at[ts, sig.symbol]
            if pd.isna(price):
                continue
            price = float(price)
            pos = holdings.pop(sig.symbol)
            proceeds = pos["shares"] * price * (1 - bp)
            cash += proceeds
            trades.append({
                "symbol": sig.symbol,
                "entry_date": pos["entry_date"], "exit_date": d,
                "entry": pos["entry"], "exit": price,
                "pnl_pct": proceeds / pos["invested"] - 1,
                "profit": proceeds - pos["invested"],
            })
        adds = [
            sig.symbol for sig in todays
            if sig.direction == BUY and sig.symbol not in holdings
            and pd.notna(closes.at[ts, sig.symbol])
        ]
        if adds and cash > 0:
            per = cash / len(adds)
            for sym in adds:
                price = float(closes.at[ts, sym])
                holdings[sym] = {
                    "shares": per * (1 - bp) / price,
                    "entry": price, "entry_date": d, "invested": per,
                }
            cash = 0.0

        value = cash
        for sym, pos in holdings.items():
            price = closes.at[ts, sym]
            if pd.notna(price):
                value += pos["shares"] * float(price)
        equity_values.append(value)

    equity = pd.Series(equity_values, index=closes.index, name="equity")
    wins = sum(1 for t in trades if t["pnl_pct"] > 0)
    open_positions = [
        {"symbol": sym, "entry_date": pos["entry_date"], "entry": pos["entry"]}
        for sym, pos in holdings.items()
    ]

    return PortfolioBacktestResult(
        strategy=strategy,
        start=closes.index[0].strftime("%Y-%m-%d"),
        end=closes.index[-1].strftime("%Y-%m-%d"),
        win_rate=wins / len(trades) if trades else 0.0,
        num_trades=len(trades),
        equity=equity,
        trades=trades,
        open_positions=open_positions,
        initial_cash=initial_cash,
    )


@dataclass
class SmartDcaResult:
    start: str
    end: str
    equity: pd.Series = field(repr=False)
    invest_dates: list[str] = field(default_factory=list)   # 正常定投日
    topup_dates: list[str] = field(default_factory=list)    # 金叉补投日
    paused_spans: list[tuple[str, str]] = field(default_factory=list)  # 死叉暂停区段
    skipped_months: int = 0
    initial_cash: float = 10_000.0

    def metrics(self) -> dict:
        m = equity_metrics(self.equity, self.initial_cash)
        return {
            "总收益": f"{m['total_return']:+.1%}",
            "年化收益": f"{m['cagr']:+.1%}",
            "最大回撤": f"{m['max_drawdown']:.1%}",
            "夏普比率": f"{m['sharpe']:.2f}",
            "暂停月数": self.skipped_months,
            "补投次数": len(self.topup_dates),
        }


def run_smart_dca_backtest(
    df: pd.DataFrame,
    fast: int = 20,
    slow: int = 60,
    initial_cash: float = 10_000.0,
    cost_bps: float = 0.0,
) -> SmartDcaResult:
    """定投+信号开关：每月首个交易日定投一份；快线<慢线（死叉期）暂停、
    份额累积为现金；金叉恢复当日把累积款一次性补投。均线不足时视为趋势向上。"""
    px = price_series(df)
    if px.empty:
        raise ValueError("行情数据为空，无法回测")
    fast_ma = px.rolling(fast).mean()
    slow_ma = px.rolling(slow).mean()
    regime = (fast_ma >= slow_ma) | slow_ma.isna()

    month_firsts = set(px.groupby([px.index.year, px.index.month]).head(1).index)
    per = initial_cash / len(month_firsts)
    bp = cost_bps / 1e4

    cash = initial_cash       # 尚未到定投日的资金池
    pending = 0.0             # 死叉期攒下的定投款
    shares = 0.0
    skipped = 0
    values, invest_dates, topup_dates, paused_spans = [], [], [], []
    prev_r = True
    pause_start: str | None = None

    for ts, price in px.items():
        price = float(price)
        d = ts.strftime("%Y-%m-%d")
        r = bool(regime[ts])

        if r and not prev_r and pending > 0:   # 金叉恢复：补投累积款
            shares += pending * (1 - bp) / price
            pending = 0.0
            topup_dates.append(d)
        if ts in month_firsts:
            cash -= per
            if r:
                shares += per * (1 - bp) / price
                invest_dates.append(d)
            else:
                pending += per
                skipped += 1

        if not r and prev_r:
            pause_start = d
        elif r and not prev_r and pause_start:
            paused_spans.append((pause_start, d))
            pause_start = None
        prev_r = r

        values.append(cash + pending + shares * price)

    if pause_start:
        paused_spans.append((pause_start, px.index[-1].strftime("%Y-%m-%d")))

    return SmartDcaResult(
        start=px.index[0].strftime("%Y-%m-%d"),
        end=px.index[-1].strftime("%Y-%m-%d"),
        equity=pd.Series(values, index=px.index, name="equity"),
        invest_dates=invest_dates,
        topup_dates=topup_dates,
        paused_spans=paused_spans,
        skipped_months=skipped,
        initial_cash=initial_cash,
    )
