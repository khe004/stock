"""信号驱动的多头回测：buy 全仓买入，sell 全部卖出，不考虑滑点手续费。"""

import math
from dataclasses import dataclass, field

import pandas as pd

from quant.strategies.base import BUY, SELL, Signal

TRADING_DAYS = 252


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
) -> BacktestResult:
    """df: 该标的日线（close 列，DatetimeIndex 升序）；signals: 该标的该策略的全部信号。"""
    sig_by_date: dict[str, str] = {}
    for s in sorted(signals, key=lambda x: x.date):
        if s.symbol == symbol and s.strategy == strategy:
            sig_by_date[s.date] = s.direction

    cash = initial_cash
    shares = 0.0
    entry_price = 0.0
    entry_date = ""
    trades: list[dict] = []
    equity_values = []

    for ts, row in df.iterrows():
        price = float(row["close"])
        direction = sig_by_date.get(ts.strftime("%Y-%m-%d"))
        if direction == BUY and shares == 0 and price > 0:
            shares = cash / price
            cash = 0.0
            entry_price = price
            entry_date = ts.strftime("%Y-%m-%d")
        elif direction == SELL and shares > 0:
            cash = shares * price
            trades.append({
                "entry_date": entry_date, "exit_date": ts.strftime("%Y-%m-%d"),
                "entry": entry_price, "exit": price,
                "pnl_pct": price / entry_price - 1,
            })
            shares = 0.0
        equity_values.append(cash + shares * price)

    open_position = {"entry_date": entry_date, "entry": entry_price} if shares > 0 else None

    equity = pd.Series(equity_values, index=df.index, name="equity")
    if equity.empty:
        raise ValueError("行情数据为空，无法回测")

    final = float(equity.iloc[-1])
    total_return = final / initial_cash - 1
    n_days = len(equity)
    cagr = (final / initial_cash) ** (TRADING_DAYS / n_days) - 1 if n_days > 0 and final > 0 else 0.0
    max_drawdown = float((equity / equity.cummax() - 1).min())
    daily_ret = equity.pct_change().dropna()
    sharpe = (
        float(daily_ret.mean() / daily_ret.std() * math.sqrt(TRADING_DAYS))
        if len(daily_ret) > 1 and daily_ret.std() > 0
        else 0.0
    )
    wins = sum(1 for t in trades if t["pnl_pct"] > 0)
    win_rate = wins / len(trades) if trades else 0.0

    return BacktestResult(
        symbol=symbol,
        strategy=strategy,
        start=df.index[0].strftime("%Y-%m-%d"),
        end=df.index[-1].strftime("%Y-%m-%d"),
        total_return=total_return,
        cagr=cagr,
        max_drawdown=max_drawdown,
        sharpe=sharpe,
        win_rate=win_rate,
        num_trades=len(trades),
        equity=equity,
        trades=trades,
        open_position=open_position,
    )
