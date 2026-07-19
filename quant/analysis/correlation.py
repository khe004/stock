"""策略相关性与组合诊断：纯计算模块，不含 UI。

把每个启用策略化简成一条日收益率序列，算 Pearson 相关矩阵，并构建等权组合
用于诊断现有策略的分散效果。

方法论口径：
- 组合类策略（momentum/dual_momentum/stock_momentum/low_vol）→ engine.run_portfolio_backtest
  取权益曲线
- smart_dca → engine.run_smart_dca_backtest 取权益曲线
- 单标的策略（sma_cross/rsi_reversal）→ 对该策略涉及的每个标的各跑
  engine.run_backtest，各标的权益曲线日收益率等权平均成该策略的收益序列
- vix_regime → 映射到 trade_symbol（默认 SPY）后按单标的模式处理

所有权益曲线统一用 pct_change() 转日收益率，成本口径统一使用 config.cost_bps。

口径局限：策略空仓（持现金）时日收益 = 0，会稀释 Pearson 相关系数——这段
"共同不动"的时间被算进去了。解读时需注意：实际仓位重叠期的相关性可能更高。
"""

from dataclasses import replace

import pandas as pd

from quant.backtest.engine import (
    equity_metrics,
    run_backtest,
    run_portfolio_backtest,
    run_smart_dca_backtest,
)
from quant.strategies.base import BUY, Signal, price_series

# 与 app.py 保持一致的分类
PORTFOLIO_STRATEGIES = {"momentum", "dual_momentum", "stock_momentum", "low_vol"}


def _portfolio_return_series(
    prices: dict[str, pd.DataFrame],
    signals: list[Signal],
    strategy: str,
    cost_bps: float,
) -> pd.Series:
    """组合类策略：用 run_portfolio_backtest 出权益曲线，转日收益率。"""
    result = run_portfolio_backtest(prices, signals, strategy, cost_bps=cost_bps)
    return result.equity.pct_change().fillna(0.0)


def _smart_dca_return_series(
    df: pd.DataFrame,
    params: dict,
    cost_bps: float,
) -> pd.Series:
    """智能定投策略：用 run_smart_dca_backtest 出权益曲线，转日收益率。"""
    fast = params.get("fast", 20)
    slow = params.get("slow", 60)
    result = run_smart_dca_backtest(df, fast, slow, cost_bps=cost_bps)
    return result.equity.pct_change().fillna(0.0)


def _single_symbol_return_series(
    prices: dict[str, pd.DataFrame],
    signals: list[Signal],
    strategy: str,
    symbols: list[str],
    cost_bps: float,
) -> pd.Series | None:
    """单标的策略：对每个标的各跑 run_backtest，日收益率等权平均。

    等权平均的含义：假设给每个标的分配等额资金独立运行该策略，
    组合日收益率 = 各标的日收益率的简单算术平均。
    """
    ret_frames = []
    for sym in symbols:
        if sym not in prices or prices[sym].empty:
            continue
        try:
            result = run_backtest(prices[sym], signals, sym, strategy,
                                  cost_bps=cost_bps)
            ret = result.equity.pct_change().fillna(0.0)
            ret.name = sym
            ret_frames.append(ret)
        except (ValueError, KeyError):
            continue
    if not ret_frames:
        return None
    # 等权平均：按日期对齐后求均值
    combined = pd.concat(ret_frames, axis=1).fillna(0.0)
    return combined.mean(axis=1)


def strategy_return_series(
    prices: dict[str, pd.DataFrame],
    strategy_signals: dict[str, list[Signal]],
    strategy_params: dict[str, dict],
    strategy_symbols: dict[str, list[str]],
    cost_bps: float,
    trade_symbol_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """把每个策略化简为一条日收益率序列，返回 DataFrame（列=策略名，行=日期）。

    参数：
        prices: symbol -> 日线 DataFrame（需含 adj_close/close 列）
        strategy_signals: strategy_name -> 该策略的全量信号列表
        strategy_params: strategy_name -> 策略参数 dict
        strategy_symbols: strategy_name -> 该策略涉及的标的列表
        cost_bps: 单边交易成本（万分之一为 1bp）
        trade_symbol_map: 需要映射到实际可交易标的的策略（如 vix_regime -> SPY）

    返回：
        pd.DataFrame，列为策略名，行为日期（DatetimeIndex），值为日收益率。
        只包含成功构建收益序列的策略。各策略的日期范围可能不同（由其行情覆盖决定）。
    """
    trade_symbol_map = trade_symbol_map or {}
    series_dict: dict[str, pd.Series] = {}

    for strategy_name, signals in strategy_signals.items():
        params = strategy_params.get(strategy_name, {})
        symbols = strategy_symbols.get(strategy_name, [])

        try:
            if strategy_name in PORTFOLIO_STRATEGIES:
                strat_prices = {s: prices[s] for s in symbols
                                if s in prices and not prices[s].empty}
                if not strat_prices:
                    continue
                ret = _portfolio_return_series(strat_prices, signals,
                                              strategy_name, cost_bps)
            elif strategy_name == "smart_dca":
                symbol = params.get("symbol", "SPY")
                if symbol not in prices or prices[symbol].empty:
                    continue
                ret = _smart_dca_return_series(prices[symbol], params, cost_bps)
            elif strategy_name == "vix_regime":
                trade_sym = trade_symbol_map.get(strategy_name,
                                                  params.get("trade_symbol", "SPY"))
                if trade_sym not in prices or prices[trade_sym].empty:
                    continue
                mapped_signals = [replace(s, symbol=trade_sym) for s in signals]
                ret = _single_symbol_return_series(
                    prices, mapped_signals, strategy_name, [trade_sym], cost_bps)
                if ret is None:
                    continue
            else:
                # 单标的策略：sma_cross, rsi_reversal 等
                ret = _single_symbol_return_series(
                    prices, signals, strategy_name, symbols, cost_bps)
                if ret is None:
                    continue
            series_dict[strategy_name] = ret
        except (ValueError, KeyError):
            continue

    if not series_dict:
        return pd.DataFrame()
    return pd.DataFrame(series_dict)


def correlation_matrix(returns_df: pd.DataFrame) -> pd.DataFrame:
    """计算策略间 Pearson 相关矩阵。

    输入 returns_df 需先按日期对齐（内连接/共同交易日）。
    返回 N×N 的相关矩阵 DataFrame（行列均为策略名）。
    对角线为 1，矩阵对称。
    """
    if returns_df.empty or returns_df.shape[1] < 2:
        return returns_df.corr() if not returns_df.empty else pd.DataFrame()
    # 内连接：只保留所有策略都有数据的交易日
    aligned = returns_df.dropna()
    if aligned.empty:
        return pd.DataFrame()
    return aligned.corr()


def combined_portfolio(
    returns_df: pd.DataFrame,
    initial_value: float = 10_000.0,
) -> tuple[pd.Series, dict]:
    """等权合成组合：各策略日收益率简单平均，还原权益曲线并算指标。

    等权组合的含义：把资金等分到 N 个策略，每天组合收益 = 各策略收益的均值。
    这是最朴素的分散方式，用于检验"多策略叠加是否比单策略更稳健"。

    参数：
        returns_df: 日收益率 DataFrame（列=策略名），已按日期对齐
        initial_value: 初始资金

    返回：
        (equity, metrics) —— 等权组合的权益曲线与风险收益指标。
    """
    # 内连接：只保留共同交易日
    aligned = returns_df.dropna()
    if aligned.empty:
        empty_eq = pd.Series(dtype=float)
        return empty_eq, {}
    avg_ret = aligned.mean(axis=1)
    equity = initial_value * (1 + avg_ret).cumprod()
    equity.name = "等权组合"
    metrics = equity_metrics(equity, initial_value)
    return equity, metrics
