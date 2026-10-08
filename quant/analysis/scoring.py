"""信号历史表现评分：逐条信号算未来 N 个交易日的表现，回答"这条信号准不准"。

与回测的区别：回测模拟机械执行整套策略（含仓位、成本、资金曲线）；这里只看
单条信号本身——发出信号后标的实际涨跌如何，不涉及仓位管理。
"""

import pandas as pd

from quant.strategies.base import BUY, Signal, price_series

DEFAULT_HORIZONS = (5, 20, 60)


def signal_forward_returns(
    signals: list[Signal],
    prices: dict[str, pd.DataFrame],
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    trade_symbol_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """逐条信号计算未来 N 个交易日的表现（用 adj_close 总回报口径）。

    收益按信号方向调整符号：buy 上涨为正、sell 下跌为正，都代表"信号判断对
    了"，因此可以直接跨方向比较正负，不用心算翻转。

    trade_symbol_map：部分策略的信号标的本身不可直接用来算收益（如 vix_regime
    的 ^VIX 只是指数），需要按 {策略名: 实际标的} 映射到可交易标的（如 SPY）。

    价格口径：``signal_price`` 是信号标的的原始收盘价；``price_now`` 是**评价标的**的最新原始
    收盘价；各 ``ret_*`` 用评价标的的复权价算（含分红）。所以信号价与现价之比不等于收益列。
    因缺行情而算不了的信号不进结果，条数记在 ``df.attrs["skipped_no_data"]``——
    "缺数据"与"未到期"（``ret_h`` 为空）是两回事，不能混在一起。
    """
    trade_symbol_map = trade_symbol_map or {}
    rows = []
    skipped = 0
    for s in signals:
        trade_symbol = trade_symbol_map.get(s.strategy, s.symbol)
        df = prices.get(trade_symbol)
        if df is None or df.empty:
            skipped += 1
            continue
        px = price_series(df)
        pos_arr = px.index.get_indexer([pd.Timestamp(s.date)])
        if pos_arr[0] == -1:
            skipped += 1
            continue  # 信号日不在该标的行情范围内
        pos = int(pos_arr[0])
        base = float(px.iloc[pos])
        if base <= 0:
            skipped += 1
            continue
        sign = 1.0 if s.direction == BUY else -1.0
        last_pos = len(px) - 1
        row = {
            "date": s.date, "symbol": s.symbol, "trade_symbol": trade_symbol,
            "strategy": s.strategy, "direction": s.direction,
            "signal_price": s.price, "reason": s.reason,
            "price_now": float(df["close"].iloc[last_pos]),
        }
        row["ret_now"] = sign * (float(px.iloc[last_pos]) / base - 1)
        for h in horizons:
            fp = pos + h
            row[f"ret_{h}"] = sign * (float(px.iloc[fp]) / base - 1) if fp <= last_pos else None
        rows.append(row)
    out = pd.DataFrame(rows)
    out.attrs["skipped_no_data"] = skipped
    return out


def summarize_scores(
    df: pd.DataFrame,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    min_samples: int = 10,
) -> pd.DataFrame:
    """按 (策略, 方向) 汇总：信号数、各周期已到期样本数/待到期数/平均收益/胜率。

    胜率与均值的分母只含该周期**已到期**的样本（``n_h``）；``pending_h`` 是还没走完 h 个
    交易日的信号数。``low_sample_h`` 按各周期自己的已到期样本数判断——信号总数够、
    但 60 日样本还没几条到期时，60 日那列同样不可信。
    """
    if df.empty:
        return pd.DataFrame()
    out = []
    for (strat, direction), g in df.groupby(["strategy", "direction"]):
        row = {"strategy": strat, "direction": direction, "n": len(g),
               "low_sample": len(g) < min_samples}
        for h in horizons:
            valid = g[f"ret_{h}"].dropna()
            row[f"n_{h}"] = len(valid)
            row[f"pending_{h}"] = len(g) - len(valid)
            row[f"low_sample_{h}"] = len(valid) < min_samples
            row[f"mean_{h}"] = float(valid.mean()) if len(valid) else None
            row[f"win_{h}"] = float((valid > 0).mean()) if len(valid) else None
        out.append(row)
    return pd.DataFrame(out)
