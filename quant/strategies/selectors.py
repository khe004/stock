"""可复用的横截面选择信号——纯计算，不含调仓日期循环、不含防守逻辑。

抽出来之前，12-1 动量公式（`adj.shift(skip) / adj.shift(lookback) - 1`）在
momentum.py / aggressive.py / cross_asset.py / canary.py / stock_momentum.py 里
各自独立抄了一遍（dual_momentum.py 用 `.pct_change(lookback)` 写法，数学上等价于
skip=0 的特例：`pct_change(n)` ≡ `shift(0)/shift(n) - 1`）；配套的"动量值→信号强度"
映射公式在其中六处也完全相同。这里统一成两个纯函数，逐策略回归测试字节级一致后接入。

【本次重构范围的边界，2026-07-30 定下】"防守机制"（避险/现金感知那部分）**没有**抽成
共享对象——虽然概念上能归纳成两类模式（自指补位型：dual_momentum/aggressive_mom/
cross_asset_mom；外部触发型：canary_mom/stock_momentum），但每个策略的 SELL/BUY
reason 文案都是手写定制的（例如 aggressive_mom 仅 SELL 就有 4 种不同措辞），而
"reason 必须是人话、含具体数值"是平台的产品要求，不是可有可无的细节。抽成通用类要么
牺牲文案质量换成模板化措辞（真实的行为倒退），要么共享类得开一堆策略专属回调钩子，
那就不是真正的共享，只是换个地方写同样多的代码——用户看过两种设计后明确选择"只做
选择信号，避险机制保持每个策略手写"。详见 memory: stock-backlog 的 refactor 条目。
"""

import pandas as pd


def momentum_return(adj: pd.DataFrame, lookback: int, skip: int = 0) -> pd.DataFrame:
    """12-1 动量（或其退化形式）：t-skip 相对 t-lookback 的收益。

    skip=0 时就是普通的"近 lookback 日收益"——数学上与 `adj.pct_change(lookback)`
    完全等价（dual_momentum 原先的写法），已用真实历史信号回归验证字节级一致。
    """
    return adj.shift(skip) / adj.shift(lookback) - 1


def momentum_strength(value: float, scale: float = 2.0) -> float:
    """动量值 → 信号强度，映射到 [0.1, 1.0]：|value|×scale 越大越强，
    超过 1/scale（默认对应 ±50% 动量）即封顶 1.0，最低不低于 0.1。

    不做 round——四舍五入统一交给调用方在构造 Signal 时做（与现状一致）。
    """
    return min(1.0, max(0.1, abs(value) * scale))
