import numpy as np
import pandas as pd
import pytest

from quant.strategies.base import BUY, SELL
from quant.strategies.momentum import Momentum
from quant.strategies.rsi_reversal import RsiReversal
from quant.strategies.sma_cross import SmaCross


def make_df(closes) -> pd.DataFrame:
    closes = np.asarray(closes, dtype=float)
    idx = pd.bdate_range("2024-01-01", periods=len(closes))
    return pd.DataFrame({
        "open": closes, "high": closes * 1.01, "low": closes * 0.99,
        "close": closes, "adj_close": closes, "volume": 1_000_000,
    }, index=idx)


def test_sma_cross_golden_cross_after_uptrend():
    down = np.linspace(120, 80, 60)
    up = np.linspace(80, 140, 60)
    df = make_df(np.concatenate([down, up]))
    signals = SmaCross(fast=5, slow=20).generate({"TEST": df})
    buys = [s for s in signals if s.direction == BUY]
    assert buys, "上涨段应出现金叉买入信号"
    turning_point = df.index[60]
    assert all(pd.Timestamp(s.date) > turning_point for s in buys)
    assert all(0 < s.strength <= 1 for s in signals)
    assert "均线" in buys[0].reason


def test_sma_cross_death_cross_after_downtrend():
    up = np.linspace(80, 140, 60)
    down = np.linspace(140, 80, 60)
    df = make_df(np.concatenate([up, down]))
    signals = SmaCross(fast=5, slow=20).generate({"TEST": df})
    assert any(s.direction == SELL for s in signals), "下跌段应出现死叉卖出信号"


def test_sma_cross_too_short_history():
    df = make_df(np.linspace(100, 110, 10))
    assert SmaCross(fast=5, slow=20).generate({"TEST": df}) == []


def test_rsi_buy_after_oversold_bounce():
    flat = [100.0] * 30
    drop = [100 * (0.98**i) for i in range(1, 11)]
    bounce = [drop[-1] * (1.03**i) for i in range(1, 9)]
    df = make_df(flat + drop + bounce)
    signals = RsiReversal(period=14).generate({"TEST": df})
    buys = [s for s in signals if s.direction == BUY]
    assert buys, "超卖反弹应出现买入信号"
    bounce_start = df.index[40]
    assert pd.Timestamp(buys[0].date) > bounce_start


def test_rsi_sell_after_overbought_pullback():
    flat = [100.0] * 30
    rally = [100 * (1.02**i) for i in range(1, 16)]
    pullback = [rally[-1] * (0.97**i) for i in range(1, 6)]
    df = make_df(flat + rally + pullback)
    signals = RsiReversal(period=14).generate({"TEST": df})
    assert any(s.direction == SELL for s in signals), "超买回落应出现卖出信号"


@pytest.fixture
def rotation_prices():
    n = 100
    steady = 100 * 1.005 ** np.arange(n)          # A: 全程缓涨
    late = np.concatenate([                        # B: 前 70 天横盘，后 30 天加速
        np.full(70, 100.0), 100 * 1.02 ** np.arange(1, 31),
    ])
    return {
        "A": make_df(steady),
        "B": make_df(late),
        "C": make_df(np.full(n, 100.0)),
        "D": make_df(np.full(n, 100.0)),
    }


def test_momentum_rotation(rotation_prices):
    signals = Momentum(lookback_days=20, top_n=1).generate(rotation_prices)
    b_buys = [s for s in signals if s.symbol == "B" and s.direction == BUY]
    a_sells = [s for s in signals if s.symbol == "A" and s.direction == SELL]
    assert b_buys, "B 加速后应进入动量榜首，产生买入信号"
    assert pd.Timestamp(b_buys[0].date) > rotation_prices["B"].index[70]
    assert a_sells, "A 被 B 挤出榜首后应产生卖出信号"


def test_momentum_needs_enough_symbols(rotation_prices):
    two = {k: rotation_prices[k] for k in ["A", "B"]}
    assert Momentum(lookback_days=20, top_n=3).generate(two) == []


def make_df_start(closes, start="2024-01-01"):
    closes = np.asarray(closes, dtype=float)
    idx = pd.bdate_range(start, periods=len(closes))
    return pd.DataFrame({
        "open": closes, "high": closes * 1.01, "low": closes * 0.99,
        "close": closes, "adj_close": closes, "volume": 1_000_000,
    }, index=idx)


def test_dual_momentum_switches_to_safe_and_back():
    from quant.strategies.dual_momentum import DualMomentum
    # 风险资产先涨（正动量持有）→ 暴跌（动量转负切避险）→ 反弹（切回）
    up = list(np.linspace(100, 150, 130))
    crash = list(np.linspace(150, 70, 65))
    recover = list(np.linspace(70, 140, 130))
    n = len(up + crash + recover)
    prices = {
        "SPY": make_df_start(up + crash + recover),
        "QQQ": make_df_start([100.0] * n),
        "TLT": make_df_start([100.0] * n),
    }
    strat = DualMomentum(lookback_days=60, risk_assets=["SPY", "QQQ"], safe_asset="TLT")
    signals = strat.generate(prices)
    buys = [s for s in signals if s.direction == BUY]
    assert buys[0].symbol == "SPY", "上涨期应持有最强风险资产"
    symbols_in_order = [s.symbol for s in buys]
    assert "TLT" in symbols_in_order, "动量转负应切换避险资产"
    assert symbols_in_order.index("TLT") > 0
    # 切避险后应有切回风险资产的一次
    after_tlt = symbols_in_order[symbols_in_order.index("TLT"):]
    assert any(sym in ("SPY", "QQQ") for sym in after_tlt[1:]), "反弹后应切回风险资产"
    sells = [s for s in signals if s.direction == SELL]
    assert sells, "换仓应先产生卖出信号"


def test_smart_dca_monthly_signals_and_pause():
    from quant.strategies.smart_dca import SmartDca
    # 横盘（正常定投）→ 急跌（死叉沉默）→ 反弹（恢复并补投）
    flat = [100.0] * 63
    crash = list(np.linspace(100, 60, 42))
    recover = list(np.linspace(60, 110, 63))
    prices = {"SPY": make_df_start(flat + crash + recover)}
    strat = SmartDca(symbol="SPY", fast=5, slow=20)
    signals = strat.generate(prices)
    assert signals and all(s.direction == BUY for s in signals)
    assert all(s.symbol == "SPY" for s in signals)
    # 死叉期应有月份沉默：信号数少于总月份数
    n_months = len({(ts.year, ts.month) for ts in prices["SPY"].index})
    assert len(signals) < n_months, "死叉期的定投日不应发信号"
    assert any("补投" in s.reason for s in signals), "恢复后应有补投信号"


def test_vix_regime_signals():
    from quant.strategies.vix_regime import VixRegime
    # 横盘 20 → 冲高 45（上穿30恐慌 + 对 VIX3M 倒挂）→ 回落到 12（穿30回补 + 穿15自满 + 倒挂解除）
    vix = [20.0] * 10 + list(np.linspace(20, 45, 10)) + list(np.linspace(45, 12, 15)) + [12.0] * 5
    vix3m = [22.0] * len(vix)
    prices = {"^VIX": make_df_start(vix), "^VIX3M": make_df_start(vix3m)}
    signals = VixRegime(vix="^VIX", vix3m="^VIX3M", panic=30, complacency=15).generate(prices)
    assert any(s.direction == SELL and "恐慌区" in s.reason for s in signals)
    assert any(s.direction == BUY and "回补" in s.reason for s in signals)
    assert any(s.direction == SELL and "倒挂" in s.reason and "解除" not in s.reason for s in signals)
    assert any(s.direction == BUY and "倒挂解除" in s.reason for s in signals)
    assert any("自满区" in s.reason for s in signals)
    assert all(s.symbol == "^VIX" for s in signals)
    dates = [s.date for s in signals]
    assert dates == sorted(dates)


def test_vix_regime_missing_data():
    from quant.strategies.vix_regime import VixRegime
    assert VixRegime().generate({}) == []
    # 只有 VIX 没有 VIX3M：仅阈值类信号，不报错
    vix = [20.0] * 10 + list(np.linspace(20, 45, 10))
    signals = VixRegime().generate({"^VIX": make_df_start(vix)})
    assert any("恐慌区" in s.reason for s in signals)
    assert not any("倒挂" in s.reason for s in signals)


@pytest.fixture
def stock_momentum_prices():
    n = 260
    def vol_df(closes, volume):
        closes = np.asarray(closes, dtype=float)
        idx = pd.bdate_range("2023-01-02", periods=len(closes))
        return pd.DataFrame({
            "open": closes, "high": closes * 1.01, "low": closes * 0.99,
            "close": closes, "adj_close": closes, "volume": float(volume),
        }, index=idx)
    return {
        "AAA": vol_df(100 * 1.004 ** np.arange(n), 5_000_000),   # 最强动量、高流动性
        "BBB": vol_df(100 * 1.002 ** np.arange(n), 5_000_000),   # 次强、高流动性
        "CCC": vol_df(np.full(n, 100.0), 5_000_000),             # 横盘、高流动性
        "DDD": vol_df(np.full(n, 100.0), 5_000_000),
        "EEE": vol_df(100 * 1.01 ** np.arange(n), 1_000),        # 动量最强但极不流动
        "SPY": vol_df(100 * 1.003 ** np.arange(n), 9_000_000),
        "TLT": vol_df(np.full(n, 100.0), 9_000_000),
    }


def _make_sm(**kw):
    from quant.strategies.stock_momentum import StockMomentum
    base = dict(
        universe=["AAA", "BBB", "CCC", "DDD", "EEE"],
        sectors={"AAA": "科技", "BBB": "科技", "CCC": "能源", "DDD": "金融", "EEE": "科技"},
        pool_size=4, liquidity_window=5, lookback_days=60, skip_days=5,
        top_n=2, max_per_sector=2, regime_symbol="SPY", regime_ma=20, safe_asset="TLT",
    )
    base.update(kw)
    return StockMomentum(**base)


def test_stock_momentum_picks_liquid_winners(stock_momentum_prices):
    signals = _make_sm().generate(stock_momentum_prices)
    buys = [s for s in signals if s.direction == BUY]
    assert buys, "应有买入信号"
    bought = {s.symbol for s in buys}
    assert "AAA" in bought and "BBB" in bought, "应选中流动性池内动量最强的两只"
    assert "EEE" not in bought, "动量最强但不流动的股票应被池子排除"
    assert any("流动性池内第 1 名" in s.reason for s in buys)


def test_stock_momentum_sector_cap(stock_momentum_prices):
    signals = _make_sm(max_per_sector=1).generate(stock_momentum_prices)
    # AAA/BBB 同为科技，限 1 只后第二席位应让给其他行业
    held_first = {s.symbol for s in signals if s.direction == BUY}
    assert "AAA" in held_first
    assert "BBB" not in held_first, "单行业上限应把同行业第二名挤出"


def test_stock_momentum_regime_off(stock_momentum_prices):
    # SPY 全程下跌 → 跌破均线后风险关闭，只应持有 TLT
    prices = dict(stock_momentum_prices)
    n = 260
    closes = 100 * 0.998 ** np.arange(n)
    idx = pd.bdate_range("2023-01-02", periods=n)
    prices["SPY"] = pd.DataFrame({
        "open": closes, "high": closes * 1.01, "low": closes * 0.99,
        "close": closes, "adj_close": closes, "volume": 9_000_000.0,
    }, index=idx)
    signals = _make_sm().generate(prices)
    buys = {s.symbol for s in signals if s.direction == BUY}
    assert buys == {"TLT"}, f"风险关闭时只应买入避险资产，实际 {buys}"
    assert any("风险关闭" in s.reason for s in signals)


def test_stock_momentum_exclude(stock_momentum_prices):
    signals = _make_sm(exclude=["AAA"]).generate(stock_momentum_prices)
    bought = {s.symbol for s in signals if s.direction == BUY}
    assert "AAA" not in bought, "被剔除的标的不应出现在持仓中"
    assert "BBB" in bought, "剔除后次强者应顶上"


# ──────────────── low_vol 低波动因子测试 ────────────────

@pytest.fixture
def low_vol_prices():
    """构造波动率明显不同的标的：
    - CALM: 每日涨 0.05%，极低波动
    - STEADY: 每日涨 0.1%，低波动
    - NORMAL: 每日涨 0.2%，中等波动
    - WILD: 每日涨跌交替 ±3%，高波动
    - CRAZY: 每日涨跌交替 ±5%，极高波动
    """
    n = 130  # 足够覆盖 90 日 lookback + 至少两个月度调仓日
    # CALM: 极低波动（每日稳定微涨）
    calm = 100 * 1.0005 ** np.arange(n)
    # STEADY: 低波动（每日稳定涨）
    steady = 100 * 1.001 ** np.arange(n)
    # NORMAL: 中等波动
    normal = 100 * 1.002 ** np.arange(n)
    # WILD: 高波动（大幅涨跌交替）
    wild_mult = np.where(np.arange(n) % 2 == 0, 1.03, 0.97)
    wild = 100 * np.cumprod(wild_mult)
    # CRAZY: 极高波动（更大幅涨跌交替）
    crazy_mult = np.where(np.arange(n) % 2 == 0, 1.05, 0.95)
    crazy = 100 * np.cumprod(crazy_mult)

    return {
        "CALM": make_df(calm),
        "STEADY": make_df(steady),
        "NORMAL": make_df(normal),
        "WILD": make_df(wild),
        "CRAZY": make_df(crazy),
    }


def test_low_vol_selects_lowest_volatility(low_vol_prices):
    """低波动策略应选中波动率最低的 top_n 只。"""
    from quant.strategies.low_vol import LowVol

    signals = LowVol(lookback_days=30, top_n=2).generate(low_vol_prices)
    buys = [s for s in signals if s.direction == BUY]
    assert buys, "应有买入信号"
    bought = {s.symbol for s in buys}
    # CALM 和 STEADY 波动率最低，应被选中
    assert "CALM" in bought, "波动最低的 CALM 应被选中"
    assert "STEADY" in bought, "波动次低的 STEADY 应被选中"
    # 高波动的不应被选中
    assert "WILD" not in bought, "高波动的 WILD 不应被选中"
    assert "CRAZY" not in bought, "极高波动的 CRAZY 不应被选中"


def test_low_vol_monthly_rebalance_buy_sell(low_vol_prices):
    """低波动策略月度调仓：进出组合应产生 BUY/SELL 信号。"""
    from quant.strategies.low_vol import LowVol

    # 用 top_n=2，先让 CALM/STEADY 被选中
    # 然后修改价格让 CALM 的波动率在后半段变高（被踢出），验证 SELL 信号
    prices = dict(low_vol_prices)
    n = 130
    # 把 CALM 后半段改成高波动
    calm_first = list(100 * 1.0005 ** np.arange(65))
    calm_wild = list(np.array(calm_first[-1:] * 65) *
                     np.cumprod(np.where(np.arange(65) % 2 == 0, 1.04, 0.96)))
    calm_combined = np.array(calm_first + list(calm_wild))[:n]
    prices["CALM"] = make_df(calm_combined)

    signals = LowVol(lookback_days=30, top_n=2).generate(prices)
    # 应有 CALM 的 SELL 信号（波动升高被踢出）
    calm_sells = [s for s in signals if s.symbol == "CALM" and s.direction == SELL]
    # 也应有某只低波动标的的 BUY 替代
    buys = [s for s in signals if s.direction == BUY]
    sells = [s for s in signals if s.direction == SELL]
    assert buys, "应有买入信号"
    # 验证有卖出信号（组合调仓）
    assert sells, "月度调仓应有卖出信号"


def test_low_vol_reason_contains_numbers(low_vol_prices):
    """低波动信号的 reason 必须含波动率数值。"""
    from quant.strategies.low_vol import LowVol

    signals = LowVol(lookback_days=30, top_n=2).generate(low_vol_prices)
    assert signals, "应有信号"
    for s in signals:
        # reason 应包含 "年化波动" 以及百分比数值
        assert "波动" in s.reason, f"reason 应含'波动'：{s.reason}"
        if s.direction == BUY:
            assert "%" in s.reason, f"买入 reason 应含波动率数值：{s.reason}"
            assert "名" in s.reason, f"买入 reason 应含排名：{s.reason}"


def test_low_vol_strength_in_range(low_vol_prices):
    """低波动信号的 strength 应在 0~1 范围。"""
    from quant.strategies.low_vol import LowVol

    signals = LowVol(lookback_days=30, top_n=2).generate(low_vol_prices)
    assert signals, "应有信号"
    for s in signals:
        assert 0 < s.strength <= 1, f"strength 应在 (0, 1]：{s.strength}"


def test_low_vol_no_signal_when_window_insufficient():
    """波动率回看窗口不足时不应发信号。"""
    from quant.strategies.low_vol import LowVol

    # 只有 20 天数据，lookback=30 不够算波动率
    short_prices = {
        "A": make_df(np.linspace(100, 110, 20)),
        "B": make_df(np.linspace(100, 105, 20)),
        "C": make_df(np.linspace(100, 108, 20)),
        "D": make_df(np.linspace(100, 103, 20)),
    }
    signals = LowVol(lookback_days=30, top_n=2).generate(short_prices)
    assert signals == [], "数据不足窗口时不应发信号"


def test_low_vol_not_enough_symbols():
    """标的数不足 top_n 时不发信号。"""
    from quant.strategies.low_vol import LowVol

    prices = {
        "A": make_df(100 * 1.001 ** np.arange(100)),
        "B": make_df(100 * 1.002 ** np.arange(100)),
    }
    signals = LowVol(lookback_days=30, top_n=3).generate(prices)
    assert signals == [], "标的数 <= top_n 时不应发信号"

