import pandas as pd

import run_daily


def _df(last: str) -> pd.DataFrame:
    idx = pd.date_range(end=last, periods=3, freq="D")
    return pd.DataFrame({"close": [1.0, 2.0, 3.0]}, index=idx)


def test_signal_date_ignores_non_strategy_symbols_dated_ahead():
    # 亚洲股/汇率/BTC 在美西下午已是次日，不能把信号日期顶到美股没数据的那天
    prices = {"SPY": _df("2026-10-06"), "QQQ": _df("2026-10-06"),
              "005930.KS": _df("2026-10-07"), "BTC-USD": _df("2026-10-07")}
    assert run_daily.latest_signal_date(prices, ["SPY", "QQQ", "TLT"]) == "2026-10-06"


def test_signal_date_falls_back_to_all_prices_without_strategy_symbols():
    prices = {"BTC-USD": _df("2026-10-07")}
    assert run_daily.latest_signal_date(prices, []) == "2026-10-07"


def test_config_strategy_symbols_exclude_display_only_groups():
    from quant.config import load_config
    cfg = load_config()
    fed = {s for _, p in cfg.enabled_strategies() for s in run_daily._strategy_symbols(cfg, p)}
    display_only = set(cfg.symbols_for(["macro", "fx_rates"]))
    assert not fed & display_only
