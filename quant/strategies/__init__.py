from quant.strategies.aggressive import AggressiveMomentum
from quant.strategies.base import BUY, SELL, Signal, Strategy
from quant.strategies.cross_asset import CrossAssetMomentum
from quant.strategies.dual_momentum import DualMomentum
from quant.strategies.low_vol import LowVol
from quant.strategies.momentum import Momentum
from quant.strategies.rsi_reversal import RsiReversal
from quant.strategies.sma_cross import SmaCross
from quant.strategies.smart_dca import SmartDca
from quant.strategies.stock_momentum import StockMomentum
from quant.strategies.vix_regime import VixRegime

REGISTRY: dict[str, type[Strategy]] = {
    cls.name: cls
    for cls in (SmaCross, Momentum, RsiReversal, SmartDca, DualMomentum,
                VixRegime, StockMomentum, LowVol, CrossAssetMomentum,
                AggressiveMomentum)
}


def build(name: str, params: dict) -> Strategy:
    if name not in REGISTRY:
        raise KeyError(f"未知策略: {name}，可用: {list(REGISTRY)}")
    # groups 是作用范围、notify 是通知开关——都不是策略参数，构造时剔除
    meta = {"groups", "notify"}
    return REGISTRY[name](**{k: v for k, v in params.items() if k not in meta})
