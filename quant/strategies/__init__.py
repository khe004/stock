from quant.strategies.base import BUY, SELL, Signal, Strategy
from quant.strategies.dual_momentum import DualMomentum
from quant.strategies.momentum import Momentum
from quant.strategies.rsi_reversal import RsiReversal
from quant.strategies.sma_cross import SmaCross
from quant.strategies.smart_dca import SmartDca
from quant.strategies.stock_momentum import StockMomentum
from quant.strategies.vix_regime import VixRegime

REGISTRY: dict[str, type[Strategy]] = {
    cls.name: cls
    for cls in (SmaCross, Momentum, RsiReversal, SmartDca, DualMomentum,
                VixRegime, StockMomentum)
}


def build(name: str, params: dict) -> Strategy:
    if name not in REGISTRY:
        raise KeyError(f"未知策略: {name}，可用: {list(REGISTRY)}")
    return REGISTRY[name](**{k: v for k, v in params.items() if k != "groups"})
