from .base import BaseStrategy
from .gbm import GBMStrategy
from .orderbook import OrderBookStrategy
from .pool_contrarian import PoolContrarianStrategy
from .mean_reversion import MeanReversionStrategy
from .follow_crowd import FollowCrowdStrategy
from .funding_rate import FundingRateStrategy
from .open_interest import OpenInterestStrategy
from .liquidation_reversal import LiquidationReversalStrategy
from .volume_breakout import VolumeBreakoutStrategy
from .rsi_reversal import RSIReversalStrategy
from .order_flow import OrderFlowStrategy
from .market_regime import MarketRegimeStrategy
from .correlation_arbitrage import CorrelationArbitrageStrategy
from .fear_greed_micro import FearGreedMicroStrategy
from .whale_signal import WhaleSignalStrategy
from .bollinger_squeeze import BollingerSqueezeStrategy

STRATEGIES = {
    "gbm": GBMStrategy,
    "orderbook": OrderBookStrategy,
    "pool_contrarian": PoolContrarianStrategy,
    "mean_reversion": MeanReversionStrategy,
    "follow_crowd": FollowCrowdStrategy,
    "funding_rate": FundingRateStrategy,
    "open_interest": OpenInterestStrategy,
    "liquidation_reversal": LiquidationReversalStrategy,
    "volume_breakout": VolumeBreakoutStrategy,
    "rsi_reversal": RSIReversalStrategy,
    "order_flow": OrderFlowStrategy,
    "market_regime": MarketRegimeStrategy,
    "correlation_arbitrage": CorrelationArbitrageStrategy,
    "fear_greed_micro": FearGreedMicroStrategy,
    "whale_signal": WhaleSignalStrategy,
    "bollinger_squeeze": BollingerSqueezeStrategy,
}
