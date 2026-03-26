from .base import BaseStrategy
from .gbm import GBMStrategy
from .orderbook import OrderBookStrategy
from .pool_contrarian import PoolContrarianStrategy
from .manual_direction import ManualDirectionStrategy
from .mean_reversion import MeanReversionStrategy
from .follow_crowd import FollowCrowdStrategy

STRATEGIES = {
    "gbm": GBMStrategy,
    "orderbook": OrderBookStrategy,
    "pool_contrarian": PoolContrarianStrategy,
    "manual_direction": ManualDirectionStrategy,
    "mean_reversion": MeanReversionStrategy,
    "follow_crowd": FollowCrowdStrategy,
}
