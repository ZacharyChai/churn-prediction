"""Opening Range Breakout (ORB) day-trading bot for ES / MES futures.

The strategy builds the high/low of the first N minutes after the New York
cash open (09:30 ET), then trades a breakout of that range with a bracket
(stop loss + profit target) and flattens before the cash close.

Modules
-------
config    : StrategyConfig and contract specifications
models    : Bar / order / position / trade dataclasses
strategy  : ORBStrategy, a pure state machine that turns bars into actions
broker    : PaperBroker, a bar-driven fill simulator used by the backtester
backtest  : run_backtest() engine
metrics   : performance statistics
data      : CSV loading and synthetic bar generation
live      : Interactive Brokers runner (requires ib_insync)
"""

from .config import CONTRACTS, ContractSpec, StrategyConfig
from .models import Bar, Side, Trade
from .backtest import BacktestResult, run_backtest
from .strategy import ORBStrategy

__all__ = [
    "CONTRACTS",
    "ContractSpec",
    "StrategyConfig",
    "Bar",
    "Side",
    "Trade",
    "BacktestResult",
    "run_backtest",
    "ORBStrategy",
]
