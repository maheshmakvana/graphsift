"""Reference data for the trading-strategy hallucination guard.

Contains ``angelbot_reference.json`` — real live-vs-backtest P&L figures used
as the default comparison baseline by :mod:`graphsift.guard`.
"""

from pathlib import Path

DEFAULT_REFERENCE_PATH = Path(__file__).parent / "angelbot_reference.json"

__all__ = ["DEFAULT_REFERENCE_PATH"]
