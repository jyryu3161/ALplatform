"""Compatibility import for the official TabM regression adapter.

The former custom MLP and Monte Carlo dropout implementation have been removed.
Use ``active_learning_docking.model.TabMRegressor`` for new code.
"""

from active_learning_docking.model import TabMRegressor


class UnifiedModel(TabMRegressor):
    """Backward-compatible class name with the explicit TabMRegressor API."""


__all__ = ["UnifiedModel", "TabMRegressor"]
