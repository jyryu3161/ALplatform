"""Compatibility entry point for the versioned, configuration-driven workflow."""

from active_learning_docking.config import ExperimentConfig
from active_learning_docking.pipeline import run_experiment


class ActiveLearningWithConformers:
    """Run an ExperimentConfig; use load_config(path) to migrate old scripts.

    Version 2 intentionally rejects the old unvalidated keyword interface.
    This avoids silently retaining label filtering or the previous MLP model.
    """

    def __init__(self, config: ExperimentConfig):
        if not isinstance(config, ExperimentConfig):
            raise TypeError("pass ExperimentConfig from active_learning_docking.config.load_config")
        self.config = config

    def run(self):
        return run_experiment(self.config)
