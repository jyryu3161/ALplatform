"""Budgeted pool-based active learning with a sealed replay evaluation partition.

The CLI configures CPU/GPU resources before importing this module; the
RDKit, PyTorch and docking modules are imported only inside the stages that
need them so that thread pools inherit the configured affinity. Libraries are
accessed through :class:`store.Library` and :class:`store.PackedFingerprints`,
so a memory-mapped store of 10^8 compounds streams through the same stages as
an in-memory CSV library.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import logging
from pathlib import Path
from typing import Any, Mapping, Protocol

import numpy as np
import pandas as pd
import yaml

from .acquisition import scheduled_beta, select_indices
from .config import ExperimentConfig
from .evaluation import discovery_metrics, regression_metrics
from .provenance import artifact_hashes, environment_manifest, utc_now, write_json
from .resources import configure_resources
from .store import Library, PackedFingerprints, is_library_store, open_library_store


class Oracle(Protocol):
    """Score one compound and report either a finite score or a structured failure."""

    def evaluate(self, cid: str, smiles: str, output_dir: Path) -> dict[str, Any]:
        """Return ``score``, ``status`` (``ok``/``failed``), ``error`` and ``pose``."""


class ReplayOracle:
    """Return recorded scores; the replay counterpart of :class:`DockingOracle`.

    Scores are looked up by binary search on the ID-sorted library column, so
    a memory-mapped store is never materialized.
    """

    def __init__(self, library: Library) -> None:
        if library.score is None:
            raise ValueError("replay requires a library with recorded scores")
        self.library = library

    def evaluate(self, cid: str, smiles: str, output_dir: Path) -> dict[str, Any]:
        ids = self.library.compound_id.values
        key = str(cid).encode("utf-8") if ids.dtype.kind == "S" else str(cid)
        position = int(np.searchsorted(ids, key))
        if position >= len(ids) or self.library.compound_id[position] != str(cid):
            raise KeyError(f"compound {cid!r} is not in the replay library")
        return {"score": float(self.library.score[position]), "status": "ok", "error": None, "pose": None}


def _open_logger(output: Path, name: str) -> tuple[logging.Logger, list[logging.Handler]]:
    logger = logging.getLogger(f"active_learning_docking.{name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handlers: list[logging.Handler] = [
        logging.FileHandler(output / "active_learning.log", encoding="utf-8"),
        logging.StreamHandler(),
    ]
    for handler in handlers:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
    return logger, handlers


class ActiveLearningExperiment:
    """One validated experiment executed inside a new output directory.

    Stages: :meth:`prepare_library` validates, featurizes and seals the
    holdout; each cycle selects queries, evaluates them with the oracle, fits
    a fresh model on all successful acquired labels and records metrics;
    :meth:`finalize` writes final predictions and the summary. Every attempted
    compound, including a failed docking, consumes one budget unit, and
    holdout labels never enter fitting, tuning, stopping or acquisition.
    Libraries are ID-sorted, so positions serve as deterministic tie keys.
    """

    def __init__(self, config: ExperimentConfig, runtime: Mapping[str, Any], output: Path,
                 logger: logging.Logger) -> None:
        self.config = config
        self.runtime = runtime
        self.output = output
        self.logger = logger
        self.observations: list[dict[str, Any]] = []
        self.cycle_metrics: list[dict[str, Any]] = []
        self.history: list[dict[str, Any]] = []
        self.model: Any = None
        self.train_indices = np.empty(0, dtype=np.int64)
        self.train_scores = np.empty(0, dtype=np.float64)

    def run(self) -> dict[str, Any]:
        """Execute all stages and return the experiment summary."""
        from .regressors import create_regressor

        # Fail on unavailable backends/invalid parameters before spending docking budget.
        preflight = create_regressor(self.config.model, seed=self.config.seed,
                                     device=self.runtime["device"],
                                     cpu_threads=self.config.resources.cpu_threads)
        if hasattr(preflight, "preflight"):
            preflight.preflight()
        del preflight
        self.prepare_library()
        self.oracle = self.make_oracle()
        seeds = np.random.SeedSequence(self.config.seed).spawn(2)
        self.initial_rng, self.acquisition_rng = [np.random.default_rng(seed) for seed in seeds]
        learning = self.config.learning
        for cycle in range(learning.cycles + 1):
            allowance = learning.budget - len(self.observations)
            if not self.remaining() or allowance <= 0:
                break
            self.run_cycle(cycle, allowance)
        return self.finalize()

    def prepare_library(self) -> None:
        """Load the canonical library and fingerprints, then seal the holdout.

        A CSV input is validated and featurized here and copied to
        ``library.csv``; a prepared library store is memory-mapped and must have
        been built with the configured fingerprint and deduplication settings.
        """
        from .data import featurize, load_library, split_holdout

        config = self.config
        if is_library_store(config.data.path):
            self.library, self.features, store = open_library_store(config.data.path)
            self._check_store(store)
            self.n_rejected = int(store["n_rejected"])
        else:
            frame, rejected = load_library(
                config.data.path, id_column=config.data.id_column,
                smiles_column=config.data.smiles_column,
                score_column=config.data.score_column if config.mode == "replay" else None,
                invalid_policy=config.data.invalid_policy, deduplicate=config.data.deduplicate,
            )
            frame.to_csv(self.output / "library.csv", index=False)
            rejected.to_csv(self.output / "rejected_rows.csv", index=False)
            self.n_rejected = len(rejected)
            self.library = Library.from_frame(frame)
            self.features = PackedFingerprints.from_dense(featurize(
                frame.smiles.tolist(), radius=config.data.fingerprint_radius,
                n_bits=config.data.fingerprint_bits, fingerprint_type=config.data.fingerprint_type))
        if config.mode == "replay" and self.library.score is None:
            raise ValueError("replay mode requires recorded scores in the library")
        if self.library.origin == "store" and config.evaluation.holdout_fraction > 0:
            self.logger.warning("Holdout splitting parses every structure of the %d-compound store in this "
                                "process; use holdout_fraction 0 for very large pools", len(self.library))
        pool, holdout = split_holdout(self.library.smiles_column, config.evaluation.holdout_fraction,
                                      config.seed, strategy=config.evaluation.split)
        self.pool, self.holdout = np.asarray(pool, dtype=np.int64), np.asarray(holdout, dtype=np.int64)
        if len(self.pool) < 3:
            raise ValueError("at least 3 eligible pool compounds are required after splitting")
        self.pool_mask = np.zeros(len(self.library), dtype=bool)
        self.pool_mask[self.pool] = True
        self.queried = np.zeros(len(self.library), dtype=bool)
        if len(self.holdout):
            partitions = pd.DataFrame({"compound_id": self.library.ids(), "partition": "pool"})
            partitions.loc[self.holdout, "partition"] = "holdout"
            partitions.to_csv(self.output / "split.csv", index=False)

    def _check_store(self, store: Mapping[str, Any]) -> None:
        data = self.config.data
        expected = {"type": data.fingerprint_type, "radius": data.fingerprint_radius, "n_bits": data.fingerprint_bits}
        if store["fingerprint"] != expected:
            raise ValueError(f"library store fingerprints {store['fingerprint']} differ from configuration {expected}")
        names = ["deduplicate", "invalid_policy", "id_column", "smiles_column"]
        if self.config.mode == "replay":
            names.append("score_column")
        for name in names:
            if store["settings"][name] != getattr(data, name):
                raise ValueError(f"library store was built with {name}={store['settings'][name]!r}; "
                                 "configure the same value or rebuild the store")

    def make_oracle(self) -> Oracle:
        """Build the score source: recorded replay scores or an actual docking engine."""
        if self.config.mode == "docking":
            from .docking import DockingOracle

            return DockingOracle(self.config.docking, seed=self.config.seed, cpu=self.config.docking.cpu)
        return ReplayOracle(self.library)

    def remaining(self) -> int:
        """Number of pool compounds that have not been queried."""
        return int(np.count_nonzero(self.pool_mask & ~self.queried))

    def run_cycle(self, cycle: int, allowance: int) -> None:
        """Select, evaluate, fit and score one cycle; cycle 0 is the initial sample."""
        cycle_dir = self.output / f"cycle_{cycle:03d}"
        cycle_dir.mkdir()
        selected, before, beta = self.select_queries(cycle, cycle_dir, allowance)
        cycle_observations = self.query_oracle(cycle, cycle_dir, selected, before)
        pd.DataFrame(cycle_observations).to_csv(cycle_dir / "queries.csv", index=False)
        pd.DataFrame(self.observations).to_csv(self.output / "observations.csv", index=False)
        self.model = self.fit_model(cycle, cycle_dir)
        self.record_metrics(cycle, cycle_dir, beta)

    def select_queries(self, cycle: int, cycle_dir: Path, allowance: int) -> tuple[np.ndarray, dict, float | None]:
        """Return library indices to query, their pre-query predictions and the beta used."""
        learning = self.config.learning
        available = np.flatnonzero(self.pool_mask & ~self.queried)
        count = min(learning.initial_size if cycle == 0 else learning.batch_size, allowance, len(available))
        if cycle == 0:
            return self.initial_rng.choice(available, size=count, replace=False), {}, None
        predictions, uncertainty = self.model.predict(self.features.take(available))
        if self.config.evaluation.save_predictions:
            pd.DataFrame({"compound_id": self.library.ids(available), "prediction": predictions,
                          "uncertainty": uncertainty}).to_csv(
                              cycle_dir / "pool_predictions_before_acquisition.csv", index=False)
        beta = scheduled_beta(cycle, learning.cycles, learning.beta, learning.beta_final, learning.beta_schedule)
        # The library is ID-sorted, so positional ties equal lexical compound-ID ties.
        positions = select_indices(predictions, uncertainty, count, strategy=learning.strategy,
                                   beta=beta, rng=self.acquisition_rng)
        before = {int(available[position]): (float(predictions[position]), float(uncertainty[position]))
                  for position in positions}
        return available[positions], before, beta

    def query_oracle(self, cycle: int, cycle_dir: Path, selected: np.ndarray,
                     before: Mapping[int, tuple[float, float]]) -> list[dict[str, Any]]:
        """Evaluate the selected compounds and append every attempt to the record."""
        rows = list(zip(self.library.ids(selected), self.library.smiles(selected)))
        results = self._evaluate_rows(rows, cycle_dir / "poses")
        cycle_observations = []
        for index, (compound_id, smiles), result in zip(selected, rows, results):
            index = int(index)
            if result["status"] == "ok" and (result.get("score") is None or not np.isfinite(result["score"])):
                raise ValueError(f"oracle returned a nonfinite successful score for {compound_id}")
            observation = {
                "compound_id": str(compound_id), "smiles": smiles,
                "library_index": index, "cycle": cycle, "attempt": len(self.observations) + 1,
                "score": result.get("score"), "status": result["status"],
                "error": result.get("error"), "pose": result.get("pose"),
                "prediction_before_query": before[index][0] if cycle else None,
                "uncertainty_before_query": before[index][1] if cycle else None,
            }
            self.observations.append(observation)
            cycle_observations.append(observation)
            self.queried[index] = True
        return cycle_observations

    def _evaluate_rows(self, rows: list[tuple[str, str]], pose_dir: Path) -> list[dict[str, Any]]:
        """Run oracle evaluations, concurrently for docking when workers > 1."""
        def evaluate(row: tuple[str, str]) -> dict[str, Any]:
            return self.oracle.evaluate(str(row[0]), str(row[1]), pose_dir)

        workers = self.config.docking.workers if self.config.mode == "docking" else 1
        if workers <= 1 or len(rows) <= 1:
            return [evaluate(row) for row in rows]
        with ThreadPoolExecutor(max_workers=workers) as executor:
            return list(executor.map(evaluate, rows))

    def fit_model(self, cycle: int, cycle_dir: Path) -> Any:
        """Fit a fresh seeded model on all successful acquired labels."""
        from .regressors import create_regressor

        successful = [row for row in self.observations if row["status"] == "ok"]
        if len(successful) < 3:
            raise RuntimeError(f"only {len(successful)} successful observations; at least 3 required for training")
        self.train_indices = np.array([row["library_index"] for row in successful], dtype=np.int64)
        self.train_scores = np.array([row["score"] for row in successful], dtype=np.float64)
        # Drop the previous cycle's GPU model/cache before allocating the new fit.
        self.model = None
        model = create_regressor(self.config.model, seed=self.config.seed + cycle,
                                 device=self.runtime["device"],
                                 cpu_threads=self.config.resources.cpu_threads)
        model.fit(self.features.take(self.train_indices), self.train_scores,
                  groups=self.library.smiles(self.train_indices))
        model.save(cycle_dir / "model")
        internal = pd.DataFrame({"compound_id": self.library.ids(self.train_indices), "partition": "unused"})
        internal.loc[model.train_indices_, "partition"] = "train"
        internal.loc[model.validation_indices_, "partition"] = "validation"
        internal.to_csv(cycle_dir / "training_split.csv", index=False)
        if hasattr(model, "members_"):
            partitions = []
            for name, member in model.members_.items():
                partition = internal.copy()
                partition["model"], partition["partition"] = name, "unused"
                partition.loc[member.train_indices_, "partition"] = "train"
                partition.loc[member.validation_indices_, "partition"] = "validation"
                partitions.append(partition)
            pd.concat(partitions, ignore_index=True).to_csv(cycle_dir / "member_training_splits.csv", index=False)
        if len(getattr(model, "weight_validation_indices_", [])):
            weighting = internal.copy()
            weighting["partition"] = "weight_train"
            weighting.loc[model.weight_validation_indices_, "partition"] = "weight_validation"
            weighting.to_csv(cycle_dir / "ensemble_validation_split.csv", index=False)
        self.history.extend({"cycle": cycle, **epoch} for epoch in model.history_)
        return model

    def record_metrics(self, cycle: int, cycle_dir: Path, beta: float | None) -> None:
        """Write cycle metrics: training summary, holdout regression and replay discovery."""
        successful = len(self.train_scores)
        metrics: dict[str, Any] = {
            "cycle": cycle, "attempted": len(self.observations), "successful": successful,
            "failed": len(self.observations) - successful, "best_score": float(min(self.train_scores)),
            "internal_validation_mse_standardized": self.model.best_validation_loss_,
            "acquisition_beta": beta,
        }
        if hasattr(self.model, "weights_"):
            for name, weight in zip(self.config.model.models, self.model.weights_):
                metrics[f"ensemble_weight_{name}"] = float(weight)
            for name, mse in self.model.validation_mse_.items():
                metrics[f"ensemble_validation_mse_{name}"] = mse
        if len(self.holdout):
            heldout_scores = self.library.scores(self.holdout)
            heldout_mean, heldout_std = self.model.predict(self.features.take(self.holdout))
            pd.DataFrame({"compound_id": self.library.ids(self.holdout), "score": heldout_scores,
                          "prediction": heldout_mean, "uncertainty": heldout_std}).to_csv(
                              cycle_dir / "holdout_predictions.csv", index=False)
            metrics.update({f"holdout_{key}": value for key, value in
                            regression_metrics(heldout_scores, heldout_mean).items()})
        if self.config.mode == "replay":
            # Pool positions are sorted, so acquired library indices map by binary search.
            discovery = discovery_metrics(self.library.scores(self.pool),
                                          np.searchsorted(self.pool, self.train_indices),
                                          self.config.evaluation.top_k)
            metrics.update({f"discovery_{key}": value for key, value in discovery.items()})
        self.cycle_metrics.append(metrics)
        write_json(cycle_dir / "metrics.json", metrics)
        pd.DataFrame(self.cycle_metrics).to_csv(self.output / "metrics.csv", index=False)
        self.logger.info("Cycle %d: attempted=%d successful=%d best_score=%.4f", cycle,
                         len(self.observations), successful, metrics["best_score"])

    def finalize(self) -> dict[str, Any]:
        """Save the final model, optional library-wide predictions, history and summary."""
        self.model.save(self.output / "model")
        final = pd.DataFrame([row for row in self.observations if row["status"] == "ok"])
        final = final.sort_values(["score", "compound_id"], kind="stable")
        final.to_csv(self.output / "final_results.csv", index=False)
        if self.config.evaluation.save_predictions:
            mean, std = self.model.predict(self.features)
            predictions = self.library.frame()[["compound_id", "smiles"]].copy()
            predictions["prediction"], predictions["uncertainty"] = mean, std
            predictions["partition"] = np.where(self.pool_mask, "pool", "holdout")
            predictions["queried"] = self.queried
            predictions.to_csv(self.output / "predictions.csv", index=False)
        pd.DataFrame(self.history).to_csv(self.output / "training_history.csv", index=False)
        learning = self.config.learning
        summary = {
            "mode": self.config.mode, "seed": self.config.seed, "strategy": learning.strategy,
            "models": self.config.model.models,
            "pool_size": len(self.pool), "holdout_size": len(self.holdout), "rejected_rows": self.n_rejected,
            "budget": learning.budget, "attempted": len(self.observations), "successful": len(final),
            "cycles_completed": len(self.cycle_metrics) - 1,
            "stop_reason": "budget" if len(self.observations) >= learning.budget else
                           "pool_exhausted" if not self.remaining() else "cycles",
            "final_metrics": self.cycle_metrics[-1],
        }
        write_json(self.output / "summary.json", summary)
        return summary


def run_experiment(config: ExperimentConfig, runtime: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Run a validated experiment in a new/empty directory and return its summary.

    ``runtime`` is the result of :func:`configure_resources`; when omitted the
    resource policy is applied here. The manifest records status, and a failure
    preserves the rejection audit and diagnostics before the error propagates.
    """
    config.validate()
    output = Path(config.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output}; choose a new directory")
    if runtime is None:
        runtime = configure_resources(config.resources, config.seed)
    manifest = environment_manifest(config, runtime)
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.resolved.yaml").write_text(yaml.safe_dump(config.to_dict(), sort_keys=False), encoding="utf-8")
    write_json(output / "manifest.json", manifest)
    logger, handlers = _open_logger(output, str(id(config)))
    try:
        summary = ActiveLearningExperiment(config, runtime, output, logger).run()
        manifest.update(status="completed", completed_at=utc_now(), summary=summary,
                        artifact_sha256=artifact_hashes(output))
        write_json(output / "manifest.json", manifest)
        return summary
    except Exception as error:
        if hasattr(error, "rejections"):
            error.rejections.to_csv(output / "rejected_rows.csv", index=False)
        manifest.update(status="failed", completed_at=utc_now(), error=f"{type(error).__name__}: {error}")
        write_json(output / "manifest.json", manifest)
        logger.exception("Experiment failed")
        raise
    finally:
        for handler in handlers:
            handler.close()
            logger.removeHandler(handler)
