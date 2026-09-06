"""Validated molecular libraries and deterministic, label-free fingerprints."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold


class LibraryValidationError(ValueError):
    """Invalid library input, with an inspectable table of rejected records."""

    def __init__(self, message: str, rejections: pd.DataFrame):
        super().__init__(message)
        self.rejections = rejections


def _molecule(smiles: str) -> Chem.Mol:
    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError("missing_smiles")
    molecule = Chem.MolFromSmiles(smiles.strip())
    if molecule is None or molecule.GetNumAtoms() == 0:
        raise ValueError("invalid_smiles")
    return molecule


def canonical_record(
    identifier: str, raw_smiles: str, raw_score: str | None,
) -> tuple[str, str | None, float | None, str | None, Chem.Mol | None]:
    """Validate one input row; return ``(id, canonical_smiles, score, reason, molecule)``.

    ``reason`` is ``None`` for an accepted row. Rejection reasons are
    ``missing_compound_id``, ``missing_smiles``, ``invalid_smiles``,
    ``invalid_score`` and ``nonfinite_score``; the score is checked only when
    ``raw_score`` is supplied. The parsed molecule is returned so callers can
    featurize without parsing the structure a second time.
    """
    compound_id = identifier.strip()
    if not compound_id:
        return compound_id, None, None, "missing_compound_id", None
    try:
        molecule = _molecule(raw_smiles)
    except ValueError as exc:
        return compound_id, None, None, str(exc), None
    smiles = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
    score = None
    if raw_score is not None:
        try:
            score = float(raw_score)
        except (ValueError, TypeError):
            return compound_id, smiles, None, "invalid_score", molecule
        if not np.isfinite(score):
            return compound_id, smiles, None, "nonfinite_score", molecule
    return compound_id, smiles, score, None, molecule


def _csv_records(path: Path, columns: list[str]):
    """Yield selected fields only after validating each complete logical record."""
    record_number = 1
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle, strict=True)
        try:
            header = next(reader, None)
            if not header:
                raise ValueError(f"CSV has no header or records: {path}:1")
            if len(set(header)) != len(header):
                raise ValueError(f"Duplicate CSV header at {path}:1")
            missing = set(columns) - set(header)
            if missing:
                raise ValueError(f"Missing required columns in {path}:1: {sorted(missing)}")
            positions = [header.index(column) for column in columns]
            record_number = 2
            for row in reader:
                if len(row) != len(header):
                    raise ValueError(
                        f"CSV field count mismatch at {path}:{record_number}: "
                        f"expected {len(header)}, got {len(row)}"
                    )
                yield record_number, [row[index] for index in positions]
                record_number += 1
        except csv.Error as exc:
            raise ValueError(f"Malformed CSV at {path}:{record_number}: {exc}") from exc


def load_library(
    path: str | Path,
    id_column: str = "Catalog ID",
    smiles_column: str = "smiles",
    score_column: str | None = "docking_score",
    invalid_policy: str = "error",
    deduplicate: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read one CSV or a sorted directory of CSVs and return library, rejections.

    Output columns are ``compound_id``, canonical isomeric ``smiles``, ``score``
    (only when requested), ``source_file`` and ``source_row``. Other input
    columns are deliberately excluded. Source rows are 1-based CSV record
    numbers, counting the header as record 1. Input identifiers remain strings.

    Invalid rows raise by default; ``skip`` records and removes them. Conflicting
    identifiers or scores for the same canonical molecule always raise, because
    choosing one silently would change the scientific target. Exact duplicate
    IDs are always removed; duplicate structures with different IDs are removed
    when ``deduplicate`` is true, retaining the lexically smallest valid ID and
    its original source record. CSV syntax, duplicate headers and record widths
    are validated even under ``skip``. Stereoisomers remain distinct. An empty result
    always raises ``LibraryValidationError`` with its ``rejections`` table.
    """
    if invalid_policy not in {"error", "skip"}:
        raise ValueError("invalid_policy must be 'error' or 'skip'")
    if not isinstance(deduplicate, bool):
        raise ValueError("deduplicate must be a boolean")
    columns = [id_column, smiles_column]
    if score_column is not None:
        columns.append(score_column)
    if any(not isinstance(column, str) or not column for column in columns):
        raise ValueError("column names must be nonempty strings")
    if len(set(columns)) != len(columns):
        raise ValueError("identifier, SMILES and score columns must be distinct")
    source = Path(path).expanduser().resolve()
    files = sorted(source.glob("*.csv")) if source.is_dir() else [source]
    if not files:
        raise ValueError(f"No CSV files in {source}")
    if not source.exists():
        raise FileNotFoundError(source)

    output_columns = ["compound_id", "smiles"]
    if score_column is not None:
        output_columns.append("score")
    output_columns.extend(["source_file", "source_row"])
    rejection_columns = output_columns + ["reason", "kept_compound_id"]
    accepted: list[dict] = []
    rejected: list[dict] = []
    identifiers: dict[str, dict] = {}
    structures: dict[str, dict] = {}

    def rejection_frame() -> pd.DataFrame:
        if deduplicate:
            for record in rejected:
                if record["reason"].startswith("duplicate_"):
                    record["kept_compound_id"] = structures[record["smiles"]]["compound_id"]
        return pd.DataFrame(
            sorted(rejected, key=lambda record: (record["source_file"], record["source_row"])),
            columns=rejection_columns,
        )

    def reject(record: dict, reason: str, kept: str = "", fatal: bool = False) -> None:
        rejected.append({**record, "reason": reason, "kept_compound_id": kept})
        if fatal or (invalid_policy == "error" and not reason.startswith("duplicate_")):
            raise LibraryValidationError(
                f"{reason} at {record['source_file']}:{record['source_row']} "
                f"(compound_id={record['compound_id']!r})",
                rejection_frame(),
            )

    for csv_file in files:
        for row_number, values in _csv_records(csv_file, columns):
            identifier, raw_smiles = values[:2]
            raw_score = values[2] if score_column is not None else None
            compound_id, smiles, score, reason, _ = canonical_record(identifier, raw_smiles, raw_score)
            record = {
                "compound_id": compound_id,
                "smiles": smiles if smiles is not None else raw_smiles,
                "source_file": str(csv_file),
                "source_row": row_number,
            }
            if score_column is not None:
                record["score"] = score if score is not None else raw_score
            if reason is not None:
                reject(record, reason)
                continue

            previous = identifiers.get(record["compound_id"])
            if previous is not None:
                kept_id = structures[previous["smiles"]]["compound_id"] if deduplicate else previous["compound_id"]
                if previous["smiles"] != record["smiles"] or previous.get("score") != record.get("score"):
                    reject(record, "conflicting_compound_id", kept_id, fatal=True)
                reject(record, "duplicate_compound_id", kept_id)
                continue
            previous = structures.get(record["smiles"])
            if previous is not None and previous.get("score") != record.get("score"):
                reject(record, "conflicting_structure_score", previous["compound_id"], fatal=True)
            identifiers[record["compound_id"]] = record
            if deduplicate and previous is not None:
                if record["compound_id"] < previous["compound_id"]:
                    structures[record["smiles"]] = record
                    reject(previous, "duplicate_structure", record["compound_id"])
                else:
                    reject(record, "duplicate_structure", previous["compound_id"])
                continue
            structures[record["smiles"]] = record
            accepted.append(record)

    if deduplicate:
        accepted = list(structures.values())
    rejections = rejection_frame()
    if not accepted:
        raise LibraryValidationError("No valid compounds remain after validation", rejections)
    library = pd.DataFrame(accepted, columns=output_columns)
    # Stable identities make random seeds and tied scores independent of CSV row order.
    library = library.sort_values("compound_id", kind="stable").reset_index(drop=True)
    logging.getLogger(__name__).info(
        "Loaded %d compounds from %d CSV files; rejected %d records",
        len(library), len(files), len(rejections),
    )
    if len(rejections):
        logging.getLogger(__name__).warning("Library rejections: %s", rejections["reason"].value_counts().to_dict())
    return library, rejections


FINGERPRINT_TYPES = ("morgan", "feature_morgan")


def fingerprint_generator(radius: int, n_bits: int, fingerprint_type: str = "morgan"):
    """Return the chirality-aware RDKit Morgan generator for the requested variant."""
    if isinstance(radius, bool) or not isinstance(radius, (int, np.integer)) or radius < 0:
        raise ValueError("radius must be a nonnegative integer")
    if isinstance(n_bits, bool) or not isinstance(n_bits, (int, np.integer)) or n_bits < 1:
        raise ValueError("n_bits must be a positive integer")
    if fingerprint_type not in FINGERPRINT_TYPES:
        raise ValueError(f"fingerprint_type must be one of {FINGERPRINT_TYPES}")
    invariants = (
        rdFingerprintGenerator.GetMorganFeatureAtomInvGen()
        if fingerprint_type == "feature_morgan"
        else None
    )
    return rdFingerprintGenerator.GetMorganGenerator(
        radius=int(radius), fpSize=int(n_bits), includeChirality=True,
        atomInvariantsGenerator=invariants,
    )


def featurize(
    smiles: Iterable[str],
    radius: int = 2,
    n_bits: int = 2048,
    fingerprint_type: str = "morgan",
) -> np.ndarray:
    """Return binary circular fingerprints as a ``uint8`` matrix with chirality.

    ``morgan`` uses connectivity invariants (ECFP-like) and ``feature_morgan``
    uses pharmacophoric feature invariants (FCFP-like); both share ``radius``
    and ``n_bits``. One byte per bit keeps large libraries in memory, and the
    model converts each batch to ``float32`` exactly. Only molecular structures
    enter this transformation; target scores and compound identifiers cannot
    become model features. Invalid structures fail instead of becoming all-zero
    vectors. Empty input has shape ``(0, n_bits)``.
    """
    generator = fingerprint_generator(radius, n_bits, fingerprint_type)
    if isinstance(smiles, str):
        raise ValueError("smiles must be a sequence, not one string")
    structures = list(smiles)
    features = np.empty((len(structures), n_bits), dtype=np.uint8)
    for index, structure in enumerate(structures):
        try:
            molecule = _molecule(structure)
        except ValueError as exc:
            raise ValueError(f"Invalid SMILES at feature row {index}: {structure!r}") from exc
        features[index] = generator.GetFingerprintAsNumPy(molecule)
    return features


def scaffold_keys(smiles: Iterable[str]) -> list[str]:
    """Return achiral Bemis–Murcko keys; all acyclic molecules share the empty key."""
    return [
        MurckoScaffold.MurckoScaffoldSmiles(mol=_molecule(value), includeChirality=False)
        for value in smiles
    ]


def split_holdout(
    smiles: Sequence[str],
    fraction: float,
    seed: int,
    strategy: str = "random",
) -> tuple[np.ndarray, np.ndarray]:
    """Return sorted candidate and holdout indices without looking at labels.

    Random splitting groups identical canonical structures. Scaffold splitting
    groups achiral Bemis–Murcko scaffolds, including one group for all acyclic
    compounds. Seeded groups are added to holdout until ``ceil(N * fraction)``
    is reached, always retaining at least one candidate group. Thus group sizes
    can make the actual holdout fraction differ from the request. A nonzero
    split with fewer than two groups raises rather than leaking structures.
    """
    if strategy not in {"random", "scaffold"}:
        raise ValueError("strategy must be 'random' or 'scaffold'")
    if not np.isfinite(fraction) or not 0 <= fraction < 1:
        raise ValueError("holdout fraction must be finite and in [0, 1)")
    if not len(smiles):
        raise ValueError("Cannot split an empty library")
    indices = np.arange(len(smiles), dtype=np.int64)
    if fraction == 0:
        return indices, np.empty(0, dtype=np.int64)
    keys = scaffold_keys(smiles) if strategy == "scaffold" else [
        Chem.MolToSmiles(_molecule(value), canonical=True, isomericSmiles=True)
        for value in smiles
    ]
    groups: dict[str, list[int]] = {}
    for index, key in enumerate(keys):
        groups.setdefault(key, []).append(index)
    if len(groups) < 2:
        raise ValueError(f"{strategy} holdout needs at least two distinct molecular groups")
    group_keys = sorted(groups)
    order = np.random.default_rng(seed).permutation(len(group_keys))
    target = int(np.ceil(len(smiles) * fraction))
    heldout: list[int] = []
    for group_index in order[:-1]:
        heldout.extend(groups[group_keys[group_index]])
        if len(heldout) >= target:
            break
    holdout = np.sort(np.asarray(heldout, dtype=np.int64))
    return np.setdiff1d(indices, holdout, assume_unique=True), holdout
