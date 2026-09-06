"""Disk-backed canonical libraries and bit-packed fingerprints for very large pools.

A library store is a directory of ID-sorted NumPy columns (``compound_id``,
``smiles``, ``source``, optional ``scores``) plus ``fingerprints.npy`` with one
bit per fingerprint position. Columns are memory-mapped at run time, so a
10^8-compound pool costs disk rather than RAM, and the model streams unpacked
batches from :class:`PackedFingerprints`. The same validation, canonicalization,
deduplication and representative rules as :func:`data.load_library` apply.
"""

from __future__ import annotations

from array import array
from collections import deque
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
from multiprocessing import get_context
from pathlib import Path
import shutil
from typing import Any, Iterator, Sequence

import numpy as np
import pandas as pd

from . import __version__
from .data import FINGERPRINT_TYPES, LibraryValidationError, _csv_records, canonical_record, fingerprint_generator
from .provenance import sha256_file, validate_new_paths, write_json

STORE_VERSION = 1
STORE_METADATA = "library.json"
_DIGEST_BYTES = 16
_DUPLICATE_REASONS = ("duplicate_compound_id", "duplicate_structure")


class PackedFingerprints:
    """Bit-packed binary fingerprints with lazy, batch-wise row access.

    ``packed`` has shape ``(n, ceil(n_bits / 8))`` and may be a memory map.
    ``rows`` optionally selects a subset without copying bits. Indexing with a
    slice, an integer index array or a boolean mask returns unpacked ``uint8``
    rows of shape ``(m, n_bits)``; :meth:`take` returns another lazy view.
    Negative indices are rejected so that both entry points agree. The model
    treats this object as a feature matrix and materializes one batch at a time.
    """

    ndim = 2

    def __init__(self, packed: np.ndarray, n_bits: int, rows: np.ndarray | None = None) -> None:
        if isinstance(n_bits, bool) or not isinstance(n_bits, (int, np.integer)) or n_bits < 1:
            raise ValueError("n_bits must be a positive integer")
        if packed.ndim != 2 or packed.dtype != np.uint8 or packed.shape[1] != -(-int(n_bits) // 8):
            raise ValueError("packed fingerprints must be a uint8 matrix with ceil(n_bits / 8) columns")
        self.packed = packed
        self.n_bits = int(n_bits)
        self.rows = None if rows is None else self._positions(rows, len(packed))
        self.shape = (len(packed) if self.rows is None else len(self.rows), self.n_bits)

    @staticmethod
    def _positions(indices: Any, limit: int) -> np.ndarray:
        indices = np.asarray(indices)
        if indices.dtype == bool:
            if indices.shape != (limit,):
                raise IndexError("boolean mask length must equal the number of rows")
            return np.flatnonzero(indices)
        if indices.dtype.kind not in "iu":
            raise IndexError("fingerprint rows must be selected with integer indices or a boolean mask")
        indices = indices.reshape(-1).astype(np.int64, copy=False)
        if len(indices) and (indices.min() < 0 or indices.max() >= limit):
            raise IndexError("fingerprint row index out of range")
        return indices

    def __len__(self) -> int:
        return self.shape[0]

    def take(self, indices: Sequence[int]) -> PackedFingerprints:
        """Return a lazy view of the selected rows without unpacking them."""
        positions = self._positions(indices, len(self))
        rows = positions if self.rows is None else self.rows[positions]
        return PackedFingerprints(self.packed, self.n_bits, rows)

    def __getitem__(self, item: Any) -> np.ndarray:
        if isinstance(item, slice):
            source = self.rows[item] if self.rows is not None else item
        else:
            positions = self._positions(item, len(self))
            source = self.rows[positions] if self.rows is not None else positions
        block = np.asarray(self.packed[source], dtype=np.uint8).reshape(-1, self.packed.shape[1])
        unpacked = np.unpackbits(block, axis=1)
        return unpacked if self.n_bits == unpacked.shape[1] else np.ascontiguousarray(unpacked[:, :self.n_bits])

    def dense(self) -> np.ndarray:
        """Materialize every row as an unpacked ``uint8`` matrix."""
        return self[0:len(self)]

    @classmethod
    def from_dense(cls, features: np.ndarray) -> PackedFingerprints:
        """Pack a binary ``(n, n_bits)`` matrix; values must be 0 or 1."""
        features = np.asarray(features)
        if features.ndim != 2 or features.shape[1] == 0:
            raise ValueError("features must have shape (n_samples, n_bits) with n_bits > 0")
        if features.dtype.kind not in "biu" or (features.size and (features.min() < 0 or features.max() > 1)):
            raise ValueError("features must be binary (0/1) integers")
        return cls(np.packbits(features.astype(np.uint8, copy=False), axis=1), features.shape[1])

    @classmethod
    def open(cls, path: str | Path, n_bits: int) -> PackedFingerprints:
        """Memory-map a packed ``.npy`` matrix written by the store builder."""
        return cls(np.load(Path(path), mmap_mode="r"), n_bits)


class StringColumn(Sequence):
    """String column backed by an object array or a fixed-width bytes memory map."""

    def __init__(self, values: np.ndarray) -> None:
        if values.ndim != 1 or values.dtype.kind not in "OS":
            raise ValueError("string columns must be 1D object or fixed-width bytes arrays")
        self.values = values

    def __len__(self) -> int:
        return len(self.values)

    def take(self, indices: Sequence[int] | slice | None = None) -> np.ndarray:
        """Return the selected values as a Python-string object array."""
        block = self.values if indices is None else self.values[indices]
        if block.dtype.kind == "S":
            return np.char.decode(block, "utf-8").astype(object)
        return np.asarray(block, dtype=object)

    def __getitem__(self, item: Any) -> Any:
        if isinstance(item, (int, np.integer)):
            value = self.values[int(item)]
            return value.decode("utf-8") if isinstance(value, bytes) else str(value)
        return self.take(item)

    def __iter__(self) -> Iterator[str]:
        for index in range(len(self)):
            yield self[index]


class Library:
    """Canonical, ID-sorted library columns from a DataFrame or a library store.

    Positional order equals lexical compound-ID order, so positions are valid
    deterministic tie keys for acquisition and discovery metrics.
    """

    def __init__(self, compound_id: StringColumn, smiles: StringColumn, score: np.ndarray | None,
                 origin: str, store: dict[str, Any] | None = None) -> None:
        if len(compound_id) != len(smiles) or (score is not None and len(score) != len(smiles)):
            raise ValueError("library columns must have equal length")
        self.compound_id, self.smiles_column, self.score = compound_id, smiles, score
        self.origin, self.store = origin, store

    def __len__(self) -> int:
        return len(self.compound_id)

    def ids(self, indices: Sequence[int] | None = None) -> np.ndarray:
        return self.compound_id.take(indices)

    def smiles(self, indices: Sequence[int] | None = None) -> np.ndarray:
        return self.smiles_column.take(indices)

    def scores(self, indices: Sequence[int] | None = None) -> np.ndarray:
        if self.score is None:
            raise ValueError("this library has no recorded scores")
        return np.asarray(self.score if indices is None else self.score[indices], dtype=np.float64)

    def frame(self, indices: Sequence[int] | None = None) -> pd.DataFrame:
        """Return ``compound_id``, ``smiles`` and, when available, ``score`` rows."""
        frame = pd.DataFrame({"compound_id": self.ids(indices), "smiles": self.smiles(indices)})
        if self.score is not None:
            frame["score"] = self.scores(indices)
        return frame

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> Library:
        score = frame["score"].to_numpy(dtype=np.float64) if "score" in frame.columns else None
        return cls(StringColumn(frame["compound_id"].to_numpy(dtype=object)),
                   StringColumn(frame["smiles"].to_numpy(dtype=object)), score, origin="csv")


def is_library_store(path: str | Path) -> bool:
    """True when ``path`` is a directory containing store metadata."""
    return (Path(path) / STORE_METADATA).is_file()


def open_library_store(directory: str | Path, verify: bool = True) -> tuple[Library, PackedFingerprints, dict[str, Any]]:
    """Memory-map a store after checking its version, file sizes and, by default, checksums.

    ``verify`` recomputes every recorded SHA-256 so same-size corruption or a
    partially copied store cannot pass silently; it costs one sequential read
    of the store files.
    """
    directory = Path(directory).resolve()
    metadata = json.loads((directory / STORE_METADATA).read_text(encoding="utf-8"))
    if metadata.get("store_version") != STORE_VERSION:
        raise ValueError(f"unsupported library store version in {directory}")
    for name, info in metadata["files"].items():
        path = directory / name
        if not path.is_file() or path.stat().st_size != info["bytes"]:
            raise ValueError(f"library store file is missing or has an unexpected size: {path}")
        if verify and sha256_file(path) != info["sha256"]:
            raise ValueError(f"library store file checksum mismatch: {path}")
    compound_id = StringColumn(np.load(directory / "compound_id.npy", mmap_mode="r"))
    smiles = StringColumn(np.load(directory / "smiles.npy", mmap_mode="r"))
    score = np.load(directory / "scores.npy", mmap_mode="r") if "scores.npy" in metadata["files"] else None
    library = Library(compound_id, smiles, score, origin="store", store=metadata)
    features = PackedFingerprints.open(directory / "fingerprints.npy", metadata["fingerprint"]["n_bits"])
    if len(features) != len(library) or len(library) != metadata["n_compounds"]:
        raise ValueError("library store columns disagree on the number of compounds")
    return library, features, metadata


_WORKER: dict[str, Any] = {}


def _init_worker(radius: int, n_bits: int, fingerprint_type: str) -> None:
    _WORKER["generator"] = fingerprint_generator(radius, n_bits, fingerprint_type)


def _process_chunk(chunk: list[tuple[int, str, str, str | None]]) -> list[tuple]:
    """Canonicalize, score-check and fingerprint one chunk of CSV records."""
    generator = _WORKER["generator"]
    results = []
    for row_number, identifier, raw_smiles, raw_score in chunk:
        compound_id, smiles, score, reason, molecule = canonical_record(identifier, raw_smiles, raw_score)
        packed = digest = b""
        if reason is None:
            packed = np.packbits(generator.GetFingerprintAsNumPy(molecule).astype(np.uint8)).tobytes()
            digest = hashlib.blake2b(smiles.encode("utf-8"), digest_size=_DIGEST_BYTES).digest()
        results.append((row_number, compound_id, smiles if smiles is not None else raw_smiles,
                        score if score is not None else raw_score, reason, packed, digest))
    return results


def _chunked_records(files: Sequence[Path], columns: list[str], chunk_size: int) -> Iterator[tuple[int, list]]:
    for file_index, csv_file in enumerate(files):
        chunk: list[tuple] = []
        for row_number, values in _csv_records(csv_file, columns):
            chunk.append((row_number, values[0], values[1], values[2] if len(values) > 2 else None))
            if len(chunk) >= chunk_size:
                yield file_index, chunk
                chunk = []
        if chunk:
            yield file_index, chunk


def _processed_chunks(files: Sequence[Path], columns: list[str], chunk_size: int, workers: int,
                      fingerprint: tuple[int, int, str]) -> Iterator[tuple[int, list[tuple]]]:
    """Yield processed chunks in input order with bounded in-flight work.

    Worker processes are spawned fresh; a worker that dies (for example from an
    out-of-memory kill) raises ``BrokenProcessPool`` instead of hanging the build.
    """
    if workers == 1:
        _init_worker(*fingerprint)
        for file_index, chunk in _chunked_records(files, columns, chunk_size):
            yield file_index, _process_chunk(chunk)
        return
    with ProcessPoolExecutor(workers, mp_context=get_context("spawn"), initializer=_init_worker,
                             initargs=fingerprint) as pool:
        pending: deque = deque()
        for file_index, chunk in _chunked_records(files, columns, chunk_size):
            pending.append((file_index, pool.submit(_process_chunk, chunk)))
            while len(pending) >= 2 * workers:
                index, future = pending.popleft()
                yield index, future.result()
        while pending:
            index, future = pending.popleft()
            yield index, future.result()


class _Accumulator:
    """Accepted-row columns kept in RAM plus temporary SMILES/fingerprint files.

    Invalid-row rejections stream to a temporary CSV in input order so that a
    library with many rejected records does not hold them all in memory.
    """

    def __init__(self, temp: Path, n_bytes: int, scored: bool, rejection_columns: list[str]) -> None:
        self.n_bytes, self.scored = n_bytes, scored
        self.ids: list[np.ndarray] = []
        self.digests: list[np.ndarray] = []
        self.file_index: list[np.ndarray] = []
        self.rows: list[np.ndarray] = []
        self.lengths: list[np.ndarray] = []
        self.scores: list[np.ndarray] = []
        self.smiles_file = (temp / "smiles.bin").open("wb")
        self.fingerprint_file = (temp / "fingerprints.bin").open("wb")
        self.rejection_path = temp / "invalid_rows.csv"
        self.rejection_file = self.rejection_path.open("w", newline="", encoding="utf-8")
        self.rejection_writer = csv.DictWriter(self.rejection_file, fieldnames=rejection_columns, lineterminator="\n")
        self.rejection_writer.writeheader()
        self.count = self.n_invalid = self.n_records = 0

    def reject(self, entry: dict[str, Any]) -> None:
        self.rejection_writer.writerow(entry)
        self.n_invalid += 1

    def invalid_rows(self) -> pd.DataFrame:
        if not self.rejection_file.closed:
            self.rejection_file.flush()
        return pd.read_csv(self.rejection_path, dtype={"compound_id": str, "smiles": str, "kept_compound_id": str},
                           keep_default_na=False)

    def add(self, file_index: int, accepted: list[tuple]) -> None:
        if not accepted:
            return
        self.ids.append(np.array([record[1].encode("utf-8") for record in accepted], dtype="S"))
        self.digests.append(np.array([record[6] for record in accepted], dtype=f"S{_DIGEST_BYTES}"))
        self.file_index.append(np.full(len(accepted), file_index, dtype=np.int64))
        self.rows.append(np.array([record[0] for record in accepted], dtype=np.int64))
        encoded = [record[2].encode("utf-8") for record in accepted]
        self.lengths.append(np.array([len(value) for value in encoded], dtype=np.int64))
        if self.scored:
            self.scores.append(np.array([record[3] for record in accepted], dtype=np.float64))
        self.smiles_file.write(b"".join(encoded))
        self.fingerprint_file.write(b"".join(record[5] for record in accepted))
        self.count += len(accepted)

    def close(self) -> None:
        self.smiles_file.close()
        self.fingerprint_file.close()
        self.rejection_file.close()

    def column(self, name: str) -> np.ndarray:
        chunks = getattr(self, name)
        empty = {"ids": "S1", "digests": f"S{_DIGEST_BYTES}", "scores": np.float64}.get(name, np.int64)
        return np.concatenate(chunks) if chunks else np.empty(0, dtype=empty)


def _write_fixed_width(path: Path, source: np.memmap, offsets: np.ndarray, lengths: np.ndarray,
                       order: np.ndarray, block: int = 1_000_000) -> int:
    """Gather variable-length byte strings into a fixed-width ``.npy`` column."""
    width = max(1, int(lengths[order].max())) if len(order) else 1
    target = np.lib.format.open_memmap(path, mode="w+", dtype=f"S{width}", shape=(len(order),))
    for start in range(0, len(order), block):
        positions = order[start:start + block]
        target[start:start + len(positions)] = [
            bytes(source[offsets[position]:offsets[position] + lengths[position]]) for position in positions
        ]
    target.flush()
    del target
    return width


def _repeated_runs(positions: np.ndarray, keys: np.ndarray) -> Iterator[np.ndarray]:
    """Yield position groups whose sorted 1D ``keys`` repeat (runs longer than one)."""
    if len(positions) < 2:
        return
    breaks = np.flatnonzero(keys[1:] != keys[:-1]) + 1
    starts, stops = np.r_[0, breaks], np.r_[breaks, len(positions)]
    for start, stop in zip(starts[stops - starts > 1], stops[stops - starts > 1]):
        yield positions[start:stop]


class _Duplicates:
    """Compact duplicate-rejection log: position, reason code and kept position."""

    def __init__(self) -> None:
        self.position, self.reason, self.kept = array("q"), array("b"), array("q")

    def add(self, position: int, reason: int, kept: int) -> None:
        self.position.append(int(position))
        self.reason.append(reason)
        self.kept.append(int(kept))

    def __len__(self) -> int:
        return len(self.position)

    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (np.frombuffer(self.position, dtype=np.int64), np.frombuffer(self.reason, dtype=np.int8),
                np.frombuffer(self.kept, dtype=np.int64))


def build_library_store(
    path: str | Path,
    output: str | Path,
    id_column: str = "Catalog ID",
    smiles_column: str = "smiles",
    score_column: str | None = None,
    invalid_policy: str = "error",
    deduplicate: bool = True,
    fingerprint_type: str = "morgan",
    radius: int = 2,
    n_bits: int = 2048,
    workers: int = 1,
    chunk_size: int = 20_000,
) -> dict[str, Any]:
    """Stream one CSV or a directory of CSVs into a new ID-sorted library store.

    Rows are validated, canonicalized and fingerprinted in ``workers`` worker
    processes with bounded in-flight chunks; only identifiers, structure
    digests, source positions and scores stay in RAM, while SMILES, packed
    fingerprints and invalid-row rejections stream through temporary files.
    Deduplication follows :func:`data.load_library`: exact duplicate IDs are
    removed, conflicting IDs or conflicting scores for one structure are fatal,
    and duplicate structures keep the lexically smallest ID when ``deduplicate``
    is true. A failure removes every generated file before the error propagates.
    Returns the metadata written to ``library.json``.
    """
    if invalid_policy not in {"error", "skip"}:
        raise ValueError("invalid_policy must be 'error' or 'skip'")
    if not isinstance(deduplicate, bool):
        raise ValueError("deduplicate must be a boolean")
    if fingerprint_type not in FINGERPRINT_TYPES:
        raise ValueError(f"fingerprint_type must be one of {FINGERPRINT_TYPES}")
    if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= 9:
        raise ValueError("workers must be an integer between 1 and 9")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    fingerprint_generator(radius, n_bits, fingerprint_type)
    columns = [id_column, smiles_column] + ([score_column] if score_column is not None else [])
    if any(not isinstance(column, str) or not column for column in columns) or len(set(columns)) != len(columns):
        raise ValueError("identifier, SMILES and score columns must be distinct nonempty strings")
    source = Path(path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    files = sorted(source.glob("*.csv")) if source.is_dir() else [source]
    if not files:
        raise ValueError(f"No CSV files in {source}")
    output = Path(output).expanduser()
    names = ["compound_id.npy", "smiles.npy", "source.npy", "fingerprints.npy", "rejected_rows.csv", STORE_METADATA]
    if score_column is not None:
        names.append("scores.npy")
    created_output = not output.exists()
    directories = [output / "tmp"] + ([] if output.is_dir() and not any(output.iterdir()) else [output])
    validate_new_paths([output / name for name in names], inputs=files, directories=directories)
    output.mkdir(parents=True, exist_ok=True)
    temp = output / "tmp"
    temp.mkdir()
    scored = score_column is not None
    rejection_columns = ["compound_id", "smiles"] + (["score"] if scored else []) + [
        "source_file", "source_row", "reason", "kept_compound_id"]
    accumulator = _Accumulator(temp, -(-int(n_bits) // 8), scored, rejection_columns)
    settings = {"id_column": id_column, "smiles_column": smiles_column, "score_column": score_column,
                "invalid_policy": invalid_policy, "deduplicate": deduplicate, "workers": workers}
    fingerprint = {"type": fingerprint_type, "radius": int(radius), "n_bits": int(n_bits)}
    try:
        for file_index, results in _processed_chunks(files, columns, chunk_size, workers,
                                                     (radius, n_bits, fingerprint_type)):
            accepted = []
            accumulator.n_records += len(results)
            for record in results:
                row_number, compound_id, smiles, score, reason = record[:5]
                if reason is None:
                    accepted.append(record)
                    continue
                entry = {"compound_id": compound_id, "smiles": smiles, "source_file": str(files[file_index]),
                         "source_row": row_number, "reason": reason, "kept_compound_id": ""}
                if scored:
                    entry["score"] = score
                accumulator.reject(entry)
                if invalid_policy == "error":
                    raise LibraryValidationError(
                        f"{reason} at {entry['source_file']}:{entry['source_row']} (compound_id={compound_id!r})",
                        accumulator.invalid_rows())
            accumulator.add(file_index, accepted)
        accumulator.close()
        metadata = _finish_store(output, temp, files, accumulator, deduplicate, fingerprint, settings)
    except BaseException:
        accumulator.close()
        shutil.rmtree(temp, ignore_errors=True)
        for name in names:
            (output / name).unlink(missing_ok=True)
        if created_output and output.is_dir() and not any(output.iterdir()):
            output.rmdir()
        raise
    shutil.rmtree(temp, ignore_errors=True)
    return metadata


def _finish_store(output: Path, temp: Path, files: Sequence[Path], acc: _Accumulator, deduplicate: bool,
                  fingerprint: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    """Deduplicate accepted rows, write ID-sorted columns and the metadata file."""
    ids, digests = acc.column("ids"), acc.column("digests")
    file_index, rows, lengths = acc.column("file_index"), acc.column("rows"), acc.column("lengths")
    scores = acc.column("scores") if acc.scored else None
    count = len(ids)
    if count == 0:
        raise LibraryValidationError("No valid compounds remain after validation", acc.invalid_rows())
    offsets = np.zeros(count + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    smiles_bytes = np.memmap(temp / "smiles.bin", dtype=np.uint8, mode="r")
    duplicates = _Duplicates()

    def smiles_at(position: int) -> str:
        return bytes(smiles_bytes[offsets[position]:offsets[position + 1]]).decode("utf-8")

    def record(position: int) -> dict[str, Any]:
        entry = {"compound_id": ids[position].decode("utf-8"), "smiles": smiles_at(position),
                 "source_file": str(files[file_index[position]]), "source_row": int(rows[position])}
        if scores is not None:
            entry["score"] = float(scores[position])
        return entry

    def rejection_rows(positions: np.ndarray, reasons: np.ndarray, kept: np.ndarray) -> Iterator[dict[str, Any]]:
        for position, reason, kept_position in zip(positions, reasons, kept):
            yield {**record(position), "reason": _DUPLICATE_REASONS[reason],
                   "kept_compound_id": ids[kept_position].decode("utf-8")}

    def fatal(reason: str, position: int, kept_position: int) -> LibraryValidationError:
        entry = {**record(position), "reason": reason, "kept_compound_id": ids[kept_position].decode("utf-8")}
        frames = [acc.invalid_rows(), pd.DataFrame(list(rejection_rows(*duplicates.arrays())) + [entry])]
        frame = pd.concat([frame for frame in frames if len(frame)])
        frame = frame.sort_values(["source_file", "source_row"], kind="stable").reset_index(drop=True)
        return LibraryValidationError(
            f"{reason} at {entry['source_file']}:{entry['source_row']} (compound_id={entry['compound_id']!r})", frame)

    def same_score(first: int, second: int) -> bool:
        return scores is None or scores[first] == scores[second]

    dropped = np.zeros(count, dtype=bool)
    by_id = np.argsort(ids, kind="stable")
    for group in _repeated_runs(by_id, ids[by_id]):
        first = group[0]
        for later in group[1:]:
            if not (digests[later] == digests[first] and smiles_at(later) == smiles_at(first)
                    and same_score(first, later)):
                raise fatal("conflicting_compound_id", later, first)
            duplicates.add(later, 0, first)
            dropped[later] = True
    kept = np.flatnonzero(~dropped)
    by_structure = kept[np.lexsort((ids[kept], digests[kept]))]
    for group in _repeated_runs(by_structure, digests[by_structure]):
        reference = smiles_at(group[0])
        if any(smiles_at(member) != reference for member in group[1:]):
            raise RuntimeError("structure digest collision; rebuild with a different library")
        in_order = sorted(group.tolist(), key=lambda member: (file_index[member], rows[member]))
        for previous, current in zip(in_order, in_order[1:]):
            if not same_score(previous, current):
                raise fatal("conflicting_structure_score", current, previous)
        if deduplicate:
            for member in group[1:]:
                duplicates.add(member, 1, group[0])
                dropped[member] = True
    positions, reasons, kept_positions = duplicates.arrays()
    if deduplicate and len(positions):
        # Duplicate IDs point at the final representative of their structure (smallest ID).
        sorted_digests = digests[by_structure]
        representative = by_structure[np.searchsorted(sorted_digests, digests[kept_positions])]
        kept_positions = np.where(reasons == 0, representative, kept_positions)
    final = np.flatnonzero(~dropped)
    order = final[np.argsort(ids[final], kind="stable")]
    by_source = np.lexsort((rows[positions], file_index[positions]))
    _write_rejections(output / "rejected_rows.csv", acc,
                      rejection_rows(positions[by_source], reasons[by_source], kept_positions[by_source]))
    id_width = max(1, int(np.char.str_len(ids[order]).max()))
    np.save(output / "compound_id.npy", ids[order].astype(f"S{id_width}"))
    smiles_width = _write_fixed_width(output / "smiles.npy", smiles_bytes, offsets[:-1], lengths, order)
    np.save(output / "source.npy", np.stack([file_index[order], rows[order]], axis=1))
    if scores is not None:
        np.save(output / "scores.npy", scores[order])
    packed = np.memmap(temp / "fingerprints.bin", dtype=np.uint8, mode="r", shape=(count, acc.n_bytes))
    target = np.lib.format.open_memmap(output / "fingerprints.npy", mode="w+", dtype=np.uint8,
                                       shape=(len(order), acc.n_bytes))
    for start in range(0, len(order), 1_000_000):
        target[start:start + 1_000_000] = packed[order[start:start + 1_000_000]]
    target.flush()
    del target, packed
    names = ["compound_id.npy", "smiles.npy", "source.npy", "fingerprints.npy", "rejected_rows.csv"]
    if scores is not None:
        names.append("scores.npy")
    metadata = {
        "store_version": STORE_VERSION, "generator_version": __version__,
        "n_compounds": int(len(order)), "n_rejected": acc.n_invalid + len(positions),
        "n_input_records": acc.n_records,
        "id_width": id_width, "smiles_width": smiles_width, "fingerprint": fingerprint, "settings": settings,
        "inputs": {str(csv_file): sha256_file(csv_file) for csv_file in files},
        "source_files": [str(csv_file) for csv_file in files],
        "files": {name: {"sha256": sha256_file(output / name), "bytes": (output / name).stat().st_size}
                  for name in names},
    }
    write_json(output / STORE_METADATA, metadata)
    return metadata


def _write_rejections(path: Path, acc: _Accumulator, duplicates: Iterator[dict[str, Any]]) -> None:
    """Merge streamed invalid rows with source-ordered duplicate rejections."""
    with acc.rejection_path.open(newline="", encoding="utf-8") as source, \
            path.open("w", newline="", encoding="utf-8") as target:
        reader = csv.DictReader(source)
        writer = csv.DictWriter(target, fieldnames=reader.fieldnames, lineterminator="\n")
        writer.writeheader()
        pending = next(reader, None)
        for entry in duplicates:
            while pending is not None and (pending["source_file"], int(pending["source_row"])) <= (
                    entry["source_file"], entry["source_row"]):
                writer.writerow(pending)
                pending = next(reader, None)
            writer.writerow(entry)
        while pending is not None:
            writer.writerow(pending)
            pending = next(reader, None)
