"""Machine-readable provenance and atomic experiment artifacts."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import subprocess
from typing import Any, Iterable, Mapping


def sha256_file(path: str | Path) -> str:
    """Return the hex SHA-256 digest of a file, streamed in 1 MiB chunks."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_new_paths(paths: Iterable[str | Path], inputs: Iterable[str | Path] = (),
                       directories: Iterable[str | Path] = ()) -> None:
    """Preflight all generated files/directories without changing the filesystem.

    Resolve aliases to detect input and output collisions, including dangling
    links. New directories may contain generated files; files cannot contain
    other targets. Call before creating even the first output or parent folder.
    """
    files = [Path(path).expanduser() for path in paths]
    folders = [Path(path).expanduser() for path in directories]
    sources = {Path(path).expanduser().resolve() for path in inputs}
    targets = files + folders
    resolved = [path.resolve() for path in targets]
    if len(set(resolved)) != len(resolved):
        raise ValueError("generated paths collide after resolving aliases")
    for path, destination in zip(targets, resolved):
        if destination in sources:
            raise ValueError(f"generated path collides with input: {path}")
        if path.is_symlink() or destination.exists():
            raise FileExistsError(f"generated path already exists or is a link: {path}")
        for parent in path.parents:
            if (parent.exists() and not parent.is_dir()) or (parent.is_symlink() and not parent.exists()):
                raise FileExistsError(f"generated path has an unusable parent: {parent}")
    for file_path in resolved[:len(files)]:
        if any(file_path in other.parents for other in resolved):
            raise ValueError(f"generated file is also an output parent: {file_path}")


def write_json(path: str | Path, value: Any) -> None:
    """Write sorted, NaN-free JSON atomically through a temporary file."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def environment_manifest(config: Any, runtime: Mapping[str, Any]) -> dict[str, Any]:
    """Capture exact inputs, installed distributions, local code and resource policy."""
    paths = [Path(config.data.path)]
    store = None
    if (paths[0] / "library.json").is_file():
        paths = [paths[0] / "library.json"]
        store = json.loads(paths[0].read_text(encoding="utf-8"))
    elif paths[0].is_dir():
        paths = sorted(paths[0].glob("*.csv"))
    if config.mode == "docking":
        paths += [Path(x) for x in (config.docking.receptor, config.docking.reference_ligand,
                                   config.docking.executable) if x]
    inputs = {str(p): sha256_file(p) for p in paths}
    package = Path(__file__).resolve().parent
    code = {p.name: sha256_file(p) for p in sorted(package.glob("*.py"))}
    try:
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=package,
                                  capture_output=True, text=True, check=True, timeout=5).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=package,
                                   capture_output=True, text=True, check=True, timeout=5).stdout.strip())
    except (OSError, subprocess.SubprocessError):
        revision, dirty = None, None
    return {
        "schema_version": 1, "status": "running", "started_at": utc_now(),
        "python": platform.python_version(), "platform": platform.platform(),
        "git_revision": revision, "git_dirty": dirty, "package_files_sha256": code,
        "input_sha256": inputs, "library_store": store,
        "distributions": dict(sorted({d.metadata["Name"]: d.version for d in metadata.distributions()
                                      if d.metadata["Name"]}.items())),
        "resources": {k: v for k, v in runtime.items() if k != "threadpool_controller"},
        "config": config.to_dict(),
    }


def artifact_hashes(directory: str | Path) -> dict[str, str]:
    """Hash every artifact except the mutable manifest and log."""
    root = Path(directory)
    return {str(p.relative_to(root)): sha256_file(p) for p in sorted(root.rglob("*"))
            if p.is_file() and p.name not in {"manifest.json", "active_learning.log"}}
