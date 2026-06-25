"""Append-only provenance ledger for benchmark experiments.

Every experiment run appends one JSON record to ``experiments.jsonl`` (under
:data:`benchmark.paths.DATA_ROOT`). The ledger is the single source of truth for the
autoresearch process and is meant to be shared as a paper supplement, so each record is
self-contained and reproducible: it pins the autoflatten **git commit SHA**, captures the
environment and seeds, and stores all metrics, artifact paths, and a repro command.

For autoresearch-loop steps, a record may also carry a **decision trace**
(hypothesis / rationale / conclusion / next_step) so the *reasoning* is auditable, not
just the numbers.

Records are never mutated. To "update" an experiment, append a new record.
"""

from __future__ import annotations

import getpass
import hashlib
import json
import platform
import socket
import subprocess
import sys
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from . import paths

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------------
# Provenance capture
# ---------------------------------------------------------------------------------
def _git(*args: str) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), *args],
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def capture_git_provenance() -> dict[str, Any]:
    """Capture the autoflatten repo commit, branch, and dirty-tree diff."""
    commit = _git("rev-parse", "HEAD")
    branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    status = _git("status", "--porcelain")
    dirty = bool(status)
    diff = _git("diff", "HEAD") if dirty else ""
    return {
        "commit": commit,
        "branch": branch,
        "dirty": dirty,
        # Truncate a runaway diff but keep enough to reconstruct intent.
        "diff": (diff or "")[:200_000],
    }


def capture_environment() -> dict[str, Any]:
    """Capture python, platform, and key package versions (incl. JAX backend)."""
    env: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "user": getpass.getuser(),
    }
    try:
        import jax

        env["jax_version"] = jax.__version__
        env["jax_backend"] = jax.default_backend()
        env["jax_devices"] = [str(d) for d in jax.devices()]
    except Exception as exc:  # pragma: no cover - environment dependent
        env["jax_error"] = repr(exc)
    for pkg in ("numpy", "scipy", "numba", "igl", "optuna"):
        try:
            mod = __import__(pkg)
            env[f"{pkg}_version"] = getattr(mod, "__version__", "unknown")
        except Exception:
            pass
    return env


def file_hash(path: str | Path, algo: str = "md5") -> Optional[str]:
    """Return a content hash of ``path`` (resolving symlinks), or None if missing."""
    p = Path(path)
    if not p.exists():
        return None
    h = hashlib.new(algo)
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return f"{algo}:{h.hexdigest()}"


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------------
# Record + ledger
# ---------------------------------------------------------------------------------
@dataclass
class ExperimentRecord:
    """One experiment run. Serialized as a single JSONL line."""

    # Identity / provenance
    experiment_id: str
    timestamp: str
    kind: str  # e.g. "baseline", "probe", "hpo_trial"
    label: str  # human-readable, e.g. "baseline:pyflatten-defaults"
    git: dict[str, Any]
    environment: dict[str, Any]

    # Inputs
    manifest_id: Optional[str] = None
    subjects: list[dict[str, Any]] = field(default_factory=list)
    method: dict[str, Any] = field(default_factory=dict)  # {name, params/config}
    seeds: dict[str, Any] = field(default_factory=dict)

    # Outputs
    metrics: dict[str, Any] = field(default_factory=dict)  # aggregate
    per_subject: list[dict[str, Any]] = field(default_factory=list)
    artifacts: list[dict[str, Any]] = field(default_factory=list)  # {path, hash, kind}
    runtime_s: Optional[float] = None
    status: str = "ok"  # "ok" | "error"
    error: Optional[str] = None

    # Repro
    repro_command: Optional[str] = None

    # Decision trace (autoresearch only)
    decision: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str)


def new_record(kind: str, label: str, **kwargs: Any) -> ExperimentRecord:
    """Build a record pre-filled with a fresh id, timestamp, and provenance."""
    return ExperimentRecord(
        experiment_id=uuid.uuid4().hex[:12],
        timestamp=_utcnow(),
        kind=kind,
        label=label,
        git=capture_git_provenance(),
        environment=capture_environment(),
        **kwargs,
    )


class Ledger:
    """Append-only JSONL ledger of experiments."""

    def __init__(self, path: str | Path = paths.LEDGER_PATH):
        self.path = Path(path)

    def append(self, record: ExperimentRecord) -> ExperimentRecord:
        """Append a record (creating the ledger dir if needed) and return it."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as f:
            f.write(record.to_json() + "\n")
        return record

    def read(self) -> list[dict[str, Any]]:
        """Read all records as dicts (oldest first). Empty if the ledger is absent."""
        if not self.path.exists():
            return []
        records = []
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    def latest(self, kind: Optional[str] = None) -> Optional[dict[str, Any]]:
        """Return the most recent record, optionally filtered by ``kind``."""
        recs = self.read()
        if kind is not None:
            recs = [r for r in recs if r.get("kind") == kind]
        return recs[-1] if recs else None
