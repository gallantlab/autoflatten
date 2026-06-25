"""Canonical paths for the AutoFlatten benchmark.

Two roots, deliberately separated (see ``benchmark/PLAN.md``):

- The **repo** holds code + markdown docs only.
- :data:`DATA_ROOT` holds *every* loop-generated artifact (ledger, manifest, configs,
  flat patches, k-ring caches, the Optuna DB, the rendered notebook).

Nothing is written outside :data:`DATA_ROOT`.
"""

from __future__ import annotations

import os
from pathlib import Path

# --- Output root (all generated artifacts live here) ------------------------------
DATA_ROOT = Path(
    os.environ.get("AUTOFLATTEN_BENCH_ROOT", "/data2/projects/autoflatten")
)

LEDGER_DIR = DATA_ROOT / "ledger"
LEDGER_PATH = LEDGER_DIR / "experiments.jsonl"
MANIFEST_PATH = DATA_ROOT / "manifest.json"
NOTEBOOK_PATH = DATA_ROOT / "NOTEBOOK.md"
RUNS_DIR = DATA_ROOT / "runs"
KRING_CACHE_DIR = DATA_ROOT / "kring_cache"
CONFIGS_DIR = DATA_ROOT / "configs"
OPTUNA_DB = DATA_ROOT / "optuna.db"

# --- Input data (read-only) -------------------------------------------------------
# Public Narratives FreeSurfer derivatives (OpenNeuro ds002345). 82 subjects already
# have {hemi}.autoflatten.patch.3d + materialized base surfaces.
NARRATIVES_FS = Path(
    os.environ.get(
        "AUTOFLATTEN_NARRATIVES_FS",
        "/data2/projects/idem/exps/narratives/datalad-narratives/derivatives/freesurfer",
    )
)


def ensure_output_dirs() -> None:
    """Create the artifact subdirectories under :data:`DATA_ROOT` if missing."""
    for d in (LEDGER_DIR, RUNS_DIR, KRING_CACHE_DIR, CONFIGS_DIR):
        d.mkdir(parents=True, exist_ok=True)
