"""Method-agnostic evaluation harness for the flattening stage.

The harness owns the *geometry* (it loads the patch + base surface and computes the
k-ring geodesic targets once, with on-disk caching), then hands a prepared
:class:`~autoflatten.flatten.algorithm.SurfaceFlattener` to a ``flatten_fn`` and scores
whatever ``uv`` it returns with the uniform metrics in :mod:`benchmark.metrics`.

A ``flatten_fn`` has signature ``flatten_fn(flattener) -> uv`` (shape ``(V, 2)``).
The current pipeline is just one such function; Tutte/LSCM/SLIM/SMACOF prototypes are
others, all scored identically.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from . import paths

FlattenFn = Callable[[Any], np.ndarray]


# ---------------------------------------------------------------------------------
# Built-in flatten_fns
# ---------------------------------------------------------------------------------
def pyflatten_flatten_fn(flattener: Any) -> np.ndarray:
    """The current FreeSurfer-clone optimizer: run the full pyflatten pipeline."""
    return flattener.run()


FLATTEN_FNS: dict[str, FlattenFn] = {
    "pyflatten": pyflatten_flatten_fn,
}


def register_flatten_fn(name: str, fn: FlattenFn) -> None:
    """Register an alternative method so it can be evaluated by name."""
    FLATTEN_FNS[name] = fn


# ---------------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------------
def load_manifest(path: str | Path = paths.MANIFEST_PATH) -> dict[str, Any]:
    """Load the benchmark manifest (subjects, hemis, patch/surface paths, splits)."""
    with open(path) as f:
        return json.load(f)


def select_entries(
    manifest: dict[str, Any],
    split: Optional[str] = None,
    subset: Optional[int] = None,
) -> list[dict[str, Any]]:
    """Filter manifest entries by split and/or take the first ``subset`` of them."""
    entries = manifest["entries"]
    if split is not None:
        entries = [e for e in entries if e.get("split") == split]
    if subset is not None:
        entries = entries[:subset]
    return entries


# ---------------------------------------------------------------------------------
# Geometry + evaluation
# ---------------------------------------------------------------------------------
def _kring_cache_path(entry: dict[str, Any], config: Any) -> Path:
    """Per-patch k-ring cache path, keyed on (subject, hemi, k_ring, n_neighbors).

    The cache is independent of energy weights, so configs that share a k-ring share
    this file — the expensive geodesic computation is paid once per (subject, hemi, k).
    """
    k = config.kring.k_ring
    n = config.kring.n_neighbors_per_ring
    nstr = "all" if n is None else str(n)
    name = f"{entry['subject']}_{entry['hemi']}.kring_k{k}_n{nstr}.npz"
    return paths.KRING_CACHE_DIR / name


def build_flattener(
    entry: dict[str, Any],
    config: Any,
    use_cache: bool = True,
) -> Any:
    """Load a patch + base surface and prepare a flattener (geometry + k-ring + JAX).

    Returns a :class:`SurfaceFlattener` ready for a ``flatten_fn``. Reuses the on-disk
    k-ring cache when available.
    """
    from autoflatten.flatten import SurfaceFlattener

    flattener = SurfaceFlattener(config)
    flattener.load_data(entry["patch_path"], entry["surface_path"])
    cache_path = str(_kring_cache_path(entry, config)) if use_cache else None
    if cache_path:
        paths.KRING_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    flattener.compute_kring_distances(cache_path=cache_path)
    flattener.prepare_optimization()
    return flattener


def evaluate_one(
    entry: dict[str, Any],
    config: Any,
    flatten_fn: FlattenFn,
    save_dir: Optional[Path] = None,
    use_cache: bool = True,
) -> dict[str, Any]:
    """Evaluate one (subject, hemi): flatten and score.

    Returns a per-patch result dict (subject/hemi + metrics + runtime + status). On
    failure, ``status="error"`` with the exception message, so a bad subject doesn't
    sink the whole run.
    """
    from .metrics import per_patch_metrics

    result: dict[str, Any] = {
        "subject": entry["subject"],
        "hemi": entry["hemi"],
        "split": entry.get("split"),
        "status": "ok",
    }
    try:
        flattener = build_flattener(entry, config, use_cache=use_cache)
        t0 = time.time()
        uv = np.asarray(flatten_fn(flattener))
        result["runtime_s"] = time.time() - t0
        result.update(per_patch_metrics(uv, flattener))

        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            out_path = save_dir / f"{entry['subject']}.{entry['hemi']}.flat.patch.3d"
            flattener.save_result(uv, str(out_path))
            result["artifact"] = str(out_path)
    except Exception as exc:  # noqa: BLE001 - record and continue
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result


def evaluate(
    entries: list[dict[str, Any]],
    config: Any,
    method: str = "pyflatten",
    save_dir: Optional[Path] = None,
    use_cache: bool = True,
) -> dict[str, Any]:
    """Evaluate a list of manifest entries with one method/config.

    Returns ``{"per_subject": [...], "aggregate": {...}}``. Aggregation is the
    multi-objective vector defined in :func:`benchmark.metrics.aggregate`.
    """
    from .metrics import aggregate

    flatten_fn = FLATTEN_FNS[method]
    per_subject = [
        evaluate_one(e, config, flatten_fn, save_dir=save_dir, use_cache=use_cache)
        for e in entries
    ]
    return {"per_subject": per_subject, "aggregate": aggregate(per_subject)}
