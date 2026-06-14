"""Quality metrics for a flattened patch, computed uniformly across methods.

Metrics are computed from the 2D ``uv`` coordinates plus the patch's geodesic targets
(the prepared :class:`~autoflatten.flatten.algorithm.SurfaceFlattener` arrays), so they
do **not** depend on *how* ``uv`` was produced. Any ``flatten_fn`` is scored identically.

Reuses the package's own metric kernels so the harness reports the same numbers the CLI
log does:

- :func:`autoflatten.flatten.algorithm.count_flipped_triangles`
- :func:`autoflatten.flatten.algorithm._compute_distance_error_jit`
"""

from __future__ import annotations

from typing import Any

import numpy as np


def per_patch_metrics(uv: np.ndarray, flattener: Any) -> dict[str, float]:
    """Compute quality metrics for one flattened patch.

    Parameters
    ----------
    uv : ndarray, shape (V, 2)
        Flattened coordinates aligned to ``flattener``'s vertex ordering.
    flattener : SurfaceFlattener
        A flattener for which ``prepare_optimization`` has run, exposing
        ``neighbors_jax``, ``targets_jax``, ``mask_jax``, and ``faces_jax``.

    Returns
    -------
    dict
        ``mean_distortion`` (mean % distance error, FreeSurfer formula),
        ``p90_distortion`` (90th-percentile per-vertex % error),
        ``n_flipped`` (flipped/negative-area triangles),
        ``frac_flipped`` (fraction of faces flipped),
        ``area_distortion`` (|total 2D area / 3D area - 1|).
    """
    import jax.numpy as jnp

    from autoflatten.flatten.algorithm import (
        _compute_distance_error_jit,
        count_flipped_triangles,
    )

    uv_jax = jnp.asarray(uv)
    faces = flattener.faces_jax
    neighbors = flattener.neighbors_jax
    targets = flattener.targets_jax
    mask = flattener.mask_jax

    mean_distortion = float(
        _compute_distance_error_jit(uv_jax, neighbors, targets, mask)
    )
    n_flipped = int(count_flipped_triangles(uv_jax, faces))
    n_faces = int(np.asarray(faces).shape[0])

    # Per-vertex distortion distribution (for robustness / tail metrics).
    p90 = _per_vertex_p90(
        np.asarray(uv), np.asarray(neighbors), np.asarray(targets), np.asarray(mask)
    )

    # Area distortion: how much total flattened area departs from the 3D patch area.
    area_distortion = _area_distortion(
        np.asarray(uv), np.asarray(faces), flattener.orig_area
    )

    return {
        "mean_distortion": mean_distortion,
        "p90_distortion": p90,
        "n_flipped": n_flipped,
        "frac_flipped": (n_flipped / n_faces) if n_faces else 0.0,
        "area_distortion": area_distortion,
    }


def _per_vertex_p90(
    uv: np.ndarray, neighbors: np.ndarray, targets: np.ndarray, mask: np.ndarray
) -> float:
    """90th percentile of per-vertex mean % distance error."""
    valid = mask & (targets > 0)
    d2d = np.linalg.norm(uv[neighbors] - uv[:, None, :], axis=-1)
    abs_err = np.where(valid, np.abs(d2d - targets), 0.0)
    n_valid = valid.sum(axis=1)
    denom = np.where(valid, targets, 0.0).sum(axis=1)
    has = (n_valid > 0) & (denom > 0)
    per_vertex = np.zeros(uv.shape[0])
    per_vertex[has] = 100.0 * abs_err.sum(axis=1)[has] / denom[has]
    if not has.any():
        return float("nan")
    return float(np.percentile(per_vertex[has], 90))


def _area_distortion(uv: np.ndarray, faces: np.ndarray, orig_area: float) -> float:
    """|sum(|2D triangle area|) / orig_3d_area - 1|."""
    v0, v1, v2 = uv[faces[:, 0]], uv[faces[:, 1]], uv[faces[:, 2]]
    areas = 0.5 * (
        (v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1])
        - (v2[:, 0] - v0[:, 0]) * (v1[:, 1] - v0[:, 1])
    )
    total_2d = float(np.abs(areas).sum())
    if not orig_area:
        return float("nan")
    return abs(total_2d / orig_area - 1.0)


# ---------------------------------------------------------------------------------
# Aggregation across subjects/hemispheres -> the multi-objective vector
# ---------------------------------------------------------------------------------
def aggregate(per_subject: list[dict[str, Any]]) -> dict[str, float]:
    """Aggregate per-patch metrics into the benchmark's multi-objective vector.

    Returns mean and worst-case distortion (robustness), total/fraction of patches with
    flips, and mean runtime.
    """
    ok = [
        r
        for r in per_subject
        if r.get("status", "ok") == "ok" and "mean_distortion" in r
    ]
    if not ok:
        return {"n_patches": 0, "n_failed": len(per_subject)}

    md = np.array([r["mean_distortion"] for r in ok], dtype=float)
    p90 = np.array([r["p90_distortion"] for r in ok], dtype=float)
    flips = np.array([r["n_flipped"] for r in ok], dtype=float)
    rt = np.array([r.get("runtime_s", np.nan) for r in ok], dtype=float)

    return {
        "n_patches": len(ok),
        "n_failed": len(per_subject) - len(ok),
        "mean_distortion": float(np.mean(md)),
        "worst_distortion": float(np.max(md)),
        "median_distortion": float(np.median(md)),
        "mean_p90_distortion": float(np.nanmean(p90)),
        "total_flipped": int(flips.sum()),
        "frac_patches_with_flips": float(np.mean(flips > 0)),
        "mean_runtime_s": float(np.nanmean(rt)),
        "total_runtime_s": float(np.nansum(rt)),
    }
