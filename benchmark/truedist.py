"""True geodesic distortion yardstick (method-agnostic quality metric).

The k-ring energy metric is miscalibrated as an *absolute* number and is not comparable
across different ``k_ring`` values (its targets change with k). To compare maps produced
under different energies/neighborhoods we score against a fixed, energy-independent
**true geodesic** reference: libigl heat-method geodesics from a set of sampled sources,
restricted to local pairs (<= R mm), comparing the 2D flat distance to the true 3D
geodesic distance.

The reference (sources + per-source geodesic fields) is precomputed once per (subject,
hemi) and cached under the k-ring cache dir as ``{subject}_{hemi}.truegeo.npz`` with keys
``srcs`` (M source indices into the *patch* vertex array), ``geo`` (M x V geodesics), and
``R`` (the local radius in mm).

Usage
-----
    from benchmark.truedist import load_truegeo, true_distortion
    ref = load_truegeo("sub-022", "lh")
    stats = true_distortion(uv, ref)        # mean/p90 true distortion over local pairs
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from . import paths


def truegeo_path(subject: str, hemi: str) -> Path:
    return paths.KRING_CACHE_DIR / f"{subject}_{hemi}.truegeo.npz"


def load_truegeo(subject: str, hemi: str) -> dict[str, Any]:
    """Load the cached true-geodesic reference for a (subject, hemi)."""
    d = np.load(truegeo_path(subject, hemi))
    return {"srcs": d["srcs"], "geo": d["geo"], "R": float(d["R"])}


def compute_truegeo(
    flattener: Any,
    n_sources: int = 200,
    radius: float = 30.0,
    seed: int = 0,
) -> dict[str, Any]:
    """Compute and cache heat-method geodesic fields from ``n_sources`` sampled sources.

    Geodesics are computed on the **fiducial** (mid-cortical) surface -- the anatomical surface
    whose distances the flattener's energy preserves and which "metric distortion" is defined
    against. The patch's own stored coordinates (``flattener.vertices``) are the FreeSurfer
    *inflated* surface (a smoothed balloon ~1.7x larger, with stretched triangles); using it
    here would score against the wrong geometry AND ill-conditions the heat solver (spurious
    near-zero / negative geodesics that blow up the relative metric). Fall back to
    ``flattener.vertices`` only if no fiducial is available.

    Sources are drawn deterministically (fixed seed). A source whose solve returns a negative
    geodesic (ill-conditioned) is resampled. The field is stored densely; scoring restricts to
    pairs within ``radius`` mm.
    """
    import igl

    surf = getattr(flattener, "fiducial_vertices", None)
    if surf is None:
        surf = flattener.vertices
    v = np.ascontiguousarray(surf, dtype=np.float64)
    f = np.ascontiguousarray(flattener.faces, dtype=np.int64)
    n_v = v.shape[0]

    rng = np.random.default_rng(seed)
    srcs = np.sort(rng.choice(n_v, size=min(n_sources, n_v), replace=False))

    data = igl.HeatGeodesicsData()
    igl.heat_geodesics_precompute(v, f, data)
    geo = np.empty((srcs.shape[0], n_v), dtype=np.float64)
    for i, s in enumerate(srcs):
        field = igl.heat_geodesics_solve(data, np.array([s], dtype=np.int64))
        # A geodesic can never be shorter than the straight-line 3D chord. The heat solver
        # occasionally emits grossly-wrong near-zero (or negative) values for far vertices
        # (ill-conditioning); these tiny denominators blow up the relative metric. Zero out any
        # estimate below HALF the chord -- a loose physical bound that flags only gross failures
        # (<0.1% of pairs; legitimate heat under-estimation stays well above 0.5*chord). Zeroed
        # entries are dropped by the scoring mask (d > 1e-6).
        chord = np.linalg.norm(v - v[s], axis=1)
        geo[i] = np.where(field < 0.5 * chord, 0.0, field)
    return {"srcs": srcs, "geo": geo, "R": float(radius)}


def true_distortion(uv: np.ndarray, ref: dict[str, Any]) -> dict[str, float]:
    """Mean/p90 absolute relative distortion of 2D distances vs true geodesics.

    For each source ``s`` and target ``j`` with ``geo[s,j] <= R`` (and > 0), the per-pair
    distortion is ``|d2d - d_geo| / d_geo`` where ``d2d`` is the Euclidean distance between
    ``uv[s]`` and ``uv[j]``. Computed on the fiducial reference (see ``compute_truegeo``) the
    raw relative error is well behaved -- no denominator floor is needed. Returns percentages.
    """
    uv = np.asarray(uv, dtype=np.float64)
    srcs = ref["srcs"]
    geo = ref["geo"]
    R = ref["R"]

    errs = []
    for i, s in enumerate(srcs):
        d_geo = geo[i]
        mask = (d_geo > 1e-6) & (d_geo <= R)
        if not np.any(mask):
            continue
        d2d = np.linalg.norm(uv[mask] - uv[s], axis=1)
        rel = np.abs(d2d - d_geo[mask]) / d_geo[mask]
        errs.append(rel)
    all_err = np.concatenate(errs)
    return {
        "true_mean_distortion": float(np.mean(all_err) * 100.0),
        "true_p90_distortion": float(np.percentile(all_err, 90) * 100.0),
        "true_median_distortion": float(np.median(all_err) * 100.0),
        "n_pairs": int(all_err.size),
    }


def true_distortion_full(uv: np.ndarray, ref: dict[str, Any]) -> dict[str, float]:
    """Local (<=R), global (all pairs), and global-at-distance-optimal-scale distortion.

    The local <=R metric is gameable (a conformal disk scores well locally while globally
    catastrophic), so the *global* all-pairs metric is the faithful objective. Also reports
    the single global scale ``s*`` that minimizes global distortion (the area-matched output
    is generally not metric-optimal) and the distortion at ``s*``.
    """
    uv = np.asarray(uv, dtype=np.float64)
    srcs = ref["srcs"]
    geo = ref["geo"]
    R = ref["R"]

    d2_all, dg_all = [], []
    for i, s in enumerate(srcs):
        dg = geo[i]
        m = dg > 1e-6
        d2_all.append(np.linalg.norm(uv[m] - uv[s], axis=1))
        dg_all.append(dg[m])
    d2 = np.concatenate(d2_all)
    dg = np.concatenate(dg_all)

    rel = np.abs(d2 - dg) / dg
    loc = dg <= R
    # optimal global scale s* minimizing mean |s*d2 - dg|/dg over a fine grid (widened range:
    # the flat map may sit at a different global scale than the reference surface)
    scales = np.linspace(0.80, 1.25, 91)
    errs = np.array([np.mean(np.abs(sc * d2 - dg) / dg) for sc in scales])
    j = int(np.argmin(errs))
    return {
        "true_local_mean": float(np.mean(rel[loc]) * 100.0),
        "true_global_mean": float(np.mean(rel) * 100.0),
        "true_global_p90": float(np.percentile(rel, 90) * 100.0),
        "opt_scale": float(scales[j]),
        "true_global_at_optscale": float(errs[j] * 100.0),
        "n_pairs_global": int(rel.size),
    }


def true_distortion_banded(
    uv: np.ndarray, ref: dict[str, Any], bands=((0, 5), (5, 15), (15, 30))
) -> dict[str, float]:
    """Mean true distortion broken down by geodesic-distance band (mm).

    Reveals *where* in the 0-R range the map distorts: a metric energy that over-fits the
    very-local scale tends to show low error in the (0,5] band but rising error at medium
    range. Returns ``band_{lo}_{hi}_mean`` percentages and pair counts.
    """
    uv = np.asarray(uv, dtype=np.float64)
    srcs = ref["srcs"]
    geo = ref["geo"]

    band_errs = {b: [] for b in bands}
    for i, s in enumerate(srcs):
        d_geo = geo[i]
        d2d_all = np.linalg.norm(uv - uv[s], axis=1)
        for lo, hi in bands:
            mask = (d_geo > max(lo, 1e-6)) & (d_geo <= hi)
            if np.any(mask):
                rel = np.abs(d2d_all[mask] - d_geo[mask]) / d_geo[mask]
                band_errs[(lo, hi)].append(rel)
    out: dict[str, float] = {}
    for (lo, hi), chunks in band_errs.items():
        if chunks:
            e = np.concatenate(chunks)
            out[f"band_{lo}_{hi}_mean"] = float(np.mean(e) * 100.0)
            out[f"band_{lo}_{hi}_n"] = int(e.size)
    return out
