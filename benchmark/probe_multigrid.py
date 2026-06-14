"""Probe: spectral (manifold-harmonic) multigrid optimization.

The FreeSurfer-style optimizer avoids folding via multi-level gradient smoothing — coherent
coarse-to-fine moves. This probe makes that explicit and geometric: parametrize the flatmap
in the **mesh-Laplacian eigenbasis** (manifold harmonics) and optimize coarse-to-fine,
adding higher-frequency modes progressively. A band-limited (few-mode) map *cannot* make
local folds, so the coarse optimization is flip-free by construction; it just can't
represent fine detail, so distortion floors out. The coarse, flip-free, distance-aware map
is then an excellent init for a short full-resolution refinement (the "fine" leg of the
multigrid V-cycle).

Two modes:
- ``--no-refine``: spectral coarse-to-fine only (fast, flip-free, band-limited).
- default: spectral coarse init -> full-resolution refinement (initial NAR disabled).

Manifold harmonics (the lowest *K* eigenvectors of the cotangent Laplacian) are cached per
mesh under the k-ring cache dir, keyed by a content hash of the vertices.

Usage
-----
    python -m benchmark.probe_multigrid --subset 1 --save
    python -m benchmark.probe_multigrid --no-refine --modes 200 --subset 1
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time

import numpy as np

from . import paths
from .harness import evaluate, load_manifest, register_flatten_fn, select_entries
from .ledger import Ledger, file_hash, new_record
from .probe_tutte_init import flipfree_init, scale_to_area


def manifold_harmonics(vertices: np.ndarray, faces: np.ndarray, k: int) -> np.ndarray:
    """Lowest-``k`` cotangent-Laplacian eigenvectors (M-orthonormal), cached by content."""
    import igl
    import scipy.sparse.linalg as sla

    v = np.ascontiguousarray(vertices, dtype=np.float64)
    f = np.ascontiguousarray(faces, dtype=np.int64)
    key = hashlib.md5(v.tobytes() + f.tobytes() + str(k).encode()).hexdigest()[:16]
    cache = paths.KRING_CACHE_DIR / f"harmonics_{key}_k{k}.npz"
    if cache.exists():
        return np.load(cache)["Phi"]

    paths.KRING_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    lap = -igl.cotmatrix(v, f)  # positive semidefinite
    mass = igl.massmatrix(v, f, igl.MASSMATRIX_TYPE_VORONOI)
    _, phi = sla.eigsh(lap.tocsc(), k=k, M=mass.tocsc(), sigma=1e-8, which="LM")
    np.savez(cache, Phi=phi.astype(np.float64))
    return phi.astype(np.float64)


def spectral_optimize(
    flattener, phi: np.ndarray, schedule, max_iter: int = 200
) -> np.ndarray:
    """Coarse-to-fine optimization of J_d in the manifold-harmonic basis."""
    import igl
    import jax
    import jax.numpy as jnp
    from scipy.optimize import minimize

    from autoflatten.flatten.energy import compute_metric_energy

    faces = np.asarray(flattener.faces)
    mass = igl.massmatrix(
        np.asarray(flattener.vertices, dtype=np.float64),
        faces.astype(np.int64),
        igl.MASSMATRIX_TYPE_VORONOI,
    )
    x0 = scale_to_area(
        flipfree_init(flattener.vertices, flattener.faces, "tutte"),
        faces,
        flattener.orig_area,
    )
    coeff = phi.T @ (mass @ x0)  # mass-weighted projection onto modes (Phi^T M x)

    phij = jnp.asarray(phi)
    nbr, tgt, msk = flattener.neighbors_jax, flattener.targets_jax, flattener.mask_jax
    avg = flattener.avg_neighbors or 1.0

    for k in schedule:
        p_k = phij[:, :k]
        energy = jax.jit(
            lambda c, p_k=p_k: compute_metric_energy(p_k @ c, nbr, tgt, msk) / avg
        )
        vg = jax.jit(jax.value_and_grad(energy))

        def vgf(z, k=k, vg=vg):
            val, grad = vg(jnp.asarray(z.reshape(k, 2)))
            return float(val), np.asarray(grad, dtype=np.float64).reshape(-1)

        res = minimize(
            vgf,
            coeff[:k].reshape(-1).astype(np.float64),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": max_iter},
        )
        coeff[:k] = res.x.reshape(k, 2)
    return np.asarray(phi @ coeff)


def make_multigrid_flatten_fn(modes: int = 200, refine: bool = True, schedule=None):
    """``flatten_fn``: spectral coarse-to-fine, then (optionally) full-resolution refine."""
    sched = schedule or [m for m in (20, 50, 100, modes) if m <= modes]

    def _fn(flattener):
        phi = manifold_harmonics(flattener.vertices, flattener.faces, modes)
        coarse = spectral_optimize(flattener, phi, sched)
        if not refine:
            return coarse
        # Coarse map is ~flip-free; skip the initial NAR and refine at full resolution.
        flattener.config.negative_area_removal.enabled = False
        flattener.initial_projection = lambda: coarse
        return flattener.run()

    return _fn


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--modes", type=int, default=200, help="number of manifold-harmonic modes"
    )
    ap.add_argument("--no-refine", action="store_true", help="spectral coarse only")
    ap.add_argument("--dev", action="store_true")
    ap.add_argument("--split", default=None, choices=["train", "holdout"])
    ap.add_argument("--subset", type=int, default=None)
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    refine = not args.no_refine
    label = "multigrid" + ("" if refine else "_coarse")
    register_flatten_fn(label, make_multigrid_flatten_fn(args.modes, refine=refine))

    from autoflatten.flatten import FlattenConfig

    cfg = FlattenConfig()
    cfg.verbose = False
    if refine:
        cfg.negative_area_removal.enabled = False

    paths.ensure_output_dirs()
    manifest = load_manifest()
    subset = 2 if args.dev else args.subset
    entries = select_entries(manifest, split=args.split, subset=subset)
    if not entries:
        print("No manifest entries selected.", file=sys.stderr)
        return 1

    record = new_record(
        kind="experiment",
        label=f"exp:{label}",
        manifest_id=manifest.get("created"),
        subjects=[
            {"subject": e["subject"], "hemi": e["hemi"], "split": e.get("split")}
            for e in entries
        ],
        method={
            "name": label,
            "modes": args.modes,
            "refine": refine,
            "config": cfg.to_dict(),
        },
        repro_command="python -m benchmark.probe_multigrid " + " ".join(sys.argv[1:]),
    )
    record.decision = {
        "hypothesis": (
            "Optimizing in the low-frequency manifold-harmonic basis is flip-free by "
            "band-limiting; the coarse, distance-aware map is a strong init for a short "
            "full-resolution refine (geometric multigrid)."
        )
    }

    print(
        f"Multigrid probe '{label}' (modes={args.modes}) on {len(entries)} hemispheres ({record.experiment_id})..."
    )
    t0 = time.time()
    save_dir = paths.RUNS_DIR / record.experiment_id if args.save else None
    result = evaluate(entries, cfg, method=label, save_dir=save_dir)
    record.per_subject = result["per_subject"]
    record.metrics = result["aggregate"]
    record.runtime_s = time.time() - t0
    if save_dir is not None:
        record.artifacts = [
            {
                "path": r["artifact"],
                "hash": file_hash(r["artifact"]),
                "kind": "flat_patch",
            }
            for r in result["per_subject"]
            if r.get("artifact")
        ]
    record.status = "ok" if result["aggregate"].get("n_failed", 0) == 0 else "partial"

    agg = result["aggregate"]
    base = Ledger().latest(kind="baseline")
    base_m = base["metrics"] if base else None
    if base_m and "mean_distortion" in agg:
        record.decision["conclusion"] = (
            f"{label}: dist {agg['mean_distortion']:.2f} vs base {base_m.get('mean_distortion'):.2f}; "
            f"flips {agg['total_flipped']} vs {base_m.get('total_flipped')}; "
            f"runtime {agg.get('mean_runtime_s', 0):.0f}s vs {base_m.get('mean_runtime_s', 0):.0f}s."
        )
    Ledger().append(record)

    print("\n=== aggregate ===")
    for k in (
        "n_patches",
        "n_failed",
        "mean_distortion",
        "total_flipped",
        "frac_patches_with_flips",
        "mean_runtime_s",
    ):
        if k in agg:
            print(f"  {k}: {agg[k]}")
    print(f"\nLogged {record.experiment_id} -> {Ledger().path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
