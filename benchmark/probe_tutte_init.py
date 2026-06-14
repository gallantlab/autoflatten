"""Probe: flip-free (Tutte/LSCM) initialization instead of FreeSurfer projection + NAR.

The current pipeline starts from a normal-axis projection that has many flipped
triangles, then spends its first ~4 minutes in "negative area removal" (NAR) un-flipping
them. A **Tutte embedding** (harmonic map with the boundary pinned to a convex circle) is
*guaranteed flip-free* for disk topology — so we can drop the initial-NAR phase entirely
and feed the flip-free map straight into the existing geodesic-stress refinement.

Hypothesis: equal-or-lower distance distortion, zero flips at init, and less runtime.

This is implemented by injecting a different initial map into the existing
``SurfaceFlattener`` (overriding ``initial_projection`` and disabling the initial NAR), so
the refinement (epochs + final NAR + spring) is shared with the baseline and the
comparison is apples-to-apples.

Usage
-----
    python -m benchmark.probe_tutte_init --dev                 # tutte init + refine
    python -m benchmark.probe_tutte_init --dev --method lscm
    python -m benchmark.probe_tutte_init --dev --no-refine     # init quality only
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from . import paths
from .harness import evaluate, load_manifest, register_flatten_fn, select_entries
from .ledger import Ledger, file_hash, new_record


# ---------------------------------------------------------------------------------
# Flip-free initialization
# ---------------------------------------------------------------------------------
def flipfree_init(
    vertices: np.ndarray, faces: np.ndarray, method: str = "tutte"
) -> np.ndarray:
    """Compute a flip-free 2D embedding of a disk-topology patch.

    Parameters
    ----------
    vertices : (V, 3) float
        3D patch vertices (use the smoothed/fiducial surface for intrinsic weights).
    faces : (F, 3) int
        Triangles (single boundary loop / disk topology).
    method : {"tutte", "lscm"}
        ``tutte`` — harmonic map with the boundary pinned to a circle. Guaranteed
        injective (Tutte's theorem) → no flipped triangles.
        ``lscm`` — least-squares conformal map (2 pinned boundary vertices). Lower angle
        distortion but *not* guaranteed flip-free.

    Returns
    -------
    (V, 2) float
        2D coordinates (unscaled).
    """
    import igl

    v = np.ascontiguousarray(vertices, dtype=np.float64)
    f = np.ascontiguousarray(faces, dtype=np.int64)
    bnd = igl.boundary_loop(f)
    if bnd is None or len(bnd) == 0:
        raise ValueError("No boundary loop found; patch is not a disk.")

    if method == "tutte":
        bc = igl.map_vertices_to_circle(v, bnd.astype(np.int32))
        uv = igl.harmonic(v, f, bnd.astype(np.int64), np.ascontiguousarray(bc), 1)
    elif method == "lscm":
        # Pin the two most distant boundary vertices to (0,0) and (1,0).
        b = np.array([bnd[0], bnd[len(bnd) // 2]], dtype=np.int64)
        bc = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)
        uv, _ = igl.lscm(v, f, b, bc)
    else:
        raise ValueError(f"Unknown init method: {method!r}")
    return np.asarray(uv, dtype=np.float64)


def scale_to_area(uv: np.ndarray, faces: np.ndarray, target_area: float) -> np.ndarray:
    """Uniformly scale ``uv`` so its total (unsigned) 2D area matches ``target_area``.

    Tutte/LSCM map to ~unit scale, while the geodesic targets are in mm; matching the 3D
    patch area gives the refinement a sensible starting scale.
    """
    v0, v1, v2 = uv[faces[:, 0]], uv[faces[:, 1]], uv[faces[:, 2]]
    area2d = float(
        np.abs(
            0.5
            * (
                (v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1])
                - (v2[:, 0] - v0[:, 0]) * (v1[:, 1] - v0[:, 1])
            )
        ).sum()
    )
    if area2d <= 0 or target_area <= 0:
        return uv
    s = np.sqrt(target_area / area2d)
    centroid = uv.mean(axis=0)
    return (uv - centroid) * s + centroid


# ---------------------------------------------------------------------------------
# flatten_fn factory
# ---------------------------------------------------------------------------------
def make_flatten_fn(method: str = "tutte", refine: bool = True):
    """Build a ``flatten_fn`` that initializes flip-free and (optionally) refines.

    When ``refine`` is True the flip-free map is injected into the existing optimizer
    with the **initial NAR phase disabled** (its purpose is moot for a flip-free start);
    the rest of the refinement (epochs, final NAR, spring) is unchanged. When False, the
    scaled init is returned directly to measure init-only quality.
    """

    def _fn(flattener):
        init = flipfree_init(flattener.vertices, flattener.faces, method=method)
        init = scale_to_area(init, np.asarray(flattener.faces), flattener.orig_area)
        if not refine:
            return init
        # Skip the initial negative-area-removal phase: a flip-free start makes it moot.
        flattener.config.negative_area_removal.enabled = False
        flattener.initial_projection = lambda: init  # injected into run()
        return flattener.run()

    return _fn


# ---------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------
def _baseline_reference():
    """Most recent baseline record's aggregate metrics, for head-to-head printing."""
    rec = Ledger().latest(kind="baseline")
    return rec["metrics"] if rec else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dev", action="store_true", help="first 2 hemispheres")
    ap.add_argument("--split", default=None, choices=["train", "holdout"])
    ap.add_argument("--subset", type=int, default=None)
    ap.add_argument("--method", default="tutte", choices=["tutte", "lscm"])
    ap.add_argument(
        "--no-refine", action="store_true", help="measure init-only quality"
    )
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    from autoflatten.flatten import FlattenConfig

    refine = not args.no_refine
    method_name = f"{args.method}_init" + ("" if refine else "_only")
    register_flatten_fn(method_name, make_flatten_fn(args.method, refine=refine))

    paths.ensure_output_dirs()
    manifest = load_manifest()
    subset = 2 if args.dev else args.subset
    entries = select_entries(manifest, split=args.split, subset=subset)
    if not entries:
        print("No manifest entries selected.", file=sys.stderr)
        return 1

    config = FlattenConfig()
    config.verbose = False
    if refine:
        config.negative_area_removal.enabled = False  # the whole point

    record = new_record(
        kind="probe",
        label=f"probe:{method_name}",
        manifest_id=manifest.get("created"),
        subjects=[
            {"subject": e["subject"], "hemi": e["hemi"], "split": e.get("split")}
            for e in entries
        ],
        method={"name": method_name, "config": config.to_dict(), "refine": refine},
        seeds={"note": "deterministic; flip-free init + CPU gradient descent"},
        repro_command="python -m benchmark.probe_tutte_init " + " ".join(sys.argv[1:]),
    )
    record.decision = {
        "hypothesis": (
            f"A flip-free {args.method} init removes the ~4-min initial NAR phase and "
            "reaches equal-or-lower distance distortion with zero flips and less runtime."
        ),
        "rationale": (
            "Tutte's theorem guarantees an injective (flip-free) embedding for disk "
            "topology when the boundary is mapped to a convex polygon, so the initial "
            "negative-area-removal phase becomes unnecessary."
        ),
    }

    print(
        f"Probe '{method_name}' on {len(entries)} hemispheres ({record.experiment_id})..."
    )
    t0 = time.time()
    save_dir = paths.RUNS_DIR / record.experiment_id if args.save else None
    result = evaluate(entries, config, method=method_name, save_dir=save_dir)
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
    n_failed = result["aggregate"].get("n_failed", 0)
    record.status = "ok" if n_failed == 0 else "partial"

    # Head-to-head conclusion vs the latest baseline.
    agg = result["aggregate"]
    base = _baseline_reference()
    if base and "mean_distortion" in agg:
        record.decision["conclusion"] = (
            f"{method_name}: mean_distortion {agg['mean_distortion']:.2f} vs baseline "
            f"{base.get('mean_distortion'):.2f}; total_flipped {agg['total_flipped']} vs "
            f"{base.get('total_flipped')}; mean_runtime {agg.get('mean_runtime_s', 0):.0f}s "
            f"vs {base.get('mean_runtime_s', 0):.0f}s."
        )

    Ledger().append(record)

    print("\n=== Probe aggregate ===")
    for k in (
        "n_patches",
        "n_failed",
        "mean_distortion",
        "worst_distortion",
        "total_flipped",
        "frac_patches_with_flips",
        "mean_runtime_s",
    ):
        if k in agg:
            print(f"  {k}: {agg[k]}")
    if base:
        print("\n=== vs baseline ===")
        for k in ("mean_distortion", "total_flipped", "mean_runtime_s"):
            print(f"  {k}: probe={agg.get(k)}  baseline={base.get(k)}")
    print(f"\nLogged to {Ledger().path}  (experiment {record.experiment_id})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
