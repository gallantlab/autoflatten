"""Phase 2: does projection refinement improve the downstream flatmap? (ablation)

Constraint (user): exactly 5 cuts, in their current anatomical positions. So the template
placement is fixed; the only question is the **refinement** that turns the raw mapped cuts
into the final patch:

- ``ensure_continuous_cuts``  -- connects disconnected mapped-cut components.
- ``refine_cuts_with_geodesic`` -- replaces each thick mapped cut blob with a thin geodesic
  shortest path between endpoints (start = farthest-from-mwall cut vertex; end = max-clearance
  medial-wall anchor).

This probe ablates those two steps and scores each resulting patch by the **downstream
flatmap distortion** (the fast flattener + global true-geodesic metric, recomputed per
variant because the patch vertex set changes) plus flips and patch size. It answers: are the
refinements necessary, and do they actually reduce distortion?

Each variant patch gets a **fresh** k-ring (``use_cache=False``) -- the on-disk cache is
keyed only on (subject, hemi, k, n), so variants would otherwise collide.

Usage
-----
    python -m benchmark.probe_refinement --hemis sub-022:lh
    python -m benchmark.probe_refinement            # all dev hemis, all variants
"""

from __future__ import annotations

import argparse
import contextlib
import io
import time

import numpy as np

from . import paths
from .ledger import Ledger, new_record
from .metrics import per_patch_metrics
from .probe_tutte_init import make_flatten_fn
from .projection import project_python
from .truedist import compute_truegeo, true_distortion_full
from .validate_speed import fast_config

from autoflatten.flatten.algorithm import count_boundary_loops

VARIANTS = [
    # shipped pipeline (Euclidean-shortest geodesic refinement)
    {"label": "geodesic", "continuity": True, "refine": True, "weight": "euclidean"},
    # thick mapped cuts, no geodesic thinning
    {
        "label": "continuity_only",
        "continuity": True,
        "refine": False,
        "weight": "euclidean",
    },
    # raw mapped cuts (no continuity, no refinement)
    {
        "label": "mapped_only",
        "continuity": False,
        "refine": False,
        "weight": "euclidean",
    },
    # (c) curvature-weighted geodesic: route cut paths along sulcal fundi
    {
        "label": "geodesic_curv",
        "continuity": True,
        "refine": True,
        "weight": "curvature",
        "alpha": 0.1,
    },
]

SURF = "fiducial"


def _surface_path(subject, hemi, subjects_dir):
    return f"{subjects_dir}/{subject}/surf/{hemi}.{SURF}"


def run_variant(subject, hemi, spec, subjects_dir):
    label = spec["label"]
    out_patch = str(paths.RUNS_DIR / f"refine_{label}_{subject}_{hemi}.patch.3d")
    with contextlib.redirect_stdout(io.StringIO()):
        proj = project_python(
            subject,
            hemi,
            subjects_dir=subjects_dir,
            continuity=spec["continuity"],
            refine_geodesic=spec["refine"],
            refine_weight=spec.get("weight", "euclidean"),
            curv_alpha=spec.get("alpha", 0.1),
            out_patch=out_patch,
        )
    n_patch = len(proj["patch_vertices"])

    entry = {
        "subject": subject,
        "hemi": hemi,
        "patch_path": proj["patch_file"],
        "surface_path": _surface_path(subject, hemi, subjects_dir),
    }

    # flatten (fresh k-ring; fast config + Tutte init)
    from .harness import build_flattener

    t0 = time.time()
    fl = build_flattener(entry, fast_config(), use_cache=False)
    # patch topology: a valid flat patch is a single boundary loop (a disk)
    n_loops, _ = count_boundary_loops(fl.faces)
    uv = np.asarray(make_flatten_fn("tutte", refine=True)(fl))
    rt = time.time() - t0

    m = per_patch_metrics(uv, fl)
    ref = compute_truegeo(fl)
    full = true_distortion_full(uv, ref)

    return {
        "label": label,
        "n_patch": n_patch,
        "n_boundary_loops": int(n_loops),
        "n_flipped": int(m["n_flipped"]),
        "true_global_mean": full["true_global_mean"],
        "true_global_at_optscale": full["true_global_at_optscale"],
        "opt_scale": full["opt_scale"],
        "runtime_s": rt,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hemis", nargs="*", default=["sub-022:lh"])
    ap.add_argument("--variants", nargs="*", default=[v["label"] for v in VARIANTS])
    ap.add_argument(
        "--subjects-dir",
        default="/data2/projects/idem/exps/narratives/datalad-narratives/derivatives/freesurfer",
    )
    args = ap.parse_args()

    paths.ensure_output_dirs()
    variants = [v for v in VARIANTS if v["label"] in set(args.variants)]
    ledger = Ledger()

    print(
        f"{'hemi':11} {'variant':16} {'n_patch':>8} {'loops':>5} {'flips':>6} "
        f"{'glob%':>7} {'opt%':>7} {'rt_s':>6}"
    )
    for spec in args.hemis:
        subject, hemi = spec.split(":")
        for vspec in variants:
            label = vspec["label"]
            try:
                r = run_variant(subject, hemi, vspec, args.subjects_dir)
            except Exception as e:  # noqa: BLE001 - record the failure, keep going
                print(f"{subject + ' ' + hemi:11} {label:16} FAILED: {e}")
                rec = new_record(
                    kind="experiment",
                    label=f"exp:probe_refinement:{subject}.{hemi}:{label}",
                    subjects=[{"subject": subject, "hemi": hemi}],
                    method={"name": "refinement_ablation", "variant": label},
                    repro_command=f"python -m benchmark.probe_refinement --hemis {spec} --variants {label}",
                )
                rec.status = "error"
                rec.decision["conclusion"] = f"{label} failed: {e}"
                ledger.append(rec)
                continue

            print(
                f"{subject + ' ' + hemi:11} {r['label']:16} {r['n_patch']:8d} "
                f"{r['n_boundary_loops']:5d} {r['n_flipped']:6d} "
                f"{r['true_global_mean']:7.2f} {r['true_global_at_optscale']:7.2f} "
                f"{r['runtime_s']:6.0f}"
            )

            rec = new_record(
                kind="experiment",
                label=f"exp:probe_refinement:{subject}.{hemi}:{label}",
                subjects=[{"subject": subject, "hemi": hemi}],
                method={
                    "name": "refinement_ablation",
                    "variant": label,
                    "continuity": vspec["continuity"],
                    "refine_geodesic": vspec["refine"],
                    "refine_weight": vspec.get("weight", "euclidean"),
                    "curv_alpha": vspec.get("alpha"),
                    "flatten": "fast_ultimate+tutte",
                },
                repro_command=f"python -m benchmark.probe_refinement --hemis {spec} --variants {label}",
            )
            rec.metrics = {k: v for k, v in r.items() if k != "label"}
            rec.per_subject = [{"subject": subject, "hemi": hemi, **rec.metrics}]
            rec.status = "ok"
            rec.decision["hypothesis"] = (
                "Geodesic refinement (thin cut paths) lowers downstream flatmap distortion "
                "vs raw/continuity-only thick mapped cuts; continuity is needed for a valid "
                "single-boundary-loop patch."
            )
            rec.decision["conclusion"] = (
                f"{label}: {r['n_boundary_loops']} boundary loop(s), {r['n_flipped']} flips, "
                f"global {r['true_global_mean']:.2f}% (opt {r['true_global_at_optscale']:.2f}%), "
                f"patch {r['n_patch']} verts."
            )
            ledger.append(rec)

    print(f"\nLogged to {ledger.path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
