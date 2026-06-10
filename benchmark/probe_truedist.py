"""Probe: optimize quality (true geodesic distortion) at the cost of a slight slowdown.

Section 8 of FINDINGS showed the *optimized* map is slightly worse than the raw Tutte init
in **true geodesic** terms (~18% vs ~16%): the k-ring energy over-fits very-local distances
and lets medium-range (10-30 mm) distances drift. This probe tests config levers that should
add medium-range constraints / reduce that drift, scored with the energy-independent true
geodesic yardstick (:mod:`benchmark.truedist`) so maps are comparable across ``k_ring``.

Each run uses the validated Tutte flip-free init and the existing refinement, varying one
knob (``--k-ring`` / ``--n-neighbors``), and logs both the k-ring metric and the true metric
to the ledger. Single-hemisphere screening (default sub-022 lh, which has a cached truegeo
reference); promote a winner to multi-hemi afterward.

Usage
-----
    python -m benchmark.probe_truedist --k-ring 7 --n-neighbors 12 --label k7_baseline
    python -m benchmark.probe_truedist --k-ring 11 --n-neighbors 12 --label k11
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from . import paths
from .harness import build_flattener, load_manifest
from .ledger import Ledger, new_record
from .metrics import per_patch_metrics
from .probe_tutte_init import make_flatten_fn
from .truedist import load_truegeo, true_distortion, true_distortion_banded


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subject", default="sub-022")
    ap.add_argument("--hemi", default="lh")
    ap.add_argument("--k-ring", type=int, default=7)
    ap.add_argument("--n-neighbors", type=int, default=12)
    ap.add_argument("--init", default="tutte", choices=["tutte", "lscm", "projection"])
    ap.add_argument(
        "--skip-epoch", action="append", choices=["epoch_1", "epoch_2", "epoch_3"]
    )
    ap.add_argument(
        "--target-scale",
        type=float,
        default=None,
        help="multiply k-ring targets by this factor (rescales the effective correction "
        "without rebuilding the cache; <1 = more compact targets)",
    )
    ap.add_argument(
        "--save-uv", action="store_true", help="save the flat patch for plotting"
    )
    ap.add_argument("--label", default=None)
    ap.add_argument("--hypothesis", default="")
    args = ap.parse_args()

    from autoflatten.flatten import FlattenConfig

    cfg = FlattenConfig()
    cfg.verbose = False
    cfg.kring.k_ring = args.k_ring
    cfg.kring.n_neighbors_per_ring = args.n_neighbors
    if args.init in ("tutte", "lscm"):
        cfg.negative_area_removal.enabled = False
    for phase in cfg.phases:
        if phase.name in (args.skip_epoch or []):
            phase.enabled = False

    label = args.label or f"truedist_{args.init}_k{args.k_ring}_n{args.n_neighbors}"

    paths.ensure_output_dirs()
    manifest = load_manifest()
    entry = [
        e
        for e in manifest["entries"]
        if e["subject"] == args.subject and e["hemi"] == args.hemi
    ]
    if not entry:
        print(f"No manifest entry for {args.subject} {args.hemi}", file=sys.stderr)
        return 1
    entry = entry[0]

    ref = load_truegeo(args.subject, args.hemi)

    record = new_record(
        kind="experiment",
        label=f"exp:{label}",
        manifest_id=manifest.get("created"),
        subjects=[
            {"subject": args.subject, "hemi": args.hemi, "split": entry.get("split")}
        ],
        method={
            "name": label,
            "init": args.init,
            "k_ring": args.k_ring,
            "n_neighbors": args.n_neighbors,
            "metric": "true_geodesic (heat, R=30mm, 200 src) + k-ring",
            "config": cfg.to_dict(),
        },
        repro_command="python -m benchmark.probe_truedist " + " ".join(sys.argv[1:]),
    )
    if args.hypothesis:
        record.decision["hypothesis"] = args.hypothesis

    print(
        f"True-distortion probe '{label}' on {args.subject} {args.hemi} ({record.experiment_id})..."
    )
    t0 = time.time()
    flatten_fn = (
        make_flatten_fn(args.init, refine=True) if args.init != "projection" else None
    )

    flattener = build_flattener(entry, cfg, use_cache=True)
    if args.target_scale is not None:
        # Rescale the effective graph-distance correction without rebuilding the cache:
        # target_new = target_old * scale  (scale<1 => more compact targets).
        flattener.targets_jax = flattener.targets_jax * args.target_scale
    t_opt = time.time()
    if flatten_fn is None:
        uv = np.asarray(flattener.run())
    else:
        uv = np.asarray(flatten_fn(flattener))
    runtime = time.time() - t_opt

    kring_m = per_patch_metrics(uv, flattener)
    true_m = true_distortion(uv, ref)
    banded = true_distortion_banded(uv, ref)

    artifact = None
    if args.save_uv:
        save_dir = paths.RUNS_DIR / record.experiment_id
        save_dir.mkdir(parents=True, exist_ok=True)
        artifact = str(save_dir / f"{args.subject}.{args.hemi}.flat.patch.3d")
        flattener.save_result(uv, artifact)
        record.artifacts = [{"path": artifact, "kind": "flat_patch"}]

    record.runtime_s = time.time() - t0
    record.metrics = {
        **{f"kring_{k}": v for k, v in kring_m.items()},
        **true_m,
        **banded,
        "opt_runtime_s": runtime,
    }
    record.per_subject = [
        {"subject": args.subject, "hemi": args.hemi, **record.metrics}
    ]
    record.status = "ok"
    record.decision["conclusion"] = (
        f"{label}: TRUE mean {true_m['true_mean_distortion']:.2f}% "
        f"(p90 {true_m['true_p90_distortion']:.2f}%); "
        f"k-ring {kring_m['mean_distortion']:.2f}%; "
        f"flips {kring_m['n_flipped']}; opt {runtime:.0f}s."
    )
    Ledger().append(record)

    print("\n=== metrics ===")
    print(f"  TRUE mean distortion:   {true_m['true_mean_distortion']:.3f}%")
    print(f"  TRUE p90 distortion:    {true_m['true_p90_distortion']:.3f}%")
    print(f"  TRUE median distortion: {true_m['true_median_distortion']:.3f}%")
    print(f"  k-ring mean distortion: {kring_m['mean_distortion']:.3f}%")
    print(f"  flipped triangles:      {kring_m['n_flipped']}")
    print("  TRUE by band:           ", end="")
    for lo, hi in ((0, 5), (5, 15), (15, 30)):
        key = f"band_{lo}_{hi}_mean"
        if key in banded:
            print(f"{lo}-{hi}mm={banded[key]:.1f}%  ", end="")
    print()
    print(f"  opt runtime:            {runtime:.0f}s")
    if artifact:
        print(f"  saved uv:               {artifact}")
    print(f"\nLogged {record.experiment_id} -> {Ledger().path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
