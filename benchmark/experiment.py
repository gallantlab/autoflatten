"""General autoresearch experiment runner.

Generalizes :mod:`benchmark.run_baseline` and :mod:`benchmark.probe_tutte_init` into one
parametrized runner so optimization ideas can be fanned out and every one is logged to the
provenance ledger with a decision trace.

Knobs:
- ``--init {projection,tutte,lscm}`` — initial map (projection = current FreeSurfer clone).
- ``--skip-initial-nar`` / ``--skip-final-nar`` / ``--skip-spring`` — drop refinement phases.
- ``--skip-epoch {epoch_1,epoch_2,epoch_3}`` (repeatable) — drop a metric epoch.
- ``--k-ring N`` / ``--n-neighbors N`` — geodesic neighborhood (changes the cache key).

Each run logs a ``kind="experiment"`` record with the full config, metrics, per-subject
results, and a decision trace (hypothesis + head-to-head conclusion vs the baseline).

Usage
-----
    python -m benchmark.experiment --init tutte --skip-final-nar --subset 1 \\
        --label tutte+nofinalnar --hypothesis "flip-free start makes final NAR removable"
"""

from __future__ import annotations

import argparse
import sys
import time

from . import paths
from .harness import (
    evaluate,
    load_manifest,
    register_flatten_fn,
    select_entries,
)
from .ledger import Ledger, file_hash, new_record
from .probe_tutte_init import make_flatten_fn


def build_config(args):
    from autoflatten.flatten import FlattenConfig

    cfg = FlattenConfig()
    cfg.verbose = False
    cfg.kring.k_ring = args.k_ring
    cfg.kring.n_neighbors_per_ring = args.n_neighbors

    # A flip-free init makes the *initial* NAR moot; allow forcing it off for projection too.
    if args.init in ("tutte", "lscm") or args.skip_initial_nar:
        cfg.negative_area_removal.enabled = False
    if args.skip_final_nar:
        cfg.final_negative_area_removal.enabled = False
    if args.skip_spring:
        cfg.spring_smoothing.enabled = False
    for phase in cfg.phases:
        if phase.name in (args.skip_epoch or []):
            phase.enabled = False
    return cfg


def resolve_method(args):
    """Return (method_name, registered) for the chosen init."""
    if args.init == "projection":
        return "pyflatten", "pyflatten"
    method_name = args.label or f"{args.init}_init"
    register_flatten_fn(method_name, make_flatten_fn(args.init, refine=True))
    return method_name, method_name


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--init", default="tutte", choices=["projection", "tutte", "lscm"])
    ap.add_argument("--skip-initial-nar", action="store_true")
    ap.add_argument("--skip-final-nar", action="store_true")
    ap.add_argument("--skip-spring", action="store_true")
    ap.add_argument(
        "--skip-epoch", action="append", choices=["epoch_1", "epoch_2", "epoch_3"]
    )
    ap.add_argument("--k-ring", type=int, default=7)
    ap.add_argument("--n-neighbors", type=int, default=12)
    ap.add_argument("--dev", action="store_true")
    ap.add_argument("--split", default=None, choices=["train", "holdout"])
    ap.add_argument("--subset", type=int, default=None)
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--label", default=None, help="short experiment label")
    ap.add_argument("--hypothesis", default="", help="what you expect and why")
    args = ap.parse_args()

    cfg = build_config(args)
    method_name, registered = resolve_method(args)
    label = args.label or method_name

    paths.ensure_output_dirs()
    manifest = load_manifest()
    subset = 2 if args.dev else args.subset
    entries = select_entries(manifest, split=args.split, subset=subset)
    if not entries:
        print("No manifest entries selected.", file=sys.stderr)
        return 1

    # Capture the toggles in the method spec for provenance.
    toggles = {
        "init": args.init,
        "skip_initial_nar": args.skip_initial_nar or args.init in ("tutte", "lscm"),
        "skip_final_nar": args.skip_final_nar,
        "skip_spring": args.skip_spring,
        "skip_epoch": args.skip_epoch or [],
        "k_ring": args.k_ring,
        "n_neighbors": args.n_neighbors,
    }
    record = new_record(
        kind="experiment",
        label=f"exp:{label}",
        manifest_id=manifest.get("created"),
        subjects=[
            {"subject": e["subject"], "hemi": e["hemi"], "split": e.get("split")}
            for e in entries
        ],
        method={"name": method_name, "toggles": toggles, "config": cfg.to_dict()},
        seeds={"note": "deterministic CPU gradient descent"},
        repro_command="python -m benchmark.experiment " + " ".join(sys.argv[1:]),
    )
    if args.hypothesis:
        record.decision["hypothesis"] = args.hypothesis

    print(
        f"Experiment '{label}' ({method_name}) on {len(entries)} hemispheres ({record.experiment_id})..."
    )
    t0 = time.time()
    save_dir = paths.RUNS_DIR / record.experiment_id if args.save else None
    result = evaluate(entries, cfg, method=registered, save_dir=save_dir)
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
        "worst_distortion",
        "total_flipped",
        "frac_patches_with_flips",
        "mean_runtime_s",
    ):
        if k in agg:
            print(f"  {k}: {agg[k]}")
    if base_m:
        print("\n=== vs baseline ===")
        for k in ("mean_distortion", "total_flipped", "mean_runtime_s"):
            print(f"  {k}: exp={agg.get(k)}  baseline={base_m.get(k)}")
    print(f"\nLogged {record.experiment_id} -> {Ledger().path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
