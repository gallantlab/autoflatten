"""Probe: flip-free (Tutte/LSCM) initialization instead of FreeSurfer projection + NAR.

This validated the change that is now shipped as the package default: ``init_method``
on ``FlattenConfig`` defaults to ``"tutte"`` and the initial negative-area-removal (NAR)
phase defaults to off (see ``autoflatten.flatten.init`` and ``benchmark/FINDINGS.md``
§1). This module now only re-exports ``flipfree_init``/``scale_to_area`` for existing
probe callers and keeps ``make_flatten_fn`` for probes that need an init-only run or a
non-default method (e.g. ``lscm``) without going through the full CLI/backend surface.

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

from autoflatten.flatten.init import flipfree_init, scale_to_area  # noqa: F401

from . import paths
from .harness import evaluate, load_manifest, register_flatten_fn, select_entries
from .ledger import Ledger, file_hash, new_record


# ---------------------------------------------------------------------------------
# flatten_fn factory
# ---------------------------------------------------------------------------------
def make_flatten_fn(method: str = "tutte", refine: bool = True):
    """Build a ``flatten_fn`` that initializes flip-free and (optionally) refines.

    When ``refine`` is True, sets ``flattener.config.init_method`` to *method* and
    disables the initial NAR phase (its purpose is moot for a flip-free start), then
    runs the shared optimizer (epochs, final NAR, spring) via ``flattener.run()``. When
    False, the scaled init is returned directly (no optimizer call) to measure
    init-only quality.
    """

    def _fn(flattener):
        if not refine:
            init = flipfree_init(flattener.vertices, flattener.faces, method=method)
            return scale_to_area(init, np.asarray(flattener.faces), flattener.orig_area)
        flattener.config.init_method = method
        flattener.config.negative_area_removal.enabled = False
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
