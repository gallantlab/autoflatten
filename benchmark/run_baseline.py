"""Run the current-default ("FreeSurfer-clone") baseline and log it to the ledger.

Establishes the reference quality + runtime the alternative methods must beat, and
(optionally) asserts the optimizer is deterministic so a single run per experiment is
defensible.

Usage
-----
    python -m benchmark.run_baseline --dev                 # 2-3 hemispheres, fast
    python -m benchmark.run_baseline --split train         # full train split
    python -m benchmark.run_baseline --dev --check-determinism
"""

from __future__ import annotations

import argparse
import sys
import time

from . import paths
from .harness import (
    evaluate,
    evaluate_one,
    load_manifest,
    pyflatten_flatten_fn,
    select_entries,
)
from .ledger import Ledger, file_hash, new_record


def _make_config(verbose: bool):
    from autoflatten.flatten import FlattenConfig

    config = FlattenConfig()
    config.verbose = verbose
    return config


def check_determinism(entry, config) -> dict:
    """Flatten one patch twice and compare the headline metrics."""
    r1 = evaluate_one(entry, config, pyflatten_flatten_fn)
    r2 = evaluate_one(entry, config, pyflatten_flatten_fn)
    same = (
        r1.get("status") == "ok" == r2.get("status")
        and r1["n_flipped"] == r2["n_flipped"]
        and abs(r1["mean_distortion"] - r2["mean_distortion"]) < 1e-6
    )
    return {
        "deterministic": bool(same),
        "run1": {
            "mean_distortion": r1.get("mean_distortion"),
            "n_flipped": r1.get("n_flipped"),
        },
        "run2": {
            "mean_distortion": r2.get("mean_distortion"),
            "n_flipped": r2.get("n_flipped"),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dev", action="store_true", help="run on a tiny subset (first 2 hemispheres)"
    )
    ap.add_argument(
        "--split",
        default=None,
        choices=["train", "holdout"],
        help="restrict to a split",
    )
    ap.add_argument("--subset", type=int, default=None, help="take first N hemispheres")
    ap.add_argument(
        "--save", action="store_true", help="also save flat patches to runs/<id>/"
    )
    ap.add_argument("--check-determinism", action="store_true")
    ap.add_argument("--verbose", action="store_true", help="print optimizer progress")
    args = ap.parse_args()

    paths.ensure_output_dirs()
    manifest = load_manifest()
    subset = 2 if args.dev else args.subset
    entries = select_entries(manifest, split=args.split, subset=subset)
    if not entries:
        print("No manifest entries selected.", file=sys.stderr)
        return 1

    config = _make_config(verbose=args.verbose)
    record = new_record(
        kind="baseline",
        label="baseline:pyflatten-defaults",
        manifest_id=manifest.get("created"),
        subjects=[
            {"subject": e["subject"], "hemi": e["hemi"], "split": e.get("split")}
            for e in entries
        ],
        method={"name": "pyflatten", "config": config.to_dict()},
        seeds={"note": "deterministic CPU gradient descent; no RNG seed"},
        repro_command="python -m benchmark.run_baseline " + " ".join(sys.argv[1:]),
    )

    print(
        f"Baseline on {len(entries)} hemispheres (experiment {record.experiment_id})..."
    )
    t0 = time.time()

    if args.check_determinism:
        det = check_determinism(entries[0], config)
        record.decision["determinism_check"] = det
        print(f"  determinism: {det['deterministic']} ({det['run1']} vs {det['run2']})")

    save_dir = paths.RUNS_DIR / record.experiment_id if args.save else None
    result = evaluate(entries, config, method="pyflatten", save_dir=save_dir)

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

    Ledger().append(record)

    agg = result["aggregate"]
    print("\n=== Baseline aggregate ===")
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
    print(f"\nLogged to {Ledger().path}  (experiment {record.experiment_id})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
