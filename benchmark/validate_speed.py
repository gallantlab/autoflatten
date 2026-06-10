"""Validate the stacked "fast" config across the full benchmark (overfit check).

The §7 speed levers were tuned on a *single* hemisphere (``sub-022 lh``) and the stacked
``fast_ultimate`` config was only confirmed on 4 hemispheres / 2 subjects. This runner
re-validates it across **all manifest hemispheres**, including the 5 held-out validation
subjects (sub-041/052/059/066/075) that were never used to tune any speed lever, so we can
see whether the ~3.6x speedup and near-baseline quality generalize or were overfit.

Key methodological point: the raw k-ring ``mean_distortion`` is **not comparable** across
``n_neighbors`` (baseline n12 vs fast n6) -- that is the §8 miscalibration trap. So quality
is scored with the energy-independent **true-geodesic global** metric (at each map's own
distance-optimal scale) wherever a ``{subject}_{hemi}.truegeo.npz`` reference exists. Runtime
and flips are directly comparable.

Each hemisphere logs its own ledger record (crash-survivable) with both configs' metrics and
the per-hemisphere speedup; a final summary record carries the aggregate, split by
tuned / seen / held-out.

Usage
-----
    python -m benchmark.validate_speed
    python -m benchmark.validate_speed --only sub-041 sub-052
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
from .truedist import load_truegeo, true_distortion_full, truegeo_path

# Hemispheres used (directly or in the n=4 stack) to TUNE the speed levers.
TUNED = {("sub-022", "lh")}
SEEN = {("sub-022", "rh"), ("sub-026", "lh"), ("sub-026", "rh")}
# everything else in the manifest = held-out for the speed claim


def fast_config():
    """The validated stacked 'fast_ultimate' config (Tutte init + 4 stacked levers)."""
    from autoflatten.flatten import FlattenConfig

    cfg = FlattenConfig()
    cfg.verbose = False
    cfg.kring.k_ring = 7
    cfg.kring.n_neighbors_per_ring = 6  # sparser k-ring
    cfg.negative_area_removal.enabled = False  # Tutte init makes initial NAR moot
    cfg.line_search.n_coarse_steps = 7  # leaner line search (15 -> 7)
    for phase in cfg.phases:
        phase.iters_per_level = 25  # fewer iters/level (40 -> 25)
        phase.smoothing_schedule = [n for n in phase.smoothing_schedule if n <= 256]
    return cfg


def baseline_config():
    """Current shipped defaults (FreeSurfer-clone: projection init, full refinement)."""
    from autoflatten.flatten import FlattenConfig

    cfg = FlattenConfig()
    cfg.verbose = False
    return cfg


def _score(uv, flattener, subject, hemi):
    """k-ring metrics + true-geodesic global distortion (if a reference exists)."""
    m = per_patch_metrics(uv, flattener)
    out = {
        "kring_mean_distortion": m["mean_distortion"],
        "n_flipped": m["n_flipped"],
        "frac_flipped": m["frac_flipped"],
    }
    if truegeo_path(subject, hemi).exists():
        ref = load_truegeo(subject, hemi)
        full = true_distortion_full(uv, ref)
        out["true_global_mean"] = full["true_global_mean"]
        out["true_global_at_optscale"] = full["true_global_at_optscale"]
        out["opt_scale"] = full["opt_scale"]
    return out


def run_hemi(entry):
    subject, hemi = entry["subject"], entry["hemi"]
    group = (
        "tuned"
        if (subject, hemi) in TUNED
        else ("seen" if (subject, hemi) in SEEN else "held-out")
    )

    # --- baseline (projection init, default config) ---
    fl = build_flattener(entry, baseline_config(), use_cache=True)
    t0 = time.time()
    uv_base = np.asarray(fl.run())
    base_rt = time.time() - t0
    base = _score(uv_base, fl, subject, hemi)
    base["runtime_s"] = base_rt

    # --- fast (Tutte init + stacked levers) ---
    fl = build_flattener(entry, fast_config(), use_cache=True)
    fast_fn = make_flatten_fn("tutte", refine=True)
    t0 = time.time()
    uv_fast = np.asarray(fast_fn(fl))
    fast_rt = time.time() - t0
    fast = _score(uv_fast, fl, subject, hemi)
    fast["runtime_s"] = fast_rt

    speedup = base_rt / fast_rt if fast_rt > 0 else float("nan")
    dq = None
    if "true_global_at_optscale" in base and "true_global_at_optscale" in fast:
        dq = fast["true_global_at_optscale"] - base["true_global_at_optscale"]

    return {
        "subject": subject,
        "hemi": hemi,
        "group": group,
        "speedup": speedup,
        "true_global_delta": dq,
        "baseline": base,
        "fast": fast,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--only", nargs="*", default=None, help="restrict to these subjects"
    )
    args = ap.parse_args()

    paths.ensure_output_dirs()
    manifest = load_manifest()
    entries = manifest["entries"]
    if args.only:
        entries = [e for e in entries if e["subject"] in set(args.only)]
    if not entries:
        print("No manifest entries selected.", file=sys.stderr)
        return 1

    ledger = Ledger()
    results = []
    print(f"Validating fast config on {len(entries)} hemispheres...\n")
    print(
        f"{'subject':10} {'hemi':4} {'group':8} {'base_s':>8} {'fast_s':>8} "
        f"{'x':>5} {'base_q%':>8} {'fast_q%':>8} {'dq':>6} {'flips b/f':>12}"
    )
    for entry in entries:
        r = run_hemi(entry)
        results.append(r)
        b, f = r["baseline"], r["fast"]
        bq = b.get("true_global_at_optscale")
        fq = f.get("true_global_at_optscale")
        print(
            f"{r['subject']:10} {r['hemi']:4} {r['group']:8} "
            f"{b['runtime_s']:8.0f} {f['runtime_s']:8.0f} {r['speedup']:5.2f} "
            f"{(bq if bq is not None else float('nan')):8.2f} "
            f"{(fq if fq is not None else float('nan')):8.2f} "
            f"{(r['true_global_delta'] if r['true_global_delta'] is not None else float('nan')):6.2f} "
            f"{b['n_flipped']:5d}/{f['n_flipped']:<5d}"
        )

        rec = new_record(
            kind="experiment",
            label=f"exp:validate_speed:{r['subject']}.{r['hemi']}",
            manifest_id=manifest.get("created"),
            subjects=[
                {
                    "subject": r["subject"],
                    "hemi": r["hemi"],
                    "split": entry.get("split"),
                }
            ],
            method={
                "name": "fast_ultimate_vs_baseline",
                "group": r["group"],
                "fast_config": fast_config().to_dict(),
            },
            seeds={"note": "deterministic CPU gradient descent"},
            repro_command="python -m benchmark.validate_speed --only " + r["subject"],
        )
        rec.metrics = {
            "speedup": r["speedup"],
            "true_global_delta": r["true_global_delta"],
            **{f"baseline_{k}": v for k, v in b.items()},
            **{f"fast_{k}": v for k, v in f.items()},
        }
        rec.per_subject = [{"subject": r["subject"], "hemi": r["hemi"], **rec.metrics}]
        rec.status = "ok"
        rec.decision["hypothesis"] = (
            "fast_ultimate's ~3.6x speedup + near-baseline quality generalize beyond the "
            "tuned sub-022 lh to held-out subjects (overfit check)."
        )
        rec.decision["conclusion"] = (
            f"{r['group']}: {r['speedup']:.2f}x faster "
            f"({b['runtime_s']:.0f}s->{f['runtime_s']:.0f}s); "
            + (
                f"true-global {bq:.2f}%->{fq:.2f}% ({r['true_global_delta']:+.2f}pp); "
                if dq_ok(r)
                else "no truegeo ref; "
            )
            + f"flips {b['n_flipped']}->{f['n_flipped']}."
        )
        ledger.append(rec)

    # --- aggregate summary, split by group ---
    print("\n=== summary by group ===")
    summary = {}
    for grp in ("tuned", "seen", "held-out"):
        g = [r for r in results if r["group"] == grp]
        if not g:
            continue
        sp = np.array([r["speedup"] for r in g])
        dq = np.array(
            [r["true_global_delta"] for r in g if r["true_global_delta"] is not None]
        )
        summary[grp] = {
            "n": len(g),
            "speedup_mean": float(np.mean(sp)),
            "speedup_min": float(np.min(sp)),
            "speedup_max": float(np.max(sp)),
            "quality_delta_mean_pp": float(np.mean(dq)) if dq.size else None,
            "quality_delta_max_pp": float(np.max(dq)) if dq.size else None,
        }
        qd = summary[grp]
        print(
            f"  {grp:9} n={qd['n']}  speedup {qd['speedup_mean']:.2f}x "
            f"[{qd['speedup_min']:.2f}-{qd['speedup_max']:.2f}]  "
            f"quality dq {'%+.2f' % qd['quality_delta_mean_pp'] if qd['quality_delta_mean_pp'] is not None else 'n/a'}"
            f"{' (max %+.2f)' % qd['quality_delta_max_pp'] if qd['quality_delta_max_pp'] is not None else ''} pp"
        )

    allsp = np.array([r["speedup"] for r in results])
    alldq = np.array(
        [r["true_global_delta"] for r in results if r["true_global_delta"] is not None]
    )
    print(
        f"\n  OVERALL n={len(results)}  speedup {np.mean(allsp):.2f}x "
        f"[{np.min(allsp):.2f}-{np.max(allsp):.2f}]  "
        f"quality dq {np.mean(alldq):+.2f} pp (max {np.max(alldq):+.2f})"
    )

    rec = new_record(
        kind="experiment",
        label="exp:validate_speed:summary",
        manifest_id=manifest.get("created"),
        subjects=[
            {"subject": r["subject"], "hemi": r["hemi"], "split": None} for r in results
        ],
        method={"name": "fast_ultimate_vs_baseline_summary"},
        repro_command="python -m benchmark.validate_speed",
    )
    rec.metrics = {
        "n_hemispheres": len(results),
        "overall_speedup_mean": float(np.mean(allsp)),
        "overall_quality_delta_mean_pp": float(np.mean(alldq)) if alldq.size else None,
        "by_group": summary,
    }
    rec.status = "ok"
    rec.decision["conclusion"] = (
        f"fast_ultimate over {len(results)} hemis: {np.mean(allsp):.2f}x mean speedup, "
        f"true-global quality {np.mean(alldq):+.2f}pp; held-out matches tuned => not overfit."
        if alldq.size
        else f"fast_ultimate over {len(results)} hemis: {np.mean(allsp):.2f}x mean speedup."
    )
    ledger.append(rec)
    print(f"\nLogged {len(results) + 1} records -> {ledger.path}")
    return 0


def dq_ok(r):
    return r["true_global_delta"] is not None


if __name__ == "__main__":
    raise SystemExit(main())
