"""Validate the FreeSurfer-free projection across the full benchmark (Phase 1).

The projection phase's only FreeSurfer dependency is
``core.map_cuts_to_subject`` -> ``mri_label2label --regmethod surface``. This runner
swaps it for the pure-Python KDTree mapper in :mod:`benchmark.projection` and checks that
the **final patch is bit-identical** (exact included-vertex set) to the cached FreeSurfer
patch on every manifest hemisphere, including held-out subjects -- the projection analogue
of the flattening overfit check.

It also records the per-hemisphere timing breakdown (mapping vs continuity+refine), so the
speed win on the mapping step (FreeSurfer ~36s subprocess -> ~0.3s KDTree) is logged.

Each hemisphere logs its own ledger record (crash-survivable); a final summary record
carries the aggregate exact-match count and timing.

Usage
-----
    python -m benchmark.validate_projection
    python -m benchmark.validate_projection --only sub-041 sub-052
"""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
import time

import numpy as np

from . import paths
from .harness import load_manifest
from .ledger import Ledger, new_record
from .projection import (
    DEFAULT_SUBJECTS_DIR,
    map_cuts_to_subject_python,
    project_python,
)

from autoflatten.freesurfer import read_patch


def _included_set(patch_path):
    _, orig_idx, _ = read_patch(patch_path)
    return set(int(i) for i in orig_idx)


def run_hemi(entry, subjects_dir):
    subject, hemi = entry["subject"], entry["hemi"]

    # time the mapping step on its own (the part that replaces FreeSurfer)
    from .projection import _load_template_vertex_dict

    vd = _load_template_vertex_dict(hemi)
    t0 = time.time()
    map_cuts_to_subject_python(vd, subject, hemi, subjects_dir=subjects_dir)
    map_s = time.time() - t0

    # full FS-free projection -> patch
    out_patch = str(paths.RUNS_DIR / f"projection_{subject}_{hemi}.patch.3d")
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        res = project_python(
            subject, hemi, subjects_dir=subjects_dir, out_patch=out_patch
        )
    total_s = time.time() - t0

    py = _included_set(res["patch_file"])
    fs = _included_set(entry["patch_path"])
    inter, union = len(py & fs), len(py | fs)
    jaccard = inter / union if union else float("nan")
    exact = py == fs

    return {
        "subject": subject,
        "hemi": hemi,
        "split": entry.get("split"),
        "exact": bool(exact),
        "jaccard": jaccard,
        "n_py": len(py),
        "n_fs": len(fs),
        "n_diff": len(py ^ fs),
        "map_s": map_s,
        "total_s": total_s,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="*", default=None, help="restrict to subjects")
    ap.add_argument("--subjects-dir", default=DEFAULT_SUBJECTS_DIR)
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
    print(f"Validating FS-free projection on {len(entries)} hemispheres...\n")
    print(
        f"{'subject':10} {'hemi':4} {'split':8} {'map_s':>6} {'total_s':>8} "
        f"{'n_inc':>8} {'jaccard':>8} {'exact':>6}"
    )
    for entry in entries:
        r = run_hemi(entry, args.subjects_dir)
        results.append(r)
        print(
            f"{r['subject']:10} {r['hemi']:4} {str(r['split']):8} "
            f"{r['map_s']:6.2f} {r['total_s']:8.1f} {r['n_py']:8d} "
            f"{r['jaccard'] * 100:7.2f}% {('YES' if r['exact'] else 'NO'):>6}"
        )

        rec = new_record(
            kind="experiment",
            label=f"exp:validate_projection:{r['subject']}.{r['hemi']}",
            manifest_id=manifest.get("created"),
            subjects=[
                {"subject": r["subject"], "hemi": r["hemi"], "split": r["split"]}
            ],
            method={
                "name": "fsfree_projection_vs_freesurfer",
                "mapper": "python_kdtree_union_push_pull",
                "subjects_dir": args.subjects_dir,
            },
            seeds={"note": "deterministic KDTree + NetworkX shortest paths"},
            repro_command="python -m benchmark.validate_projection --only "
            + r["subject"],
        )
        rec.metrics = {
            "exact_match": r["exact"],
            "jaccard": r["jaccard"],
            "n_included_py": r["n_py"],
            "n_included_fs": r["n_fs"],
            "n_symmetric_diff": r["n_diff"],
            "mapping_s": r["map_s"],
            "total_s": r["total_s"],
        }
        rec.per_subject = [{"subject": r["subject"], "hemi": r["hemi"], **rec.metrics}]
        rec.status = "ok"
        rec.decision["hypothesis"] = (
            "mri_label2label --regmethod surface is reproducible in pure Python as a "
            "union(push, pull) KDTree mapping on sphere.reg, yielding a bit-identical "
            "patch with no FreeSurfer dependency."
        )
        rec.decision["conclusion"] = (
            f"{r['split']}: exact patch match={r['exact']} (jaccard {r['jaccard'] * 100:.2f}%, "
            f"{r['n_diff']} vertices differ); mapping {r['map_s']:.2f}s "
            f"(FreeSurfer ~36s), total {r['total_s']:.1f}s."
        )
        ledger.append(rec)

    n_exact = sum(r["exact"] for r in results)
    map_times = np.array([r["map_s"] for r in results])
    tot_times = np.array([r["total_s"] for r in results])
    print(
        f"\nEXACT match: {n_exact}/{len(results)} hemispheres | "
        f"mapping mean {map_times.mean():.2f}s | total mean {tot_times.mean():.1f}s"
    )

    rec = new_record(
        kind="experiment",
        label="exp:validate_projection:summary",
        manifest_id=manifest.get("created"),
        subjects=[
            {"subject": r["subject"], "hemi": r["hemi"], "split": r["split"]}
            for r in results
        ],
        method={"name": "fsfree_projection_summary"},
        repro_command="python -m benchmark.validate_projection",
    )
    rec.metrics = {
        "n_hemispheres": len(results),
        "n_exact_match": int(n_exact),
        "all_exact": n_exact == len(results),
        "mapping_mean_s": float(map_times.mean()),
        "total_mean_s": float(tot_times.mean()),
    }
    rec.status = "ok"
    rec.decision["conclusion"] = (
        f"FS-free projection reproduces the FreeSurfer patch exactly on "
        f"{n_exact}/{len(results)} hemispheres; mapping {map_times.mean():.2f}s vs "
        f"FreeSurfer ~36s. Projection no longer requires FreeSurfer."
    )
    ledger.append(rec)
    print(f"\nLogged {len(results) + 1} records -> {ledger.path}")
    return 0 if n_exact == len(results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
