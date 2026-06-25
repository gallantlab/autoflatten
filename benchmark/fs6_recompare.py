"""Re-verify the FreeSurfer-6 vs pyflatten distortion comparison on the CORRECTED metric.

The old ``fs6_compare/true_comparison_20subj.csv`` scored both methods against truegeo built on
the *inflated* surface (the surface bug). This rebuilds a fiducial + chord-sanitized truegeo for
each FS6 hemi and re-scores BOTH the FreeSurfer flat (``{hemi}.flat``, never re-run) and a fresh
pyflatten ``robust_fast`` flat on the *same* patch, each at its own distance-optimal scale (wide
bracket, so neither method is penalized for a global scale offset).

Per hemi (``fs6_compare/runs/<subject>.<hemi>/``): ``{hemi}.patch.3d`` (shared input),
``{hemi}.flat`` (FreeSurfer output), ``{hemi}.smoothwm`` (symlink -> the subject's fiducial).
All three share vertex order, so the FS UV maps directly onto the truegeo columns.

4 hemis concurrent at 8 CPUs. Resumable (a hemi already in the CSV is skipped). Output ->
``fs6_compare/true_comparison_fiducial.csv``.

Usage
-----
    python -m benchmark.fs6_recompare --runs-dir /data2/projects/autoflatten/fs6_compare/runs
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
import time
from pathlib import Path

from . import paths
from .group_flatten import LANES, N_LANES, pin_cpus
from .time_cores import make_config

OUT_CSV = paths.DATA_ROOT / "fs6_compare" / "true_comparison_fiducial.csv"
FIELDS = [
    "subject",
    "hemi",
    "method",
    "true_local_at_optscale",
    "true_global_at_optscale",
    "opt_scale",
    "n_pairs",
    "status",
    "error",
]


def _score(uv, ref):
    # Use the shared metric (benchmark.truedist) so the FS6 comparison stays on the same
    # definition as the group/cores tables. The bracket is widened to [0.5, 2.0]: FreeSurfer
    # does not apply the distance-optimal expansion, so each method is scored at its own
    # optimum. local-at-optscale is true_distortion on the scale-applied uv (identical to
    # masking dg<=R after the global fit).
    from . import truedist

    full = truedist.true_distortion_full(
        uv, ref, scale_bracket=(0.5, 2.0), n_scales=151
    )
    opt = full["opt_scale"]
    loc = truedist.true_distortion(uv * opt, ref)
    return {
        "opt_scale": round(opt, 4),
        "true_global_at_optscale": round(float(full["true_global_at_optscale"]), 4),
        "true_local_at_optscale": round(float(loc["true_mean_distortion"]), 4),
        "n_pairs": int(full["n_pairs_global"]),
    }


def _append(row):
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    new = not OUT_CSV.exists()
    with open(OUT_CSV, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in FIELDS})


def run_worker(args) -> int:
    pin_cpus([int(c) for c in args.cpus.split(",")])
    import numpy as np
    from autoflatten.flatten import SurfaceFlattener
    from autoflatten.freesurfer import read_patch
    from . import truedist
    from .probe_tutte_init import make_flatten_fn

    run_dir = Path(args.run_dir)
    subj, hemi = args.subject, args.hemi
    base_row = {"subject": subj, "hemi": hemi}
    try:
        patch = run_dir / f"{hemi}.patch.3d"
        base = run_dir / f"{hemi}.smoothwm"  # symlink -> fiducial
        fl = SurfaceFlattener(make_config("robust_fast"))
        fl.load_data(str(patch), str(base))
        ref = truedist.compute_truegeo(
            fl
        )  # fiducial + chord-sanitized (stamps provenance)
        np.savez(run_dir / "truegeo_fiducial.npz", **ref)

        # FreeSurfer flat (never re-run)
        uv_fs = read_patch(str(run_dir / f"{hemi}.flat"))[0][:, :2].astype(float)
        # pyflatten robust_fast on the SAME patch
        fl.compute_kring_distances(cache_path=str(run_dir / "kring_fiducial.npz"))
        fl.prepare_optimization()
        uv_py = np.asarray(make_flatten_fn("tutte", refine=True)(fl))

        for method, uv in (("freesurfer6", uv_fs), ("robust_fast", uv_py)):
            s = _score(uv, ref)
            _append({**base_row, "method": method, "status": "ok", **s})
            print(
                f"[ok] {subj} {hemi} {method}: local@opt={s['true_local_at_optscale']}% "
                f"global@opt={s['true_global_at_optscale']}% scale={s['opt_scale']}",
                flush=True,
            )
    except Exception as exc:  # noqa: BLE001
        _append(
            {
                **base_row,
                "method": "ERROR",
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        print(f"[error] {subj} {hemi}: {exc}", flush=True)
    return 0


def _done() -> set[tuple[str, str]]:
    # A (subject, hemi) is done only when BOTH methods (FS6 + pyflatten) recorded status=ok.
    from collections import Counter

    c: Counter = Counter()
    if OUT_CSV.exists():
        with open(OUT_CSV) as fh:
            for r in csv.DictReader(fh):
                if r.get("status") == "ok":
                    c[(r["subject"], r["hemi"])] += 1
    return {k for k, n in c.items() if n >= 2}


def run_driver(args) -> int:
    runs = Path(args.runs_dir)
    hemis = []
    for d in sorted(runs.glob("sub-*.??")):
        m = re.match(r"(sub-\d+)\.(lh|rh)$", d.name)
        if (
            m
            and (d / f"{m.group(2)}.flat").exists()
            and (d / f"{m.group(2)}.patch.3d").exists()
        ):
            hemis.append((m.group(1), m.group(2), d))
    done = _done()
    todo = [(s, h, d) for (s, h, d) in hemis if (s, h) not in done]
    print(
        f"fs6_recompare: {len(hemis)} hemis, {len(done)} done, {len(todo)} to run "
        f"({N_LANES} lanes). Out -> {OUT_CSV}",
        flush=True,
    )

    queue = list(todo)
    running: dict[int, subprocess.Popen] = {}
    repo = Path(__file__).resolve().parent.parent

    def launch(lane, cell):
        s, h, d = cell
        cpus = ",".join(str(x) for x in LANES[lane])
        cmd = [
            sys.executable,
            "-m",
            "benchmark.fs6_recompare",
            "--worker",
            "--run-dir",
            str(d),
            "--subject",
            s,
            "--hemi",
            h,
            "--cpus",
            cpus,
        ]
        return subprocess.Popen(cmd, cwd=repo)

    for lane in range(min(N_LANES, len(queue))):
        running[lane] = launch(lane, queue.pop(0))
    while running:
        time.sleep(2)
        for lane in list(running):
            if running[lane].poll() is None:
                continue
            if queue:
                running[lane] = launch(lane, queue.pop(0))
            else:
                del running[lane]
    print(f"\nDone -> {OUT_CSV}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-dir", default=str(paths.DATA_ROOT / "fs6_compare" / "runs"))
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--run-dir")
    ap.add_argument("--subject")
    ap.add_argument("--hemi", choices=["lh", "rh"])
    ap.add_argument("--cpus")
    args = ap.parse_args()
    return run_worker(args) if args.worker else run_driver(args)


if __name__ == "__main__":
    raise SystemExit(main())
