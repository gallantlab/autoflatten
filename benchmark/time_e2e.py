"""End-to-end AutoFlatten timing at 16 cores (paper Deliverable 1).

Times the *full* pipeline -- FreeSurfer-free projection (cut mapping -> patch) + k-ring prep +
flatten -- for both shipped variants, on a fixed set of subjects, both hemispheres run
sequentially with all 16 physical cores. The per-brain headline = ``lh.total_s + rh.total_s``.

Unlike the core-scaling sweep this runs as a single long process (cores fixed at 16 throughout),
pinned once via :func:`benchmark.time_cores.pin_cores`. Resumable: a (subject, hemi, config)
already recorded ``status=ok`` is skipped. All outputs are timestamped under
``<DATA_ROOT>/paper_bench_2026/e2e_<TS>/``.

Usage
-----
    python -m benchmark.time_e2e --run-dir <root>            # fresh TS
    python -m benchmark.time_e2e --run-dir <root> --ts <TS>  # resume
    python -m benchmark.time_e2e --subjects sub-056 sub-259  # subset
"""

from __future__ import annotations

import argparse
import csv
import json
import socket
import time
from datetime import datetime
from pathlib import Path

from . import paths
from .time_cores import (
    CONFIGS,
    _git_sha,
    make_config,
    pin_cores,
    run_truegeo,
)

# Fixed end-to-end subject set (S_e2e): 10 subjects from the prior scaleup-20 cohort with no
# hemisphere overlap with the core-scaling set S_time. All have both hemispheres.
S_E2E: list[str] = [
    "sub-056",
    "sub-259",
    "sub-265",
    "sub-270",
    "sub-277",
    "sub-283",
    "sub-285",
    "sub-287",
    "sub-298",
    "sub-299",
]
HEMIS = ["lh", "rh"]
N_CORES = 16

CSV_FIELDS = [
    "timestamp",
    "subject",
    "hemi",
    "config",
    "n_cores",
    "n_vertices",
    "n_faces",
    "projection_s",
    "prep_s",
    "flatten_s",
    "total_s",  # projection + prep + flatten (this hemi)
    "kring_mean_distortion",
    "n_flipped",
    "frac_flipped",
    "true_local_at_optscale",
    "true_global_at_optscale",
    "opt_scale",
    "patch_path",
    "flat_path",
    "status",
    "error",
]


def _csv_path(run_dir: Path, config: str, ts: str) -> Path:
    return run_dir / "e2e" / f"{config}_{ts}.csv"


def _append_row(csv_path: Path, row: dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new = not csv_path.exists()
    with open(csv_path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in CSV_FIELDS})


def _done(csv_path: Path) -> set[tuple[str, str]]:
    done: set[tuple[str, str]] = set()
    if csv_path.exists():
        for r in csv.DictReader(open(csv_path, newline="")):
            if r.get("status") == "ok":
                done.add((r["subject"], r["hemi"]))
    return done


def run_hemi(subject, hemi, config_name, run_dir, ts):
    """Project (FS-free) + prep + flatten one hemi at 16 cores. Append a CSV row + ledger."""
    import numpy as np

    from autoflatten.flatten import SurfaceFlattener
    from . import projection, truedist
    from .metrics import per_patch_metrics
    from .probe_tutte_init import make_flatten_fn

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "subject": subject,
        "hemi": hemi,
        "config": config_name,
        "n_cores": N_CORES,
        "status": "ok",
    }
    try:
        patch_dir = run_dir / "patches"
        patch_dir.mkdir(parents=True, exist_ok=True)
        patch_path = patch_dir / f"{subject}_{hemi}.autoflatten.patch.3d"

        # --- projection (FreeSurfer-free cut mapping -> patch) ---
        t0 = time.time()
        projection.project_python(
            subject,
            hemi,
            subjects_dir=str(paths.NARRATIVES_FS),
            continuity=True,
            refine_geodesic=False,  # shipped continuity-only pipeline
            out_patch=str(patch_path),
            verbose=False,
        )
        row["projection_s"] = round(time.time() - t0, 3)

        surf = Path(paths.NARRATIVES_FS) / subject / "surf"
        base = surf / f"{hemi}.fiducial"
        base = base if base.exists() else surf / f"{hemi}.smoothwm"
        cfg = make_config(config_name)

        # --- prep (load + k-ring, cached: e2e measures realistic pipeline cost) ---
        # Run-local cache: the global kring_cache is keyed only on (subject,hemi,k,n) and
        # would load STALE k-rings from a run with different (e.g. geodesic-refined) patches,
        # whose vertex count differs -> broadcasting error. Keep it per-run + per-patch-set.
        t1 = time.time()
        fl = SurfaceFlattener(cfg)
        fl.load_data(str(patch_path), str(base))
        cache = (
            run_dir
            / "kring_cache"
            / (
                f"{subject}_{hemi}.kring_k{cfg.kring.k_ring}_n{cfg.kring.n_neighbors_per_ring}.npz"
            )
        )
        cache.parent.mkdir(parents=True, exist_ok=True)
        fl.compute_kring_distances(cache_path=str(cache))
        fl.prepare_optimization()
        row["prep_s"] = round(time.time() - t1, 3)
        row["n_vertices"] = int(np.asarray(fl.vertices).shape[0])
        row["n_faces"] = int(np.asarray(fl.faces).shape[0])

        # --- flatten ---
        fn = make_flatten_fn("tutte", refine=True)
        t2 = time.time()
        uv = np.asarray(fn(fl))
        row["flatten_s"] = round(time.time() - t2, 3)
        row["total_s"] = round(
            row["projection_s"] + row["prep_s"] + row["flatten_s"], 3
        )

        m = per_patch_metrics(uv, fl)
        row["kring_mean_distortion"] = round(float(m["mean_distortion"]), 4)
        row["n_flipped"] = int(m["n_flipped"])
        row["frac_flipped"] = float(m["frac_flipped"])

        ref = run_truegeo(run_dir, subject, hemi, fl)
        full = truedist.true_distortion_full(uv, ref)
        row["true_global_at_optscale"] = round(
            float(full["true_global_at_optscale"]), 4
        )
        row["opt_scale"] = round(float(full["opt_scale"]), 6)
        loc = truedist.true_distortion(uv * full["opt_scale"], ref)
        row["true_local_at_optscale"] = round(float(loc["true_mean_distortion"]), 4)

        flat_dir = run_dir / "flat"
        flat_dir.mkdir(parents=True, exist_ok=True)
        flat_path = flat_dir / f"{subject}_{hemi}_{config_name}.flat.patch.3d"
        fl.save_result(uv, str(flat_path))
        row["patch_path"] = str(patch_path)
        row["flat_path"] = str(flat_path)

        _log_ledger(subject, hemi, config_name, cfg, row, run_dir, ts)
    except Exception as exc:  # noqa: BLE001
        row["status"] = "error"
        row["error"] = f"{type(exc).__name__}: {exc}"

    _append_row(_csv_path(run_dir, config_name, ts), row)
    print(
        f"[{row['status']}] {subject} {hemi} {config_name}: "
        f"proj={row.get('projection_s')}s prep={row.get('prep_s')}s "
        f"flatten={row.get('flatten_s')}s total={row.get('total_s')}s "
        f"flips={row.get('n_flipped')}"
        + (f"  ERROR {row.get('error')}" if row["status"] == "error" else ""),
        flush=True,
    )
    return row


def _log_ledger(subject, hemi, config_name, cfg, row, run_dir, ts):
    from .ledger import Ledger, new_record

    rec = new_record(
        kind="experiment",
        label=f"paper:e2e:{config_name}:{subject}.{hemi}",
        subjects=[{"subject": subject, "hemi": hemi}],
        method={
            "name": "end_to_end_pipeline",
            "config_name": config_name,
            "init": "tutte",
            "n_cores": N_CORES,
            "stages": ["projection_fsfree", "prep_kring", "flatten"],
            "config": cfg.to_dict(),
        },
        seeds={"note": "deterministic; FS-free projection + CPU gradient descent"},
        repro_command=(
            f"python -m benchmark.time_e2e --subjects {subject} --ts {ts} "
            f"--run-dir {run_dir.parent}"
        ),
    )
    rec.metrics = {
        k: row.get(k)
        for k in (
            "projection_s",
            "prep_s",
            "flatten_s",
            "total_s",
            "n_flipped",
            "true_local_at_optscale",
            "true_global_at_optscale",
        )
    }
    rec.per_subject = [{"subject": subject, "hemi": hemi, **rec.metrics}]
    rec.runtime_s = row.get("total_s")
    rec.status = "ok"
    Ledger().append(rec)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--run-dir",
        default=str(paths.DATA_ROOT / "paper_bench_2026"),
        help="parent dir; an e2e_<TS> subdir is created under it",
    )
    ap.add_argument("--ts", default=None)
    ap.add_argument("--subjects", nargs="*", default=None, help="subset of S_e2e")
    ap.add_argument("--configs", nargs="*", default=CONFIGS, choices=CONFIGS)
    args = ap.parse_args()

    pin_cores(N_CORES)  # fixed 16 physical cores for the whole run

    ts = args.ts or datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(args.run_dir) / f"e2e_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    subjects = args.subjects or S_E2E

    meta = run_dir / "run_meta.json"
    if not meta.exists():
        meta.write_text(
            json.dumps(
                {
                    "ts": ts,
                    "benchmark": "end_to_end_pipeline",
                    "git_sha": _git_sha(),
                    "host": socket.gethostname(),
                    "n_cores": N_CORES,
                    "configs": args.configs,
                    "s_e2e": subjects,
                    "projection": "continuity_only",
                    "created": datetime.now().isoformat(timespec="seconds"),
                },
                indent=2,
            )
        )

    print(f"End-to-end @ {N_CORES} cores  TS={ts}  run_dir={run_dir}")
    for config_name in args.configs:
        done = _done(_csv_path(run_dir, config_name, ts))
        print(f"\n=== config {config_name} ({len(done)} hemis already done) ===")
        for subject in subjects:
            for hemi in HEMIS:
                if (subject, hemi) in done:
                    continue
                run_hemi(subject, hemi, config_name, run_dir, ts)
    print(f"\nDone. CSVs -> {run_dir / 'e2e'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
