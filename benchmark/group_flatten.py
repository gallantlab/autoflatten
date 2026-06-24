"""Group-scale flattening for the paper's distortion-at-scale results.

Flattens a fixed **50-subject** Narratives cohort (**100 hemispheres**) with both shipped
AutoFlatten variants (``robust_fast``, ``tutte_default``), end-to-end and FreeSurfer-free
(continuity-only projection -> k-ring prep -> flatten), and records the energy-independent
true-geodesic distortion for each hemi. This is the *quality-at-scale* deliverable -- distinct
from the timing benchmarks (``time_cores`` / ``time_e2e``); here throughput matters, not
single-job wall-clock, so we run **4 hemispheres concurrently at 8 CPUs each** (all 32 logical
CPUs, hyperthreads included -- distortion is core-count-invariant so HT is free throughput).

Cohort (fixed, written to ``COHORT.md`` at launch): the prior scaleup-20 subjects (continuity
with the timing sets + earlier qualitative work) plus 30 more drawn deterministically from the
materialized-content pool. **All 100 hemis are flattened fresh on the current code** -- no mixing
with the older (pre-rerun) scaleup outputs. The only thing reused across runs is the
``{subject}_{hemi}.truegeo.npz`` geodesic reference (a function of the surface alone, independent
of config/flattening), symlinked in as a head start where it already exists.

Resumable: a ``(subject, hemi)`` cell already ``status=ok`` in a config's CSV is skipped. All
outputs are timestamped under ``<DATA_ROOT>/paper_bench_2026/group_<TS>/``.

Usage
-----
    # full sweep (driver), backgroundable + resumable:
    python -m benchmark.group_flatten --run-dir <root>            # fresh TS
    python -m benchmark.group_flatten --run-dir <root> --ts <TS>  # resume
    python -m benchmark.group_flatten --configs robust_fast       # one config
    python -m benchmark.group_flatten --subjects sub-022 sub-026  # subset

    # one cell (worker), used by the driver:
    python -m benchmark.group_flatten --worker --subject sub-022 --hemi lh \\
        --config robust_fast --cpus 0,1,2,3,4,5,6,7 --run-dir <group_dir> --ts <TS>
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from . import paths
from .time_cores import CONFIGS, _git_sha, make_config, run_truegeo

# --- Fixed 50-subject cohort (100 hemispheres) ------------------------------------
# Prior scaleup-20 (timing sets S_time + S_e2e, + sub-295) for continuity, then 30 more taken
# from the materialized-content pool (sorted ascending). Reproduce with the snippet in COHORT.md.
S_GROUP: list[str] = [
    "sub-022",
    "sub-026",
    "sub-041",
    "sub-052",
    "sub-053",
    "sub-055",
    "sub-056",
    "sub-058",
    "sub-059",
    "sub-060",
    "sub-061",
    "sub-065",
    "sub-066",
    "sub-069",
    "sub-075",
    "sub-127",
    "sub-131",
    "sub-190",
    "sub-201",
    "sub-235",
    "sub-244",
    "sub-249",
    "sub-254",
    "sub-255",
    "sub-256",
    "sub-257",
    "sub-258",
    "sub-259",
    "sub-260",
    "sub-261",
    "sub-262",
    "sub-263",
    "sub-264",
    "sub-265",
    "sub-266",
    "sub-267",
    "sub-268",
    "sub-269",
    "sub-270",
    "sub-271",
    "sub-272",
    "sub-277",
    "sub-283",
    "sub-285",
    "sub-287",
    "sub-295",
    "sub-296",
    "sub-298",
    "sub-299",
    "sub-303",
]
# The 20 already present (older code) -- documented, NOT reused for flats (only truegeo is reused).
PRIOR_20: list[str] = [
    "sub-055",
    "sub-056",
    "sub-066",
    "sub-190",
    "sub-201",
    "sub-259",
    "sub-264",
    "sub-265",
    "sub-268",
    "sub-270",
    "sub-271",
    "sub-277",
    "sub-283",
    "sub-285",
    "sub-287",
    "sub-295",
    "sub-296",
    "sub-298",
    "sub-299",
    "sub-303",
]
HEMIS = ["lh", "rh"]

# 4 lanes x 8 logical CPUs = all 32 logical CPUs (physical 0-15 + their HT siblings 16-31).
N_LANES = 4
CPUS_PER_LANE = 8
LANES: list[list[int]] = [
    list(range(i * CPUS_PER_LANE, (i + 1) * CPUS_PER_LANE)) for i in range(N_LANES)
]

CSV_FIELDS = [
    "timestamp",
    "subject",
    "hemi",
    "config",
    "n_cpus",
    "lane_cpus",
    "n_vertices",
    "n_faces",
    "projection_s",
    "prep_s",
    "flatten_s",
    "total_s",
    "kring_mean_distortion",
    "kring_p90_distortion",
    "n_flipped",
    "frac_flipped",
    "area_distortion",
    "true_local_at_optscale",
    "true_global_at_optscale",
    "opt_scale",
    "n_pairs_global",
    "patch_path",
    "flat_path",
    "status",
    "error",
]

# Existing run dirs that may already hold {subject}_{hemi}.truegeo.npz to reuse as a head start.
_TRUEGEO_SOURCES = [
    paths.DATA_ROOT / "paper_bench_2026" / "20260617-123120" / "truegeo",
    paths.DATA_ROOT / "paper_bench_2026" / "e2e_20260617-123120" / "truegeo",
    paths.DATA_ROOT / "paper_bench_2026" / "20260615-084508" / "truegeo",
    paths.DATA_ROOT / "paper_bench_2026" / "e2e_20260615-084508" / "truegeo",
    paths.DATA_ROOT / "paper_bench_2026" / "20260614-105856" / "truegeo",
    paths.DATA_ROOT / "paper_bench_2026" / "e2e_20260614-105856" / "truegeo",
]


def pin_cpus(cpus: list[int]) -> int:
    """Restrict this process to an explicit CPU set (affinity + thread env). Returns the count
    actually pinned. MUST run before importing JAX / Numba."""
    try:
        os.sched_setaffinity(0, set(cpus))
    except (AttributeError, OSError):
        pass
    n = str(len(cpus))
    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMBA_NUM_THREADS",
    ):
        os.environ[var] = n
    os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=true"
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return len(cpus)


# ---------------------------------------------------------------------------------
# CSV helpers (one CSV per config)
# ---------------------------------------------------------------------------------
def _csv_path(run_dir: Path, config: str, ts: str) -> Path:
    return run_dir / "results" / f"group_{config}_{ts}.csv"


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
        with open(csv_path, newline="") as fh:
            for r in csv.DictReader(fh):
                if r.get("status") == "ok":
                    done.add((r["subject"], r["hemi"]))
    return done


def _seed_truegeo(run_dir: Path) -> int:
    """Symlink existing per-hemi truegeo references into this run (config-independent reuse)."""
    dst = run_dir / "truegeo"
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for subject in S_GROUP:
        for hemi in HEMIS:
            name = f"{subject}_{hemi}.truegeo.npz"
            target = dst / name
            if target.exists():
                continue
            for src in _TRUEGEO_SOURCES:
                cand = src / name
                if cand.exists():
                    target.symlink_to(cand.resolve())
                    n += 1
                    break
    return n


# ---------------------------------------------------------------------------------
# Worker: one (subject, hemi, config) cell -- project + prep + flatten + metrics
# ---------------------------------------------------------------------------------
def run_worker(args) -> int:
    cpus = [int(c) for c in args.cpus.split(",")]
    n_pinned = pin_cpus(cpus)  # MUST precede JAX/Numba import

    import numpy as np

    from autoflatten.flatten import SurfaceFlattener
    from . import projection, truedist
    from .metrics import per_patch_metrics
    from .probe_tutte_init import make_flatten_fn

    try:
        import numba

        numba.set_num_threads(min(len(cpus), numba.get_num_threads()))
    except Exception:  # noqa: BLE001
        pass

    run_dir = Path(args.run_dir)
    row: dict = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "subject": args.subject,
        "hemi": args.hemi,
        "config": args.config,
        "n_cpus": n_pinned,
        "lane_cpus": args.cpus,
        "status": "ok",
    }
    try:
        patch_dir = run_dir / "patches"
        patch_dir.mkdir(parents=True, exist_ok=True)
        patch_path = patch_dir / f"{args.subject}_{args.hemi}.autoflatten.patch.3d"

        # --- projection (FreeSurfer-free, continuity-only -- the shipped default) ---
        t0 = time.time()
        if not patch_path.exists():
            projection.project_python(
                args.subject,
                args.hemi,
                subjects_dir=str(paths.NARRATIVES_FS),
                continuity=True,
                refine_geodesic=False,
                out_patch=str(patch_path),
                verbose=False,
            )
        row["projection_s"] = round(time.time() - t0, 3)

        surf = Path(paths.NARRATIVES_FS) / args.subject / "surf"
        base = surf / f"{args.hemi}.fiducial"
        base = base if base.exists() else surf / f"{args.hemi}.smoothwm"
        cfg = make_config(args.config)

        # --- prep (load + k-ring; run-local per-patch cache to avoid stale cross-run k-rings) ---
        t1 = time.time()
        fl = SurfaceFlattener(cfg)
        fl.load_data(str(patch_path), str(base))
        cache = (
            run_dir
            / "kring_cache"
            / (
                f"{args.subject}_{args.hemi}.kring_k{cfg.kring.k_ring}"
                f"_n{cfg.kring.n_neighbors_per_ring}.npz"
            )
        )
        cache.parent.mkdir(parents=True, exist_ok=True)
        fl.compute_kring_distances(cache_path=str(cache))
        fl.prepare_optimization()
        row["prep_s"] = round(time.time() - t1, 3)
        row["n_vertices"] = int(np.asarray(fl.vertices).shape[0])
        row["n_faces"] = int(np.asarray(fl.faces).shape[0])

        # --- flatten (Tutte flip-free init + refinement) ---
        fn = make_flatten_fn("tutte", refine=True)
        t2 = time.time()
        uv = np.asarray(fn(fl))
        row["flatten_s"] = round(time.time() - t2, 3)
        row["total_s"] = round(
            row["projection_s"] + row["prep_s"] + row["flatten_s"], 3
        )

        # --- k-ring native diagnostic ---
        m = per_patch_metrics(uv, fl)
        row["kring_mean_distortion"] = round(float(m["mean_distortion"]), 4)
        row["kring_p90_distortion"] = round(float(m["p90_distortion"]), 4)
        row["n_flipped"] = int(m["n_flipped"])
        row["frac_flipped"] = float(m["frac_flipped"])
        row["area_distortion"] = round(float(m["area_distortion"]), 6)

        # --- true-geodesic distortion (energy-independent, the reported metric) ---
        ref = run_truegeo(run_dir, args.subject, args.hemi, fl)
        full = truedist.true_distortion_full(uv, ref)
        row["true_global_at_optscale"] = round(
            float(full["true_global_at_optscale"]), 4
        )
        row["opt_scale"] = round(float(full["opt_scale"]), 6)
        row["n_pairs_global"] = int(full["n_pairs_global"])
        loc = truedist.true_distortion(uv * full["opt_scale"], ref)
        row["true_local_at_optscale"] = round(float(loc["true_mean_distortion"]), 4)

        # --- save flat patch (for per-hemi flatmaps / qualitative panels) ---
        flat_dir = run_dir / "flat"
        flat_dir.mkdir(parents=True, exist_ok=True)
        flat_path = flat_dir / f"{args.subject}_{args.hemi}_{args.config}.flat.patch.3d"
        fl.save_result(uv, str(flat_path))
        row["patch_path"] = str(patch_path)
        row["flat_path"] = str(flat_path)

        _log_ledger(args, cfg, row)
    except Exception as exc:  # noqa: BLE001 - record and keep the sweep going
        row["status"] = "error"
        row["error"] = f"{type(exc).__name__}: {exc}"

    _append_row(_csv_path(run_dir, args.config, args.ts), row)
    print(
        f"[{row['status']}] {args.subject} {args.hemi} {args.config}: "
        f"proj={row.get('projection_s')}s prep={row.get('prep_s')}s "
        f"flatten={row.get('flatten_s')}s "
        f"true_local@opt={row.get('true_local_at_optscale')}% "
        f"flips={row.get('n_flipped')}"
        + (f"  ERROR {row.get('error')}" if row["status"] == "error" else ""),
        flush=True,
    )
    return 0 if row["status"] == "ok" else 1


def _log_ledger(args, cfg, row: dict) -> None:
    from .ledger import Ledger, new_record

    rec = new_record(
        kind="experiment",
        label=f"paper:group:{args.config}:{args.subject}.{args.hemi}",
        subjects=[{"subject": args.subject, "hemi": args.hemi}],
        method={
            "name": "group_flatten_quality",
            "config_name": args.config,
            "init": "tutte",
            "stages": ["projection_fsfree", "prep_kring", "flatten"],
            "config": cfg.to_dict(),
        },
        seeds={"note": "deterministic; FS-free projection + CPU gradient descent"},
        repro_command=(
            f"python -m benchmark.group_flatten --worker --subject {args.subject} "
            f"--hemi {args.hemi} --config {args.config} --cpus {args.cpus} "
            f"--run-dir {args.run_dir} --ts {args.ts}"
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
            "frac_flipped",
            "true_local_at_optscale",
            "true_global_at_optscale",
            "kring_mean_distortion",
        )
    }
    rec.per_subject = [{"subject": args.subject, "hemi": args.hemi, **rec.metrics}]
    rec.runtime_s = row.get("total_s")
    rec.status = "ok"
    Ledger().append(rec)


# ---------------------------------------------------------------------------------
# Driver: 4-lane concurrent sweep
# ---------------------------------------------------------------------------------
def _write_cohort_doc(run_dir: Path) -> None:
    doc = run_dir / "COHORT.md"
    if doc.exists():
        return
    extra = [s for s in S_GROUP if s not in PRIOR_20]
    doc.write_text(
        "# Group-results cohort (50 subjects / 100 hemispheres)\n\n"
        "Fixed cohort for the paper's distortion-at-scale results. **All 100 hemispheres are\n"
        "flattened fresh on the current code** (continuity-only FS-free projection + pyflatten);\n"
        "the older pre-rerun `scaleup_*` outputs are NOT reused.\n\n"
        f"## Prior 20 (continuity with timing sets; flats re-run, not reused)\n{' '.join(PRIOR_20)}\n\n"
        f"## 30 added (deterministic, sorted ascending from the materialized-content pool)\n{' '.join(extra)}\n\n"
        f"## Full cohort ({len(S_GROUP)})\n{' '.join(S_GROUP)}\n\n"
        "## Selection rule (reproducible)\n"
        "Pool = Narratives FS-derivatives subjects with BOTH hemispheres' `sphere.reg` + `fiducial`\n"
        "materialized on disk (82 subjects). Cohort = sorted(PRIOR_20 + first-30(sorted(pool - PRIOR_20))).\n"
    )


def run_driver(args) -> int:
    ts = args.ts or datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(args.run_dir) / f"group_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    subjects = args.subjects or S_GROUP
    configs = args.configs

    meta = run_dir / "run_meta.json"
    if not meta.exists():
        meta.write_text(
            json.dumps(
                {
                    "ts": ts,
                    "benchmark": "group_flatten_quality",
                    "git_sha": _git_sha(),
                    "host": socket.gethostname(),
                    "configs": configs,
                    "n_subjects": len(subjects),
                    "n_hemis": len(subjects) * len(HEMIS),
                    "n_lanes": N_LANES,
                    "cpus_per_lane": CPUS_PER_LANE,
                    "lanes": LANES,
                    "projection": "continuity_only",
                    "kring_cache": "run_local",
                    "created": datetime.now().isoformat(timespec="seconds"),
                },
                indent=2,
            )
        )
    _write_cohort_doc(run_dir)
    seeded = _seed_truegeo(run_dir)
    print(f"Seeded {seeded} existing truegeo references (head start).", flush=True)

    # Build todo cells (robust_fast first -> a complete shipping-default group lands early).
    cells: list[tuple[str, str, str]] = []
    for config in configs:
        done = _done(_csv_path(run_dir, config, ts))
        for subject in subjects:
            for hemi in HEMIS:
                if (subject, hemi) not in done:
                    cells.append((subject, hemi, config))
    total = len(subjects) * len(HEMIS) * len(configs)
    print(
        f"Group sweep TS={ts}  run_dir={run_dir}\n"
        f"{total} cells, {total - len(cells)} done, {len(cells)} to run "
        f"on {N_LANES} lanes x {CPUS_PER_LANE} CPUs.\n",
        flush=True,
    )

    queue = list(cells)
    running: dict[int, tuple[subprocess.Popen, tuple]] = {}  # lane -> (proc, cell)
    repo = Path(__file__).resolve().parent.parent
    launched = 0

    def launch(lane: int, cell: tuple) -> subprocess.Popen:
        nonlocal launched
        launched += 1
        s, h, c = cell
        cpus = ",".join(str(x) for x in LANES[lane])
        print(
            f"--- [{launched}/{len(cells)}] lane{lane} cpus={cpus}: {s} {h} {c} ---",
            flush=True,
        )
        cmd = [
            sys.executable,
            "-m",
            "benchmark.group_flatten",
            "--worker",
            "--subject",
            s,
            "--hemi",
            h,
            "--config",
            c,
            "--cpus",
            cpus,
            "--run-dir",
            str(run_dir),
            "--ts",
            ts,
        ]
        return subprocess.Popen(cmd, cwd=repo)

    # Prime the lanes, then refill as each finishes.
    for lane in range(min(N_LANES, len(queue))):
        running[lane] = (launch(lane, queue.pop(0)), None)

    while running:
        time.sleep(2)
        for lane in list(running):
            proc, _ = running[lane]
            if proc.poll() is None:
                continue
            if queue:
                running[lane] = (launch(lane, queue.pop(0)), None)
            else:
                del running[lane]

    print(f"\nDone. CSVs -> {run_dir / 'results'}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--worker", action="store_true", help="run a single cell (internal)"
    )
    ap.add_argument("--subject")
    ap.add_argument("--hemi", choices=HEMIS)
    ap.add_argument("--config", choices=CONFIGS)
    ap.add_argument(
        "--cpus", help="worker: comma-separated CPU ids to pin this cell to"
    )
    ap.add_argument(
        "--run-dir",
        default=str(paths.DATA_ROOT / "paper_bench_2026"),
        help="driver: parent dir (a group_<TS> subdir is created); worker: the exact group_<TS> dir",
    )
    ap.add_argument(
        "--ts", default=None, help="run timestamp; omit on driver to mint a fresh one"
    )
    ap.add_argument("--subjects", nargs="*", default=None, help="subset of the cohort")
    ap.add_argument("--configs", nargs="*", default=CONFIGS, choices=CONFIGS)
    args = ap.parse_args()

    if args.worker:
        if not all([args.subject, args.hemi, args.config, args.cpus, args.ts]):
            ap.error(
                "--worker requires --subject --hemi --config --cpus --ts --run-dir"
            )
        return run_worker(args)
    return run_driver(args)


if __name__ == "__main__":
    raise SystemExit(main())
