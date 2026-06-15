"""Core-scaling timing for the flatten-only stage (paper Deliverable 2).

Times the *flattening* stage (k-ring prep + optimization, projection excluded) for the two
shipped AutoFlatten variants at 1 / 8 / 16 CPU cores, on a fixed set of hemispheres, and
records distortion at every core count (it should be core-count-invariant -- recording it
confirms determinism). The k-ring cache is **disabled** so the Numba-parallel prep actually
runs and scales (a cache hit would zero out ``prep_s``).

Core count is enforced by **CPU affinity** (``os.sched_setaffinity`` to physical cores
``0..n-1``; on this box logical CPUs 0-15 are distinct physical cores, 16-31 their hyperthread
siblings) plus matching thread-count env vars, set **before** JAX/Numba import. Affinity caps
Numba, XLA and BLAS uniformly and is robust to XLA-flag churn (the package's
``configure_threading`` sets an ``--xla_cpu_multi_thread_eigen_thread_count`` flag that aborts
on JAX 0.6+). Each (hemi, config, n_cores) cell runs in a fresh subprocess (``--worker``); the
driver spawns them one at a time (true single-job wall-clock) and is resumable: a cell already
present with ``status=ok`` in the CSV is skipped.

Two AutoFlatten variants (both Tutte flip-free init + refinement, initial NAR off):
  * ``robust_fast``    -- shipping default: k=7, n_neighbors=6, line-search n_coarse_steps=7,
                          full smoothing, iters/level=40.
  * ``tutte_default``  -- quality-first: k=7, n_neighbors=12, full default refinement.

All outputs are timestamped under ``<DATA_ROOT>/paper_bench_2026/<TS>/``.

Usage
-----
    # one cell (worker), used by the driver:
    python -m benchmark.time_cores --worker --subject sub-055 --hemi lh \\
        --config robust_fast --n-cores 8 --run-dir <dir> --ts <TS>

    # full sweep (driver), backgroundable + resumable:
    python -m benchmark.time_cores --run-dir <dir>            # fresh TS
    python -m benchmark.time_cores --run-dir <dir> --ts <TS>  # resume
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

# Fixed core-scaling hemisphere set (S_time). Reuses the prior speed_1core hemis so the
# existing FS6 1-/8-core timings overlay on the figure without re-running mris_flatten.
S_TIME: list[tuple[str, str]] = [
    ("sub-055", "lh"),
    ("sub-055", "rh"),
    ("sub-066", "rh"),
    ("sub-190", "rh"),
    ("sub-201", "rh"),
    ("sub-264", "rh"),
    ("sub-268", "rh"),
    ("sub-271", "lh"),
    ("sub-296", "rh"),
    ("sub-303", "lh"),
]

CONFIGS = ["robust_fast", "tutte_default"]
CORE_COUNTS = [1, 8, 16]  # capped at the 16 physical cores (no hyperthreads)

CSV_FIELDS = [
    "timestamp",
    "subject",
    "hemi",
    "config",
    "n_cores",
    "n_cores_pinned",
    "n_vertices",
    "n_faces",
    "prep_s",
    "flatten_s",
    "total_s",
    # k-ring native diagnostic (NOT the reported metric -- biased by n_neighbors)
    "kring_mean_distortion",
    "kring_p90_distortion",
    "n_flipped",
    "frac_flipped",
    "area_distortion",
    # true-geodesic metrics (energy-independent; the reported distortion)
    "true_local_mean",
    "true_local_at_optscale",
    "true_global_mean",
    "true_global_at_optscale",
    "opt_scale",
    "n_pairs_global",
    "flat_path",
    "status",
    "error",
]


# ---------------------------------------------------------------------------------
# Core pinning (must run before importing JAX / Numba)
# ---------------------------------------------------------------------------------
_THREAD_ENV = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMBA_NUM_THREADS",
)


def pin_cores(n: int) -> int:
    """Restrict this process to ``n`` physical cores (affinity + thread env). Returns the
    number of cores actually pinned. Call before any JAX/Numba import."""
    cpus = set(range(n))
    try:
        os.sched_setaffinity(0, cpus)
    except (AttributeError, OSError):
        pass
    for var in _THREAD_ENV:
        os.environ[var] = str(n)
    # Keep XLA's CPU thread pool on (sized from the affinity mask); avoid the bad flag.
    os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=true"
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return n


# ---------------------------------------------------------------------------------
# Config + entry construction
# ---------------------------------------------------------------------------------
def make_config(name: str):
    """Reconstruct the named AutoFlatten variant (configs lived in the retired scaleup.py).

    Both use Tutte flip-free init (initial NAR off); they differ only in k-ring density and
    line-search points. Definitions per FINDINGS.md / the robust_fast shipping-default note.
    """
    from autoflatten.flatten import FlattenConfig

    cfg = FlattenConfig()
    cfg.verbose = False
    cfg.kring.k_ring = 7
    cfg.negative_area_removal.enabled = (
        False  # flip-free Tutte start makes initial NAR moot
    )
    if name == "robust_fast":
        cfg.kring.n_neighbors_per_ring = 6
        cfg.line_search.n_coarse_steps = 7
    elif name == "tutte_default":
        cfg.kring.n_neighbors_per_ring = 12
        # else: full defaults (n_coarse_steps=15, iters/level=40, full smoothing)
    else:
        raise ValueError(f"unknown config '{name}'")
    return cfg


def build_entry(subject: str, hemi: str, patch_dir: str | None = None) -> dict:
    """Build a manifest-style entry.

    Patch comes from ``patch_dir`` (re-projected continuity-only patches) when given,
    else from the Narratives FreeSurfer derivatives. Base surface is always the derivatives
    fiducial.
    """
    surf = Path(paths.NARRATIVES_FS) / subject / "surf"
    if patch_dir:
        patch = Path(patch_dir) / f"{subject}_{hemi}.autoflatten.patch.3d"
    else:
        patch = surf / f"{hemi}.autoflatten.patch.3d"
    fid = surf / f"{hemi}.fiducial"
    base = fid if fid.exists() else surf / f"{hemi}.smoothwm"
    if not patch.exists():
        raise FileNotFoundError(f"missing patch: {patch}")
    if not base.exists():
        raise FileNotFoundError(f"missing base surface for {subject} {hemi}")
    return {
        "subject": subject,
        "hemi": hemi,
        "patch_path": str(patch),
        "surface_path": str(base),
        "surface_kind": base.suffix.lstrip("."),
    }


def reproject_continuity_only(subject: str, hemi: str, patch_dir: Path) -> Path:
    """Re-project one hemi with the shipped continuity-only pipeline (geodesic refine off)."""
    from . import projection

    patch_dir.mkdir(parents=True, exist_ok=True)
    out = patch_dir / f"{subject}_{hemi}.autoflatten.patch.3d"
    if not out.exists():
        projection.project_python(
            subject,
            hemi,
            subjects_dir=str(paths.NARRATIVES_FS),
            continuity=True,
            refine_geodesic=False,
            out_patch=str(out),
            verbose=False,
        )
    return out


def run_truegeo(run_dir: Path, subject: str, hemi: str, flattener):
    """Run-local true-geodesic reference (avoids clobbering the shared kring_cache)."""
    import numpy as np

    from . import truedist

    tg = run_dir / "truegeo" / f"{subject}_{hemi}.truegeo.npz"
    if tg.exists():
        d = np.load(tg)
        return {"srcs": d["srcs"], "geo": d["geo"], "R": float(d["R"])}
    ref = truedist.compute_truegeo(flattener)
    tg.parent.mkdir(parents=True, exist_ok=True)
    np.savez(tg, **ref)
    return ref


# ---------------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------------
def _csv_path(run_dir: Path, ts: str) -> Path:
    return run_dir / "cores" / f"cores_{ts}.csv"


def _append_row(csv_path: Path, row: dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new = not csv_path.exists()
    with open(csv_path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in CSV_FIELDS})


def _done_cells(csv_path: Path) -> set[tuple[str, str, str, str]]:
    """(subject, hemi, config, n_cores) cells already recorded with status=ok."""
    done: set[tuple[str, str, str, str]] = set()
    if not csv_path.exists():
        return done
    with open(csv_path, newline="") as fh:
        for r in csv.DictReader(fh):
            if r.get("status") == "ok":
                done.add((r["subject"], r["hemi"], r["config"], str(r["n_cores"])))
    return done


# ---------------------------------------------------------------------------------
# Worker: one (subject, hemi, config, n_cores) cell
# ---------------------------------------------------------------------------------
def run_worker(args) -> int:
    # MUST pin cores before importing JAX/Numba-backed modules.
    n_pinned = pin_cores(args.n_cores)

    import numpy as np

    from autoflatten.flatten import SurfaceFlattener
    from .metrics import per_patch_metrics
    from .probe_tutte_init import make_flatten_fn
    from . import truedist

    try:  # cap Numba's pool too (env is set, but be explicit)
        import numba

        numba.set_num_threads(min(args.n_cores, numba.get_num_threads()))
    except Exception:  # noqa: BLE001
        pass

    run_dir = Path(args.run_dir)
    csv_path = _csv_path(run_dir, args.ts)
    iso = datetime.now().isoformat(timespec="seconds")
    row: dict = {
        "timestamp": iso,
        "subject": args.subject,
        "hemi": args.hemi,
        "config": args.config,
        "n_cores": args.n_cores,
        "n_cores_pinned": n_pinned,
        "status": "ok",
    }
    try:
        entry = build_entry(args.subject, args.hemi, patch_dir=args.patch_dir)
        cfg = make_config(args.config)

        # --- prep (load + k-ring geodesics, cache DISABLED + JAX setup) ---
        t0 = time.time()
        fl = SurfaceFlattener(cfg)
        fl.load_data(entry["patch_path"], entry["surface_path"])
        fl.compute_kring_distances(cache_path=None)  # fresh: measure Numba scaling
        fl.prepare_optimization()
        row["prep_s"] = round(time.time() - t0, 3)
        row["n_vertices"] = int(np.asarray(fl.vertices).shape[0])
        row["n_faces"] = int(np.asarray(fl.faces).shape[0])

        # --- flatten (Tutte flip-free init + refinement) ---
        fn = make_flatten_fn("tutte", refine=True)
        t1 = time.time()
        uv = np.asarray(fn(fl))
        row["flatten_s"] = round(time.time() - t1, 3)
        row["total_s"] = round(row["prep_s"] + row["flatten_s"], 3)

        # --- k-ring native diagnostic ---
        m = per_patch_metrics(uv, fl)
        row["kring_mean_distortion"] = round(float(m["mean_distortion"]), 4)
        row["kring_p90_distortion"] = round(float(m["p90_distortion"]), 4)
        row["n_flipped"] = int(m["n_flipped"])
        row["frac_flipped"] = float(m["frac_flipped"])
        row["area_distortion"] = round(float(m["area_distortion"]), 6)

        # --- true-geodesic metrics (energy-independent, reported) ---
        ref = run_truegeo(run_dir, args.subject, args.hemi, fl)
        full = truedist.true_distortion_full(uv, ref)
        row["true_local_mean"] = round(float(full["true_local_mean"]), 4)
        row["true_global_mean"] = round(float(full["true_global_mean"]), 4)
        row["true_global_at_optscale"] = round(
            float(full["true_global_at_optscale"]), 4
        )
        row["opt_scale"] = round(float(full["opt_scale"]), 6)
        row["n_pairs_global"] = int(full["n_pairs_global"])
        # local distortion at the (global-)optimal scale, for an apples-to-apples local number
        loc = truedist.true_distortion(uv * full["opt_scale"], ref)
        row["true_local_at_optscale"] = round(float(loc["true_mean_distortion"]), 4)

        # --- save flat patch on the max-core run (for the per-hemi flatmap figures) ---
        if args.n_cores == max(CORE_COUNTS):
            flat_dir = run_dir / "flat"
            flat_dir.mkdir(parents=True, exist_ok=True)
            flat_path = (
                flat_dir / f"{args.subject}_{args.hemi}_{args.config}.flat.patch.3d"
            )
            fl.save_result(uv, str(flat_path))
            row["flat_path"] = str(flat_path)

        _log_ledger(args, entry, cfg, row)
    except Exception as exc:  # noqa: BLE001 - record and keep the sweep going
        row["status"] = "error"
        row["error"] = f"{type(exc).__name__}: {exc}"

    _append_row(csv_path, row)
    print(
        f"[{row['status']}] {args.subject} {args.hemi} {args.config} "
        f"n={args.n_cores}: prep={row.get('prep_s')}s flatten={row.get('flatten_s')}s "
        f"true_local@opt={row.get('true_local_at_optscale')}% "
        f"true_global@opt={row.get('true_global_at_optscale')}% flips={row.get('n_flipped')}"
        + (f"  ERROR {row.get('error')}" if row["status"] == "error" else "")
    )
    return 0 if row["status"] == "ok" else 1


def _log_ledger(args, entry: dict, cfg, row: dict) -> None:
    from .ledger import Ledger, new_record

    rec = new_record(
        kind="experiment",
        label=f"paper:core_scaling:{args.config}:{args.subject}.{args.hemi}:n{args.n_cores}",
        subjects=[{"subject": args.subject, "hemi": args.hemi}],
        method={
            "name": "core_scaling_flatten_only",
            "config_name": args.config,
            "init": "tutte",
            "n_cores": args.n_cores,
            "kring_cache": "disabled",
            "config": cfg.to_dict(),
        },
        seeds={
            "note": "deterministic CPU gradient descent; n_cores affects threading only"
        },
        repro_command=(
            f"python -m benchmark.time_cores --worker --subject {args.subject} "
            f"--hemi {args.hemi} --config {args.config} --n-cores {args.n_cores} "
            f"--run-dir {args.run_dir} --ts {args.ts}"
        ),
    )
    rec.metrics = {
        k: row.get(k)
        for k in (
            "prep_s",
            "flatten_s",
            "total_s",
            "n_flipped",
            "frac_flipped",
            "true_local_at_optscale",
            "true_global_at_optscale",
            "opt_scale",
            "kring_mean_distortion",
        )
    }
    rec.per_subject = [{"subject": args.subject, "hemi": args.hemi, **rec.metrics}]
    rec.runtime_s = row.get("total_s")
    rec.status = "ok"
    Ledger().append(rec)


# ---------------------------------------------------------------------------------
# Driver: full sweep
# ---------------------------------------------------------------------------------
def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent.parent
            )
            .decode()
            .strip()
        )
    except Exception:  # noqa: BLE001
        return "unknown"


def run_driver(args) -> int:
    ts = args.ts or datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(args.run_dir) / ts if args.run_dir_is_root else Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = _csv_path(run_dir, ts)

    meta_path = run_dir / "run_meta.json"
    if not meta_path.exists():
        meta_path.write_text(
            json.dumps(
                {
                    "ts": ts,
                    "benchmark": "core_scaling_flatten_only",
                    "git_sha": _git_sha(),
                    "host": socket.gethostname(),
                    "configs": CONFIGS,
                    "core_counts": CORE_COUNTS,
                    "s_time": [f"{s}.{h}" for s, h in S_TIME],
                    "projection": "continuity_only"
                    if args.reproject
                    else "derivatives_patches",
                    "kring_cache": "disabled",
                    "created": datetime.now().isoformat(timespec="seconds"),
                },
                indent=2,
            )
        )

    # re-project S_time with the shipped continuity-only pipeline (geodesic refine off)
    patch_dir = None
    if args.reproject:
        patch_dir = run_dir / "patches"
        print(
            f"Re-projecting {len(S_TIME)} hemis continuity-only -> {patch_dir}",
            flush=True,
        )
        for s, h in S_TIME:
            reproject_continuity_only(s, h, patch_dir)

    done = _done_cells(csv_path)
    # 1-core dominates wall-clock -> run small core counts last so quick cells land first.
    cells = [
        (s, h, c, n)
        for n in sorted(CORE_COUNTS, reverse=True)
        for c in CONFIGS
        for (s, h) in S_TIME
    ]
    todo = [
        cell for cell in cells if (cell[0], cell[1], cell[2], str(cell[3])) not in done
    ]
    print(
        f"Core-scaling sweep TS={ts}  run_dir={run_dir}\n"
        f"{len(cells)} cells, {len(done)} done, {len(todo)} to run.\n"
    )
    for i, (s, h, c, n) in enumerate(todo, 1):
        print(f"--- [{i}/{len(todo)}] {s} {h} {c} n={n} ---", flush=True)
        cmd = [
            sys.executable,
            "-m",
            "benchmark.time_cores",
            "--worker",
            "--subject",
            s,
            "--hemi",
            h,
            "--config",
            c,
            "--n-cores",
            str(n),
            "--run-dir",
            str(run_dir),
            "--ts",
            ts,
        ]
        if patch_dir is not None:
            cmd += ["--patch-dir", str(patch_dir)]
        subprocess.run(cmd, check=False, cwd=Path(__file__).resolve().parent.parent)
    print(f"\nDone. CSV -> {csv_path}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--worker", action="store_true", help="run a single cell (internal)"
    )
    ap.add_argument("--subject")
    ap.add_argument("--hemi", choices=["lh", "rh"])
    ap.add_argument("--config", choices=CONFIGS)
    ap.add_argument("--n-cores", type=int)
    ap.add_argument(
        "--run-dir",
        default=str(paths.DATA_ROOT / "paper_bench_2026"),
        help="driver: parent dir (a <TS> subdir is created); worker: the exact <TS> dir",
    )
    ap.add_argument(
        "--ts", default=None, help="run timestamp; omit on driver to mint a fresh one"
    )
    ap.add_argument(
        "--patch-dir",
        default=None,
        help="worker: dir of re-projected {subject}_{hemi}.autoflatten.patch.3d "
        "(else use Narratives derivatives)",
    )
    ap.add_argument(
        "--reproject",
        action="store_true",
        help="driver: re-project S_time continuity-only into <run>/patches first",
    )
    args = ap.parse_args()

    if args.worker:
        if not all([args.subject, args.hemi, args.config, args.n_cores, args.ts]):
            ap.error(
                "--worker requires --subject --hemi --config --n-cores --ts --run-dir"
            )
        return run_worker(args)

    # driver: --run-dir is the *root*; a <TS> subdir is created under it.
    args.run_dir_is_root = True
    return run_driver(args)


if __name__ == "__main__":
    raise SystemExit(main())
