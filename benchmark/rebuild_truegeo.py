"""Rebuild per-hemi true-geodesic references on the FIDUCIAL surface.

Background: ``compute_truegeo`` previously ran heat geodesics on ``flattener.vertices``, which
``load_data`` sets to the patch's stored coords = the FreeSurfer *inflated* surface (~1.7x larger,
stretched triangles that ill-condition the heat solver). The fixed ``compute_truegeo`` runs on the
fiducial surface. This tool regenerates the cached ``{subject}_{hemi}.truegeo.npz`` files for an
existing run with the corrected reference (heat re-solved; no re-flattening needed).

4 hemis concurrent at 8 CPUs each (same lanes as group_flatten). Resumable: a hemi whose truegeo
already carries ``surface='fiducial'`` metadata is skipped. Writes into ``<run>/truegeo`` (the old
inflated files are backed up once to ``<run>/truegeo_inflated``).

Usage
-----
    python -m benchmark.rebuild_truegeo --run-dir <group_or_run_dir>            # all hemis w/ a flat
    python -m benchmark.rebuild_truegeo --run-dir <dir> --worker --subject sub-022 --hemi lh --cpus 0,1,..
"""

from __future__ import annotations

import argparse
import glob
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from . import paths
from .group_flatten import LANES, N_LANES, pin_cpus


def _hemis_with_flat(run_dir: Path) -> list[tuple[str, str]]:
    out = set()
    for f in glob.glob(str(run_dir / "flat" / "*.flat.patch.3d")):
        m = re.search(r"/(sub-\d+)_(lh|rh)_", f)
        if m:
            out.add((m.group(1), m.group(2)))
    return sorted(out)


def run_worker(args) -> int:
    pin_cpus([int(c) for c in args.cpus.split(",")])
    import numpy as np
    from autoflatten.flatten import FlattenConfig, SurfaceFlattener
    from . import truedist

    run_dir = Path(args.run_dir)
    subj, hemi = args.subject, args.hemi
    surf = Path(paths.NARRATIVES_FS) / subj / "surf"
    base = surf / f"{hemi}.fiducial"
    base = base if base.exists() else surf / f"{hemi}.smoothwm"
    patch = run_dir / "patches" / f"{subj}_{hemi}.autoflatten.patch.3d"
    out = run_dir / "truegeo" / f"{subj}_{hemi}.truegeo.npz"

    fl = SurfaceFlattener(FlattenConfig())
    fl.load_data(str(patch), str(base))
    ref = truedist.compute_truegeo(fl)  # now heat on fiducial
    # negative-geo sanity (should be ~0 on fiducial)
    n_neg = int(sum(ref["geo"][i].min() < -0.1 for i in range(len(ref["srcs"]))))
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, srcs=ref["srcs"], geo=ref["geo"], R=ref["R"], surface="fiducial")
    print(
        f"[ok] {subj} {hemi}: rebuilt on fiducial, neg-geo sources={n_neg}", flush=True
    )
    return 0


def _needs_rebuild(run_dir: Path, subj: str, hemi: str) -> bool:
    import numpy as np

    p = run_dir / "truegeo" / f"{subj}_{hemi}.truegeo.npz"
    if not p.exists() or p.is_symlink():  # symlinked-in old refs must be replaced
        return True
    try:
        d = np.load(p)
        return str(d.get("surface", "")) != "fiducial"
    except Exception:  # noqa: BLE001
        return True


def run_driver(args) -> int:
    run_dir = Path(args.run_dir)
    hemis = _hemis_with_flat(run_dir)
    # back up the old (inflated) truegeo dir once
    bak = run_dir / "truegeo_inflated"
    tg = run_dir / "truegeo"
    if tg.exists() and not bak.exists():
        shutil.copytree(tg, bak, symlinks=True)
        print(f"backed up inflated truegeo -> {bak}")
    todo = [(s, h) for (s, h) in hemis if _needs_rebuild(run_dir, s, h)]
    print(
        f"rebuild_truegeo: {len(hemis)} hemis, {len(todo)} to rebuild on FIDUCIAL "
        f"({N_LANES} lanes).",
        flush=True,
    )

    queue = list(todo)
    running: dict[int, subprocess.Popen] = {}
    repo = Path(__file__).resolve().parent.parent
    done = 0

    def launch(lane, cell):
        s, h = cell
        cpus = ",".join(str(x) for x in LANES[lane])
        cmd = [
            sys.executable,
            "-m",
            "benchmark.rebuild_truegeo",
            "--worker",
            "--run-dir",
            str(run_dir),
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
            done += 1
            if queue:
                running[lane] = launch(lane, queue.pop(0))
            else:
                del running[lane]
    print(f"\nDone. Rebuilt {len(todo)} fiducial truegeo refs in {run_dir / 'truegeo'}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--subject")
    ap.add_argument("--hemi", choices=["lh", "rh"])
    ap.add_argument("--cpus")
    args = ap.parse_args()
    return run_worker(args) if args.worker else run_driver(args)


if __name__ == "__main__":
    raise SystemExit(main())
