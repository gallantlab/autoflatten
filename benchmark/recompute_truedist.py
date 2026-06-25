"""Recompute true-geodesic distortion columns in a results CSV from saved flat patches.

The true-geodesic metric is a pure post-hoc function of the saved flat map (``uv``) and the
cached per-hemi geodesic reference (``truegeo.npz``). When the metric definition changes (e.g.
the 5 mm denominator floor in :mod:`benchmark.truedist`), this rewrites the affected columns --
``opt_scale``, ``true_local_at_optscale``, ``true_global_at_optscale`` (and ``true_local_mean`` /
``true_global_mean`` if present) -- WITHOUT re-flattening. The original CSV is preserved as
``<name>.unfloored.csv`` (once), and the recomputed table is written to ``<name>.floored.csv``
(or in place with ``--in-place``).

Flat-patch resolution per row: use the row's ``flat_path`` if present and on disk; else build
``<flat-dir>/<subject>_<hemi>_<config>.flat.patch.3d``. Rows with no resolvable flat patch (e.g.
the 1-/8-core core-scaling rows, which share the 16-core map by determinism) fall back to the
``--config`` sibling flat in ``--flat-dir``.

Usage
-----
    python -m benchmark.recompute_truedist --csv <results.csv> \\
        --flat-dir <run>/flat --truegeo-dir <run>/truegeo [--in-place]
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import numpy as np

from autoflatten.freesurfer import read_patch

from . import truedist

_RECOMPUTED = (
    "opt_scale",
    "true_local_at_optscale",
    "true_global_at_optscale",
    "true_local_mean",
    "true_global_mean",
)


def _resolve_flat(row, flat_dir: Path) -> Path | None:
    fp = row.get("flat_path") or ""
    if fp and Path(fp).exists():
        return Path(fp)
    cand = flat_dir / f"{row['subject']}_{row['hemi']}_{row['config']}.flat.patch.3d"
    return cand if cand.exists() else None


def _recompute_row(row, flat_dir: Path, truegeo_dir: Path) -> str:
    if row.get("status") != "ok":
        return "skip(not-ok)"
    flat = _resolve_flat(row, flat_dir)
    if flat is None:
        return "skip(no-flat)"
    tg = truegeo_dir / f"{row['subject']}_{row['hemi']}.truegeo.npz"
    if not tg.exists():
        return "skip(no-truegeo)"
    d = np.load(tg)
    ref = {"srcs": d["srcs"], "geo": d["geo"], "R": float(d["R"])}
    uv = read_patch(str(flat))[0][:, :2].astype(np.float64)

    try:
        full = truedist.true_distortion_full(uv, ref)
        opt = full["opt_scale"]
        loc = truedist.true_distortion(uv * opt, ref)
    except ValueError as exc:
        # misaligned flat/reference or no valid pairs -- leave this row's columns untouched
        print(f"  skip {row['subject']} {row['hemi']} {row['config']}: {exc}")
        return "skip(error)"
    row["opt_scale"] = round(float(opt), 6)
    row["true_global_at_optscale"] = round(float(full["true_global_at_optscale"]), 4)
    row["true_local_at_optscale"] = round(float(loc["true_mean_distortion"]), 4)
    if "true_global_mean" in row:
        row["true_global_mean"] = round(float(full["true_global_mean"]), 4)
    if "true_local_mean" in row:
        row["true_local_mean"] = round(float(full["true_local_mean"]), 4)
    return "ok"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--flat-dir", required=True)
    ap.add_argument("--truegeo-dir", required=True)
    ap.add_argument(
        "--in-place", action="store_true", help="overwrite the CSV (keeps a backup)"
    )
    args = ap.parse_args()

    csv_path = Path(args.csv)
    flat_dir = Path(args.flat_dir)
    truegeo_dir = Path(args.truegeo_dir)

    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        fields = reader.fieldnames
        rows = list(reader)

    from collections import Counter

    tally = Counter()
    before, after = [], []
    for r in rows:
        if r.get("status") == "ok" and r.get("true_local_at_optscale"):
            before.append(float(r["true_local_at_optscale"]))
        tally[_recompute_row(r, flat_dir, truegeo_dir)] += 1
        if r.get("status") == "ok" and r.get("true_local_at_optscale"):
            after.append(float(r["true_local_at_optscale"]))

    if args.in_place:
        backup = csv_path.with_suffix(".unfloored.csv")
        if not backup.exists():
            shutil.copy2(csv_path, backup)
        out = csv_path
    else:
        out = csv_path.with_name(csv_path.stem + ".floored.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(f"{csv_path.name}: {dict(tally)}")
    if before and after:
        print(
            f"  local@opt median {np.median(before):.2f} -> {np.median(after):.2f} %  "
            f"mean {np.mean(before):.2f} -> {np.mean(after):.2f} %  max {max(before):.1f} -> {max(after):.1f} %"
        )
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
