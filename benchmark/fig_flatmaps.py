"""Per-hemisphere Fischl-1999-style flatmap+histogram figures (paper Deliverable 5).

One image per hemisphere in S_time, with the three methods side by side
(robust_fast / tutte_default / freesurfer6). Each column = the flattened patch colored by
per-vertex metric distortion (top) above a histogram of that distortion (bottom), the spatial +
distributional views of the same per-vertex quantity (Fischl 1999 Fig 9 + Fig 11).

Fair, dense per-vertex distortion: ``viz.compute_kring_distortion`` recomputes each map's distortion
from the saved flat patch + base surface using a **fixed (k, n) neighborhood + optimal scale,
identical for all three methods** -- so the comparison is not contaminated by each optimizer's own
n_neighbors (the bias trap). FreeSurfer6 flat patches are reused from the existing comparison.

Usage
-----
    python -m benchmark.fig_flatmaps --cores-dir <cores_TS_dir> --ts <TS> \\
        --out-dir <cores_TS_dir>/flatmaps
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import paths  # noqa: E402
from .time_cores import S_TIME  # noqa: E402

# Fixed neighborhood for the per-vertex distortion field (identical for every method).
KRING_K = 2
KRING_N = None  # all neighbors in the ring -> dense, 100% coverage

METHODS = ["robust_fast", "tutte_default", "freesurfer6"]
MLABEL = {
    "robust_fast": "AutoFlatten\n(robust_fast)",
    "tutte_default": "AutoFlatten\n(tutte_default)",
    "freesurfer6": "FreeSurfer 6",
}
MCOL = {
    "robust_fast": "#2e6f95",  # blue
    "tutte_default": "#8a4f9e",  # purple
    "freesurfer6": "#d1495b",  # red
}
MLEG = {
    "robust_fast": "AutoFlatten (robust_fast)",
    "tutte_default": "AutoFlatten (tutte_default)",
    "freesurfer6": "FreeSurfer 6",
}


def _flat_path(cores_dir: Path, subject: str, hemi: str, method: str) -> Path | None:
    if method == "freesurfer6":
        p = (
            paths.DATA_ROOT
            / "fs6_compare"
            / "runs"
            / f"{subject}.{hemi}"
            / f"{hemi}.flat"
        )
    else:
        p = cores_dir / "flat" / f"{subject}_{hemi}_{method}.flat.patch.3d"
    return p if p.exists() else None


def _base_surface(subject: str, hemi: str) -> Path:
    surf = Path(paths.NARRATIVES_FS) / subject / "surf"
    fid = surf / f"{hemi}.fiducial"
    return fid if fid.exists() else surf / f"{hemi}.smoothwm"


def _per_vertex_distortion(flat_path: Path, base_path: Path):
    """(xy, distortion%) for one map, fixed-neighborhood + optimal scale (fair across methods)."""
    import autoflatten.freesurfer as fs
    from autoflatten.viz import compute_kring_distortion

    coords, orig_idx, _border = fs.read_patch(str(flat_path))
    xy = np.ascontiguousarray(coords[:, :2], dtype=np.float64)
    surf = fs.read_surface(str(base_path))
    bv = np.ascontiguousarray(surf[0], dtype=np.float64)
    bf = np.ascontiguousarray(
        surf[1], dtype=np.int64
    )  # cast big-endian -> native int64
    orig = np.ascontiguousarray(orig_idx, dtype=np.int64)
    # compute_kring_distortion returns (vertex_distortion, mean_distortion)
    vertex_dist, _mean = compute_kring_distortion(
        xy,
        bv,
        bf,
        orig,
        k=KRING_K,
        n_samples_per_ring=KRING_N,
        optimal_scale=True,
        signed=False,
        verbose=False,
    )
    return xy, np.asarray(vertex_dist, dtype=np.float64)


def render_hemi(
    subject: str, hemi: str, cores_dir: Path, out_dir: Path, ts: str
) -> None:
    panels = {}
    for m in METHODS:
        fp = _flat_path(cores_dir, subject, hemi, m)
        if fp is None:
            print(f"  [skip] {subject} {hemi} {m}: no flat patch")
            continue
        try:
            panels[m] = _per_vertex_distortion(fp, _base_surface(subject, hemi))
        except Exception as exc:  # noqa: BLE001
            print(f"  [skip] {subject} {hemi} {m}: {type(exc).__name__}: {exc}")
    if not panels:
        return

    # Shared scales across the (available) methods within this hemi.
    alld = np.concatenate([d for _, d in panels.values()])
    vmax = float(np.percentile(alld, 98))
    bins = np.linspace(0, vmax, 61)

    present = [m for m in METHODS if m in panels]
    fig = plt.figure(figsize=(2.5 * len(present), 4.9))
    gs = fig.add_gridspec(
        2, len(present), height_ratios=[2.4, 1.35], hspace=0.28, wspace=0.08
    )

    # --- top row: spatial distortion map per method (shared color scale) ---
    top_axes = []
    sc = None
    for j, m in enumerate(present):
        xy, dist = panels[m]
        ax = fig.add_subplot(gs[0, j])
        sc = ax.scatter(
            xy[:, 0],
            xy[:, 1],
            c=dist,
            s=0.5,
            cmap="inferno",
            vmin=0,
            vmax=vmax,
            linewidths=0,
            rasterized=True,
        )
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(MLABEL[m], fontsize=7.5, color=MCOL[m])
        top_axes.append(ax)
    if sc is not None:
        cbar = fig.colorbar(sc, ax=top_axes, fraction=0.022, pad=0.01)
        cbar.set_label("metric distortion (%)", fontsize=7)

    # --- bottom: all method histograms overlaid on one axis (% of vertices) ---
    axh = fig.add_subplot(gs[1, :])
    for m in present:
        dist = panels[m][1]
        w = np.full(len(dist), 100.0 / len(dist))  # -> % of vertices (fair across N)
        mean, p90 = float(np.mean(dist)), float(np.percentile(dist, 90))
        axh.hist(
            dist,
            bins=bins,
            weights=w,
            histtype="stepfilled",
            color=MCOL[m],
            alpha=0.18,
            linewidth=0,
        )
        axh.hist(
            dist,
            bins=bins,
            weights=w,
            histtype="step",
            color=MCOL[m],
            linewidth=1.6,
            label=f"{MLEG[m]}  (mean {mean:.1f}%, p90 {p90:.1f}%)",
        )
        axh.axvline(mean, color=MCOL[m], lw=0.8, ls="--", alpha=0.7)
    axh.set_xlim(0, vmax)
    axh.set_xlabel("metric distortion (%)", fontsize=8)
    axh.set_ylabel("% of vertices", fontsize=8)
    axh.legend(loc="upper right", frameon=False, fontsize=6.5)
    axh.spines["top"].set_visible(False)
    axh.spines["right"].set_visible(False)

    fig.suptitle(
        f"{subject} {hemi} -- per-vertex metric distortion (k={KRING_K}, optimal scale)",
        fontsize=9,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.text(
        0.006,
        0.995,
        f"fig_flatmaps  TS={ts}",
        fontsize=4.5,
        color="0.6",
        ha="left",
        va="top",
    )
    stem = f"{subject}_{hemi}_3method_{ts}"
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {stem}.pdf / .png  ({len(present)} methods)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--cores-dir", required=True, help="core-scaling run dir (<TS>) with flat/"
    )
    ap.add_argument("--ts", required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument(
        "--only", nargs="*", default=None, help="restrict to subjects, e.g. sub-055"
    )
    args = ap.parse_args()

    cores_dir = Path(args.cores_dir)
    out_dir = Path(args.out_dir) if args.out_dir else cores_dir / "flatmaps"
    hemis = [(s, h) for s, h in S_TIME if not args.only or s in set(args.only)]
    print(f"Rendering {len(hemis)} per-hemi flatmap figures -> {out_dir}")
    for subject, hemi in hemis:
        render_hemi(subject, hemi, cores_dir, out_dir, args.ts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
