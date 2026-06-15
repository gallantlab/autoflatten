"""AutoFlatten pipeline schematic (paper overview figure).

A visual pipeline with real thumbnails -- cortical surface -> 3D patch (medial wall + cuts) ->
flat map -- connected by labelled step arrows, with the two phases called out:
FreeSurfer-free **projection** and JAX **flattening** (pyflatten). The cut boundary is drawn in
red so it can be traced through the patch and the flat map.

Usage
-----
    python -m benchmark.fig_pipeline --subject sub-055 --hemi lh \\
        --flat <…/flat/sub-055_lh_robust_fast.flat.patch.3d> --out-dir <dir> --ts <TS>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from . import paths  # noqa: E402

GRAY = "#9aa0a6"
CUT = "#d1495b"
PROJ_C = "#2e6f95"  # projection phase
FLAT_C = "#5a9367"  # flattening phase


def _scatter(ax, xy, border, title, size=0.6, sub=8):
    """Gray point cloud with the cut boundary in red; equal aspect, no axes."""
    idx = np.arange(xy.shape[0])
    body = idx[~border][::sub]
    ax.scatter(
        xy[body, 0], xy[body, 1], s=size, color=GRAY, linewidths=0, rasterized=True
    )
    bd = idx[border]
    if len(bd):
        ax.scatter(
            xy[bd, 0],
            xy[bd, 1],
            s=size * 2.2,
            color=CUT,
            linewidths=0,
            rasterized=True,
            zorder=3,
        )
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=8)


def _phase_band(fig, x0, x1, y, label, color):
    fig.add_artist(
        FancyBboxPatch(
            (x0, y),
            x1 - x0,
            0.05,
            boxstyle="round,pad=0.004,rounding_size=0.02",
            transform=fig.transFigure,
            facecolor=color,
            alpha=0.18,
            edgecolor=color,
            linewidth=0.8,
            zorder=0,
        )
    )
    fig.text(
        (x0 + x1) / 2,
        y + 0.025,
        label,
        ha="center",
        va="center",
        fontsize=8,
        color=color,
        fontweight="bold",
    )


def _arrow(fig, x0, x1, y, steps, color):
    fig.add_artist(
        FancyArrowPatch(
            (x0, y),
            (x1, y),
            transform=fig.transFigure,
            arrowstyle="-|>",
            mutation_scale=14,
            color=color,
            linewidth=1.6,
            zorder=2,
        )
    )
    fig.text(
        (x0 + x1) / 2,
        y - 0.04,
        steps,
        ha="center",
        va="top",
        fontsize=6.2,
        color="0.25",
        linespacing=1.35,
    )


def render(subject, hemi, flat_path, out_dir: Path, ts: str) -> None:
    import autoflatten.freesurfer as fs

    surf = Path(paths.NARRATIVES_FS) / subject / "surf"
    base = surf / f"{hemi}.fiducial"
    base = base if base.exists() else surf / f"{hemi}.smoothwm"
    patch3d = surf / f"{hemi}.autoflatten.patch.3d"

    bv, _bf = fs.read_surface(str(base))[:2]
    pc, _po, pb = fs.read_patch(str(patch3d))
    fc, _fo, fb = fs.read_patch(str(flat_path))

    bv = np.asarray(bv, float)
    # lateral view (anterior-posterior horizontal, superior-inferior vertical)
    surf_xy = np.column_stack([bv[:, 1], bv[:, 2]])
    surf_border = np.zeros(bv.shape[0], bool)
    patch_xy = np.column_stack([pc[:, 1], pc[:, 2]])
    flat_xy = np.asarray(fc[:, :2], float)

    fig = plt.figure(figsize=(9.2, 3.4))
    ax1 = fig.add_axes([0.015, 0.30, 0.235, 0.52])
    ax2 = fig.add_axes([0.385, 0.30, 0.235, 0.52])
    ax3 = fig.add_axes([0.755, 0.30, 0.235, 0.52])

    _scatter(ax1, surf_xy, surf_border, "Cortical surface\n(white / fiducial)")
    _scatter(ax2, patch_xy, pb, "3D patch\n(medial wall + cuts)")
    _scatter(ax3, flat_xy, fb, "Flat map")

    # phase bands across the two transitions
    _phase_band(fig, 0.255, 0.62, 0.86, "PROJECTION  (FreeSurfer-free)", PROJ_C)
    _phase_band(fig, 0.625, 0.99, 0.86, "FLATTENING  (pyflatten, JAX)", FLAT_C)

    # step arrows
    _arrow(
        fig,
        0.265,
        0.375,
        0.56,
        "map fsaverage cuts (sphere.reg)\n→ ensure continuity\n"
        "→ geodesic refinement\n→ cut into patch",
        PROJ_C,
    )
    _arrow(
        fig,
        0.635,
        0.745,
        0.56,
        "k-ring geodesic targets\n→ Tutte flip-free init\n"
        "→ metric + area optimization\n→ spring smoothing",
        FLAT_C,
    )

    fig.text(0.0125, 0.20, "input: subject FreeSurfer surface", fontsize=6, color="0.4")
    fig.text(
        0.9875,
        0.20,
        f"output: {hemi}.flat.patch.3d",
        fontsize=6,
        color="0.4",
        ha="right",
    )
    fig.text(
        0.5, 0.95, "AutoFlatten pipeline", ha="center", fontsize=11, fontweight="bold"
    )
    fig.text(
        0.5,
        0.05,
        f"red = cut boundary / medial wall   ·   example: {subject} {hemi}",
        ha="center",
        fontsize=6,
        color="0.45",
    )
    fig.text(
        0.006,
        0.992,
        f"fig_pipeline  TS={ts}",
        fontsize=4.5,
        color="0.6",
        ha="left",
        va="top",
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"fig_pipeline_{ts}"
    for ext in ("pdf", "png"):
        fig.savefig(
            out_dir / f"{stem}.{ext}", bbox_inches="tight", pad_inches=0.1, dpi=300
        )
    plt.close(fig)
    print(f"  wrote {stem}.pdf / .png")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subject", default="sub-055")
    ap.add_argument("--hemi", default="lh", choices=["lh", "rh"])
    ap.add_argument("--flat", required=True, help="flattened patch (.flat.patch.3d)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--ts", required=True)
    args = ap.parse_args()
    render(args.subject, args.hemi, args.flat, Path(args.out_dir), args.ts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
