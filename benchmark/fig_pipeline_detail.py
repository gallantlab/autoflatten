"""AutoFlatten low-level pipeline flowchart (technical detail figure).

A vertical block diagram of every stage with its function, data structures and parameters,
grouped into the two phases (FreeSurfer-free projection, pyflatten flattening). Companion to the
high-level visual `fig_pipeline`.

Usage
-----
    python -m benchmark.fig_pipeline_detail --hemi lh --out-dir <dir> --ts <TS>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import to_rgba  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

PROJ_C = "#2e6f95"
FLAT_C = "#5a9367"
IO_C = "#b07d2b"  # inputs / outputs
ROBUST_C = "#2e6f95"  # robust_fast
TUTTE_C = "#8a4f9e"  # tutte_default


def _steps(hemi):
    return [
        (
            "proj",
            "io",
            "Inputs",
            f"{hemi}.sphere.reg · fsaverage {hemi}.sphere.reg\n"
            "template cuts JSON  (medial wall + 5 anatomical cuts)",
        ),
        (
            "proj",
            "step",
            "map_cuts_to_subject",
            "union push/pull nearest-neighbour on sphere.reg\n"
            "(FreeSurfer-free reimplementation of mri_label2label)",
        ),
        (
            "proj",
            "step",
            "ensure_continuous_cuts",
            "connect cut components (NetworkX shortest paths) — cuts kept in place\n"
            "[continuity-only; geodesic cut refinement available but off by default]",
        ),
        ("proj", "io", "create_patch_file", f"→  {hemi}.autoflatten.patch.3d"),
        (
            "flat",
            "step",
            "compute_kring_distances",
            "Numba Dijkstra · k = 7 rings · n neighbours/ring\n"
            "→ per-vertex geodesic targets  d⁰",
        ),
        (
            "flat",
            "step",
            "Initialization  (both configs)",
            "Tutte flip-free embedding (libigl harmonic)\n"
            "[package legacy option: FreeSurfer normal-projection init]",
        ),
        (
            "flat",
            "step",
            "3-epoch optimization  (JAX)",
            "energy  J_d (metric: Σ (d−d⁰)²)  +  J_a (area / no flips)\n"
            "epoch 1 area-dominant → 2 balanced → 3 distance-dominant\n"
            "vectorised log-spaced line search",
        ),
        (
            "flat",
            "step",
            "Negative-area removal",
            "final pass — eliminate flipped triangles\n"
            "(initial NAR off — the flip-free Tutte start makes it moot)",
        ),
        ("flat", "step", "Spring smoothing", "Laplacian smoothing (visual quality)"),
        ("flat", "io", "Distance-optimal rescale", f"→  {hemi}.flat.patch.3d"),
    ]


def _color(kind, phase):
    if kind == "io":
        return IO_C
    return PROJ_C if phase == "proj" else FLAT_C


def render(hemi, out_dir: Path, ts: str) -> None:
    steps = _steps(hemi)
    box_w = 7.4
    pitch = 1.70
    top = len(steps) * pitch
    fig_h = 0.66 * len(steps) + 2.4
    fig, ax = plt.subplots(figsize=(7.4, fig_h))
    ax.set_xlim(0, 10)
    ax.set_ylim(-2.9, top + 1.0)
    ax.axis("off")

    centers = []
    for i, (phase, kind, title, detail) in enumerate(steps):
        cy = top - i * pitch
        centers.append(cy)
        c = _color(kind, phase)
        h = 1.34
        ax.add_patch(
            FancyBboxPatch(
                (5 - box_w / 2, cy - h / 2),
                box_w,
                h,
                boxstyle="round,pad=0.02,rounding_size=0.10",
                facecolor=to_rgba(c, 0.12),
                edgecolor=c,
                linewidth=1.3,
                zorder=2,
            )
        )
        ax.text(
            5,
            cy + h / 2 - 0.22,
            title,
            ha="center",
            va="top",
            fontsize=8.5,
            fontweight="bold",
            color=c,
            zorder=3,
        )
        ax.text(
            5,
            cy + h / 2 - 0.52,
            detail,
            ha="center",
            va="top",
            fontsize=6.0,
            color="0.18",
            linespacing=1.4,
            zorder=3,
        )
        if i > 0:
            ax.add_patch(
                FancyArrowPatch(
                    (5, centers[i - 1] - h / 2),
                    (5, cy + h / 2),
                    arrowstyle="-|>",
                    mutation_scale=11,
                    color="0.55",
                    lw=1.2,
                    zorder=1,
                )
            )

    # phase labels on the left, spanning their boxes
    def _band(idx0, idx1, label, color):
        y0 = centers[idx1] - 0.7
        y1 = centers[idx0] + 0.7
        ax.add_patch(
            FancyBboxPatch(
                (0.15, y0),
                0.62,
                y1 - y0,
                boxstyle="round,pad=0.01,rounding_size=0.06",
                facecolor=to_rgba(color, 0.16),
                edgecolor=color,
                linewidth=0.8,
                zorder=0,
            )
        )
        ax.text(
            0.46,
            (y0 + y1) / 2,
            label,
            rotation=90,
            ha="center",
            va="center",
            fontsize=8.5,
            fontweight="bold",
            color=color,
        )

    _band(0, 3, "PROJECTION  (FreeSurfer-free)", PROJ_C)
    _band(4, 9, "FLATTENING  (pyflatten · JAX)", FLAT_C)

    # config-comparison callout below the flow
    cy0 = centers[-1] - 0.67
    ax.add_patch(
        FancyBboxPatch(
            (5 - box_w / 2, cy0 - 1.9),
            box_w,
            1.7,
            boxstyle="round,pad=0.02,rounding_size=0.10",
            facecolor="#f7f5ef",
            edgecolor="0.55",
            linewidth=1.0,
            linestyle="--",
            zorder=2,
        )
    )
    ax.text(
        5,
        cy0 - 0.36,
        "Two configurations  (both Tutte init · initial NAR off)",
        ha="center",
        va="top",
        fontsize=8,
        fontweight="bold",
        color="0.25",
        zorder=3,
    )
    ax.text(
        5,
        cy0 - 0.82,
        "robust_fast  (shipping default):  k-ring n = 6 · line-search 7  "
        "→  ~3× faster, ~same accuracy",
        ha="center",
        va="top",
        fontsize=6.4,
        color=ROBUST_C,
        zorder=3,
    )
    ax.text(
        5,
        cy0 - 1.24,
        "tutte_default  (quality-first):  k-ring n = 12 · full refinement  "
        "→  lowest distortion, slower",
        ha="center",
        va="top",
        fontsize=6.4,
        color=TUTTE_C,
        zorder=3,
    )

    ax.text(
        5,
        top + 0.8,
        "AutoFlatten — detailed pipeline",
        ha="center",
        fontsize=12,
        fontweight="bold",
    )
    fig.text(
        0.006,
        0.995,
        f"fig_pipeline_detail  TS={ts}",
        fontsize=4.5,
        color="0.6",
        ha="left",
        va="top",
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"fig_pipeline_detail_{ts}"
    for ext in ("pdf", "png"):
        fig.savefig(
            out_dir / f"{stem}.{ext}", bbox_inches="tight", pad_inches=0.12, dpi=300
        )
    plt.close(fig)
    print(f"  wrote {stem}.pdf / .png")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hemi", default="lh", choices=["lh", "rh"])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--ts", required=True)
    args = ap.parse_args()
    render(args.hemi, Path(args.out_dir), args.ts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
