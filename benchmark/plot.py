"""Render flatmap images for a logged experiment — visual verification.

Metrics can hide a tangled map, so every saved run can be eyeballed. Given an experiment
id, this finds the flat patches that run saved (from its ledger record), looks up each
patch's base surface from the manifest, and renders the flat mesh with flipped triangles
highlighted in red.

By default this uses a **fast** single-panel renderer (mesh + flips + boundary), pulling
the distortion/flip numbers from the ledger record so nothing is recomputed. ``--full``
uses the package's slower three-panel :func:`autoflatten.viz.plot_flatmap` (which
recomputes per-vertex distortion — minutes on a full hemisphere).

Usage
-----
    python -m benchmark.plot <experiment_id>
    python -m benchmark.plot --latest probe
    python -m benchmark.plot <experiment_id> --full
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np

from . import paths
from .harness import load_manifest
from .ledger import Ledger


def _surface_lookup(manifest: dict) -> dict[tuple[str, str], str]:
    return {(e["subject"], e["hemi"]): e["surface_path"] for e in manifest["entries"]}


def _parse_subject_hemi(flat_path: str) -> tuple[str, str]:
    """``.../sub-022.lh.flat.patch.3d`` -> ("sub-022", "lh")."""
    parts = Path(flat_path).name.split(".")
    return parts[0], parts[1]


def fast_flatmap(
    flat_path: str,
    surface_path: str,
    out_path: str,
    title: str,
    subtitle: str = "",
) -> str:
    """Fast single-panel flatmap: mesh in gray, flipped triangles in red, boundary blue.

    Skips the expensive per-vertex distortion recompute (the bottleneck in
    :func:`autoflatten.viz.plot_flatmap`); the numbers come from the ledger instead.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri

    from autoflatten.freesurfer import extract_patch_faces, read_patch, read_surface

    flat_vertices, orig_indices, is_border = read_patch(flat_path)
    _, base_faces = read_surface(surface_path)
    faces = extract_patch_faces(base_faces, orig_indices)
    xy = flat_vertices[:, :2]

    v0, v1, v2 = xy[faces[:, 0]], xy[faces[:, 1]], xy[faces[:, 2]]
    areas = 0.5 * (
        (v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1])
        - (v2[:, 0] - v0[:, 0]) * (v1[:, 1] - v0[:, 1])
    )
    flipped = areas < 0
    n_flipped = int(flipped.sum())

    fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)
    triang = mtri.Triangulation(xy[:, 0], xy[:, 1], faces)
    # Gray fill for all faces, then overlay flipped faces in red.
    ax.tripcolor(
        triang,
        facecolors=np.where(flipped, 1.0, 0.0),
        cmap="Greys" if n_flipped == 0 else "Reds",
        vmin=0,
        vmax=1,
        edgecolors="none",
    )
    ax.triplot(triang, color="0.6", linewidth=0.1)
    if np.sum(is_border) > 0:
        ax.scatter(xy[is_border, 0], xy[is_border, 1], s=1.5, c="tab:blue", zorder=5)
    ax.set_aspect("equal")
    ax.axis("off")
    full_title = title + (f"\n{subtitle}" if subtitle else "")
    ax.set_title(full_title, fontsize=10)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_experiment(
    experiment_id: str,
    out_root: Optional[Path] = None,
    full: bool = False,
) -> list[str]:
    """Render flatmaps for every flat patch an experiment saved. Returns the PNG paths."""
    records = Ledger().read()
    record = next((r for r in records if r["experiment_id"] == experiment_id), None)
    if record is None:
        raise SystemExit(f"No ledger record for experiment {experiment_id!r}")
    artifacts = [
        a for a in record.get("artifacts", []) if a.get("kind") == "flat_patch"
    ]
    if not artifacts:
        raise SystemExit(
            f"Experiment {experiment_id} has no saved flat patches "
            "(re-run with --save to render flatmaps)."
        )

    surfaces = _surface_lookup(load_manifest())
    per_subject = {(r["subject"], r["hemi"]): r for r in record.get("per_subject", [])}
    out_dir = (out_root or paths.DATA_ROOT / "figures") / experiment_id
    out_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for art in artifacts:
        flat = art["path"]
        subject, hemi = _parse_subject_hemi(flat)
        surface = surfaces.get((subject, hemi))
        if surface is None:
            print(f"  ! no surface in manifest for {subject} {hemi}; skipping")
            continue
        out = out_dir / f"{subject}.{hemi}.png"
        title = f"{subject} {hemi} — {record['label']}"
        if full:
            _full_plot(flat, surface, str(out), title)
        else:
            ps: dict[str, Any] = per_subject.get((subject, hemi), {})
            sub = _subtitle(ps)
            fast_flatmap(flat, surface, str(out), title, subtitle=sub)
        print(f"  wrote {out}")
        written.append(str(out))
    return written


def _subtitle(ps: dict[str, Any]) -> str:
    if not ps:
        return ""
    bits = []
    if "mean_distortion" in ps:
        bits.append(f"{ps['mean_distortion']:.2f}% dist")
    if "n_flipped" in ps:
        bits.append(f"{ps['n_flipped']} flipped")
    if "runtime_s" in ps:
        bits.append(f"{ps['runtime_s']:.0f}s")
    return "  |  ".join(bits)


def _full_plot(flat: str, surface: str, out: str, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from autoflatten.viz import plot_flatmap

    plot_flatmap(flat, base_surface_path=surface, output_path=out, title=title)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("experiment_id", nargs="?", help="experiment id to plot")
    ap.add_argument(
        "--latest", metavar="KIND", help="plot the latest record of this kind"
    )
    ap.add_argument(
        "--full",
        action="store_true",
        help="use the slow 3-panel plot_flatmap (recomputes distortion)",
    )
    args = ap.parse_args()

    if args.latest:
        rec = Ledger().latest(kind=args.latest)
        if rec is None:
            print(f"No '{args.latest}' experiments in the ledger.", file=sys.stderr)
            return 1
        eid = rec["experiment_id"]
    elif args.experiment_id:
        eid = args.experiment_id
    else:
        ap.error("provide an experiment_id or --latest KIND")

    print(f"Plotting experiment {eid}...")
    written = plot_experiment(eid, full=args.full)
    print(f"Wrote {len(written)} figure(s) to {paths.DATA_ROOT / 'figures' / eid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
