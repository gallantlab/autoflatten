"""Pipeline-schematic panels for the paper (compose in Illustrator).

Renders the AutoFlatten pipeline as a set of **individual, transparent-background,
label-free** panels (left hemisphere, medial view) so they can be laid out by hand:

1. ``fsaverage`` inflated with the template cut pattern superimposed (the cuts that
   define the patch).
2. For each example participant: their inflated surface with the *projected* patch
   superimposed.
3. For each example participant: the resulting flatmap.

All panels show binarized curvature (sulci dark / gyri light). On the inflated views,
faces touching a removed vertex (cuts + medial wall) are painted red -- the same
convention as :func:`autoflatten.viz.plot_projection`. Each panel is saved as both a
300-dpi PNG (transparent) and a PDF, with no axes, title, or colorbar.

Usage
-----
    python -m benchmark.fig_pipeline_panels \\
        --subjects sub-022 sub-041 sub-052 --hemi lh --config robust_fast
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import nibabel
import numpy as np
from matplotlib.collections import PolyCollection

# Import only the lightweight binary-format readers. autoflatten.viz is NOT imported here
# because it eagerly pulls in the JAX/igl flattening chain, which these panels do not need;
# the two helpers used (curvature loading + view angles) are inlined below.
from autoflatten.freesurfer import extract_patch_faces, read_patch, read_surface

from . import paths


def load_curvature(curv_path: str) -> np.ndarray:
    """Per-vertex FreeSurfer curvature (positive = sulci). Mirrors autoflatten.viz."""
    return nibabel.freesurfer.read_morph_data(curv_path)


def _get_view_angles(hemi: str, view: str) -> tuple[float, float]:
    """(elev, azim) for a view, matching autoflatten.viz.plot_projection conventions."""
    if view == "medial":
        return (0, 0) if hemi == "lh" else (0, 180)
    if view == "lateral":
        return (0, 180) if hemi == "lh" else (0, 0)
    if view == "ventral":
        return (-90, 180)
    if view == "frontal":
        return (0, -90)
    raise ValueError(f"Unknown view {view!r} (medial/lateral/ventral/frontal)")


# Binarized-curvature greyscale (FreeSurfer convention: curv > 0 = sulcus = dark).
GRAY_SULCUS = 0.35
GRAY_GYRUS = 0.75
# Lambertian shading for the inflated views (matches viz.plot_projection).
AMBIENT, DIFFUSE = 0.45, 0.55
LIGHT_DIR = np.array([0.3, 0.5, 0.8]) / np.linalg.norm([0.3, 0.5, 0.8])

DEFAULT_RUN = paths.DATA_ROOT / "paper_bench_2026" / "group_20260624-060144"
TEMPLATE = (
    Path(__file__).resolve().parent.parent
    / "autoflatten"
    / "default_templates"
    / "fsaverage_cuts_template.json"
)


def _base_gray(face_curv: np.ndarray) -> np.ndarray:
    """Per-face binarized-curvature grey level."""
    return np.where(face_curv > 0, GRAY_SULCUS, GRAY_GYRUS)


def _rotation(hemi: str, view: str) -> np.ndarray:
    """View rotation matrix (same construction as viz.plot_projection)."""
    elev, azim = _get_view_angles(hemi, view)
    er, ar = np.radians(elev), np.radians(azim)
    ca, sa = np.cos(ar), np.sin(ar)
    ce, se = np.cos(er), np.sin(er)
    r_base = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    rz = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, ce, -se], [0, se, ce]])
    return rx @ r_base @ rz


def render_inflated(
    vertices: np.ndarray,
    faces: np.ndarray,
    curv: np.ndarray,
    removed_mask: np.ndarray,
    hemi: str,
    out_stem: Path,
    view: str = "medial",
    dpi: int = 300,
) -> None:
    """Render one inflated-surface panel: curvature grey + removed faces red.

    ``removed_mask`` is a per-vertex boolean (cuts + medial wall). A face is drawn red
    if any of its vertices is removed -- the patch boundary then reads as the red region's
    edge. Transparent background, no axes/labels.
    """
    face_curv = curv[faces].mean(axis=1)
    base_gray = _base_gray(face_curv)
    face_is_cut = removed_mask[faces].any(axis=1)

    verts_per_face = vertices[faces]
    v0, v1, v2 = verts_per_face[:, 0], verts_per_face[:, 1], verts_per_face[:, 2]
    normals = np.cross(v1 - v0, v2 - v0)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-10
    centroids = verts_per_face.mean(axis=1)

    r = _rotation(hemi, view)
    rverts = vertices @ r.T
    rface = verts_per_face @ r.T
    rnorm = normals @ r.T
    rcent = centroids @ r.T

    vis = rnorm[:, 2] > 0  # back-face culling
    shading = AMBIENT + DIFFUSE * np.clip(rnorm[vis] @ LIGHT_DIR, 0, 1)

    colors = np.ones((vis.sum(), 4))
    cut_v = face_is_cut[vis]
    g = base_gray[vis] * shading
    colors[~cut_v, :3] = g[~cut_v, None]
    colors[cut_v, 0] = np.clip(0.8 + 0.2 * shading[cut_v], 0, 1)
    colors[cut_v, 1] = 0.0
    colors[cut_v, 2] = 0.0

    order = np.argsort(rcent[vis, 2])
    fig, ax = plt.subplots(figsize=(5, 6))
    ax.add_collection(
        PolyCollection(
            rface[vis][order][:, :, :2],
            facecolors=colors[order],
            edgecolors="face",
            linewidths=0,
            antialiaseds=False,  # AA on a transparent canvas blends seams -> see-through look
        )
    )
    margin = 5
    ax.set_xlim(rverts[:, 0].min() - margin, rverts[:, 0].max() + margin)
    ax.set_ylim(rverts[:, 1].min() - margin, rverts[:, 1].max() + margin)
    ax.set_aspect("equal")
    ax.axis("off")
    _save(fig, out_stem, dpi)


def _orient_flat(xy: np.ndarray, anat: np.ndarray, hemi: str) -> np.ndarray:
    """Rotate flat coords so the anatomical anterior-posterior axis is horizontal.

    The optimizer leaves each flat map at an arbitrary in-plane orientation. Using each
    patch vertex's anatomical position on the base surface (FreeSurfer RAS, +Y = anterior),
    fit the flat-space direction of increasing anterior and rotate it to point LEFT for the
    left hemisphere (anterior/frontal left, posterior/occipital right); the right hemisphere
    is mirrored (anterior right). Pure rotation about the centroid -- shape and chirality of
    the map are unchanged.
    """
    xy_c = xy - xy.mean(axis=0)
    ap = anat[:, 1] - anat[:, 1].mean()  # FreeSurfer RAS Y: anterior positive
    coeffs, *_ = np.linalg.lstsq(xy_c, ap, rcond=None)  # flat gradient of anterior
    phi = np.arctan2(coeffs[1], coeffs[0])
    target = np.pi if hemi == "lh" else 0.0  # anterior -> left (lh) / right (rh)
    theta = target - phi
    c, s = np.cos(theta), np.sin(theta)
    return xy_c @ np.array([[c, -s], [s, c]]).T


def render_flatmap(
    flat_patch_path: Path,
    base_surface_path: Path,
    curv_path: Path,
    out_stem: Path,
    hemi: str = "lh",
    orient: bool = True,
    dpi: int = 300,
) -> None:
    """Render a flatmap panel colored by binarized curvature. Transparent, no labels.

    When ``orient`` is set, the map is rotated so the anatomical A-P axis is horizontal
    (anterior left for lh; see :func:`_orient_flat`).
    """
    flat_vertices, orig_indices, _ = read_patch(str(flat_patch_path))
    base_vertices, base_faces = read_surface(str(base_surface_path))
    curv = load_curvature(str(curv_path))

    faces = extract_patch_faces(base_faces, orig_indices)
    xy = flat_vertices[:, :2]
    if orient:
        xy = _orient_flat(xy, base_vertices[orig_indices], hemi)
    face_curv = curv[orig_indices][faces].mean(axis=1)
    base_gray = _base_gray(face_curv)

    triang = tri.Triangulation(xy[:, 0], xy[:, 1], faces)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.tripcolor(triang, facecolors=base_gray, cmap="gray", vmin=0.0, vmax=1.0)
    ax.set_aspect("equal")
    ax.axis("off")
    _save(fig, out_stem, dpi)


def _save(fig, out_stem: Path, dpi: int) -> None:
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(
            f"{out_stem}.{ext}",
            dpi=dpi,
            transparent=True,
            bbox_inches="tight",
            pad_inches=0,
        )
    plt.close(fig)
    print(f"  wrote {out_stem.name}.png / .pdf")


def _removed_from_patch(patch_path: Path, n_vertices: int) -> np.ndarray:
    """Per-vertex removed mask: vertices NOT retained in the patch."""
    _, orig_indices, _ = read_patch(str(patch_path))
    mask = np.ones(n_vertices, dtype=bool)
    mask[orig_indices] = False
    return mask


def _removed_from_template(hemi: str, n_vertices: int) -> np.ndarray:
    """Per-vertex removed mask from the fsaverage template (mwall + all cuts)."""
    tmpl = json.loads(TEMPLATE.read_text())
    removed = np.zeros(n_vertices, dtype=bool)
    for key, idx in tmpl.items():
        if key.startswith(f"{hemi}_"):
            removed[np.asarray(idx, dtype=np.int64)] = True
    return removed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--subjects", nargs="+", default=["sub-022", "sub-041", "sub-052"])
    ap.add_argument("--hemi", default="lh", choices=["lh", "rh"])
    ap.add_argument("--config", default="robust_fast")
    ap.add_argument("--run-dir", default=str(DEFAULT_RUN))
    ap.add_argument("--fs-dir", default=str(paths.NARRATIVES_FS))
    ap.add_argument(
        "--out-dir", default=str(DEFAULT_RUN / "figures" / "pipeline_panels")
    )
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument(
        "--no-orient",
        action="store_true",
        help="keep the optimizer's raw flat orientation (default: A-P horizontal)",
    )
    args = ap.parse_args()

    hemi = args.hemi
    run_dir = Path(args.run_dir)
    fs_dir = Path(args.fs_dir)
    out_dir = Path(args.out_dir)

    # --- fsaverage inflated + template patch ---
    print("fsaverage:")
    fsa = fs_dir / "fsaverage" / "surf"
    verts, faces = read_surface(str(fsa / f"{hemi}.inflated"))
    curv = load_curvature(str(fsa / f"{hemi}.curv"))
    removed = _removed_from_template(hemi, len(verts))
    render_inflated(
        verts,
        faces,
        curv,
        removed,
        hemi,
        out_dir / f"fsaverage_{hemi}_inflated_patch",
        dpi=args.dpi,
    )

    # --- per subject: inflated + projected patch, and flatmap ---
    for subj in args.subjects:
        print(f"{subj}:")
        surf = fs_dir / subj / "surf"
        verts, faces = read_surface(str(surf / f"{hemi}.inflated"))
        curv = load_curvature(str(surf / f"{hemi}.curv"))
        patch = run_dir / "patches" / f"{subj}_{hemi}.autoflatten.patch.3d"
        removed = _removed_from_patch(patch, len(verts))
        render_inflated(
            verts,
            faces,
            curv,
            removed,
            hemi,
            out_dir / f"{subj}_{hemi}_inflated_patch",
            dpi=args.dpi,
        )
        flat = run_dir / "flat" / f"{subj}_{hemi}_{args.config}.flat.patch.3d"
        render_flatmap(
            flat,
            surf / f"{hemi}.fiducial",
            surf / f"{hemi}.curv",
            out_dir / f"{subj}_{hemi}_flatmap",
            hemi=hemi,
            orient=not args.no_orient,
            dpi=args.dpi,
        )

    print(f"\nDone -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
