"""Template-flexibility demo: flatten an arbitrary anatomical parcel as the patch.

AutoFlatten ships one template (the fsaverage medial wall + 5 anatomical cuts), but the
template format is general: a template JSON is just ``{hemi}_<region>`` -> fsaverage
vertex indices to *exclude*, and the patch is whatever survives. This script demonstrates
that any region can become a template by deriving one from an off-the-shelf anatomical
parcellation:

    take one parcel of an fsaverage ``.annot`` -> its **complement** is the cut/removed
    region (``{hemi}_mwall``) -> run the standard FreeSurfer-free pipeline (project ->
    flatten) -> the parcel alone comes out as a flatmap.

For each parcel we write the fsaverage-space template JSON (the reusable artifact), then
for each subject we project it (validated ``sphere.reg`` KDTree mapper, no FreeSurfer),
flatten with the shipping ``robust_fast`` config, and render Illustrator-ready panels
(transparent background, no labels): the fsaverage and per-subject inflated surfaces with
the cut-out (removed) region washed semi-transparent red over the shaded curvature (so the
curvature still reads underneath) and the patch left as full, un-washed curvature, and the
resulting flatmap.

Each inflated panel is shown from a per-parcel view (the ``PARCEL_VIEW`` dict): either a
named aspect (medial/lateral/ventral/frontal) or, for parcels that sit off the cardinal
aspects, an explicit ``(elev, azim)`` angle pair. ``--inflated-view`` overrides it.

Flat patches are reused if they already exist (so the panels can be re-rendered cheaply);
pass ``--reflatten`` to force re-running the flattening optimization.

A largest-connected-component guard is applied to the projected patch: the mapped parcel
boundary can leave a stray island that would disconnect the patch and break flattening.

Usage
-----
    python -m benchmark.parcel_template_demo                       # 4 parcels x 3 subjects
    python -m benchmark.parcel_template_demo --parcels fusiform --subjects sub-022
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import nibabel as nib
import nibabel.freesurfer.io as fsio
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import PolyCollection
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from autoflatten.core import fill_holes_in_patch
from autoflatten.freesurfer import create_patch_file, load_surface

from . import paths
from .fig_pipeline_panels import (
    AMBIENT,
    DIFFUSE,
    LIGHT_DIR,
    _base_gray,
    _rotation,
    _save,
    load_curvature,
    render_flatmap,
)
from .probe_tutte_init import make_flatten_fn
from .projection import map_cuts_to_subject_python
from .time_cores import make_config

# fsaverage parcellations ship with FreeSurfer; the datalad fsaverage's annot content is
# an unfetched annex stub, so read the annot from the FreeSurfer-6 install (identical
# standard 163842-vertex fsaverage indexing as the narratives sphere.reg, verified).
FS6_FSAVERAGE_LABEL = Path("/data2/freesurfer-6.0/subjects/fsaverage/label")

# The four large, disc-like Desikan-Killiany (aparc) parcels chosen for the demo.
DEFAULT_PARCELS = [
    "lateraloccipital",
    "superiorfrontal",
    "superiorparietal",
    "fusiform",
]
DEFAULT_SUBJECTS = ["sub-022", "sub-041", "sub-052"]

# Natural inflated-surface view per parcel: the aspect where the parcel is most fully
# visible. A value is either a named view (medial/lateral/ventral/frontal) or an explicit
# (elev, azim) pair for parcels that sit off the cardinal aspects. Projection is orthographic,
# so *elevation* foreshortens the A-P extent (a high-tilt camera looks "squashed"); azimuth
# just orbits the vertical axis and keeps proportions natural. superiorfrontal is largely a
# *medial* structure (a lateral view catches only its dorsal lip), so it is shown medially;
# fusiform is ventral; lateraloccipital wraps the occipital pole, so it is orbited toward the
# pole at zero elevation (no squash); superiorparietal is dorsal-posterior and needs a modest
# tilt. Angles are for the left hemisphere. Overridden for all parcels by --inflated-view.
PARCEL_VIEW: dict[str, str | tuple[float, float]] = {
    "lateraloccipital": (0.0, 135.0),
    "superiorfrontal": "medial",
    "superiorparietal": (20.0, 200.0),
    "fusiform": "ventral",
}

DEFAULT_OUT = paths.DATA_ROOT / "paper_bench_2026" / "parcel_template_demo"

# Cut-out (removed) region wash: red, matching the "removed = red" convention of
# fig_pipeline_panels. Painted semi-transparently OVER the shaded curvature so the
# curvature still reads underneath; the patch itself keeps full, un-washed curvature.
REMOVED_RGB = np.array([0.85, 0.12, 0.12])
REMOVED_ALPHA = 0.55


def parcel_vertices(hemi: str, parcel: str, annot: str = "aparc") -> np.ndarray:
    """fsaverage vertex indices belonging to ``parcel`` in the given ``.annot``."""
    path = FS6_FSAVERAGE_LABEL / f"{hemi}.{annot}.annot"
    labels, _, names = fsio.read_annot(str(path))
    names = [n.decode() if isinstance(n, bytes) else n for n in names]
    if parcel not in names:
        raise ValueError(f"parcel {parcel!r} not in {path.name}; available: {names}")
    return np.nonzero(labels == names.index(parcel))[0].astype(np.int64)


def largest_cc(kept: np.ndarray, faces: np.ndarray, n_vertices: int) -> np.ndarray:
    """Largest connected component of ``kept`` over faces lying entirely within ``kept``.

    Drops stray islands and kept vertices that no fully-kept face touches, guaranteeing a
    single connected patch (a disc) that the flattener can handle.
    """
    kept = np.unique(np.asarray(kept, dtype=np.int64))
    in_kept = np.zeros(n_vertices, dtype=bool)
    in_kept[kept] = True
    fk = faces[in_kept[faces].all(axis=1)]
    if len(fk) == 0:
        return kept
    edges = np.vstack([fk[:, [0, 1]], fk[:, [1, 2]], fk[:, [0, 2]]])
    adj = coo_matrix(
        (np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
        shape=(n_vertices, n_vertices),
    )
    adj = adj + adj.T
    _, comp = connected_components(adj, directed=False)
    comp_of_kept = comp[kept]
    vals, counts = np.unique(comp_of_kept, return_counts=True)
    biggest = vals[np.argmax(counts)]
    return kept[comp_of_kept == biggest]


def write_template_json(
    hemi: str, parcel: str, parcel_idx: np.ndarray, n_vertices: int, out_path: Path
) -> Path:
    """Write the reusable fsaverage-space template: mwall = complement of the parcel."""
    complement = np.setdiff1d(np.arange(n_vertices, dtype=np.int64), parcel_idx)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({f"{hemi}_mwall": complement.tolist()}))
    return out_path


def project_parcel(
    subject: str,
    hemi: str,
    parcel_fsavg: np.ndarray,
    fs_dir: Path,
    out_patch: Path,
) -> tuple[Path, np.ndarray, int]:
    """Project the parcel to ``subject`` (FS-free) and build a single-component patch.

    Returns ``(patch_file, kept_vertices, n_surface)``.
    """
    mapped = map_cuts_to_subject_python(
        {"parcel": parcel_fsavg}, subject, hemi, subjects_dir=str(fs_dir)
    )["parcel"]
    pts, polys = load_surface(subject, "inflated", hemi)
    polys = np.asarray(polys, dtype=np.int64)
    n = len(pts)

    kept = largest_cc(mapped, polys, n)
    excluded = set(np.setdiff1d(np.arange(n, dtype=np.int64), kept).tolist())
    holes = fill_holes_in_patch(polys, excluded)
    if holes:
        excluded |= {int(v) for v in holes}
    kept = np.array(sorted(set(range(n)) - excluded), dtype=np.int64)

    out_patch.parent.mkdir(parents=True, exist_ok=True)
    create_patch_file(str(out_patch), pts, polys, {"mwall": np.array(sorted(excluded))})
    return out_patch, kept, n


def flatten_patch(
    patch_file: Path,
    subject: str,
    hemi: str,
    config: str,
    fs_dir: Path,
    cache_dir: Path,
    out_flat: Path,
    reuse: bool = True,
) -> Path:
    """Flatten a patch with the named config (Tutte init + refinement); save flat patch.

    When ``reuse`` and the flat patch already exists, skip the (expensive) optimization and
    return it -- lets the panels be re-rendered cheaply without re-flattening.
    """
    if reuse and out_flat.exists():
        print(f"    reusing existing flat: {out_flat.name}")
        return out_flat
    from autoflatten.flatten import SurfaceFlattener

    surf = fs_dir / subject / "surf"
    base = surf / f"{hemi}.fiducial"
    base = base if base.exists() else surf / f"{hemi}.smoothwm"
    cfg = make_config(config)

    fl = SurfaceFlattener(cfg)
    fl.load_data(str(patch_file), str(base))
    cache_dir.mkdir(parents=True, exist_ok=True)
    # Cache key includes the patch stem: the k-ring distances are patch-specific, so
    # different parcels for the same subject/hemi must not share a cache file.
    cache = cache_dir / (
        f"{patch_file.stem}.kring_k{cfg.kring.k_ring}"
        f"_n{cfg.kring.n_neighbors_per_ring}.npz"
    )
    fl.compute_kring_distances(cache_path=str(cache))
    fl.prepare_optimization()

    uv = np.asarray(make_flatten_fn("tutte", refine=True)(fl))
    out_flat.parent.mkdir(parents=True, exist_ok=True)
    fl.save_result(uv, str(out_flat))
    return out_flat


def _view_rotation(hemi: str, view: str | tuple[float, float]) -> np.ndarray:
    """Rotation matrix for a named view or an explicit ``(elev, azim)`` pair.

    Named views delegate to fig_pipeline_panels (medial/lateral/ventral/frontal); a tuple
    builds the same matrix construction directly from the given angles, letting a parcel be
    shown from an oblique angle facing its surface.
    """
    if isinstance(view, str):
        return _rotation(hemi, view)
    elev, azim = view
    er, ar = np.radians(elev), np.radians(azim)
    ca, sa = np.cos(ar), np.sin(ar)
    ce, se = np.cos(er), np.sin(er)
    r_base = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    rz = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, ce, -se], [0, se, ce]])
    return rx @ r_base @ rz


def render_cut_inflated(
    vertices: np.ndarray,
    faces: np.ndarray,
    curv: np.ndarray,
    patch_mask: np.ndarray,
    hemi: str,
    out_stem: Path,
    view: str | tuple[float, float] = "lateral",
    dpi: int = 300,
) -> None:
    """Inflated-surface panel: the cut-out region washed red, the patch left as curvature.

    Mirrors fig_pipeline_panels.render_inflated (back-face culling + Lambertian shading +
    depth sort). A face belongs to the patch only if all three vertices are kept, so the
    wash stops cleanly at the patch boundary; every other face is the removed region and
    gets a semi-transparent red wash painted over its shaded curvature -- the curvature
    still reads underneath, while the patch alone shows full, un-washed curvature.
    Opaque surface, transparent background, no axes/labels.
    """
    face_curv = curv[faces].mean(axis=1)
    base_gray = _base_gray(face_curv)
    face_patch = patch_mask[faces].all(axis=1)

    vpf = vertices[faces]
    v0, v1, v2 = vpf[:, 0], vpf[:, 1], vpf[:, 2]
    normals = np.cross(v1 - v0, v2 - v0)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-10
    centroids = vpf.mean(axis=1)

    r = _view_rotation(hemi, view)
    rverts = vertices @ r.T
    rface = vpf @ r.T
    rnorm = normals @ r.T
    rcent = centroids @ r.T

    vis = rnorm[:, 2] > 0  # back-face culling
    shading = AMBIENT + DIFFUSE * np.clip(rnorm[vis] @ LIGHT_DIR, 0, 1)

    # Shaded greyscale curvature for every visible face (the patch keeps this as-is).
    g = (base_gray[vis] * shading)[:, None]
    colors = np.ones((vis.sum(), 4))
    colors[:, :3] = g
    # Removed faces: alpha-composite red over the shaded curvature (curvature shows through).
    removed_v = ~face_patch[vis]
    colors[removed_v, :3] = (
        REMOVED_ALPHA * REMOVED_RGB[None, :] + (1 - REMOVED_ALPHA) * g[removed_v]
    )

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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parcels", nargs="+", default=DEFAULT_PARCELS)
    ap.add_argument("--subjects", nargs="+", default=DEFAULT_SUBJECTS)
    ap.add_argument("--hemi", default="lh", choices=["lh", "rh"])
    ap.add_argument(
        "--annot", default="aparc", help="fsaverage annotation (default aparc)"
    )
    ap.add_argument("--config", default="robust_fast")
    ap.add_argument("--fs-dir", default=str(paths.NARRATIVES_FS))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument(
        "--reflatten",
        action="store_true",
        help="re-run flattening even if the flat patch already exists (default: reuse)",
    )
    ap.add_argument(
        "--no-orient",
        action="store_true",
        help="keep the optimizer's raw flat orientation (default: A-P horizontal)",
    )
    ap.add_argument(
        "--inflated-view",
        default=None,
        help="view for the inflated panels; default is per-parcel (see PARCEL_VIEW)",
    )
    args = ap.parse_args()

    hemi = args.hemi
    fs_dir = Path(args.fs_dir)
    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figures"

    # fsaverage inflated + curvature (shared across parcels) for the template panel.
    fsa = fs_dir / "fsaverage" / "surf"
    fsa_verts, fsa_faces = nib.freesurfer.read_geometry(str(fsa / f"{hemi}.inflated"))
    fsa_faces = np.asarray(fsa_faces, dtype=np.int64)
    fsa_curv = load_curvature(str(fsa / f"{hemi}.curv"))
    n_fsavg = len(fsa_verts)

    for parcel in args.parcels:
        view = args.inflated_view or PARCEL_VIEW.get(parcel, "lateral")
        print(f"\n=== parcel: {parcel} ({hemi}, {view} view) ===")
        p_idx = parcel_vertices(hemi, parcel, args.annot)
        print(f"  fsaverage parcel: {p_idx.size} vertices")

        write_template_json(
            hemi,
            parcel,
            p_idx,
            n_fsavg,
            out_dir / "templates" / f"{hemi}_{parcel}.json",
        )

        # fsaverage template panel: the parcel is the patch (full curvature), the
        # complement (= the template's cut region) washed red.
        patch_mask = np.zeros(n_fsavg, dtype=bool)
        patch_mask[largest_cc(p_idx, fsa_faces, n_fsavg)] = True
        render_cut_inflated(
            np.asarray(fsa_verts),
            fsa_faces,
            fsa_curv,
            patch_mask,
            hemi,
            fig_dir / f"fsaverage_{hemi}_{parcel}_template",
            view=view,
            dpi=args.dpi,
        )

        for subj in args.subjects:
            print(f"  {subj}:")
            surf = fs_dir / subj / "surf"
            patch, kept, n = project_parcel(
                subj,
                hemi,
                p_idx,
                fs_dir,
                out_dir / "patches" / f"{subj}_{hemi}_{parcel}.patch.3d",
            )
            print(f"    projected patch: {kept.size}/{n} vertices kept")

            # subject inflated panel: projected patch keeps curvature, cut region washed red
            s_verts, s_faces = nib.freesurfer.read_geometry(
                str(surf / f"{hemi}.inflated")
            )
            s_curv = load_curvature(str(surf / f"{hemi}.curv"))
            s_mask = np.zeros(n, dtype=bool)
            s_mask[kept] = True
            render_cut_inflated(
                np.asarray(s_verts),
                np.asarray(s_faces, dtype=np.int64),
                s_curv,
                s_mask,
                hemi,
                fig_dir / f"{subj}_{hemi}_{parcel}_inflated_patch",
                view=view,
                dpi=args.dpi,
            )

            flat = flatten_patch(
                patch,
                subj,
                hemi,
                args.config,
                fs_dir,
                out_dir / "kring_cache",
                out_dir
                / "flat"
                / f"{subj}_{hemi}_{parcel}_{args.config}.flat.patch.3d",
                reuse=not args.reflatten,
            )
            render_flatmap(
                flat,
                surf / f"{hemi}.fiducial",
                surf / f"{hemi}.curv",
                fig_dir / f"{subj}_{hemi}_{parcel}_flatmap",
                hemi=hemi,
                orient=not args.no_orient,
                dpi=args.dpi,
            )

    print(f"\nDone -> {fig_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
