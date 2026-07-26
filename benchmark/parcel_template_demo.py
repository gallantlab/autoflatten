"""Template-flexibility demo: flatten an arbitrary anatomical parcel as the patch.

AutoFlatten ships one template (the fsaverage medial wall + 5 anatomical cuts), but the
template format is general: a template JSON is just ``{hemi}_<region>`` -> fsaverage
vertex indices to *exclude*, and the patch is whatever survives. This script demonstrates
that any region can become a template by deriving one from an off-the-shelf anatomical
parcellation:

    take one parcel of an fsaverage ``.annot`` -> its **complement** is the cut/removed
    region (a neutral ``{hemi}_excluded`` template key) -> run the standard FreeSurfer-free
    pipeline (project -> flatten) -> the parcel alone comes out as a flatmap.

For each parcel we write the fsaverage-space template JSON (the reusable artifact) and read
it straight back, so everything downstream consumes the file rather than the in-memory
indices -- the demo shows template JSON in, flatmap out. Then, for each subject, we project
that template (validated ``sphere.reg`` KDTree mapper, no FreeSurfer), flatten with the
shipping ``robust_fast`` config, and render Illustrator-ready panels (transparent
background, no labels): the fsaverage and per-subject inflated surfaces with the cut-out
(removed) region washed semi-transparent red over the shaded curvature (so the curvature
still reads underneath) and the patch left as full, un-washed curvature, and the resulting
flatmap.

Each inflated panel is shown from a per-parcel view (the ``PARCEL_VIEW`` dict): either a
named aspect (medial/lateral/ventral/frontal) or, for parcels that sit off the cardinal
aspects, an explicit ``(elev, azim)`` angle pair. ``--inflated-view`` overrides it.

Flat patches are reused if they already exist (so the panels can be re-rendered cheaply);
pass ``--reflatten`` to force re-running the flattening optimization.

A largest-connected-component guard is applied to the projected patch: the mapped parcel
boundary can leave a stray island that would disconnect the patch and break flattening.

A ``--parcels`` name may also be a *composite region* (``COMPOSITE_REGIONS``): a union of
aparc parcels, optionally cropped to a ball around the temporal pole -- e.g.
``anteriortemporal`` (the temporal-pole cap). With ``--relax-cut`` each patch gets a relaxation
(relief) cut: a thin slit authored on fsaverage (boundary -> centroid, ``{hemi}_relaxcut``
key), mapped, re-knit with continuity repair, and subtracted; the patch is flattened with
and without it and the true-global distortion change is reported. A relief cut only helps
on an intrinsically curved cap (the anterior temporal lobe drops ~1 pp), not a near-flat
gyral parcel (distortion rises -- only the cross-cut penalty, no curvature to relieve).

Usage
-----
    python -m benchmark.parcel_template_demo                       # 4 parcels x 3 subjects
    python -m benchmark.parcel_template_demo --parcels fusiform --subjects sub-022
    python -m benchmark.parcel_template_demo --parcels anteriortemporal \\
        --subjects sub-022 --relax-cut --relax-cut-width 1        # relief-cut comparison
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import networkx as nx
import nibabel as nib
import nibabel.freesurfer.io as fsio
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import PolyCollection
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from autoflatten.core import ensure_continuous_cuts, fill_holes_in_patch
from autoflatten.freesurfer import create_patch_from_keep, load_surface

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

# Hardcoded fallback for the fsaverage label dir (the box where this benchmark was authored).
_FS6_FSAVERAGE_LABEL = Path("/data2/freesurfer-6.0/subjects/fsaverage/label")


def _find_fsaverage_label_dir(preferred: Path | None = None) -> Path:
    """Locate an fsaverage ``label/`` dir that has *real* ``.annot`` content.

    fsaverage parcellations ship with every FreeSurfer install, but the datalad-managed
    fsaverage under ``SUBJECTS_DIR`` keeps its annots as unfetched git-annex symlinks (the
    dir exists; the ``.annot`` is a broken link). So we probe candidate roots and accept the
    first whose ``lh.aparc.annot`` *resolves* (``Path.exists`` follows symlinks, returning
    False for a broken annex stub): ``preferred`` (if given, e.g. derived from ``--fs-dir``),
    then ``FREESURFER_HOME`` (canonical, always real), then ``SUBJECTS_DIR``, then the
    authoring box's FreeSurfer-6 install.
    """
    candidates = []
    if preferred is not None:
        candidates.append(preferred)
    fs_home = os.environ.get("FREESURFER_HOME")
    if fs_home:
        candidates.append(Path(fs_home) / "subjects" / "fsaverage" / "label")
    subjects_dir = os.environ.get("SUBJECTS_DIR")
    if subjects_dir:
        candidates.append(Path(subjects_dir) / "fsaverage" / "label")
    candidates.append(_FS6_FSAVERAGE_LABEL)
    for cand in candidates:
        if (cand / "lh.aparc.annot").exists():  # follows symlinks: real content only
            return cand
    return _FS6_FSAVERAGE_LABEL


# Demo regions: four large disc-like Desikan-Killiany (aparc) parcels plus one composite
# region (the anterior temporal lobe, a curved temporal-pole cap; see COMPOSITE_REGIONS).
DEFAULT_PARCELS = [
    "lateraloccipital",
    "superiorfrontal",
    "superiorparietal",
    "fusiform",
    "anteriortemporal",
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
# tilt. Tuple angles are tuned for the left hemisphere; _view_rotation mirrors them across the
# sagittal plane (azim -> 180 - azim) for the right. Overridden for all parcels by --inflated-view.
PARCEL_VIEW: dict[str, str | tuple[float, float]] = {
    "lateraloccipital": (0.0, 135.0),
    "superiorfrontal": "medial",
    "superiorparietal": (20.0, 200.0),
    "fusiform": "ventral",
    "anteriortemporal": (
        -25.0,
        215.0,
    ),  # antero-latero-inferior, facing the temporal pole
}

DEFAULT_OUT = paths.DATA_ROOT / "paper_bench_2026" / "parcel_template_demo"

# Cut-out (removed) region wash: red, matching the "removed = red" convention of
# fig_pipeline_panels. Painted semi-transparently OVER the shaded curvature so the
# curvature still reads underneath; the patch itself keeps full, un-washed curvature.
REMOVED_RGB = np.array([0.85, 0.12, 0.12])
REMOVED_ALPHA = 0.55


def parcel_vertices(
    hemi: str, parcel: str, annot: str = "aparc", label_dir: Path | None = None
) -> np.ndarray:
    """fsaverage vertex indices belonging to ``parcel`` in the given ``.annot``.

    Parameters
    ----------
    hemi : str
        Hemisphere, ``"lh"`` or ``"rh"``.
    parcel : str
        Parcel name as it appears in the ``.annot`` color table.
    annot : str, optional
        fsaverage annotation name (default ``"aparc"``).
    label_dir : Path, optional
        Directory holding the fsaverage ``.annot`` files. If None, resolved via
        :func:`_find_fsaverage_label_dir`.
    """
    if label_dir is None:
        label_dir = _find_fsaverage_label_dir()
    path = label_dir / f"{hemi}.{annot}.annot"
    labels, _, names = fsio.read_annot(str(path))
    names = [n.decode() if isinstance(n, bytes) else n for n in names]
    if parcel not in names:
        raise ValueError(f"parcel {parcel!r} not in {path.name}; available: {names}")
    return np.nonzero(labels == names.index(parcel))[0].astype(np.int64)


# Composite regions: a union of aparc parcels, optionally cropped to a ball around the
# temporal pole (``pole_radius_mm``). The anterior temporal lobe is a curved 3D cap (the
# temporal pole), so it flattens with high distortion -- the case where a relief cut earns
# its keep, unlike a near-developable single gyral parcel.
COMPOSITE_REGIONS: dict[str, dict] = {
    "anteriortemporal": {
        # Lateral + ventral anterior temporal lobe. entorhinal is deliberately excluded: it
        # sits on the medial wall, so including it pokes a thin strip onto the medial surface
        # that flattening stretches into a spike.
        "parcels": [
            "temporalpole",
            "superiortemporal",
            "middletemporal",
            "inferiortemporal",
            "fusiform",
        ],
        # Crop a *geodesic* ball around the temporal pole (mm along the inflated surface), not
        # an axis-aligned slice or a Euclidean ball: surface distance never grabs the far bank
        # of a sulcus (Euclidean-close, surface-far), so the cap stays a clean disc with no thin
        # cross-fold necks that break topology / stretch into spikes when flattened.
        "pole_radius_mm": 55.0,
    },
}


def region_vertices(
    hemi: str,
    name: str,
    annot: str,
    surf_coords: np.ndarray,
    faces: np.ndarray,
    label_dir: Path | None = None,
) -> np.ndarray:
    """Vertices of a composite region: union of parcels, cropped to a geodesic pole ball."""
    spec = COMPOSITE_REGIONS[name]
    idx = np.unique(
        np.concatenate(
            [parcel_vertices(hemi, p, annot, label_dir) for p in spec["parcels"]]
        )
    )
    faces = np.asarray(faces, dtype=np.int64)
    n = surf_coords.shape[0]
    radius = spec.get("pole_radius_mm")
    if radius is not None:
        # Geodesic ball: shortest-path distance from the temporal pole along the region's own
        # mesh edges (weighted by Euclidean edge length on the inflated surface), keeping
        # vertices within ``radius``. Staying on-surface never grabs the far bank of a sulcus
        # (Euclidean-close, surface-far), so the cap has no thin cross-fold necks.
        in_reg = np.zeros(n, dtype=bool)
        in_reg[idx] = True
        fr = faces[in_reg[faces].all(axis=1)]
        e = np.vstack([fr[:, [0, 1]], fr[:, [1, 2]], fr[:, [0, 2]]])
        w = np.linalg.norm(surf_coords[e[:, 0]] - surf_coords[e[:, 1]], axis=1)
        g = nx.Graph()
        g.add_nodes_from(idx.tolist())
        g.add_weighted_edges_from(zip(e[:, 0].tolist(), e[:, 1].tolist(), w.tolist()))
        pole = int(idx[np.argmax(surf_coords[idx, 1])])  # most anterior vertex
        if pole in g:
            dist = nx.single_source_dijkstra_path_length(
                g, pole, cutoff=radius, weight="weight"
            )
            idx = np.array(sorted(dist), dtype=np.int64)
    return largest_cc(idx, faces, n)


def resolve_region(
    name: str,
    hemi: str,
    annot: str,
    surf_coords: np.ndarray,
    faces: np.ndarray,
    label_dir: Path | None = None,
) -> np.ndarray:
    """fsaverage vertices for a region name: a composite if known, else a single parcel."""
    if name in COMPOSITE_REGIONS:
        return region_vertices(hemi, name, annot, surf_coords, faces, label_dir)
    return parcel_vertices(hemi, name, annot, label_dir)


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


def relaxation_cut(
    parcel_idx: np.ndarray, faces: np.ndarray, surf_coords: np.ndarray, width: int = 0
) -> np.ndarray:
    """Author an illustrative relaxation (relief) cut for a parcel, on fsaverage.

    A thin slit from the parcel boundary inward to its centroid: removing it lets a dome-like
    patch open and relax instead of shearing. Returned as fsaverage vertex indices. The slit
    is the within-parcel shortest path (edge weights = Euclidean distance on ``surf_coords``)
    from the boundary vertex farthest from the centroid to the centroid vertex; ``width``
    dilates it by that many mesh rings (kept inside the parcel). Anchoring on the boundary is
    what keeps it a *slit* (one boundary loop) rather than an interior hole.
    """
    faces = np.asarray(faces, dtype=np.int64)
    n = surf_coords.shape[0]
    # Reduce to a single connected component first: a disconnected parcel would put the slit
    # endpoints in different graph components (no path) -- composites are already largest_cc'd,
    # but a bare single parcel from parcel_vertices() is not.
    parcel_idx = largest_cc(np.unique(np.asarray(parcel_idx, dtype=np.int64)), faces, n)
    in_p = np.zeros(n, dtype=bool)
    in_p[parcel_idx] = True

    # Parcel-internal edges -> weighted graph (the slit must stay inside the parcel).
    fp = faces[in_p[faces].all(axis=1)]
    edges = np.vstack([fp[:, [0, 1]], fp[:, [1, 2]], fp[:, [0, 2]]])
    w = np.linalg.norm(surf_coords[edges[:, 0]] - surf_coords[edges[:, 1]], axis=1)
    g = nx.Graph()
    g.add_nodes_from(parcel_idx.tolist())
    g.add_weighted_edges_from(
        zip(edges[:, 0].tolist(), edges[:, 1].tolist(), w.tolist())
    )

    # Centroid vertex: parcel vertex nearest the parcel's mean position.
    centroid_xyz = surf_coords[parcel_idx].mean(axis=0)
    centroid_v = int(
        parcel_idx[
            np.argmin(np.linalg.norm(surf_coords[parcel_idx] - centroid_xyz, axis=1))
        ]
    )

    # Boundary vertices: parcel vertices in a face that straddles the parcel edge.
    straddle = faces[in_p[faces].any(axis=1) & ~in_p[faces].all(axis=1)]
    bmask = np.zeros(n, dtype=bool)
    for col in range(3):
        cv = straddle[:, col]
        bmask[cv[in_p[cv]]] = True
    boundary = np.nonzero(bmask)[0]
    if boundary.size == 0:
        raise ValueError("parcel has no boundary (covers the whole surface?)")
    # Farthest boundary vertex from the centroid -> the longest radial slit.
    start = int(
        boundary[
            np.argmax(
                np.linalg.norm(surf_coords[boundary] - surf_coords[centroid_v], axis=1)
            )
        ]
    )

    path = nx.shortest_path(g, start, centroid_v, weight="weight")
    cut = set(path)
    for _ in range(max(0, width)):  # dilate within the parcel
        cut |= {nb for v in list(cut) for nb in g.neighbors(v)}
    return np.array(sorted(cut), dtype=np.int64)


def write_template_json(
    hemi: str,
    parcel_idx: np.ndarray,
    n_vertices: int,
    out_path: Path,
    cut_idx: np.ndarray | None = None,
) -> Path:
    """Write the reusable fsaverage-space template excluding the parcel's complement.

    The exclusion list lives under a neutral ``{hemi}_excluded`` key rather than
    ``{hemi}_mwall``: the template loaders treat every ``{hemi}_<region>`` key the same
    (union into the excluded set), and the ``mwall`` name is only meaningful to the
    geodesic-refine barrier -- which a parcel complement (no anatomical medial wall) should
    not trigger anyway. An optional relaxation cut is written under ``{hemi}_relaxcut`` as a
    separate thin-cut key (a 1D slit inside the parcel, repaired by continuity, not part of
    the solid complement).
    """
    complement = np.setdiff1d(np.arange(n_vertices, dtype=np.int64), parcel_idx)
    tmpl = {f"{hemi}_excluded": complement.tolist()}
    if cut_idx is not None and len(cut_idx):
        tmpl[f"{hemi}_relaxcut"] = np.asarray(cut_idx, dtype=np.int64).tolist()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(tmpl))
    return out_path


def read_template_json(
    path: Path, hemi: str, n_vertices: int
) -> tuple[np.ndarray, np.ndarray | None]:
    """Read back a template JSON written by :func:`write_template_json`.

    Parameters
    ----------
    path : Path
        Template JSON file to read.
    hemi : str
        Hemisphere key prefix (``"lh"`` or ``"rh"``).
    n_vertices : int
        Number of fsaverage vertices, used to complement ``{hemi}_excluded`` back into a
        keep-set.

    Returns
    -------
    keep : np.ndarray
        fsaverage vertex indices to keep (the complement of ``{hemi}_excluded``), as
        ``np.int64``.
    cut_idx : np.ndarray or None
        fsaverage vertex indices of the ``{hemi}_relaxcut`` key, as ``np.int64``, or None
        if the key is absent or empty.
    """
    tmpl = json.loads(path.read_text())
    excluded = np.asarray(tmpl[f"{hemi}_excluded"], dtype=np.int64)
    keep = np.setdiff1d(np.arange(n_vertices, dtype=np.int64), excluded)
    relaxcut = tmpl.get(f"{hemi}_relaxcut")
    cut_idx = np.asarray(relaxcut, dtype=np.int64) if relaxcut else None
    return keep, cut_idx


def project_parcel(
    subject: str,
    hemi: str,
    parcel_fsavg: np.ndarray,
    fs_dir: Path,
    out_patch: Path,
    relax_cut_fsavg: np.ndarray | None = None,
) -> tuple[Path, np.ndarray, int]:
    """Project the parcel to ``subject`` (FS-free) and build a single-component patch.

    If ``relax_cut_fsavg`` is given, that fsaverage relaxation cut is mapped alongside the
    parcel, repaired with :func:`ensure_continuous_cuts` (the mapped thin path can fragment,
    and with geodesic refine off, continuity repair is what re-knits the slit), and subtracted
    from the patch -- demonstrating a template-defined relief cut on an arbitrary patch.

    Note: with a relax cut, ``ensure_continuous_cuts`` loads the subject surfaces via the
    ``SUBJECTS_DIR`` env var (it takes no subjects-dir arg), so callers must have it pointing at
    ``fs_dir``. ``main`` sets it; a direct caller passing a different ``fs_dir`` must too.

    Returns ``(patch_file, kept_vertices, n_surface)``.
    """
    to_map = {"parcel": parcel_fsavg}
    if relax_cut_fsavg is not None and len(relax_cut_fsavg):
        to_map["relaxcut"] = relax_cut_fsavg
    mapped = map_cuts_to_subject_python(to_map, subject, hemi, subjects_dir=str(fs_dir))

    pts, polys = load_surface(subject, "inflated", hemi, subjects_dir=str(fs_dir))
    polys = np.asarray(polys, dtype=np.int64)
    n = len(pts)

    kept0 = largest_cc(mapped["parcel"], polys, n)
    excluded = set(np.setdiff1d(np.arange(n, dtype=np.int64), kept0).tolist())

    if "relaxcut" in mapped and len(mapped["relaxcut"]):
        # Continuity repair on the mapped slit (de-hardcoded: a non-anatomical cut key), then
        # fold it into the exclusion set *before* hole filling. A boundary-connected slit just
        # notches the outer boundary (one loop, preserved); an interior portion -- where the
        # mapped endpoint landed just inside the patch boundary -- becomes a hole that the
        # fill below closes, keeping the patch a single disc instead of a 5-loop mess.
        cut_dict = ensure_continuous_cuts(
            {"relaxcut": mapped["relaxcut"]}, subject, hemi
        )
        excluded |= {int(v) for v in cut_dict["relaxcut"]}

    holes = fill_holes_in_patch(polys, excluded)
    if holes:
        excluded |= {int(v) for v in holes}
    kept = largest_cc(
        np.array(sorted(set(range(n)) - excluded), dtype=np.int64), polys, n
    )

    out_patch.parent.mkdir(parents=True, exist_ok=True)
    create_patch_from_keep(str(out_patch), pts, polys, kept)
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


def true_global_distortion(
    patch_3d: Path, flat_patch: Path, subject: str, hemi: str, fs_dir: Path
) -> float | None:
    """True-geodesic global distortion (%) of a flat patch, at the distance-optimal scale.

    Energy-independent quality yardstick (see :mod:`benchmark.truedist`): heat-method
    geodesics on the fiducial surface vs. 2D flat distances. Returns ``None`` (with a note) if
    the geodesic backend (libigl) is unavailable, so the comparison never breaks the demo.
    """
    try:
        from autoflatten.flatten import FlattenConfig, SurfaceFlattener
        from autoflatten.freesurfer import read_patch

        from . import truedist

        base = fs_dir / subject / "surf" / f"{hemi}.fiducial"
        fl = SurfaceFlattener(FlattenConfig())
        fl.load_data(
            str(patch_3d), str(base)
        )  # geometry/fiducial (shares orig indices)
        ref = truedist.compute_truegeo(fl)
        uv = read_patch(str(flat_patch))[0][:, :2].astype(np.float64)
        return float(truedist.true_distortion_full(uv, ref)["true_global_at_optscale"])
    except Exception as exc:  # noqa: BLE001 - distortion is a best-effort extra
        print(f"    (true-distortion skipped: {exc})")
        return None


def _process_one(
    subj: str,
    hemi: str,
    p_idx: np.ndarray,
    parcel: str,
    view: str | tuple[float, float],
    fs_dir: Path,
    out_dir: Path,
    fig_dir: Path,
    args,
    relax_cut_fsavg: np.ndarray | None,
    tag: str,
) -> tuple[Path, Path]:
    """Project -> render inflated -> flatten -> render flatmap for one subject/parcel variant.

    ``tag`` ("", "_nocut", "_relaxcut") disambiguates the output filenames. Returns
    ``(patch_3d_path, flat_patch_path)``.
    """
    surf = fs_dir / subj / "surf"
    patch, kept, n = project_parcel(
        subj,
        hemi,
        p_idx,
        fs_dir,
        out_dir / "patches" / f"{subj}_{hemi}_{parcel}{tag}.patch.3d",
        relax_cut_fsavg=relax_cut_fsavg,
    )
    print(f"    {tag or 'patch'}: {kept.size}/{n} vertices kept")

    s_verts, s_faces = nib.freesurfer.read_geometry(str(surf / f"{hemi}.inflated"))
    s_curv = load_curvature(str(surf / f"{hemi}.curv"))
    s_mask = np.zeros(n, dtype=bool)
    s_mask[kept] = True
    render_cut_inflated(
        np.asarray(s_verts),
        np.asarray(s_faces, dtype=np.int64),
        s_curv,
        s_mask,
        hemi,
        fig_dir / f"{subj}_{hemi}_{parcel}{tag}_inflated_patch",
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
        out_dir / "flat" / f"{subj}_{hemi}_{parcel}{tag}_{args.config}.flat.patch.3d",
        reuse=not args.reflatten,
    )
    render_flatmap(
        flat,
        surf / f"{hemi}.fiducial",
        surf / f"{hemi}.curv",
        fig_dir / f"{subj}_{hemi}_{parcel}{tag}_flatmap",
        hemi=hemi,
        orient=not args.no_orient,
        dpi=args.dpi,
    )
    return patch, flat


def _view_rotation(hemi: str, view: str | tuple[float, float]) -> np.ndarray:
    """Rotation matrix for a named view or an explicit ``(elev, azim)`` pair.

    Named views delegate to fig_pipeline_panels (medial/lateral/ventral/frontal); a tuple
    builds the same matrix construction directly from the given angles, letting a parcel be
    shown from an oblique angle facing its surface.

    The PARCEL_VIEW tuples are tuned for the left hemisphere. For the right hemisphere they are
    mirrored across the sagittal plane, which is exactly ``azim -> 180 - azim`` (elevation
    unchanged): this matches the lateral/medial hemisphere flip in _get_view_angles and is a
    proper rotation, so back-face culling stays correct.
    """
    if isinstance(view, str):
        return _rotation(hemi, view)
    elev, azim = view
    if hemi == "rh":
        azim = 180.0 - azim
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
    ap.add_argument(
        "--annot-dir",
        default=None,
        help="directory holding the fsaverage `.annot` files; default: auto-detected "
        "from --fs-dir / FREESURFER_HOME / SUBJECTS_DIR",
    )
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
    ap.add_argument(
        "--relax-cut",
        action="store_true",
        help="add a template-defined relaxation (relief) cut to each parcel, flatten with "
        "and without it, and report the true-distortion change",
    )
    ap.add_argument(
        "--relax-cut-width",
        type=int,
        default=0,
        help="dilate the relaxation slit by this many mesh rings (default 0 = 1-wide)",
    )
    args = ap.parse_args()

    hemi = args.hemi
    fs_dir = Path(args.fs_dir)
    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figures"
    # ensure_continuous_cuts loads subject surfaces via SUBJECTS_DIR; pin it to --fs-dir.
    os.environ["SUBJECTS_DIR"] = str(fs_dir)

    # fsaverage inflated + curvature (shared across parcels) for the template panel.
    fsa = fs_dir / "fsaverage" / "surf"
    fsa_verts, fsa_faces = nib.freesurfer.read_geometry(str(fsa / f"{hemi}.inflated"))
    fsa_faces = np.asarray(fsa_faces, dtype=np.int64)
    fsa_curv = load_curvature(str(fsa / f"{hemi}.curv"))
    n_fsavg = len(fsa_verts)

    label_dir = (
        Path(args.annot_dir)
        if args.annot_dir
        else _find_fsaverage_label_dir(fs_dir / "fsaverage" / "label")
    )
    print(f"fsaverage parcellations: {label_dir}")

    for parcel in args.parcels:
        view = args.inflated_view or PARCEL_VIEW.get(parcel, "lateral")
        print(f"\n=== parcel: {parcel} ({hemi}, {view} view) ===")
        p_idx = resolve_region(
            parcel, hemi, args.annot, np.asarray(fsa_verts), fsa_faces, label_dir
        )
        print(f"  fsaverage parcel: {p_idx.size} vertices")

        cut_idx = None
        if args.relax_cut:
            # Must be computed from the original p_idx, before the write/round-trip below.
            cut_idx = relaxation_cut(
                p_idx, fsa_faces, np.asarray(fsa_verts), width=args.relax_cut_width
            )
            print(f"  relaxation cut: {cut_idx.size} fsaverage vertices")

        tmpl_path = write_template_json(
            hemi,
            p_idx,
            n_fsavg,
            out_dir / "templates" / f"{hemi}_{parcel}.json",
            cut_idx=cut_idx,
        )
        # Round-trip through the written template so everything downstream consumes the same
        # artifact a user would supply -- the demo proves "template JSON in -> flatmap out",
        # not just "in-memory indices in -> flatmap out".
        p_idx, cut_idx = read_template_json(tmpl_path, hemi, n_fsavg)

        # fsaverage template panel: the parcel is the patch (full curvature), the complement
        # (and the relaxation slit, if any) washed red.
        patch_mask = np.zeros(n_fsavg, dtype=bool)
        patch_mask[largest_cc(p_idx, fsa_faces, n_fsavg)] = True
        if cut_idx is not None:
            patch_mask[cut_idx] = False
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
            if not args.relax_cut:
                _process_one(
                    subj,
                    hemi,
                    p_idx,
                    parcel,
                    view,
                    fs_dir,
                    out_dir,
                    fig_dir,
                    args,
                    relax_cut_fsavg=None,
                    tag="",
                )
                continue

            # With/without comparison: flatten the parcel both ways and report distortion.
            patch0, flat0 = _process_one(
                subj,
                hemi,
                p_idx,
                parcel,
                view,
                fs_dir,
                out_dir,
                fig_dir,
                args,
                relax_cut_fsavg=None,
                tag="_nocut",
            )
            patch1, flat1 = _process_one(
                subj,
                hemi,
                p_idx,
                parcel,
                view,
                fs_dir,
                out_dir,
                fig_dir,
                args,
                relax_cut_fsavg=cut_idx,
                # Width is in the tag so a different --relax-cut-width gets its own patch/flat
                # /kring-cache filenames and is never served a stale reused flat.
                tag=f"_relaxcut_w{args.relax_cut_width}",
            )
            d0 = true_global_distortion(patch0, flat0, subj, hemi, fs_dir)
            d1 = true_global_distortion(patch1, flat1, subj, hemi, fs_dir)
            if d0 is not None and d1 is not None:
                print(
                    f"    true-global distortion: no-cut {d0:.2f}%  ->  "
                    f"relaxcut {d1:.2f}%  ({d1 - d0:+.2f} pp)"
                )

    print(f"\nDone -> {fig_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
