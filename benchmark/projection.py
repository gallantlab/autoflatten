"""FreeSurfer-free cut projection (Phase 1 of the projection autoresearch).

The only FreeSurfer dependency in the projection phase is
``autoflatten.core.map_cuts_to_subject`` -> ``mri_label2label --regmethod surface``.
Underneath, surface-registration label mapping is just a nearest-neighbour lookup on
the registered sphere (``{hemi}.sphere.reg``): for each *target* vertex, find the
closest *source* (fsaverage) vertex on the sphere, and include the target vertex in the
mapped label if its nearest source vertex is in the source label. This is the
**target-driven (pull)** convention -- it is the only one consistent with the observed
projection-log counts (e.g. calcarine 124 src -> 156 trg: a forward push of 124 source
vertices can hit at most 124 unique targets, so the extra vertices can only come from a
target-driven map, which is natural here because individual surfaces (~204k verts) are
denser than fsaverage (~164k)).

Both ``sphere.reg`` files are on disk for every benchmark subject, so this removes the
FreeSurfer requirement and is validated against the cached FreeSurfer patches.

Reference (the FS call this replaces): ``autoflatten/core.py:map_cuts_to_subject``.
"""

from __future__ import annotations

import os

import nibabel as nib
import numpy as np
from scipy.spatial import cKDTree

import autoflatten.core as _core
from autoflatten.core import (
    ensure_continuous_cuts,
    fill_holes_in_patch,
    refine_cuts_with_geodesic,
)
from autoflatten.freesurfer import create_patch_file, load_surface
from autoflatten.utils import load_json

# Root of the FreeSurfer derivatives tree (contains fsaverage + all subjects).
DEFAULT_SUBJECTS_DIR = (
    "/data2/projects/idem/exps/narratives/datalad-narratives/derivatives/freesurfer"
)


def sphere_reg_path(subject, hemi, subjects_dir=None):
    subjects_dir = subjects_dir or os.environ.get("SUBJECTS_DIR", DEFAULT_SUBJECTS_DIR)
    return os.path.join(subjects_dir, subject, "surf", f"{hemi}.sphere.reg")


def load_sphere_reg(subject, hemi, subjects_dir=None):
    """Return the registered-sphere vertex coordinates (N, 3) for a subject/hemi."""
    coords, _ = nib.freesurfer.read_geometry(
        sphere_reg_path(subject, hemi, subjects_dir)
    )
    return np.asarray(coords, dtype=np.float64)


def map_label_surface(src_sphere, trg_sphere, src_label_idx, nearest_src=None):
    """Map a label from source to target, reproducing ``mri_label2label`` surface mode.

    FreeSurfer's surface-registration label mapping is the **union of two passes**:

    - *pull* (target-driven): for every target vertex, find its nearest source vertex on
      the registered sphere; include the target vertex if that source is in the label.
    - *push* (source-driven): for every source label vertex, include its nearest target
      vertex.

    The pull pass handles the bulk; the push pass recovers boundary target vertices whose
    own nearest source falls just outside the label. The union reproduces FreeSurfer's
    mapped-vertex counts exactly (validated on all dev hemispheres).

    Parameters
    ----------
    src_sphere : ndarray (Ns, 3)
        Source (fsaverage) ``sphere.reg`` coordinates.
    trg_sphere : ndarray (Nt, 3)
        Target (subject) ``sphere.reg`` coordinates.
    src_label_idx : array-like of int
        Source-subject vertex indices in the label.
    nearest_src : ndarray (Nt,), optional
        Precomputed target->nearest-source index map (shared across cuts for speed).

    Returns
    -------
    ndarray of int
        Sorted target-subject vertex indices in the mapped label.
    """
    src_label = np.unique(np.asarray(src_label_idx, dtype=np.int64))
    if src_label.size == 0:
        return np.array([], dtype=np.int64)

    # pull: each target vertex -> nearest source; include if source in label
    if nearest_src is None:
        _, nearest_src = cKDTree(src_sphere).query(trg_sphere, k=1)
    in_label = np.zeros(src_sphere.shape[0], dtype=bool)
    in_label[src_label] = True
    pull = np.nonzero(in_label[nearest_src])[0]

    # push: each source label vertex -> nearest target vertex
    _, push = cKDTree(trg_sphere).query(src_sphere[src_label], k=1)

    return np.union1d(pull, push).astype(np.int64)


def map_cuts_to_subject_python(
    vertex_dict,
    target_subject,
    hemi,
    source_subject="fsaverage",
    subjects_dir=None,
    trg_sphere=None,
    src_sphere=None,
):
    """Drop-in, FreeSurfer-free replacement for ``core.map_cuts_to_subject``.

    Maps every label in ``vertex_dict`` (fsaverage indices) to ``target_subject``
    indices via a single shared KDTree query (one NN lookup reused for all cuts).
    """
    if src_sphere is None:
        src_sphere = load_sphere_reg(source_subject, hemi, subjects_dir)
    if trg_sphere is None:
        trg_sphere = load_sphere_reg(target_subject, hemi, subjects_dir)

    # Build both KDTrees once and reuse across every cut.
    _, nearest_src = cKDTree(src_sphere).query(trg_sphere, k=1)
    trg_tree = cKDTree(trg_sphere)

    mapped = {}
    for cut_name, vertices in vertex_dict.items():
        src_label = np.unique(np.asarray(vertices, dtype=np.int64))
        if src_label.size == 0:
            mapped[cut_name] = []
            continue
        in_label = np.zeros(src_sphere.shape[0], dtype=bool)
        in_label[src_label] = True
        pull = np.nonzero(in_label[nearest_src])[0]
        _, push = trg_tree.query(src_sphere[src_label], k=1)
        mapped[cut_name] = np.union1d(pull, push).astype(np.int64)
    return mapped


def _curvature_weighted_graph_builder(subject, hemi, subjects_dir, alpha, morph="sulc"):
    """Factory for a ``(pts, polys) -> nx.Graph`` builder with curvature-weighted edges.

    Phase 2 hypothesis (c): the shipped geodesic refinement routes cuts along the
    *Euclidean-shortest* path, which is curvature-blind and pulls cuts off the sulcal fundi
    they should track. Reweighting edges by ``length * exp(-alpha * sulc_edge)`` makes the
    shortest path *prefer deep sulci* (high ``sulc``) and avoid gyral crowns. The returned
    builder matches ``core._build_surface_graph``'s signature so it can be monkeypatched in
    for the refinement call only (every endpoint/trapped-vertex heuristic stays identical;
    only the path's edge weights change).
    """
    import networkx as nx

    sulc = nib.freesurfer.read_morph_data(
        os.path.join(subjects_dir, subject, "surf", f"{hemi}.{morph}")
    )

    def build(pts, polys):
        polys = np.asarray(polys)
        edges = np.vstack([polys[:, [0, 1]], polys[:, [0, 2]], polys[:, [1, 2]]])
        length = np.linalg.norm(pts[edges[:, 0]] - pts[edges[:, 1]], axis=1)
        s_edge = 0.5 * (sulc[edges[:, 0]] + sulc[edges[:, 1]])
        weights = length * np.exp(-alpha * s_edge)  # high sulc -> low cost
        G = nx.Graph()
        G.add_nodes_from(range(len(pts)))
        G.add_weighted_edges_from(
            zip(edges[:, 0].tolist(), edges[:, 1].tolist(), weights.tolist())
        )
        return G

    return build


def _load_template_vertex_dict(hemi, template_file=None):
    """Load fsaverage cut/mwall labels for a hemisphere from the JSON template."""
    if template_file is None:
        from autoflatten.config import fsaverage_cut_template

        template_file = fsaverage_cut_template
    template_data = load_json(str(template_file))
    prefix = f"{hemi}_"
    return {
        key[len(prefix) :]: np.array(value)
        for key, value in template_data.items()
        if key.startswith(prefix)
    }


def project_python(
    subject,
    hemi,
    subjects_dir=None,
    template_file=None,
    continuity=True,
    refine_geodesic=True,
    refine_weight="euclidean",
    curv_alpha=0.1,
    curv_morph="sulc",
    out_patch=None,
    verbose=False,
):
    """Run the full projection phase **without FreeSurfer**.

    Mirrors ``autoflatten.cli.cmd_project`` (map -> continuity -> geodesic refine ->
    hole fill -> patch) but swaps the ``mri_label2label`` mapping for the validated
    Python KDTree mapper. Every downstream step is already pure Python.

    The ``continuity`` and ``refine_geodesic`` toggles exist for the Phase 2 refinement
    ablation (are these steps necessary / do they improve downstream distortion?). Both
    default to True (the shipped pipeline).

    Returns
    -------
    dict
        ``{"vertex_dict": ..., "patch_vertices": ..., "patch_file": ..., "n_surface": ...}``.
    """
    subjects_dir = subjects_dir or os.environ.get("SUBJECTS_DIR", DEFAULT_SUBJECTS_DIR)
    os.environ.setdefault("SUBJECTS_DIR", subjects_dir)

    vertex_dict = _load_template_vertex_dict(hemi, template_file)
    mapped = map_cuts_to_subject_python(
        vertex_dict, subject, hemi, subjects_dir=subjects_dir
    )

    fixed = (
        ensure_continuous_cuts(dict(mapped), subject, hemi)
        if continuity
        else dict(mapped)
    )
    if refine_geodesic:
        if refine_weight == "curvature":
            # Route cut paths along sulcal fundi by monkeypatching the graph builder
            # used inside refine_cuts_with_geodesic (endpoint/trapped logic unchanged).
            builder = _curvature_weighted_graph_builder(
                subject, hemi, subjects_dir, curv_alpha, curv_morph
            )
            orig = _core._build_surface_graph
            _core._build_surface_graph = builder
            try:
                fixed = refine_cuts_with_geodesic(
                    fixed, subject, hemi, medial_wall_vertices=fixed.get("mwall")
                )
            finally:
                _core._build_surface_graph = orig
        else:
            fixed = refine_cuts_with_geodesic(
                fixed, subject, hemi, medial_wall_vertices=fixed.get("mwall")
            )

    pts, polys = load_surface(subject, "inflated", hemi)
    excluded = set()
    for vertices in fixed.values():
        excluded.update(int(v) for v in vertices)
    hole_vertices = fill_holes_in_patch(polys, excluded)
    if hole_vertices:
        fixed["_hole_fill"] = np.array(list(hole_vertices))

    if out_patch is None:
        out_patch = f"/tmp/{subject}.{hemi}.fsfree.patch.3d"
    patch_file, patch_vertices = create_patch_file(out_patch, pts, polys, fixed)

    return {
        "vertex_dict": fixed,
        "patch_vertices": patch_vertices,
        "patch_file": patch_file,
        "n_surface": len(pts),
    }
