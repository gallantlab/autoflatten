"""Flip-free (Tutte/LSCM) initial projection for surface flattening.

The FreeSurfer-style normal-axis projection (:func:`freesurfer_projection` in
:mod:`.algorithm`) produces many flipped triangles, which the initial negative-area-removal
(NAR) phase then spends time un-flipping. A **Tutte embedding** (harmonic map with the
boundary pinned to a convex circle) is *guaranteed flip-free* for disk topology, so starting
from it lets the initial-NAR phase be skipped entirely and feeds a clean map straight into
the existing geodesic-stress refinement (epochs + final NAR + spring).

This is the default initial projection used by :class:`SurfaceFlattener`
(``FlattenConfig.init_method == "tutte"``); see ``benchmark/FINDINGS.md`` for the
validation that motivated shipping it.
"""

from __future__ import annotations

import numpy as np


def flipfree_init(
    vertices: np.ndarray, faces: np.ndarray, method: str = "tutte"
) -> np.ndarray:
    """Compute a flip-free 2D embedding of a disk-topology patch.

    Parameters
    ----------
    vertices : (V, 3) float
        3D patch vertices (use the smoothed/fiducial surface for intrinsic weights).
    faces : (F, 3) int
        Triangles (single boundary loop / disk topology).
    method : {"tutte", "lscm"}
        ``tutte`` — harmonic map with the boundary pinned to a circle. Guaranteed
        injective (Tutte's theorem) → no flipped triangles.
        ``lscm`` — least-squares conformal map (2 pinned boundary vertices). Lower angle
        distortion but *not* guaranteed flip-free.

    Returns
    -------
    (V, 2) float
        2D coordinates (unscaled).
    """
    import igl

    v = np.ascontiguousarray(vertices, dtype=np.float64)
    f = np.ascontiguousarray(faces, dtype=np.int64)
    bnd = igl.boundary_loop(f)
    if bnd is None or len(bnd) == 0:
        raise ValueError("No boundary loop found; patch is not a disk.")

    if method == "tutte":
        bc = igl.map_vertices_to_circle(v, bnd.astype(np.int32))
        uv = igl.harmonic(v, f, bnd.astype(np.int64), np.ascontiguousarray(bc), 1)
    elif method == "lscm":
        # Pin the two most distant boundary vertices to (0,0) and (1,0).
        b = np.array([bnd[0], bnd[len(bnd) // 2]], dtype=np.int64)
        bc = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)
        uv, _ = igl.lscm(v, f, b, bc)
    else:
        raise ValueError(f"Unknown init method: {method!r}")
    return np.asarray(uv, dtype=np.float64)


def scale_to_area(uv: np.ndarray, faces: np.ndarray, target_area: float) -> np.ndarray:
    """Uniformly scale ``uv`` so its total (unsigned) 2D area matches ``target_area``.

    Tutte/LSCM map to ~unit scale, while the geodesic targets are in mm; matching the 3D
    patch area gives the refinement a sensible starting scale.
    """
    v0, v1, v2 = uv[faces[:, 0]], uv[faces[:, 1]], uv[faces[:, 2]]
    area2d = float(
        np.abs(
            0.5
            * (
                (v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1])
                - (v2[:, 0] - v0[:, 0]) * (v1[:, 1] - v0[:, 1])
            )
        ).sum()
    )
    if area2d <= 0 or target_area <= 0:
        return uv
    s = np.sqrt(target_area / area2d)
    centroid = uv.mean(axis=0)
    return (uv - centroid) * s + centroid
