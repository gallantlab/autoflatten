#!/usr/bin/env python
"""Test script to compare native FreeSurfer medial wall vs projected fsaverage medial wall.

This script compares two approaches for defining the medial wall in cortical flattening:
1. Projected: Medial wall projected from fsaverage using mri_label2label
2. Native: Medial wall from FreeSurfer's aparc.a2009s.annot (vertices with label=-1)

Usage:
    python scripts/test_native_medial_wall.py /path/to/subject --hemi lh --output-dir ./comparison_results
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import nibabel as nib
import numpy as np

# Add parent directory to path so we can import autoflatten
sys.path.insert(0, str(Path(__file__).parent.parent))

from autoflatten.config import fsaverage_cut_template
from autoflatten.core import (
    ensure_continuous_cuts,
    fill_holes_in_patch,
    map_cuts_to_subject,
    refine_cuts_with_geodesic,
)
from autoflatten.freesurfer import (
    create_patch_file,
    load_surface,
    read_patch,
    setup_freesurfer,
)
from autoflatten.utils import load_json
from autoflatten.viz import compute_triangle_areas


def get_native_medial_wall(subject_dir, hemi, annot_file="aparc.a2009s.annot"):
    """
    Read medial wall vertices from FreeSurfer annotation file.

    The medial wall in FreeSurfer annotation files is identified by label index -1.

    Parameters
    ----------
    subject_dir : str
        Path to FreeSurfer subject directory
    hemi : str
        Hemisphere ('lh' or 'rh')
    annot_file : str
        Annotation file name (default: aparc.a2009s.annot)

    Returns
    -------
    np.ndarray
        Array of vertex indices in the medial wall (where label == -1)
    """
    annot_path = os.path.join(subject_dir, "label", f"{hemi}.{annot_file}")
    if not os.path.exists(annot_path):
        raise FileNotFoundError(f"Annotation file not found: {annot_path}")

    labels, ctab, names = nib.freesurfer.read_annot(annot_path)
    medial_wall_vertices = np.where(labels == -1)[0]
    return medial_wall_vertices


def check_medial_wall_topology(vertices, polys):
    """
    Verify medial wall forms a single connected component.

    Parameters
    ----------
    vertices : array-like
        Array of vertex indices in the medial wall
    polys : array-like
        Array of face indices with shape (n_faces, 3)

    Returns
    -------
    is_single : bool
        True if medial wall is a single connected component
    n_components : int
        Number of connected components found
    component_sizes : list
        List of sizes for each connected component
    """
    vertex_set = set(vertices)
    G = nx.Graph()
    G.add_nodes_from(vertices)

    # Add edges from faces where both vertices are in medial wall
    for face in polys:
        for i in range(3):
            v1, v2 = int(face[i]), int(face[(i + 1) % 3])
            if v1 in vertex_set and v2 in vertex_set:
                G.add_edge(v1, v2)

    components = list(nx.connected_components(G))
    n_components = len(components)
    component_sizes = sorted([len(c) for c in components], reverse=True)

    return n_components == 1, n_components, component_sizes


def compare_medial_walls(projected_mwall, native_mwall):
    """
    Compare projected and native medial wall vertex sets.

    Parameters
    ----------
    projected_mwall : array-like
        Vertex indices from projected fsaverage medial wall
    native_mwall : array-like
        Vertex indices from native FreeSurfer annotation

    Returns
    -------
    dict
        Dictionary containing comparison metrics:
        - n_projected: Number of vertices in projected medial wall
        - n_native: Number of vertices in native medial wall
        - n_intersection: Number of vertices in both
        - n_only_projected: Vertices only in projected
        - n_only_native: Vertices only in native
        - dice: Dice coefficient
        - jaccard: Jaccard index (IoU)
    """
    proj_set = set(projected_mwall)
    native_set = set(native_mwall)

    intersection = proj_set & native_set
    union = proj_set | native_set
    only_projected = proj_set - native_set
    only_native = native_set - proj_set

    # Dice coefficient: 2*|A∩B| / (|A| + |B|)
    dice = (
        2 * len(intersection) / (len(proj_set) + len(native_set))
        if (len(proj_set) + len(native_set)) > 0
        else 0
    )

    # Jaccard index: |A∩B| / |A∪B|
    jaccard = len(intersection) / len(union) if len(union) > 0 else 0

    return {
        "n_projected": len(proj_set),
        "n_native": len(native_set),
        "n_intersection": len(intersection),
        "n_only_projected": len(only_projected),
        "n_only_native": len(only_native),
        "only_projected": np.array(list(only_projected)),
        "only_native": np.array(list(only_native)),
        "dice": dice,
        "jaccard": jaccard,
    }


# ============================================================================
# Geodesic Boundary Smoothing Functions
# ============================================================================


def build_adjacency(polys, n_vertices):
    """
    Build vertex adjacency list from face connectivity.

    Parameters
    ----------
    polys : ndarray
        Array of face indices with shape (n_faces, 3)
    n_vertices : int
        Total number of vertices

    Returns
    -------
    list of sets
        adjacency[v] contains the set of vertices adjacent to v
    """
    adjacency = [set() for _ in range(n_vertices)]
    for face in polys:
        for i in range(3):
            v = int(face[i])
            adjacency[v].update(int(face[j]) for j in range(3) if j != i)
    return adjacency


def extract_mwall_boundary(mwall_vertices, polys, n_vertices, adjacency=None):
    """
    Extract the ordered boundary vertices of the medial wall.

    The boundary consists of medial wall vertices that are adjacent
    to at least one cortex (non-medial wall) vertex.

    Parameters
    ----------
    mwall_vertices : array-like
        Vertex indices in the medial wall
    polys : ndarray
        Surface triangles
    n_vertices : int
        Total number of vertices
    adjacency : list of sets, optional
        Pre-computed adjacency list. Built if not provided.

    Returns
    -------
    list
        Ordered list of boundary vertices forming a closed loop
    int
        Number of connected boundary components found
    """
    mwall_set = set(int(v) for v in mwall_vertices)

    if adjacency is None:
        adjacency = build_adjacency(polys, n_vertices)

    # Find boundary vertices (mwall vertices with ≥1 cortex neighbor)
    boundary_vertices = set()
    for v in mwall_set:
        for neighbor in adjacency[v]:
            if neighbor not in mwall_set:
                boundary_vertices.add(v)
                break

    if len(boundary_vertices) == 0:
        return [], 0

    # Build boundary graph (edges between adjacent boundary vertices)
    boundary_graph = nx.Graph()
    boundary_graph.add_nodes_from(boundary_vertices)
    for v in boundary_vertices:
        for neighbor in adjacency[v]:
            if neighbor in boundary_vertices:
                boundary_graph.add_edge(v, neighbor)

    # Check number of connected components
    components = list(nx.connected_components(boundary_graph))
    n_components = len(components)

    # Use the largest component
    if n_components > 1:
        largest_component = max(components, key=len)
        boundary_vertices = largest_component
        # Rebuild graph for largest component only
        boundary_graph = nx.Graph()
        boundary_graph.add_nodes_from(boundary_vertices)
        for v in boundary_vertices:
            for neighbor in adjacency[v]:
                if neighbor in boundary_vertices:
                    boundary_graph.add_edge(v, neighbor)

    # Trace the boundary loop using DFS
    # Start from the vertex with minimum index (for reproducibility)
    start = min(boundary_vertices)
    visited = {start}
    path = [start]
    current = start

    while True:
        # Find unvisited neighbor in boundary
        found_next = False
        neighbors = list(boundary_graph.neighbors(current))
        # Sort for reproducibility
        neighbors.sort()
        for neighbor in neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                path.append(neighbor)
                current = neighbor
                found_next = True
                break
        if not found_next:
            break

    return path, n_components


def sample_boundary_anchors(boundary_path, n_anchors=40, pts=None):
    """
    Sample evenly-spaced anchor points along the boundary.

    Parameters
    ----------
    boundary_path : list
        Ordered list of boundary vertex indices
    n_anchors : int
        Number of anchor points to sample
    pts : ndarray, optional
        Vertex coordinates (currently unused, kept for API compatibility).

    Returns
    -------
    list
        Vertex indices of sampled anchor points
    """
    n = len(boundary_path)
    if n == 0:
        return []
    if n <= n_anchors:
        return boundary_path.copy()

    # Uniform spacing by index - guarantees exactly n_anchors points
    indices = np.linspace(0, n - 1, n_anchors, dtype=int)
    # Ensure unique indices (linspace with int can sometimes produce duplicates at edges)
    indices = np.unique(indices)
    return [boundary_path[i] for i in indices]


def smooth_boundary_geodesic(anchors, pts, polys, surface_graph=None):
    """
    Connect anchor points with geodesic shortest paths.

    Parameters
    ----------
    anchors : list
        Ordered anchor vertex indices (closed loop, last connects to first)
    pts : ndarray
        Surface coordinates (fiducial preferred)
    polys : ndarray
        Surface triangles
    surface_graph : nx.Graph, optional
        Pre-built surface graph with edge weights. Built if not provided.

    Returns
    -------
    list
        Ordered vertices forming the smoothed boundary loop
    """
    if len(anchors) < 2:
        return anchors.copy()

    if surface_graph is None:
        surface_graph = nx.Graph()
        for triangle in polys:
            for i in range(3):
                v1 = int(triangle[i])
                for j in range(i + 1, 3):
                    v2 = int(triangle[j])
                    weight = np.linalg.norm(pts[v1] - pts[v2])
                    surface_graph.add_edge(v1, v2, weight=weight)

    smoothed_path = []
    n_anchors = len(anchors)

    for i in range(n_anchors):
        start = anchors[i]
        end = anchors[(i + 1) % n_anchors]  # Wrap to first anchor

        try:
            path = nx.shortest_path(surface_graph, start, end, weight="weight")
            # Exclude end to avoid duplicates (it will be start of next segment)
            smoothed_path.extend(path[:-1])
        except nx.NetworkXNoPath:
            # If no path found, just use the anchor
            smoothed_path.append(start)

    return smoothed_path


def fill_smoothed_boundary(
    smoothed_boundary, original_mwall, polys, n_vertices, pts, adjacency=None
):
    """
    Fill the interior of the smoothed boundary to get the new medial wall.

    Uses flood-fill from a seed point in the original medial wall.
    Validates the result by comparing centroids with the original medial wall.

    Parameters
    ----------
    smoothed_boundary : list
        Ordered vertices forming the smoothed boundary loop
    original_mwall : array-like
        Original medial wall vertices (used to find interior seed)
    polys : ndarray
        Surface triangles
    n_vertices : int
        Total number of vertices
    pts : ndarray
        Vertex coordinates for centroid validation
    adjacency : list of sets, optional
        Pre-computed adjacency list

    Returns
    -------
    ndarray
        Vertex indices of the smoothed medial wall (boundary + interior)
    """
    boundary_set = set(smoothed_boundary)
    original_set = set(int(v) for v in original_mwall)

    if adjacency is None:
        adjacency = build_adjacency(polys, n_vertices)

    # Compute centroid of original medial wall for validation
    original_centroid = pts[list(original_set)].mean(axis=0)

    # Find a seed point: original mwall vertex closest to the centroid
    # This ensures we start from a "central" point in the medial wall
    seed = None
    min_dist = float("inf")
    for v in original_set:
        if v not in boundary_set:
            dist = np.linalg.norm(pts[v] - original_centroid)
            if dist < min_dist:
                min_dist = dist
                seed = v

    if seed is None:
        # All original vertices are on boundary, return just boundary
        return np.array(smoothed_boundary)

    # Flood fill from seed, stopping at boundary
    interior = set()
    queue = [seed]
    visited = set(boundary_set)  # Treat boundary as already visited (barrier)

    while queue:
        v = queue.pop()
        if v in visited:
            continue
        visited.add(v)
        interior.add(v)

        for neighbor in adjacency[v]:
            if neighbor not in visited:
                queue.append(neighbor)

    # Result is boundary + interior
    result = boundary_set | interior

    # Simple sanity check: if result is larger than 1.5x original, we filled wrong side
    # The smoothed medial wall should be similar in size to the original
    if len(result) > 1.5 * len(original_set):
        # We filled the cortex instead of the medial wall
        # Take the complement: everything NOT in what we filled
        all_vertices = set(range(n_vertices))
        result = all_vertices - interior
        # Boundary vertices should be included
        result = result | boundary_set

    return np.array(list(result))


def smooth_medial_wall_morphological(
    mwall_vertices, polys, n_vertices, iterations=1, verbose=True
):
    """
    Smooth the medial wall boundary using morphological operations.

    Uses "opening" (erosion followed by dilation) to remove small protrusions.

    Parameters
    ----------
    mwall_vertices : array-like
        Vertex indices of the original medial wall
    polys : ndarray
        Surface triangles
    n_vertices : int
        Total number of vertices in the surface
    iterations : int
        Number of erosion/dilation iterations (default: 1)
    verbose : bool
        Whether to print progress messages

    Returns
    -------
    ndarray
        Vertex indices of the smoothed medial wall
    dict
        Smoothing statistics
    """
    if verbose:
        print(f"\n=== Smoothing medial wall with morphological operations ===")
        print(f"Original medial wall: {len(mwall_vertices)} vertices")
        print(f"Iterations: {iterations}")

    mwall_set = set(int(v) for v in mwall_vertices)
    adjacency = build_adjacency(polys, n_vertices)

    # Count original boundary
    original_boundary = sum(
        1 for v in mwall_set if any(n not in mwall_set for n in adjacency[v])
    )

    if verbose:
        print(f"Original boundary vertices: {original_boundary}")

    # Morphological opening: erosion then dilation
    # This removes small protrusions while preserving the overall shape
    current = mwall_set.copy()

    for i in range(iterations):
        if verbose:
            print(f"\nIteration {i + 1}:")

        # Erosion: remove boundary vertices (vertices with any cortex neighbor)
        eroded = set()
        for v in current:
            # Keep vertex only if ALL neighbors are in the set
            if all(n in current for n in adjacency[v]):
                eroded.add(v)

        if verbose:
            print(
                f"  After erosion: {len(eroded)} vertices (removed {len(current) - len(eroded)})"
            )

        # Dilation: add vertices adjacent to the eroded set
        dilated = eroded.copy()
        for v in eroded:
            for n in adjacency[v]:
                dilated.add(n)

        if verbose:
            print(
                f"  After dilation: {len(dilated)} vertices (added {len(dilated) - len(eroded)})"
            )

        current = dilated

    # Count final boundary
    final_boundary = sum(
        1 for v in current if any(n not in current for n in adjacency[v])
    )

    if verbose:
        print(f"\nFinal medial wall: {len(current)} vertices")
        print(f"Final boundary vertices: {final_boundary}")
        diff = len(current) - len(mwall_vertices)
        if diff > 0:
            print(f"Net change: added {diff} vertices")
        elif diff < 0:
            print(f"Net change: removed {-diff} vertices")

    stats = {
        "n_original": len(mwall_vertices),
        "n_smoothed": len(current),
        "n_boundary_original": original_boundary,
        "n_boundary_smoothed": final_boundary,
        "iterations": iterations,
    }

    return np.array(list(current)), stats


def smooth_medial_wall_geodesic(mwall_vertices, pts, polys, n_anchors=40, verbose=True):
    """
    Smooth the medial wall boundary using geodesic refinement.

    NOTE: This function has issues with complex boundaries. Consider using
    smooth_medial_wall_morphological() instead.

    Parameters
    ----------
    mwall_vertices : array-like
        Vertex indices of the original medial wall
    pts : ndarray
        Surface coordinates (fiducial preferred for accurate geodesics)
    polys : ndarray
        Surface triangles
    n_anchors : int
        Number of anchor points for geodesic smoothing (default: 40)
    verbose : bool
        Whether to print progress messages

    Returns
    -------
    ndarray
        Vertex indices of the smoothed medial wall
    dict
        Smoothing statistics
    """
    # Use morphological smoothing instead - geodesic has boundary tracing issues
    n_vertices = len(pts)
    # Convert n_anchors to iterations: fewer anchors = more smoothing = more iterations
    iterations = max(1, 50 // n_anchors)
    return smooth_medial_wall_morphological(
        mwall_vertices, polys, n_vertices, iterations=iterations, verbose=verbose
    )


def get_projected_medial_wall(subject, hemi, template_file=None):
    """
    Get the projected medial wall from fsaverage using the standard pipeline.

    Parameters
    ----------
    subject : str
        Subject ID
    hemi : str
        Hemisphere ('lh' or 'rh')
    template_file : str, optional
        Path to template file. If None, uses default fsaverage template.

    Returns
    -------
    np.ndarray
        Array of vertex indices for the projected medial wall
    """
    if template_file is None:
        template_file = fsaverage_cut_template

    template_data = load_json(template_file)

    # Extract just the medial wall for this hemisphere
    mwall_key = f"{hemi}_mwall"
    if mwall_key not in template_data:
        raise ValueError(f"Key {mwall_key} not found in template file")

    mwall_vertices = np.array(template_data[mwall_key])

    # Map to subject using mri_label2label
    vertex_dict = {"mwall": mwall_vertices}
    mapped = map_cuts_to_subject(vertex_dict, subject, hemi)

    return np.array(mapped["mwall"])


def create_vertex_dict_with_native_mwall(
    subject,
    hemi,
    subject_dir,
    template_file=None,
    refine_geodesic=True,
    smooth_mwall=False,
    smooth_n_anchors=40,
):
    """
    Create a vertex dict using native medial wall but projected anatomical cuts.

    Parameters
    ----------
    subject : str
        Subject ID
    hemi : str
        Hemisphere ('lh' or 'rh')
    subject_dir : str
        Path to FreeSurfer subject directory
    template_file : str, optional
        Path to template file. If None, uses default fsaverage template.
    refine_geodesic : bool
        Whether to refine cuts with geodesic shortest paths
    smooth_mwall : bool
        Whether to apply geodesic boundary smoothing to native medial wall
    smooth_n_anchors : int
        Number of anchor points for geodesic smoothing

    Returns
    -------
    dict
        Vertex dictionary with 'mwall' from native annotation and cuts from projection
    """
    if template_file is None:
        template_file = fsaverage_cut_template

    template_data = load_json(template_file)

    # Extract anatomical cuts (everything except mwall)
    prefix = f"{hemi}_"
    vertex_dict = {}
    for key, value in template_data.items():
        if key.startswith(prefix):
            new_key = key[len(prefix) :]
            if new_key != "mwall":
                vertex_dict[new_key] = np.array(value)

    # Map cuts to subject
    vertex_dict_mapped = map_cuts_to_subject(vertex_dict, subject, hemi)

    # Get native medial wall
    native_mwall = get_native_medial_wall(subject_dir, hemi)

    # Optionally smooth the native medial wall
    if smooth_mwall:
        # Load surface for smoothing
        pts, polys = load_surface(subject, "inflated", hemi)

        # Try fiducial surface for accurate geodesics
        surf_dir = os.path.join(subject_dir, "surf")
        fiducial_path = os.path.join(surf_dir, f"{hemi}.fiducial")
        if os.path.exists(fiducial_path):
            pts_geo, _ = load_surface(subject, "fiducial", hemi)
        else:
            pts_geo, _ = load_surface(subject, "smoothwm", hemi)

        native_mwall, _ = smooth_medial_wall_geodesic(
            native_mwall, pts_geo, polys, n_anchors=smooth_n_anchors, verbose=False
        )

    # Add native medial wall to the dict
    vertex_dict_mapped["mwall"] = native_mwall

    # Ensure cuts are continuous
    vertex_dict_fixed = ensure_continuous_cuts(vertex_dict_mapped.copy(), subject, hemi)

    # Optionally refine with geodesic paths
    if refine_geodesic:
        vertex_dict_fixed = refine_cuts_with_geodesic(
            vertex_dict_fixed,
            subject,
            hemi,
            medial_wall_vertices=vertex_dict_fixed.get("mwall"),
        )

    return vertex_dict_fixed


def create_vertex_dict_with_projected_mwall(
    subject, hemi, template_file=None, refine_geodesic=True
):
    """
    Create a vertex dict using the standard projected medial wall from fsaverage.

    This mimics the standard autoflatten projection pipeline.

    Parameters
    ----------
    subject : str
        Subject ID
    hemi : str
        Hemisphere ('lh' or 'rh')
    template_file : str, optional
        Path to template file. If None, uses default fsaverage template.
    refine_geodesic : bool
        Whether to refine cuts with geodesic shortest paths

    Returns
    -------
    dict
        Vertex dictionary with projected medial wall and cuts
    """
    if template_file is None:
        template_file = fsaverage_cut_template

    template_data = load_json(template_file)

    # Extract all components for this hemisphere
    prefix = f"{hemi}_"
    vertex_dict = {}
    for key, value in template_data.items():
        if key.startswith(prefix):
            new_key = key[len(prefix) :]
            vertex_dict[new_key] = np.array(value)

    # Map to subject
    vertex_dict_mapped = map_cuts_to_subject(vertex_dict, subject, hemi)

    # Ensure cuts are continuous
    vertex_dict_fixed = ensure_continuous_cuts(vertex_dict_mapped.copy(), subject, hemi)

    # Optionally refine with geodesic paths
    if refine_geodesic:
        vertex_dict_fixed = refine_cuts_with_geodesic(
            vertex_dict_fixed,
            subject,
            hemi,
            medial_wall_vertices=vertex_dict_fixed.get("mwall"),
        )

    return vertex_dict_fixed


def run_flattening_comparison(
    subject_dir,
    hemi,
    output_dir,
    refine_geodesic=True,
    backend=None,
    verbose=True,
    smooth_mwall=False,
    smooth_n_anchors=40,
):
    """
    Run flattening with both projected and native medial wall approaches.

    Parameters
    ----------
    subject_dir : str
        Path to FreeSurfer subject directory
    hemi : str
        Hemisphere ('lh' or 'rh')
    output_dir : str
        Directory to save output files
    refine_geodesic : bool
        Whether to refine cuts with geodesic shortest paths
    backend : str, optional
        Backend to use for flattening ('pyflatten' or 'freesurfer')
    verbose : bool
        Whether to print progress messages
    smooth_mwall : bool
        Whether to apply geodesic boundary smoothing to native medial wall
    smooth_n_anchors : int
        Number of anchor points for geodesic smoothing

    Returns
    -------
    dict
        Dictionary with paths to output files and metrics
    """
    from autoflatten.cli import run_flatten_backend

    subject = Path(subject_dir).name
    os.makedirs(output_dir, exist_ok=True)

    # Load surface for patch creation
    pts, polys = load_surface(subject, "inflated", hemi)

    # Get fiducial/smoothwm for flattening
    surf_dir = os.path.join(subject_dir, "surf")
    fiducial_path = os.path.join(surf_dir, f"{hemi}.fiducial")
    smoothwm_path = os.path.join(surf_dir, f"{hemi}.smoothwm")
    base_surface = fiducial_path if os.path.exists(fiducial_path) else smoothwm_path

    results = {}

    # ============ PROJECTED MEDIAL WALL ============
    if verbose:
        print("\n" + "=" * 60)
        print("PROJECTED MEDIAL WALL (fsaverage)")
        print("=" * 60)

    proj_vertex_dict = create_vertex_dict_with_projected_mwall(
        subject, hemi, refine_geodesic=refine_geodesic
    )

    # Fill holes
    excluded_proj = set()
    for vertices in proj_vertex_dict.values():
        excluded_proj.update(int(v) for v in vertices)
    hole_vertices_proj = fill_holes_in_patch(polys, excluded_proj)
    if hole_vertices_proj:
        proj_vertex_dict["_hole_fill"] = np.array(list(hole_vertices_proj))

    # Create patch file
    proj_patch_path = os.path.join(output_dir, f"{hemi}.projected.patch.3d")
    create_patch_file(proj_patch_path, pts, polys, proj_vertex_dict)

    # Flatten
    proj_flat_path = os.path.join(output_dir, f"{hemi}.projected.flat.patch.3d")
    run_flatten_backend(
        proj_patch_path,
        base_surface,
        proj_flat_path,
        backend_name=backend,
        verbose=verbose,
    )

    results["projected"] = {
        "patch_path": proj_patch_path,
        "flat_path": proj_flat_path,
        "mwall_vertices": len(proj_vertex_dict.get("mwall", [])),
        "vertex_dict": proj_vertex_dict,
    }

    # ============ NATIVE MEDIAL WALL ============
    if verbose:
        print("\n" + "=" * 60)
        print("NATIVE MEDIAL WALL (aparc.a2009s.annot)")
        print("=" * 60)

    native_vertex_dict = create_vertex_dict_with_native_mwall(
        subject,
        hemi,
        subject_dir,
        refine_geodesic=refine_geodesic,
        smooth_mwall=smooth_mwall,
        smooth_n_anchors=smooth_n_anchors,
    )

    # Fill holes
    excluded_native = set()
    for vertices in native_vertex_dict.values():
        excluded_native.update(int(v) for v in vertices)
    hole_vertices_native = fill_holes_in_patch(polys, excluded_native)
    if hole_vertices_native:
        native_vertex_dict["_hole_fill"] = np.array(list(hole_vertices_native))

    # Create patch file
    native_patch_path = os.path.join(output_dir, f"{hemi}.native.patch.3d")
    create_patch_file(native_patch_path, pts, polys, native_vertex_dict)

    # Flatten
    native_flat_path = os.path.join(output_dir, f"{hemi}.native.flat.patch.3d")
    run_flatten_backend(
        native_patch_path,
        base_surface,
        native_flat_path,
        backend_name=backend,
        verbose=verbose,
    )

    results["native"] = {
        "patch_path": native_patch_path,
        "flat_path": native_flat_path,
        "mwall_vertices": len(native_vertex_dict.get("mwall", [])),
        "vertex_dict": native_vertex_dict,
    }

    return results


def plot_medial_view(
    subject_dir,
    hemi,
    output_dir,
    projected_mwall,
    native_mwall,
    native_mwall_smoothed=None,
    comparison_metrics=None,
    verbose=True,
):
    """
    Create medial view plots comparing projected vs native medial wall.

    This is a lightweight visualization that doesn't require running flattening.

    Parameters
    ----------
    subject_dir : str
        Path to FreeSurfer subject directory
    hemi : str
        Hemisphere ('lh' or 'rh')
    output_dir : str
        Directory to save plots
    projected_mwall : array-like
        Vertex indices from projected fsaverage medial wall
    native_mwall : array-like
        Vertex indices from native FreeSurfer annotation (raw, before smoothing)
    native_mwall_smoothed : array-like, optional
        Vertex indices from native medial wall after smoothing
    comparison_metrics : dict, optional
        Metrics from compare_medial_walls
    verbose : bool
        Whether to print progress messages
    """
    from matplotlib.colors import ListedColormap

    subject = Path(subject_dir).name

    # Load inflated surface for visualization
    pts, polys = load_surface(subject, "inflated", hemi)
    n_vertices = len(pts)

    # For medial view, we need to look at the surface from the medial side
    # FreeSurfer coordinates: x=left-right, y=posterior-anterior, z=inferior-superior
    # For lh: medial view is from +x (looking toward -x)
    # For rh: medial view is from -x (looking toward +x)
    # Use y-z plane for display

    # Determine how many panels we need
    has_smoothed = native_mwall_smoothed is not None
    n_cols = 3 if has_smoothed else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 8))

    # Custom colormap: gray (cortex), blue (mwall)
    colors_mwall = ["lightgray", "steelblue"]
    cmap_mwall = ListedColormap(colors_mwall)

    # Custom colormap for comparison: gray, blue (both), orange (only proj), green (only native)
    colors_comp = ["lightgray", "steelblue", "darkorange", "forestgreen"]
    cmap_comp = ListedColormap(colors_comp)

    # Filter triangles to show only medial-facing ones
    # Compute face centroids in x direction
    face_centroids_x = pts[polys, 0].mean(axis=1)

    # For lh medial view: show faces with positive x (medial side)
    # For rh medial view: show faces with negative x (medial side)
    if hemi == "lh":
        medial_faces = polys[face_centroids_x > 0]
        # For display, flip y so anterior is up
        display_coords = pts[:, [1, 2]]  # y, z
    else:
        medial_faces = polys[face_centroids_x < 0]
        # For rh, flip y axis for proper orientation
        display_coords = pts[:, [1, 2]]
        display_coords = display_coords * np.array([-1, 1])  # Flip y for mirror view

    # Panel 1: Projected medial wall
    ax = axes[0]
    overlay = np.zeros(n_vertices)
    for v in projected_mwall:
        overlay[int(v)] = 1

    triang = plt.matplotlib.tri.Triangulation(
        display_coords[:, 0], display_coords[:, 1], medial_faces
    )
    ax.tripcolor(triang, overlay, cmap=cmap_mwall, vmin=0, vmax=1)
    ax.set_title(f"Projected ({hemi})\n{len(projected_mwall):,} vertices", fontsize=12)
    ax.set_aspect("equal")
    ax.axis("off")

    # Panel 2: Native medial wall (raw or smoothed)
    ax = axes[1]
    native_to_show = native_mwall_smoothed if has_smoothed else native_mwall
    native_label = "Native (smoothed)" if has_smoothed else "Native"

    overlay = np.zeros(n_vertices)
    for v in native_to_show:
        overlay[int(v)] = 1

    triang = plt.matplotlib.tri.Triangulation(
        display_coords[:, 0], display_coords[:, 1], medial_faces
    )
    ax.tripcolor(triang, overlay, cmap=cmap_mwall, vmin=0, vmax=1)
    ax.set_title(
        f"{native_label} ({hemi})\n{len(native_to_show):,} vertices", fontsize=12
    )
    ax.set_aspect("equal")
    ax.axis("off")

    # Panel 3 (if smoothed): Comparison view
    if has_smoothed:
        ax = axes[2]

        proj_set = set(int(v) for v in projected_mwall)
        native_set = set(int(v) for v in native_to_show)

        # 0=cortex, 1=both, 2=only projected, 3=only native
        overlay = np.zeros(n_vertices)
        for v in proj_set & native_set:
            overlay[v] = 1
        for v in proj_set - native_set:
            overlay[v] = 2
        for v in native_set - proj_set:
            overlay[v] = 3

        triang = plt.matplotlib.tri.Triangulation(
            display_coords[:, 0], display_coords[:, 1], medial_faces
        )
        ax.tripcolor(triang, overlay, cmap=cmap_comp, vmin=0, vmax=3)
        ax.set_title(
            f"Comparison ({hemi})\nBlue: Both | Orange: Only proj | Green: Only native",
            fontsize=10,
        )
        ax.set_aspect("equal")
        ax.axis("off")

    # Add metrics text below the figure if available
    if comparison_metrics is not None:
        metrics_text = (
            f"Projected: {comparison_metrics['n_projected']:,} | "
            f"Native: {comparison_metrics['n_native']:,} | "
            f"Dice: {comparison_metrics['dice']:.3f} | "
            f"Jaccard: {comparison_metrics['jaccard']:.3f}"
        )
        fig.text(
            0.5, 0.02, metrics_text, ha="center", fontsize=10, fontfamily="monospace"
        )

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    output_path = os.path.join(output_dir, f"{hemi}_medial_wall_comparison.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    if verbose:
        print(f"\nMedial view plot saved to: {output_path}")
    plt.close()

    return output_path


def plot_comparison(
    results, subject_dir, hemi, output_dir, comparison_metrics, verbose=True
):
    """
    Create comparison plots for projected vs native medial wall results.

    Parameters
    ----------
    results : dict
        Results from run_flattening_comparison
    subject_dir : str
        Path to FreeSurfer subject directory
    hemi : str
        Hemisphere ('lh' or 'rh')
    output_dir : str
        Directory to save plots
    comparison_metrics : dict
        Metrics from compare_medial_walls
    verbose : bool
        Whether to print progress messages
    """
    from autoflatten.freesurfer import read_surface, extract_patch_faces

    subject = Path(subject_dir).name
    surf_dir = os.path.join(subject_dir, "surf")

    # Get base surface for plotting
    fiducial_path = os.path.join(surf_dir, f"{hemi}.fiducial")
    smoothwm_path = os.path.join(surf_dir, f"{hemi}.smoothwm")
    base_surface = fiducial_path if os.path.exists(fiducial_path) else smoothwm_path

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))

    # Plot projected flatmap
    ax = axes[0, 0]
    try:
        # Read the data for custom plotting
        # read_patch returns (vertices, original_indices, is_border)
        vertices, orig_idx, is_border = read_patch(results["projected"]["flat_path"])
        vertices_2d = vertices[:, :2]

        _, orig_faces = read_surface(base_surface)
        faces = extract_patch_faces(orig_faces, orig_idx)

        triang = plt.matplotlib.tri.Triangulation(
            vertices_2d[:, 0], vertices_2d[:, 1], faces
        )
        ax.triplot(triang, "k-", lw=0.1, alpha=0.3)
        areas = compute_triangle_areas(vertices_2d, faces)
        flipped = np.sum(areas < 0)
        ax.set_title(
            f"Projected ({hemi})\n"
            f"Patch vertices: {len(orig_idx)}\n"
            f"Flipped triangles: {flipped}"
        )
        ax.set_aspect("equal")
    except Exception as e:
        ax.text(
            0.5, 0.5, f"Error: {e}", ha="center", va="center", transform=ax.transAxes
        )
        ax.set_title(f"Projected ({hemi}) - Error")

    # Plot native flatmap
    ax = axes[0, 1]
    try:
        # read_patch returns (vertices, original_indices, is_border)
        vertices, orig_idx, is_border = read_patch(results["native"]["flat_path"])
        vertices_2d = vertices[:, :2]

        _, orig_faces = read_surface(base_surface)
        faces = extract_patch_faces(orig_faces, orig_idx)

        triang = plt.matplotlib.tri.Triangulation(
            vertices_2d[:, 0], vertices_2d[:, 1], faces
        )
        ax.triplot(triang, "k-", lw=0.1, alpha=0.3)
        areas = compute_triangle_areas(vertices_2d, faces)
        flipped = np.sum(areas < 0)
        ax.set_title(
            f"Native ({hemi})\n"
            f"Patch vertices: {len(orig_idx)}\n"
            f"Flipped triangles: {flipped}"
        )
        ax.set_aspect("equal")
    except Exception as e:
        ax.text(
            0.5, 0.5, f"Error: {e}", ha="center", va="center", transform=ax.transAxes
        )
        ax.set_title(f"Native ({hemi}) - Error")

    # Plot medial wall comparison on inflated surface (medial view)
    ax = axes[1, 0]
    pts, polys = load_surface(subject, "inflated", hemi)
    n_vertices = len(pts)

    # Filter to medial-facing triangles
    face_centroids_x = pts[polys, 0].mean(axis=1)
    if hemi == "lh":
        medial_faces = polys[face_centroids_x > 0]
        display_coords = pts[:, [1, 2]]
    else:
        medial_faces = polys[face_centroids_x < 0]
        display_coords = pts[:, [1, 2]] * np.array([-1, 1])

    # Color code: 0=cortex, 1=both mwalls, 2=only projected, 3=only native
    overlay = np.zeros(n_vertices)

    proj_mwall = set(results["projected"]["vertex_dict"].get("mwall", []))
    native_mwall = set(results["native"]["vertex_dict"].get("mwall", []))

    for v in proj_mwall & native_mwall:
        overlay[int(v)] = 1  # Both
    for v in proj_mwall - native_mwall:
        overlay[int(v)] = 2  # Only projected
    for v in native_mwall - proj_mwall:
        overlay[int(v)] = 3  # Only native

    # Plot inflated surface with overlay (medial view)
    triang = plt.matplotlib.tri.Triangulation(
        display_coords[:, 0], display_coords[:, 1], medial_faces
    )
    ax.tripcolor(triang, overlay, cmap="RdYlBu", vmin=0, vmax=3)
    ax.set_title(
        f"Medial Wall Comparison ({hemi})\n"
        f"Blue: Both | Yellow: Only projected | Red: Only native"
    )
    ax.set_aspect("equal")
    ax.axis("off")

    # Summary metrics
    ax = axes[1, 1]
    ax.axis("off")
    metrics_text = f"""
    COMPARISON METRICS ({hemi})
    {"=" * 40}

    Medial Wall Vertices:
      Projected: {comparison_metrics["n_projected"]:,}
      Native:    {comparison_metrics["n_native"]:,}
      Difference: {comparison_metrics["n_native"] - comparison_metrics["n_projected"]:+,}

    Overlap:
      Intersection: {comparison_metrics["n_intersection"]:,}
      Only projected: {comparison_metrics["n_only_projected"]:,}
      Only native: {comparison_metrics["n_only_native"]:,}

    Similarity:
      Dice coefficient: {comparison_metrics["dice"]:.4f}
      Jaccard index:    {comparison_metrics["jaccard"]:.4f}

    Patch Vertices:
      Projected: {results["projected"]["mwall_vertices"]:,} mwall vertices
      Native:    {results["native"]["mwall_vertices"]:,} mwall vertices
    """
    ax.text(
        0.1,
        0.9,
        metrics_text,
        transform=ax.transAxes,
        fontfamily="monospace",
        fontsize=10,
        verticalalignment="top",
    )

    plt.tight_layout()
    output_path = os.path.join(output_dir, f"{hemi}_comparison.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    if verbose:
        print(f"\nComparison plot saved to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Compare native vs projected medial wall for cortical flattening",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot medial wall comparison only (fast, no flattening)
  python scripts/test_native_medial_wall.py /path/to/subject --hemi lh --plot-only

  # Plot with medial wall smoothing
  python scripts/test_native_medial_wall.py /path/to/subject --hemi lh --plot-only --smooth-mwall

  # Full comparison with flattening
  python scripts/test_native_medial_wall.py /path/to/subject --hemi lh --output-dir ./results

  # Compare metrics only (no plots, no flattening)
  python scripts/test_native_medial_wall.py /path/to/subject --hemi lh --compare-only
        """,
    )
    parser.add_argument("subject_dir", help="Path to FreeSurfer subject directory")
    parser.add_argument(
        "--hemi",
        choices=["lh", "rh", "both"],
        default="lh",
        help="Hemisphere to process (default: lh)",
    )
    parser.add_argument(
        "--output-dir",
        default="./medial_wall_comparison",
        help="Output directory for results (default: ./medial_wall_comparison)",
    )
    parser.add_argument(
        "--backend",
        choices=["pyflatten", "freesurfer"],
        default=None,
        help="Flattening backend (default: auto-detect)",
    )
    parser.add_argument(
        "--no-refine-geodesic",
        action="store_true",
        help="Disable geodesic refinement of cuts",
    )
    parser.add_argument(
        "--compare-only",
        action="store_true",
        help="Only compare medial walls without running flattening",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Only plot medial wall comparison (medial view) without running flattening",
    )
    parser.add_argument(
        "--annot-file",
        default="aparc.a2009s.annot",
        help="Annotation file for native medial wall (default: aparc.a2009s.annot)",
    )
    parser.add_argument(
        "--smooth-mwall",
        action="store_true",
        help="Enable geodesic boundary smoothing for native medial wall",
    )
    parser.add_argument(
        "--smooth-n-anchors",
        type=int,
        default=40,
        help="Number of anchor points for geodesic smoothing (default: 40)",
    )

    args = parser.parse_args()

    # Validate subject directory
    subject_dir = os.path.abspath(args.subject_dir)
    if not os.path.isdir(subject_dir):
        print(f"Error: Subject directory not found: {subject_dir}")
        sys.exit(1)

    # Check FreeSurfer setup
    if not setup_freesurfer():
        print("Error: FreeSurfer environment not properly configured")
        sys.exit(1)

    subject = Path(subject_dir).name
    os.makedirs(args.output_dir, exist_ok=True)

    hemispheres = ["lh", "rh"] if args.hemi == "both" else [args.hemi]

    for hemi in hemispheres:
        print(f"\n{'#' * 60}")
        print(f"# Processing {hemi} hemisphere for subject {subject}")
        print(f"{'#' * 60}")

        # Load surface
        pts, polys = load_surface(subject, "inflated", hemi)

        # Get native medial wall
        print(f"\nLoading native medial wall from {args.annot_file}...")
        try:
            native_mwall = get_native_medial_wall(
                subject_dir, hemi, annot_file=args.annot_file
            )
            print(f"  Native medial wall: {len(native_mwall)} vertices")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            continue

        # Check topology
        print("\nChecking native medial wall topology...")
        is_single, n_components, component_sizes = check_medial_wall_topology(
            native_mwall, polys
        )
        if is_single:
            print(f"  ✓ Single connected component ({len(native_mwall)} vertices)")
        else:
            print(f"  ✗ Multiple components: {n_components}")
            print(f"  Component sizes: {component_sizes[:5]}...")  # Show top 5

        # Keep raw native medial wall for comparison
        native_mwall_raw = native_mwall.copy()
        native_mwall_smoothed = None

        # Optionally smooth the native medial wall
        smoothing_stats = None
        if args.smooth_mwall:
            # Load fiducial surface for accurate geodesic distances
            surf_dir = os.path.join(subject_dir, "surf")
            fiducial_path = os.path.join(surf_dir, f"{hemi}.fiducial")
            smoothwm_path = os.path.join(surf_dir, f"{hemi}.smoothwm")
            if os.path.exists(fiducial_path):
                pts_geo, _ = load_surface(subject, "fiducial", hemi)
            else:
                pts_geo, _ = load_surface(subject, "smoothwm", hemi)

            native_mwall_smoothed, smoothing_stats = smooth_medial_wall_geodesic(
                native_mwall,
                pts_geo,
                polys,
                n_anchors=args.smooth_n_anchors,
                verbose=True,
            )
            print(f"\n  Smoothed medial wall: {len(native_mwall_smoothed)} vertices")
            # Update native_mwall to point to smoothed version for downstream use
            native_mwall = native_mwall_smoothed

        # Get projected medial wall
        print("\nGetting projected medial wall from fsaverage...")
        projected_mwall = get_projected_medial_wall(subject, hemi)
        print(f"  Projected medial wall: {len(projected_mwall)} vertices")

        # Compare medial walls
        print("\nComparing medial walls...")
        comparison = compare_medial_walls(projected_mwall, native_mwall)
        print(f"  Intersection: {comparison['n_intersection']} vertices")
        print(f"  Only in projected: {comparison['n_only_projected']} vertices")
        print(f"  Only in native: {comparison['n_only_native']} vertices")
        print(f"  Dice coefficient: {comparison['dice']:.4f}")
        print(f"  Jaccard index: {comparison['jaccard']:.4f}")

        # Handle plot-only mode
        if args.plot_only:
            print("\n--plot-only flag set, generating medial view plot...")
            plot_medial_view(
                subject_dir,
                hemi,
                args.output_dir,
                projected_mwall,
                native_mwall_raw,
                native_mwall_smoothed=native_mwall_smoothed,
                comparison_metrics=comparison,
                verbose=True,
            )
            continue

        if args.compare_only:
            print("\n--compare-only flag set, skipping flattening")
            continue

        # Run full comparison with flattening
        print("\nRunning flattening comparison...")
        results = run_flattening_comparison(
            subject_dir,
            hemi,
            args.output_dir,
            refine_geodesic=not args.no_refine_geodesic,
            backend=args.backend,
            verbose=True,
            smooth_mwall=args.smooth_mwall,
            smooth_n_anchors=args.smooth_n_anchors,
        )

        # Create comparison plots
        print("\nGenerating comparison plots...")
        plot_comparison(results, subject_dir, hemi, args.output_dir, comparison)

        # Print summary
        print(f"\n{'=' * 60}")
        print(f"SUMMARY ({hemi})")
        print(f"{'=' * 60}")
        print(f"  Projected patch: {results['projected']['patch_path']}")
        print(f"  Native patch: {results['native']['patch_path']}")
        print(f"  Projected flat: {results['projected']['flat_path']}")
        print(f"  Native flat: {results['native']['flat_path']}")

    print(f"\n\nAll results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
