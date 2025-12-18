"""Visualization utilities for flattened cortical surfaces.

This module provides matplotlib-based visualization for flat patches,
showing the mesh with flipped triangles highlighted and metric distortion.
"""

import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np

from autoflatten.freesurfer import read_patch, read_surface, extract_patch_faces
from autoflatten.backends import find_base_surface
from autoflatten.flatten.distance import (
    compute_kring_geodesic_distances,
    compute_kring_geodesic_distances_angular,
)


def compute_triangle_areas(vertices_2d, faces):
    """Compute signed area of each triangle.

    Parameters
    ----------
    vertices_2d : ndarray of shape (N, 2)
        2D vertex coordinates
    faces : ndarray of shape (F, 3)
        Triangle indices

    Returns
    -------
    areas : ndarray of shape (F,)
        Signed area of each triangle (negative = flipped)
    """
    v0 = vertices_2d[faces[:, 0]]
    v1 = vertices_2d[faces[:, 1]]
    v2 = vertices_2d[faces[:, 2]]

    # Signed area using cross product
    areas = 0.5 * (
        (v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1])
        - (v2[:, 0] - v0[:, 0]) * (v1[:, 1] - v0[:, 1])
    )
    return areas


def compute_kring_distortion(
    xy,
    base_vertices,
    base_faces,
    orig_indices,
    k=2,
    n_samples_per_ring=None,
    verbose=True,
):
    """Compute per-vertex metric distortion using k-ring geodesic distances.

    Uses the same approach as autoflatten optimization:
    - Computes geodesic distances to k-ring neighbors on 3D surface
    - Compares with Euclidean distances on 2D flatmap
    - Returns percentage error per vertex: 100 * mean(|d_2D - d_3D|) / mean(d_3D)

    This should match plots in Fischl et al., 1999, and the computation
    implemented in FreeSurfer.

    Parameters
    ----------
    xy : ndarray of shape (N, 2)
        2D flat patch vertex coordinates
    base_vertices : ndarray of shape (M, 3)
        Full base surface 3D vertex coordinates
    base_faces : ndarray of shape (F, 3)
        Full base surface face indices
    orig_indices : ndarray of shape (N,)
        Mapping from patch vertex indices to full surface indices
    k : int
        Number of rings to include (default: 2, fast and accurate)
    n_samples_per_ring : int or None
        Angular samples per ring. If None, use all neighbors without angular
        sampling (default: None, faster). Use 12 for pyflatten-style sampling.
    verbose : bool
        Print progress messages

    Returns
    -------
    vertex_distortion : ndarray of shape (N,)
        Percentage distortion at each vertex
    mean_distortion : float
        Overall mean percentage distortion (same formula as autoflatten)
    """
    n_patch_vertices = len(xy)

    # Build mapping from full surface indices to patch indices
    # orig_indices[patch_idx] = full_idx, so we need the inverse
    full_to_patch = np.zeros(orig_indices.max() + 1, dtype=np.int64)
    full_to_patch[orig_indices] = np.arange(len(orig_indices))

    # Extract the subgraph of the base surface that corresponds to the patch
    # K-ring distances are computed on the patch subgraph only
    patch_vertices_3d = base_vertices[orig_indices]

    # Extract faces that are entirely within the patch
    patch_face_mask = np.all(np.isin(base_faces, orig_indices), axis=1)
    patch_faces_full = base_faces[patch_face_mask]

    # Remap face indices from full surface to patch indices (vectorized)
    patch_faces = full_to_patch[patch_faces_full]

    # Compute k-ring geodesic distances on the patch subgraph
    if n_samples_per_ring is None:
        # Use all neighbors (no angular sampling); avoids angular-sampling overhead
        if verbose:
            print(f"Computing {k}-ring geodesic distances (all neighbors)...")
        k_rings, target_distances = compute_kring_geodesic_distances(
            patch_vertices_3d,
            patch_faces,
            k=k,
            use_numba=True,
            tqdm_position=0,
        )
    else:
        # Use angular sampling
        if verbose:
            print(
                f"Computing {k}-ring geodesic distances ({n_samples_per_ring} samples/ring)..."
            )
        k_rings, target_distances = compute_kring_geodesic_distances_angular(
            patch_vertices_3d,
            patch_faces,
            k=k,
            n_samples_per_ring=n_samples_per_ring,
            use_numba=True,
            tqdm_position=0,
        )

    if verbose:
        print("Computing per-vertex distortion...")

    # Compute per-vertex distortion
    vertex_distortion = np.zeros(n_patch_vertices)
    total_abs_error = 0.0
    total_target = 0.0

    for v in range(n_patch_vertices):
        neighbors = k_rings[v]
        targets = target_distances[v]

        if len(neighbors) == 0:
            vertex_distortion[v] = 0.0
            continue

        # Compute 2D Euclidean distances to neighbors
        d_2d = np.linalg.norm(xy[neighbors] - xy[v], axis=1)

        # Compute per-vertex distortion: 100 * mean(|d_2D - d_3D|) / mean(d_3D)
        mean_target = np.mean(targets)
        if mean_target > 0.0:
            abs_errors = np.abs(d_2d - targets)
            vertex_distortion[v] = 100.0 * np.mean(abs_errors) / mean_target

            # Accumulate for global mean
            total_abs_error += np.sum(abs_errors)
            total_target += np.sum(targets)
        else:
            # If all target distances are zero, define local distortion as zero
            vertex_distortion[v] = 0.0

    # Global mean distortion (same formula as autoflatten)
    mean_distortion = (
        100.0 * total_abs_error / total_target if total_target > 0 else 0.0
    )

    return vertex_distortion, mean_distortion


def parse_log_file(log_path):
    """Parse optimization results from log file.

    Parameters
    ----------
    log_path : str
        Path to the log file

    Returns
    -------
    dict
        Dictionary with keys: 'distance_error', 'flipped', 'subject', 'hemisphere'.
        Missing values are omitted from the dict.
    """
    result = {}
    try:
        with open(log_path, "r") as f:
            content = f.read()

        # Look for final result section (pyflatten format)
        final_match = re.search(
            r"FINAL RESULT.*?"
            r"Flipped triangles:.*?->\s+(\d+)\s*\n"
            r"Mean % distance error:\s+([\d.]+)%",
            content,
            re.DOTALL,
        )
        if final_match:
            result["flipped"] = int(final_match.group(1))
            result["distance_error"] = float(final_match.group(2))

        # Extract subject and hemisphere from input patch path
        input_match = re.search(r"Input patch:\s*(.+)", content)
        if input_match:
            input_path = Path(input_match.group(1).strip())
            result["subject"] = input_path.parent.name
            filename = input_path.name
            if filename.startswith("lh."):
                result["hemisphere"] = "lh"
            elif filename.startswith("rh."):
                result["hemisphere"] = "rh"

    except FileNotFoundError:
        # Log file doesn't exist - this is expected and normal
        pass
    except OSError as e:
        # File exists but couldn't be read (permissions, I/O error, etc.)
        import warnings

        warnings.warn(
            f"Could not read log file {log_path}: {e}",
            UserWarning,
            stacklevel=2,
        )

    return result


def plot_flatmap(
    flat_patch_path,
    base_surface_path=None,
    output_path=None,
    title=None,
    figsize=(14, 5),
    show_flipped=True,
    show_boundary=True,
    distortion_cmap="viridis",
    distance_method="fast",
    dpi=150,
):
    """
    Plot a flattened cortical surface using matplotlib.

    Creates a three-panel figure showing:
    - Left: Mesh with flipped triangles highlighted in red
    - Center: Per-vertex metric distortion (percentage error between 2D and 3D distances)
    - Right: Histogram of distortion distribution

    Parameters
    ----------
    flat_patch_path : str
        Path to the flat patch file (e.g., lh.flat.patch.3d)
    base_surface_path : str, optional
        Path to the base surface file (e.g., lh.fiducial).
        If None, will auto-detect from flat_patch_path location.
    output_path : str, optional
        If provided, save figure to this path. Otherwise returns figure.
    title : str, optional
        Custom title (e.g., "S01 lh")
    figsize : tuple
        Figure size in inches
    show_flipped : bool
        Highlight flipped triangles in red
    show_boundary : bool
        Show boundary vertices as dots
    distortion_cmap : str
        Colormap for the distortion visualization (default: viridis)
    distance_method : str
        Method for computing distortion: "fast" (2-ring, all neighbors, default)
        or "pyflatten" (7-ring, 12 angular samples per ring, more accurate but slower)
    dpi : int
        Resolution for saved figure

    Returns
    -------
    str or matplotlib.figure.Figure
        If output_path is provided, returns the output path.
        Otherwise returns the matplotlib figure.
    """
    # Auto-detect base surface if not provided
    if base_surface_path is None:
        # For flat patches, we need the original patch to find the surface
        # Try to find it by replacing .flat.patch.3d with .patch.3d
        orig_patch_path = flat_patch_path.replace(".flat.patch.3d", ".patch.3d")
        base_surface_path = find_base_surface(orig_patch_path)
        if base_surface_path is None:
            # Try the flat patch path directly
            base_surface_path = find_base_surface(flat_patch_path)
        if base_surface_path is None:
            raise ValueError(
                f"Could not auto-detect base surface for {flat_patch_path}. "
                "Please provide base_surface_path."
            )

    # Read flat patch
    flat_vertices, orig_indices, is_border = read_patch(flat_patch_path)

    # Read base surface to get vertices and faces
    base_vertices, base_faces = read_surface(base_surface_path)

    # Extract patch faces
    faces = extract_patch_faces(base_faces, orig_indices)

    # Get 2D coordinates (x, y from flat patch)
    xy = flat_vertices[:, :2]

    # Compute triangle areas
    areas = compute_triangle_areas(xy, faces)
    n_flipped = np.sum(areas < 0)

    # Compute per-vertex distortion
    if distance_method == "pyflatten":
        # Accurate 7-ring geodesic distances (same method as pyflatten, slower)
        k, n_samples = 7, 12
    elif distance_method == "fast":
        # Fast 2-ring distances (default) - all neighbors, no angular sampling
        k, n_samples = 2, None
    else:
        raise ValueError(
            f"Invalid distance_method: {distance_method!r}. "
            "Must be 'fast' or 'pyflatten'."
        )

    vertex_dist, mean_dist = compute_kring_distortion(
        xy,
        base_vertices,
        base_faces,
        orig_indices,
        k=k,
        n_samples_per_ring=n_samples,
        verbose=True,
    )

    # Try to parse log file for optimization results
    log_path = flat_patch_path + ".log"
    log_results = parse_log_file(log_path)

    # Build title
    if title:
        main_title = title
    elif log_results.get("subject") and log_results.get("hemisphere"):
        main_title = f"{log_results['subject']} {log_results['hemisphere']}"
    else:
        main_title = Path(flat_patch_path).name

    # Add results to subtitle - use log file error if available, otherwise computed
    log_error = log_results.get("distance_error")
    if log_error is not None:
        error_str = f"{log_error:.2f}% error"
    else:
        error_str = f"{mean_dist:.2f}% error"
    subtitle_parts = [
        f"{len(flat_vertices):,} vertices, {len(faces):,} faces",
        f"{error_str}, {n_flipped} flipped",
    ]
    subtitle = ", ".join(subtitle_parts)

    # Create figure with 3 panels
    fig, axes = plt.subplots(1, 3, figsize=figsize, constrained_layout=True)

    # Create triangulation
    triang = tri.Triangulation(xy[:, 0], xy[:, 1], faces)

    # Left plot: Mesh colored by area, flipped in red
    ax = axes[0]

    # Color triangles: normal areas in gray, flipped in red
    face_colors = np.ones((len(faces), 4))  # RGBA
    face_colors[:, :3] = 0.8  # Light gray for normal triangles
    face_colors[:, 3] = 1.0  # Full opacity

    if show_flipped and n_flipped > 0:
        flipped_mask = areas < 0
        face_colors[flipped_mask] = [1.0, 0.0, 0.0, 1.0]  # Red for flipped

    # Use tripcolor with face colors
    ax.tripcolor(triang, facecolors=face_colors[:, 0], cmap="gray", vmin=0, vmax=1)

    # Draw flipped triangles explicitly on top (more visible)
    if show_flipped and n_flipped > 0:
        flipped_mask = areas < 0
        flipped_faces = faces[flipped_mask]
        for face in flipped_faces:
            triangle = plt.Polygon(
                xy[face],
                facecolor="red",
                edgecolor="darkred",
                alpha=0.9,
                linewidth=1.0,
                zorder=10,
            )
            ax.add_patch(triangle)

        # Also mark centroids for visibility when zoomed out
        flipped_centroids = np.mean(xy[flipped_faces], axis=1)
        ax.scatter(
            flipped_centroids[:, 0],
            flipped_centroids[:, 1],
            c="yellow",
            s=20,
            marker="o",
            edgecolors="red",
            linewidths=1,
            zorder=11,
            label=f"Flipped ({n_flipped})",
        )

    if show_boundary and np.sum(is_border) > 0:
        boundary_xy = xy[is_border]
        ax.scatter(
            boundary_xy[:, 0],
            boundary_xy[:, 1],
            c="blue",
            s=1,
            alpha=0.5,
            label=f"Boundary ({np.sum(is_border)} vertices)",
        )

    ax.set_aspect("equal")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_title("Flatmap")
    if (show_flipped and n_flipped > 0) or (show_boundary and np.sum(is_border) > 0):
        ax.legend(loc="upper right", fontsize=8)

    # Center plot: Per-vertex metric distortion (percentage)
    ax = axes[1]

    # Fixed color limits: 0-100%
    vmin = 0
    vmax = 100

    # Use tripcolor with vertex values for smooth interpolation
    tpc = ax.tripcolor(
        triang,
        vertex_dist,
        shading="gouraud",
        cmap=distortion_cmap,
        vmin=vmin,
        vmax=vmax,
    )

    # Create colorbar
    fig.colorbar(tpc, ax=ax, label="Distortion (%)", shrink=0.8)

    ax.set_aspect("equal")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_title(f"Metric Distortion ({k}-ring)")

    # Right plot: Histogram of distortion distribution
    ax = axes[2]

    # Compute histogram (clip to 0-100 range for display)
    n_bins = 50
    vertex_dist_clipped = np.clip(vertex_dist, vmin, vmax)
    hist, bin_edges = np.histogram(vertex_dist_clipped, bins=n_bins, range=(vmin, vmax))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Color bars by distortion value using same colormap
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap_obj = plt.colormaps[distortion_cmap]
    colors = cmap_obj(norm(bin_centers))

    ax.bar(
        bin_centers, hist, width=np.diff(bin_edges)[0], color=colors, edgecolor="none"
    )

    # Add mean and median lines (use weighted mean_dist from compute_kring_distortion)
    median_dist = np.median(vertex_dist)
    ax.axvline(
        x=mean_dist,
        color="black",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {mean_dist:.1f}%",
    )
    ax.axvline(
        x=median_dist,
        color="gray",
        linestyle=":",
        linewidth=1.5,
        label=f"Median: {median_dist:.1f}%",
    )

    ax.set_xlabel("Distortion (%)")
    ax.set_ylabel("Vertex Count")
    ax.set_title(f"Distortion Distribution ({k}-ring)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(vmin, vmax)

    # Add main title
    fig.suptitle(f"{main_title}\n{subtitle}", fontsize=12)

    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved figure to {output_path}")
        plt.close()
        return output_path
    else:
        return fig


def plot_patch(
    patch_file,
    subject,
    subject_dir,
    output_dir=None,
    surface="lh.inflated",
    trim=True,
):
    """
    Generate a PNG image of a FreeSurfer patch file.

    This is a compatibility wrapper that uses matplotlib-based plotting
    instead of the original FreeView-based approach.

    Parameters
    ----------
    patch_file : str
        Path to the input patch file (e.g., *.flat.patch.3d).
    subject : str
        FreeSurfer subject identifier (used for title).
    subject_dir : str
        Path to the specific subject's surf directory within SUBJECTS_DIR.
    output_dir : str or None, optional
        Directory where the output PNG image will be saved.
        If None, the image is saved in the same directory as `patch_file`.
    surface : str, optional
        The surface file to use for face information (default is 'lh.inflated').
        This should be relative to `subject_dir`.
    trim : bool, optional
        Ignored (kept for backward compatibility).

    Returns
    -------
    str
        Path to the generated PNG image.

    Raises
    ------
    FileNotFoundError
        If the input patch file does not exist.
    """
    if not os.path.exists(patch_file):
        raise FileNotFoundError(f"Patch file not found: {patch_file}")

    # Default output directory to patch file's directory if None
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(patch_file))

    os.makedirs(output_dir, exist_ok=True)
    final_img_name = os.path.join(
        output_dir, os.path.basename(patch_file).replace(".3d", ".png")
    )

    if os.path.exists(final_img_name):
        print(
            f"Image already exists: {final_img_name}. "
            "Deleting it if you want to re-run."
        )
        return final_img_name

    # Determine hemisphere from filename
    basename = os.path.basename(patch_file)
    if basename.startswith("lh."):
        hemi = "lh"
    elif basename.startswith("rh."):
        hemi = "rh"
    else:
        hemi = surface.split(".")[0] if "." in surface else "lh"

    # Find base surface
    base_surface_path = os.path.join(subject_dir, f"{hemi}.fiducial")
    if not os.path.exists(base_surface_path):
        base_surface_path = os.path.join(subject_dir, f"{hemi}.white")
    if not os.path.exists(base_surface_path):
        base_surface_path = os.path.join(subject_dir, surface)

    # Generate title
    title = f"{subject} {hemi}"

    # Use the new matplotlib-based plotting
    plot_flatmap(
        flat_patch_path=patch_file,
        base_surface_path=base_surface_path,
        output_path=final_img_name,
        title=title,
    )

    return final_img_name
