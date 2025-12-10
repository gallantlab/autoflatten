"""Visualization utilities for flattened cortical surfaces.

This module provides matplotlib-based visualization for flat patches,
showing the mesh with flipped triangles highlighted and triangle areas.
"""

import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as tri
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

from autoflatten.freesurfer import read_patch, read_surface, extract_patch_faces
from autoflatten.backends import find_base_surface


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

    except (FileNotFoundError, IOError):
        # Log file doesn't exist or can't be read - return empty result
        pass

    return result


def plot_flatmap(
    flat_patch_path,
    base_surface_path=None,
    output_path=None,
    title=None,
    figsize=(12, 6),
    show_flipped=True,
    show_boundary=True,
    cmap="viridis",
    dpi=150,
):
    """
    Plot a flattened cortical surface using matplotlib.

    Creates a two-panel figure showing:
    - Left: Mesh with flipped triangles highlighted in red
    - Right: Triangle areas on log scale

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
    cmap : str
        Colormap for the mesh area visualization
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

    # Read base surface to get faces
    _, base_faces = read_surface(base_surface_path)

    # Extract patch faces
    faces = extract_patch_faces(base_faces, orig_indices)

    # Get 2D coordinates (x, y from flat patch)
    xy = flat_vertices[:, :2]

    # Compute triangle areas
    areas = compute_triangle_areas(xy, faces)
    n_flipped = np.sum(areas < 0)

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

    # Add optimization results to subtitle
    subtitle_parts = [f"{len(flat_vertices):,} vertices, {len(faces):,} faces"]
    if log_results.get("distance_error") is not None:
        subtitle_parts.append(
            f"{log_results['distance_error']}% error, "
            f"{log_results.get('flipped', n_flipped)} flipped"
        )
    else:
        subtitle_parts.append(f"{n_flipped:,} flipped")

    subtitle = ", ".join(subtitle_parts)

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=figsize)

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

    # Right plot: Triangle areas on log scale
    ax = axes[1]

    # Color by triangle area (log scale for visibility)
    log_areas = np.log10(np.abs(areas) + 1e-10)

    # Use tripcolor for area visualization
    tpc = ax.tripcolor(triang, log_areas, shading="flat", cmap=cmap)

    # Create colorbar with matching height using axes_grid1
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    fig.colorbar(tpc, cax=cax, label="log10(area)")

    # Mark flipped triangles
    if n_flipped > 0:
        flipped_centroids = np.mean(xy[faces[areas < 0]], axis=1)
        ax.scatter(
            flipped_centroids[:, 0],
            flipped_centroids[:, 1],
            c="red",
            s=20,
            marker="x",
            linewidths=1.5,
            label=f"Flipped ({n_flipped})",
            zorder=10,
        )
        ax.legend(loc="upper right", fontsize=8)

    ax.set_aspect("equal")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_title("Triangle Areas (log scale)")

    # Add main title
    fig.suptitle(f"{main_title}\n{subtitle}", fontsize=12)
    plt.tight_layout()

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
