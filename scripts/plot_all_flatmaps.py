#!/usr/bin/env python3
"""Plot all flatmaps from autoflatten results in a single figure.

This script creates a multi-panel figure showing all participants' flatmaps,
with left hemisphere rotated 90° CCW and right hemisphere rotated 90° CW.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np


from autoflatten.freesurfer import read_patch, read_surface, extract_patch_faces
from autoflatten.viz import compute_triangle_areas, parse_log_file


def rotate_coords(xy, angle_deg):
    """Rotate 2D coordinates by given angle in degrees.

    Parameters
    ----------
    xy : ndarray of shape (N, 2)
        2D vertex coordinates
    angle_deg : float
        Rotation angle in degrees (positive = counter-clockwise)

    Returns
    -------
    rotated : ndarray of shape (N, 2)
        Rotated coordinates
    """
    angle_rad = np.deg2rad(angle_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    return xy @ rotation_matrix.T


def find_flat_patches(subject_dir):
    """Find autoflatten flat patches for a subject.

    Parameters
    ----------
    subject_dir : Path
        Path to subject directory

    Returns
    -------
    tuple or None
        (lh_path, rh_path) if both hemispheres found, None otherwise
    """
    subject_dir = Path(subject_dir)

    # Check FreeSurfer structure: subject/surf/
    surf_dir = subject_dir / "surf"
    if surf_dir.exists():
        lh_flat = surf_dir / "lh.autoflatten.flat.patch.3d"
        rh_flat = surf_dir / "rh.autoflatten.flat.patch.3d"
        if lh_flat.exists() and rh_flat.exists():
            return (lh_flat, rh_flat)

    # Check flat structure: subject/
    lh_flat = subject_dir / "lh.autoflatten.flat.patch.3d"
    rh_flat = subject_dir / "rh.autoflatten.flat.patch.3d"
    if lh_flat.exists() and rh_flat.exists():
        return (lh_flat, rh_flat)

    return None


def load_flatmap_data(flat_patch_path, subject, subjects_dir, hemi):
    """Load flatmap data including vertices, faces, areas, and log info.

    Parameters
    ----------
    flat_patch_path : str or Path
        Path to the flat patch file
    subject : str
        Subject name (directory name in subjects_dir)
    subjects_dir : str or Path
        Path to FreeSurfer subjects directory
    hemi : str
        Hemisphere ('lh' or 'rh')

    Returns
    -------
    dict
        Dictionary with keys: xy, faces, areas, n_flipped, log_results
    """
    flat_patch_path = str(flat_patch_path)
    subjects_dir = Path(subjects_dir)

    # Find base surface in subjects_dir
    base_surface_path = subjects_dir / subject / "surf" / f"{hemi}.fiducial"
    if not base_surface_path.exists():
        base_surface_path = subjects_dir / subject / "surf" / f"{hemi}.white"
    if not base_surface_path.exists():
        raise ValueError(
            f"Could not find base surface for {subject} {hemi} in {subjects_dir}"
        )

    # Read flat patch
    flat_vertices, orig_indices, is_border = read_patch(flat_patch_path)

    # Read base surface to get faces
    _, base_faces = read_surface(base_surface_path)

    # Extract patch faces
    faces = extract_patch_faces(base_faces, orig_indices)

    # Get 2D coordinates
    xy = flat_vertices[:, :2]

    # Compute triangle areas
    areas = compute_triangle_areas(xy, faces)
    n_flipped = np.sum(areas < 0)

    # Parse log file
    log_path = flat_patch_path + ".log"
    log_results = parse_log_file(log_path)

    return {
        "xy": xy,
        "faces": faces,
        "areas": areas,
        "n_flipped": n_flipped,
        "log_results": log_results,
    }


def plot_single_flatmap(ax, xy, faces, areas, rotation_deg, cmap="viridis"):
    """Plot a single flatmap on an axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to plot on
    xy : ndarray of shape (N, 2)
        2D vertex coordinates
    faces : ndarray of shape (F, 3)
        Triangle indices
    areas : ndarray of shape (F,)
        Signed triangle areas
    rotation_deg : float
        Rotation angle in degrees
    cmap : str
        Colormap for area visualization

    Returns
    -------
    mappable
        The tripcolor mappable for colorbar creation
    """
    # Rotate coordinates
    xy_rot = rotate_coords(xy, rotation_deg)

    # Center coordinates using bounding box center (not centroid)
    bbox_center = (xy_rot.max(axis=0) + xy_rot.min(axis=0)) / 2
    xy_rot = xy_rot - bbox_center

    # Create triangulation
    triang = tri.Triangulation(xy_rot[:, 0], xy_rot[:, 1], faces)

    # Color by triangle area (log scale)
    log_areas = np.log10(np.abs(areas) + 1e-10)

    # Plot area visualization
    tpc = ax.tripcolor(triang, log_areas, shading="flat", cmap=cmap)

    # Mark flipped triangles with red points
    n_flipped = np.sum(areas < 0)
    if n_flipped > 0:
        flipped_mask = areas < 0
        flipped_faces = faces[flipped_mask]
        # Get centroids of flipped triangles
        flipped_centroids = np.mean(xy_rot[flipped_faces], axis=1)
        ax.scatter(
            flipped_centroids[:, 0],
            flipped_centroids[:, 1],
            c="red",
            s=5,
            marker="o",
            zorder=10,
            alpha=0.8,
        )

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")

    return tpc


def plot_all_flatmaps(
    data_dir,
    subjects_dir=None,
    output_path=None,
    ncols=4,
    figsize_per_cell=(3, 2.5),
    cmap="viridis",
    dpi=150,
    scale_mode="individual",
):
    """Plot all flatmaps in a single figure with multi-column grid layout.

    Parameters
    ----------
    data_dir : str or Path
        Directory containing participant subdirectories with flatmap results.
        Can be a FreeSurfer subjects directory (patches in surf/) or a
        separate output directory (patches directly in subject folders).
    subjects_dir : str or Path, optional
        Path to FreeSurfer subjects directory. If None, defaults to data_dir
        (assumes data_dir is the FreeSurfer subjects directory).
    output_path : str or Path, optional
        If provided, save figure to this path
    ncols : int
        Number of columns in the grid (each cell = one hemisphere)
    figsize_per_cell : tuple
        Width, height per cell in inches
    cmap : str
        Colormap for area visualization
    dpi : int
        Resolution for saved figure
    scale_mode : str
        How to scale flatmaps: "individual" (each fills subplot) or "global" (preserve mm)

    Returns
    -------
    matplotlib.figure.Figure
        The figure object
    """
    data_dir = Path(data_dir)
    # Default subjects_dir to data_dir if not specified
    if subjects_dir is None:
        subjects_dir = data_dir
    else:
        subjects_dir = Path(subjects_dir)

    # Find all participant directories with flat patches
    participants = {}  # participant_name -> (lh_path, rh_path)
    for subdir in sorted(data_dir.iterdir()):
        if subdir.is_dir():
            flat_paths = find_flat_patches(subdir)
            if flat_paths is not None:
                participants[subdir.name] = flat_paths

    if not participants:
        raise ValueError(f"No valid participant data found in {data_dir}")

    n_participants = len(participants)
    print(f"Found {n_participants} participants: {', '.join(participants.keys())}")

    # Each participant has 2 hemispheres, arranged in grid
    n_cells = n_participants * 2
    nrows = int(np.ceil(n_cells / ncols))

    # Create figure with subplots using constrained_layout for better spacing
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(figsize_per_cell[0] * ncols, figsize_per_cell[1] * nrows),
        squeeze=False,
        constrained_layout=True,
    )

    # Track min/max log areas for consistent colorbar
    all_log_areas = []

    # First pass: load all data and collect area ranges
    participant_data = {}
    for participant, (lh_path, rh_path) in participants.items():
        try:
            lh_data = load_flatmap_data(lh_path, participant, subjects_dir, "lh")
            rh_data = load_flatmap_data(rh_path, participant, subjects_dir, "rh")
            participant_data[participant] = {"lh": lh_data, "rh": rh_data}

            all_log_areas.extend(np.log10(np.abs(lh_data["areas"]) + 1e-10))
            all_log_areas.extend(np.log10(np.abs(rh_data["areas"]) + 1e-10))
        except Exception as e:
            print(f"Warning: Could not load data for {participant}: {e}")
            participant_data[participant] = None

    # Check that at least one participant loaded successfully
    if not all_log_areas:
        raise RuntimeError(
            "No participant data could be loaded; cannot compute area percentiles."
        )
    # Compute consistent color limits
    vmin = np.percentile(all_log_areas, 1)
    vmax = np.percentile(all_log_areas, 99)

    # Second pass: plot
    # Build list of (participant, hemi, data) tuples
    plot_items = []
    for participant in participants.keys():
        data = participant_data.get(participant)
        if data is not None:
            plot_items.append((participant, "lh", data["lh"]))
            plot_items.append((participant, "rh", data["rh"]))
        else:
            plot_items.append((participant, "lh", None))
            plot_items.append((participant, "rh", None))

    mappables = []
    plotted_axes = []
    axes_bounds = []  # Store individual bounds for each axis
    global_xmin, global_xmax = np.inf, -np.inf
    global_ymin, global_ymax = np.inf, -np.inf

    for idx, (participant, hemi, hemi_data) in enumerate(plot_items):
        row = idx // ncols
        col = idx % ncols
        ax = axes[row, col]

        if hemi_data is None:
            ax.text(
                0.5,
                0.5,
                "Load failed",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        # Rotation: LH 90° CCW, RH 90° CW
        rotation_deg = 90 if hemi == "lh" else -90

        tpc = plot_single_flatmap(
            ax,
            hemi_data["xy"],
            hemi_data["faces"],
            hemi_data["areas"],
            rotation_deg=rotation_deg,
            cmap=cmap,
        )
        tpc.set_clim(vmin, vmax)
        mappables.append(tpc)
        plotted_axes.append(ax)

        # Track bounds (after rotation and centering with bbox center)
        xy_rot = rotate_coords(hemi_data["xy"], rotation_deg)
        bbox_center = (xy_rot.max(axis=0) + xy_rot.min(axis=0)) / 2
        xy_centered = xy_rot - bbox_center
        xmin, xmax = xy_centered[:, 0].min(), xy_centered[:, 0].max()
        ymin, ymax = xy_centered[:, 1].min(), xy_centered[:, 1].max()
        axes_bounds.append((xmin, xmax, ymin, ymax))

        # Track global bounds for "global" scale mode
        global_xmin = min(global_xmin, xmin)
        global_xmax = max(global_xmax, xmax)
        global_ymin = min(global_ymin, ymin)
        global_ymax = max(global_ymax, ymax)

        # Get metrics from log file
        log_results = hemi_data["log_results"]
        err = log_results.get("distance_error", "?")
        flipped = log_results.get("flipped", hemi_data["n_flipped"])

        # Title: participant name, hemisphere, metrics
        hemi_label = "LH" if hemi == "lh" else "RH"
        ax.set_title(
            f"{participant} {hemi_label}\n{err}% err, {flipped} flip", fontsize=9
        )

    # Apply axis limits based on scale_mode
    padding = 0.02  # 2% padding
    if scale_mode == "global":
        # All flatmaps use same limits (preserve relative sizes in mm)
        x_range = global_xmax - global_xmin
        y_range = global_ymax - global_ymin
        for ax in plotted_axes:
            ax.set_xlim(
                global_xmin - padding * x_range, global_xmax + padding * x_range
            )
            ax.set_ylim(
                global_ymin - padding * y_range, global_ymax + padding * y_range
            )
    else:
        # "individual" mode: each flatmap fills its subplot
        # Use the maximum extent across all flatmaps for consistent visual size
        if not axes_bounds:
            print("Warning: No flatmap data was loaded; nothing to plot.")
            return fig
        max_extent = max(
            max(xmax - xmin, ymax - ymin) for xmin, xmax, ymin, ymax in axes_bounds
        )
        half_extent = max_extent / 2 * (1 + padding)
        for ax in plotted_axes:
            ax.set_xlim(-half_extent, half_extent)
            ax.set_ylim(-half_extent, half_extent)

    # Hide any empty axes
    for idx in range(len(plot_items), nrows * ncols):
        row = idx // ncols
        col = idx % ncols
        axes[row, col].axis("off")

    # Add colorbar using constrained_layout-compatible method
    if mappables:
        fig.colorbar(mappables[0], ax=axes, shrink=0.6, label="log10(area)")

    # Main title
    fig.suptitle(
        "Autoflatten Results - All Participants", fontsize=14, fontweight="bold"
    )

    if output_path:
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved figure to {output_path}")

    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Plot all flatmaps from autoflatten results"
    )
    parser.add_argument(
        "data_dir",
        type=str,
        help="Directory containing participant subdirectories with flatmap results. "
        "Can be a FreeSurfer subjects directory (patches in surf/) or a separate "
        "output directory (patches directly in subject folders).",
    )
    parser.add_argument(
        "--subjects-dir",
        type=str,
        default=None,
        help="Path to FreeSurfer subjects directory. If not specified, defaults to "
        "data_dir (assumes data_dir is the FreeSurfer subjects directory).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output path for the figure (default: display interactively)",
    )
    parser.add_argument(
        "--ncols",
        type=int,
        default=4,
        help="Number of columns in the grid (default: 4)",
    )
    parser.add_argument(
        "--figsize-width",
        type=float,
        default=3,
        help="Width per cell in inches (default: 3)",
    )
    parser.add_argument(
        "--figsize-height",
        type=float,
        default=2.5,
        help="Height per cell in inches (default: 2.5)",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="viridis",
        help="Colormap for area visualization (default: viridis)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI for saved figure (default: 150)",
    )
    parser.add_argument(
        "--scale-mode",
        type=str,
        choices=["individual", "global"],
        default="individual",
        help="Scale mode: 'individual' (each flatmap fills subplot) or 'global' (preserve mm)",
    )

    args = parser.parse_args()

    plot_all_flatmaps(
        args.data_dir,
        subjects_dir=args.subjects_dir,
        output_path=args.output,
        ncols=args.ncols,
        figsize_per_cell=(args.figsize_width, args.figsize_height),
        cmap=args.cmap,
        dpi=args.dpi,
        scale_mode=args.scale_mode,
    )

    if args.output is None:
        plt.show()


if __name__ == "__main__":
    main()
