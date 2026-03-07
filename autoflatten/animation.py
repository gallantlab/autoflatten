"""Animation utilities for visualizing the flattening process.

This module provides tools for capturing intermediate optimization states
during surface flattening and rendering them as animation frames.

Typical workflow::

    # Step 1: Run flattening with snapshot saving
    autoflatten flatten lh.patch.3d --save-snapshots snapshots.npz

    # Step 2: Render frames from snapshots
    autoflatten render-snapshots snapshots.npz --subject-dir /path/to/subject

    # Step 3: Assemble into video
    ffmpeg -r 15 -i flatten_frames/frame_%04d.png \\
        -c:v libx264 -pix_fmt yuv420p flatten.mp4
"""

import json
import os
import shutil

import numpy as np

# Human-readable phase labels for display
_PHASE_LABELS = {
    "initial": "Initial Projection",
    "nar": "Neg. Area Removal",
    "epoch_1": "Epoch 1",
    "epoch_2": "Epoch 2",
    "epoch_3": "Epoch 3",
    "final_nar": "Final NAR",
    "smoothing": "Spring Smoothing",
}


class SnapshotCollector:
    """Collects UV coordinate snapshots during optimization.

    Used as a callback for ``SurfaceFlattener.run()`` to capture
    intermediate optimization states at regular intervals.

    Parameters
    ----------
    every_n : int
        Save a snapshot every *every_n* callback invocations.
        Set to 1 to capture every iteration (large memory usage).

    Examples
    --------
    >>> collector = SnapshotCollector(every_n=10)
    >>> uv = flattener.run(snapshot_callback=collector)
    >>> collector.save("snapshots.npz", flattener.vertices, flattener.faces,
    ...               flattener.orig_indices)
    """

    def __init__(self, every_n: int = 10):
        self.every_n = max(1, every_n)
        self._snapshots = []
        self._metadata = []
        self._call_count = 0

    def __call__(self, uv: np.ndarray, metadata: dict | None = None) -> None:
        """Record a snapshot if the call count is a multiple of *every_n*.

        Parameters
        ----------
        uv : ndarray of shape (V, 2)
            Current UV coordinates.
        metadata : dict, optional
            Per-snapshot metadata (e.g., phase name, energy values).
        """
        self._call_count += 1
        if self._call_count % self.every_n == 0 or self._call_count == 1:
            self._snapshots.append(uv.astype(np.float32).copy())
            self._metadata.append(metadata or {})

    def save(
        self,
        path: str,
        vertices_3d: np.ndarray,
        faces: np.ndarray,
        orig_indices: np.ndarray,
    ) -> str:
        """Save collected snapshots and mesh metadata to an ``.npz`` file.

        Parameters
        ----------
        path : str
            Output file path (should end in ``.npz``).
        vertices_3d : ndarray of shape (V, 3)
            Original 3D vertex positions from the patch.
        faces : ndarray of shape (F, 3)
            Patch-local face indices.
        orig_indices : ndarray of shape (V,)
            Mapping from patch vertices to full-surface vertex indices.

        Returns
        -------
        str
            The output file path.
        """
        snapshots = np.stack(self._snapshots, axis=0)  # (M, V, 2)
        metadata_json = np.array(json.dumps(self._metadata))
        np.savez_compressed(
            path,
            snapshots=snapshots,
            vertices_3d=vertices_3d.astype(np.float32),
            faces=faces.astype(np.int32),
            orig_indices=orig_indices.astype(np.int32),
            metadata_json=metadata_json,
        )
        print(f"Saved {len(self._snapshots)} snapshots to {path}")
        return path

    @property
    def n_snapshots(self) -> int:
        """Number of snapshots collected so far."""
        return len(self._snapshots)


def render_snapshot_frames(
    npz_path: str,
    output_dir: str = "flatten_frames",
    n_frames: int = 120,
    curv_path: str | None = None,
    subject_dir: str | None = None,
    figsize: float = 6.0,
    dpi: int = 150,
    overwrite: bool = False,
    fps: float = 15.0,
    hold_start: float = 1.5,
    hold_phase_transition: float = 0.75,
    hold_end: float = 2.0,
    color_mode: str = "curvature",
) -> list[str]:
    """Render animation frames from saved optimization snapshots.

    Reads an ``.npz`` file produced by :class:`SnapshotCollector` and
    renders a subset of the snapshots as PNG frames suitable for
    assembling into a video with ffmpeg.

    Parameters
    ----------
    npz_path : str
        Path to the ``.npz`` snapshot file.
    output_dir : str
        Directory for output PNGs (created if needed).
    n_frames : int
        Number of frames to render. If fewer snapshots are available,
        all snapshots are rendered.
    curv_path : str, optional
        Path to a FreeSurfer curvature file (e.g., ``lh.curv``).
        If not provided and *subject_dir* is given, auto-detected
        from the hemisphere prefix in the snapshot filename.
    subject_dir : str, optional
        FreeSurfer subject directory for auto-detecting curvature.
    figsize : float
        Figure width and height in inches.
    dpi : int
        Resolution for saved frames.
    overwrite : bool
        Whether to overwrite existing frame files.
    fps : float
        Frames per second (used in the suggested ffmpeg command).
    hold_start : float
        Seconds to hold the first frame (initial projection).
    hold_phase_transition : float
        Seconds to hold at each phase boundary.
    hold_end : float
        Seconds to hold the last frame (final result).
    color_mode : str
        Face coloring mode. One of:

        - ``"curvature"`` (default): sulcal/gyral shading from curvature file
        - ``"distortion"``: per-face area distortion (log ratio of 2D/3D area)
          with flipped triangles in red

    Returns
    -------
    list of str
        Paths to the generated frame files.
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    # Load snapshots
    data = np.load(npz_path, allow_pickle=True)
    snapshots = data["snapshots"]  # (M, V, 2)
    faces = data["faces"]  # (F, 3)
    orig_indices = data["orig_indices"]  # (V,)

    # Load metadata if available (backward compat with old .npz files)
    all_metadata = None
    if "metadata_json" in data:
        try:
            all_metadata = json.loads(str(data["metadata_json"]))
        except (json.JSONDecodeError, ValueError):
            pass

    n_total = len(snapshots)
    print(
        f"Loaded {n_total} snapshots, {len(faces)} faces, {len(orig_indices)} vertices"
    )

    # Subsample to n_frames evenly spaced
    if n_total <= n_frames:
        indices = np.arange(n_total)
    else:
        indices = np.round(np.linspace(0, n_total - 1, n_frames)).astype(int)
        indices = np.unique(indices)  # remove duplicates

    # Expand frame list with hold frames for pacing
    indices = _expand_frames_with_holds(
        indices, all_metadata, fps, hold_start, hold_phase_transition, hold_end
    )

    # Prepare face coloring
    if color_mode == "distortion":
        vertices_3d = data["vertices_3d"]
        areas_3d = _compute_face_areas_3d(vertices_3d, faces)
        face_colors = None  # computed per-frame
    else:
        face_colors = _load_face_colors(curv_path, subject_dir, orig_indices, faces)
        areas_3d = None

    # Compute per-frame bounding boxes with consistent aspect ratio
    # Use a smooth transition so the "camera" follows the mesh
    all_selected = snapshots[indices]
    per_frame_mins = all_selected.min(axis=1)  # (n_frames, 2)
    per_frame_maxs = all_selected.max(axis=1)  # (n_frames, 2)
    per_frame_centers = (per_frame_mins + per_frame_maxs) / 2
    per_frame_extents = per_frame_maxs - per_frame_mins
    # Use max of x/y extent per frame for square aspect
    per_frame_size = per_frame_extents.max(axis=1)  # (n_frames,)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    frame_paths = []
    rendered_cache = {}  # snap_idx -> frame_path (first rendered copy)

    for frame_idx, snap_idx in enumerate(indices):
        frame_path = os.path.join(output_dir, f"frame_{frame_idx:04d}.png")
        frame_paths.append(frame_path)

        if os.path.exists(frame_path) and not overwrite:
            continue

        # Copy from cache if this snap_idx was already rendered
        if snap_idx in rendered_cache:
            shutil.copy2(rendered_cache[snap_idx], frame_path)
            continue

        uv = snapshots[snap_idx]  # (V, 2)

        fig, ax = plt.subplots(1, 1, figsize=(figsize, figsize))

        # Build polygon vertices for each face
        verts_per_face = uv[faces]  # (F, 3, 2)

        # Compute per-frame colors for distortion mode
        if color_mode == "distortion":
            frame_colors, flipped_mask = _compute_distortion_colors(uv, faces, areas_3d)
        else:
            frame_colors = face_colors
            flipped_mask = None

        poly = PolyCollection(
            verts_per_face,
            facecolors=frame_colors,
            edgecolors="none",
            linewidths=0,
            antialiaseds=False,
        )
        ax.add_collection(poly)

        # Draw flipped triangles on top (matching viz.py style)
        if flipped_mask is not None and np.any(flipped_mask):
            _draw_flipped_triangles(ax, uv, faces, flipped_mask)

        # Per-frame bounding box (square, centered on mesh)
        center = per_frame_centers[frame_idx]
        half_size = per_frame_size[frame_idx] / 2 * 1.1  # 10% margin
        ax.set_xlim(center[0] - half_size, center[0] + half_size)
        ax.set_ylim(center[1] - half_size, center[1] + half_size)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_facecolor("white")

        # Add colorbar for distortion mode
        if color_mode == "distortion":
            _add_distortion_colorbar(fig, ax)

        # Draw stage label in bottom-right corner
        if all_metadata is not None and snap_idx < len(all_metadata):
            meta = all_metadata[snap_idx]
            _draw_label(ax, meta)

        fig.savefig(
            frame_path,
            dpi=dpi,
            facecolor="white",
        )
        plt.close(fig)
        rendered_cache[snap_idx] = frame_path

        if (frame_idx + 1) % 20 == 0 or frame_idx == len(indices) - 1:
            print(f"  Rendered frame {frame_idx + 1}/{len(indices)}")

    fps_int = int(fps) if fps == int(fps) else fps
    print(f"\nGenerated {len(frame_paths)} frames in {output_dir}")
    print(
        f"Suggested command:\n"
        f"  ffmpeg -r {fps_int} -i {output_dir}/frame_%04d.png "
        f"-c:v libx264 -pix_fmt yuv420p flatten.mp4"
    )
    return frame_paths


def _expand_frames_with_holds(
    indices: np.ndarray,
    all_metadata: list[dict] | None,
    fps: float,
    hold_start: float,
    hold_phase_transition: float,
    hold_end: float,
) -> np.ndarray:
    """Insert duplicate frame indices to create holds at key moments.

    Returns an expanded array of snapshot indices where duplicates
    produce hold/pause effects at uniform fps playback.
    """
    if len(indices) == 0:
        return indices

    expanded = []

    # Hold on the first frame
    n_hold_start = max(0, int(hold_start * fps))
    expanded.extend([indices[0]] * n_hold_start)

    for i, snap_idx in enumerate(indices):
        expanded.append(snap_idx)

        # Detect phase transitions
        if (
            hold_phase_transition > 0
            and all_metadata is not None
            and i < len(indices) - 1
        ):
            cur_phase = (
                all_metadata[snap_idx].get("phase", "")
                if snap_idx < len(all_metadata)
                else ""
            )
            next_snap = indices[i + 1]
            next_phase = (
                all_metadata[next_snap].get("phase", "")
                if next_snap < len(all_metadata)
                else ""
            )
            if cur_phase and next_phase and cur_phase != next_phase:
                n_hold = max(0, int(hold_phase_transition * fps))
                expanded.extend([snap_idx] * n_hold)

    # Hold on the last frame
    n_hold_end = max(0, int(hold_end * fps))
    expanded.extend([indices[-1]] * n_hold_end)

    n_added = len(expanded) - len(indices)
    if n_added > 0:
        print(f"  Added {n_added} hold frames ({len(expanded)} total)")

    return np.array(expanded, dtype=indices.dtype)


def _compute_face_areas_3d(vertices_3d, faces):
    """Compute 3D triangle areas from vertices and faces."""
    v0 = vertices_3d[faces[:, 0]]
    v1 = vertices_3d[faces[:, 1]]
    v2 = vertices_3d[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    return 0.5 * np.linalg.norm(cross, axis=1)


def _compute_distortion_colors(uv, faces, areas_3d):
    """Compute per-face area distortion colors and flipped triangle mask.

    Uses log2(area_2d / area_3d) mapped to the plasma colormap.

    Returns
    -------
    colors : ndarray of shape (F, 4)
        RGBA face colors from the distortion colormap.
    flipped : ndarray of shape (F,)
        Boolean mask of flipped (negative signed area) triangles.
    """
    import matplotlib.cm as cm

    # Signed 2D areas
    v0 = uv[faces[:, 0]]
    v1 = uv[faces[:, 1]]
    v2 = uv[faces[:, 2]]
    signed_areas_2d = 0.5 * (
        (v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1])
        - (v2[:, 0] - v0[:, 0]) * (v1[:, 1] - v0[:, 1])
    )

    flipped = signed_areas_2d < 0
    areas_2d = np.abs(signed_areas_2d)

    # Log ratio: positive = expanded, negative = compressed
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(areas_3d > 0, areas_2d / areas_3d, 1.0)
        log_ratio = np.log2(np.clip(ratio, 1e-6, 1e6))

    # Map to colormap: clip to [-3, 3] (8x compression to 8x expansion)
    vmin, vmax = -3.0, 3.0
    normalized = np.clip((log_ratio - vmin) / (vmax - vmin), 0, 1)
    colors = cm.viridis(normalized)  # dark=compressed, bright=expanded

    return colors, flipped


def _draw_flipped_triangles(ax, uv, faces, flipped_mask):
    """Draw flipped triangles as red polygons with yellow centroid markers."""
    from matplotlib.collections import PolyCollection

    flipped_faces = faces[flipped_mask]
    flipped_verts = uv[flipped_faces]  # (N, 3, 2)

    # Red polygons with dark red edges
    poly = PolyCollection(
        flipped_verts,
        facecolors="red",
        edgecolors="darkred",
        alpha=0.9,
        linewidths=0.5,
        zorder=10,
    )
    ax.add_collection(poly)

    # Yellow centroid markers for visibility when zoomed out
    centroids = flipped_verts.mean(axis=1)  # (N, 2)
    ax.scatter(
        centroids[:, 0],
        centroids[:, 1],
        c="yellow",
        s=8,
        marker="o",
        edgecolors="red",
        linewidths=0.5,
        zorder=11,
    )


def _add_distortion_colorbar(fig, ax):
    """Add a horizontal colorbar for area distortion below the plot."""
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    vmin, vmax = -3.0, 3.0
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    sm = cm.ScalarMappable(cmap=cm.viridis, norm=norm)
    sm.set_array([])

    # Position colorbar below the axes
    bbox = ax.get_position()
    cbar_ax = fig.add_axes([bbox.x0 + 0.1, bbox.y0 - 0.02, bbox.width - 0.2, 0.015])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_label(
        "log₂(area 2D / area 3D)",
        fontsize=8,
        fontfamily="monospace",
    )
    cbar.set_ticks([-3, -2, -1, 0, 1, 2, 3])
    cbar.set_ticklabels(["⅛×", "¼×", "½×", "1×", "2×", "4×", "8×"])
    cbar.ax.tick_params(labelsize=7)


def _draw_label(ax, meta: dict) -> None:
    """Draw a stage label with key values in the bottom-right corner."""
    phase = meta.get("phase", "")
    label = _PHASE_LABELS.get(phase, phase.replace("_", " ").title())

    lines = [label]
    if "J_d" in meta:
        lines.append(f"J_d = {meta['J_d']:.4f}")
    if "J_a" in meta:
        lines.append(f"J_a = {meta['J_a']:.4f}")
    if "n_flipped" in meta:
        lines.append(f"Flipped: {meta['n_flipped']}")

    label_text = "  |  ".join(lines)
    ax.set_title(
        label_text,
        fontsize=10,
        fontfamily="monospace",
        pad=8,
    )


def _load_face_colors(curv_path, subject_dir, orig_indices, faces):
    """Load curvature and compute per-face colors.

    Returns an (F, 4) RGBA array with sulci in dark gray and gyri in
    light gray.
    """
    curv = None

    if curv_path is not None:
        curv = _read_curv(curv_path, orig_indices)
    elif subject_dir is not None:
        # Auto-detect hemisphere and curvature
        curv = _auto_detect_curv(subject_dir, orig_indices)

    n_faces = len(faces)
    colors = np.ones((n_faces, 4))  # RGBA

    if curv is not None:
        # Per-face curvature: average of vertex curvatures
        face_curv = curv[faces].mean(axis=1)
        # Sulci (positive curv) = dark, gyri (negative) = light
        gray = np.where(face_curv > 0, 0.3, 0.7)
        colors[:, 0] = gray
        colors[:, 1] = gray
        colors[:, 2] = gray
    else:
        # Uniform light gray
        colors[:, :3] = 0.6

    return colors


def _read_curv(curv_path, orig_indices):
    """Read curvature file and extract patch vertices."""
    try:
        import nibabel.freesurfer

        curv_full = nibabel.freesurfer.read_morph_data(curv_path)
        return curv_full[orig_indices]
    except Exception as e:
        print(f"Warning: Could not read curvature file {curv_path}: {e}")
        return None


def _auto_detect_curv(subject_dir, orig_indices):
    """Try to auto-detect and load curvature from subject directory."""
    from pathlib import Path

    surf_dir = Path(subject_dir) / "surf"
    if not surf_dir.is_dir():
        surf_dir = Path(subject_dir)

    for hemi in ("lh", "rh"):
        curv_path = surf_dir / f"{hemi}.curv"
        if curv_path.exists():
            curv = _read_curv(str(curv_path), orig_indices)
            if curv is not None:
                return curv
    return None
