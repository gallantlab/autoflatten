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

    # Load curvature for face coloring
    face_colors = _load_face_colors(curv_path, subject_dir, orig_indices, faces)

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
    n_digits = len(str(len(indices) - 1))

    for frame_idx, snap_idx in enumerate(indices):
        frame_path = os.path.join(output_dir, f"frame_{frame_idx:04d}.png")
        frame_paths.append(frame_path)

        if os.path.exists(frame_path) and not overwrite:
            continue

        uv = snapshots[snap_idx]  # (V, 2)

        fig, ax = plt.subplots(1, 1, figsize=(figsize, figsize))

        # Build polygon vertices for each face
        verts_per_face = uv[faces]  # (F, 3, 2)

        poly = PolyCollection(
            verts_per_face,
            facecolors=face_colors,
            edgecolors="none",
            linewidths=0,
            antialiaseds=False,
        )
        ax.add_collection(poly)

        # Per-frame bounding box (square, centered on mesh)
        center = per_frame_centers[frame_idx]
        half_size = per_frame_size[frame_idx] / 2 * 1.1  # 10% margin
        ax.set_xlim(center[0] - half_size, center[0] + half_size)
        ax.set_ylim(center[1] - half_size, center[1] + half_size)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_facecolor("white")

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
