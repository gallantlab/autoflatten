#!/usr/bin/env python
"""Create a half-sphere test mesh, flatten it with snapshots, and render a video."""

import os
import tempfile

import nibabel.freesurfer
import numpy as np

from autoflatten.animation import SnapshotCollector, render_snapshot_frames
from autoflatten.flatten import FlattenConfig, SurfaceFlattener
from autoflatten.freesurfer import write_patch


def make_half_sphere(n_subdivisions=20):
    """Create a half-sphere mesh with disk topology (open, not wrapped).

    Returns vertices (V,3), faces (F,3), curvature (V,).
    """
    # Create a disk-topology hemisphere by NOT wrapping theta around
    n_rings = n_subdivisions  # rings from equator to pole
    n_sectors = n_subdivisions  # sectors around

    radius = 50.0  # mm

    vertices = []
    # Pole vertex (top)
    vertices.append([0.0, 0.0, radius])

    # Rings from near-pole down to equator
    for i in range(1, n_rings + 1):
        phi = (np.pi / 2) * (1.0 - i / n_rings)  # pi/2 (pole) to 0 (equator)
        for j in range(n_sectors):
            theta = 2 * np.pi * j / n_sectors
            x = radius * np.cos(phi) * np.cos(theta)
            y = radius * np.cos(phi) * np.sin(theta)
            z = radius * np.sin(phi)
            vertices.append([x, y, z])

    vertices = np.array(vertices, dtype=np.float32)
    n_verts = len(vertices)

    faces = []
    # Connect pole to first ring (fan triangles)
    for j in range(n_sectors):
        j_next = (j + 1) % n_sectors
        faces.append([0, 1 + j, 1 + j_next])

    # Connect ring i to ring i+1
    for i in range(n_rings - 1):
        ring_start = 1 + i * n_sectors
        next_ring_start = 1 + (i + 1) * n_sectors
        for j in range(n_sectors):
            j_next = (j + 1) % n_sectors
            v0 = ring_start + j
            v1 = ring_start + j_next
            v2 = next_ring_start + j
            v3 = next_ring_start + j_next
            faces.append([v0, v1, v2])
            faces.append([v1, v3, v2])

    faces = np.array(faces, dtype=np.int32)

    # Create curvature: alternating bands based on ring index
    curv = np.zeros(n_verts, dtype=np.float32)
    curv[0] = 1.0  # pole
    for i in range(n_rings):
        ring_start = 1 + i * n_sectors
        val = 1.0 if (i % 3 < 2) else -1.0
        curv[ring_start : ring_start + n_sectors] = val

    # Identify border vertices (equator ring = last ring)
    is_border = np.zeros(n_verts, dtype=bool)
    last_ring_start = 1 + (n_rings - 1) * n_sectors
    is_border[last_ring_start : last_ring_start + n_sectors] = True

    return vertices, faces, curv, is_border


def main():
    tmpdir = tempfile.mkdtemp(prefix="autoflatten_demo_")
    print(f"Working directory: {tmpdir}")

    surf_dir = os.path.join(tmpdir, "surf")
    os.makedirs(surf_dir)

    # Create half-sphere mesh
    print("Creating half-sphere mesh...")
    vertices, faces, curv, is_border = make_half_sphere(n_subdivisions=30)
    print(f"  {len(vertices)} vertices, {len(faces)} faces")

    # Write surface files
    surface_path = os.path.join(surf_dir, "lh.fiducial")
    nibabel.freesurfer.write_geometry(surface_path, vertices, faces)
    print(f"  Saved surface: {surface_path}")

    # Write curvature
    curv_path = os.path.join(surf_dir, "lh.curv")
    nibabel.freesurfer.write_morph_data(curv_path, curv)
    print(f"  Saved curvature: {curv_path}")

    # Create patch file (all vertices, border = equator ring)
    orig_indices = np.arange(len(vertices), dtype=np.int32)

    patch_path = os.path.join(surf_dir, "lh.autoflatten.patch.3d")
    write_patch(patch_path, vertices, orig_indices, is_border)
    print(f"  Saved patch: {patch_path}")

    # Run flattening with snapshots
    print("\nRunning flattening with snapshot capture...")
    snapshot_path = os.path.join(tmpdir, "snapshots.npz")
    output_path = os.path.join(surf_dir, "lh.autoflatten.flat.patch.3d")

    config = FlattenConfig()
    # Use fast settings for the demo
    config.kring.k_ring = 3
    config.kring.n_neighbors_per_ring = 8
    config.verbose = True
    config.print_every = 10

    flattener = SurfaceFlattener(config)
    flattener.load_data(patch_path, surface_path)
    flattener.compute_kring_distances()
    flattener.prepare_optimization()

    # Capture every iteration for smooth animation
    collector = SnapshotCollector(every_n=1)
    uv = flattener.run(snapshot_callback=collector)
    flattener.save_result(uv, output_path)

    print(f"\nCollected {collector.n_snapshots} snapshots")

    # Save snapshots
    collector.save(
        snapshot_path,
        vertices_3d=flattener.vertices,
        faces=flattener.faces,
        orig_indices=flattener.orig_indices,
    )

    # Render frames
    print("\nRendering frames...")
    fps = 8
    frames_dir = os.path.join(tmpdir, "frames")
    frames = render_snapshot_frames(
        npz_path=snapshot_path,
        output_dir=frames_dir,
        n_frames=60,
        curv_path=curv_path,
        figsize=6.0,
        dpi=100,
        fps=fps,
    )

    # Assemble video
    print("\nAssembling video...")
    mp4_path = os.path.join(tmpdir, "flatten.mp4")
    gif_path = os.path.join(tmpdir, "flatten.gif")
    try:
        import subprocess

        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-r",
                str(fps),
                "-i",
                os.path.join(frames_dir, "frame_%04d.png"),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                mp4_path,
            ],
            check=True,
            capture_output=True,
        )
        print(f"Video saved to: {mp4_path}")
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        print(f"ffmpeg failed ({e}), trying GIF fallback...")
        try:
            import imageio.v3 as iio

            frame_images = [iio.imread(fp)[:, :, :3] for fp in frames]
            duration_ms = int(1000 / fps)
            iio.imwrite(
                gif_path, frame_images, duration=duration_ms, loop=0, plugin="pillow"
            )
            print(f"GIF saved to: {gif_path}")
        except Exception as e2:
            print(f"Could not create video ({e2}), frames are in: {frames_dir}")

    print(f"\nAll outputs in: {tmpdir}")
    return tmpdir


if __name__ == "__main__":
    main()
