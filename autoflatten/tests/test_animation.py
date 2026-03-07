"""Tests for the animation module."""

import json
import os
import tempfile

import numpy as np
import pytest

from autoflatten.animation import (
    SnapshotCollector,
    _compute_distortion_colors,
    _compute_face_areas_3d,
    _draw_flipped_triangles,
    _draw_label,
    _expand_frames_with_holds,
    _load_face_colors,
    render_snapshot_frames,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_quad_mesh():
    """Create a simple quad (2-triangle) mesh for testing.

    Returns vertices_3d (4,3), uv (4,2), faces (2,3).
    """
    vertices_3d = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.float32
    )
    uv = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
    faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    return vertices_3d, uv, faces


def _make_snapshot_npz(path, n_snapshots=5, n_verts=4, faces=None, metadata=None):
    """Write a minimal .npz file that render_snapshot_frames can load."""
    if faces is None:
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)

    rng = np.random.RandomState(42)
    snapshots = rng.randn(n_snapshots, n_verts, 2).astype(np.float32)
    vertices_3d = rng.randn(n_verts, 3).astype(np.float32)
    orig_indices = np.arange(n_verts, dtype=np.int32)

    if metadata is None:
        metadata = [{"phase": "initial"}] * n_snapshots

    np.savez_compressed(
        path,
        snapshots=snapshots,
        faces=faces,
        vertices_3d=vertices_3d,
        orig_indices=orig_indices,
        metadata_json=np.array(json.dumps(metadata)),
    )


# ---------------------------------------------------------------------------
# SnapshotCollector
# ---------------------------------------------------------------------------


class TestSnapshotCollector:
    def test_collects_first_call(self):
        collector = SnapshotCollector(every_n=5)
        uv = np.zeros((10, 2))
        collector(uv, {"phase": "initial"})
        assert collector.n_snapshots == 1

    def test_every_n(self):
        collector = SnapshotCollector(every_n=3)
        for i in range(9):
            collector(np.zeros((4, 2)), {"phase": "epoch_1"})
        # call 1 (first), call 3, call 6, call 9 → 4 snapshots
        assert collector.n_snapshots == 4

    def test_save_and_load(self):
        collector = SnapshotCollector(every_n=1)
        uv1 = np.array([[0, 0], [1, 0]], dtype=np.float32)
        uv2 = np.array([[0, 1], [1, 1]], dtype=np.float32)
        collector(uv1, {"phase": "initial"})
        collector(uv2, {"phase": "epoch_1"})

        verts = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
        faces = np.array([[0, 1, 0]], dtype=np.int32)
        orig = np.array([0, 1], dtype=np.int32)

        with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as f:
            path = f.name
        try:
            collector.save(path, verts, faces, orig)
            data = np.load(path, allow_pickle=True)
            assert data["snapshots"].shape == (2, 2, 2)
            meta = json.loads(str(data["metadata_json"]))
            assert meta[0]["phase"] == "initial"
            assert meta[1]["phase"] == "epoch_1"
        finally:
            os.unlink(path)

    def test_minimum_every_n(self):
        collector = SnapshotCollector(every_n=0)
        assert collector.every_n == 1

    def test_copies_uv(self):
        collector = SnapshotCollector(every_n=1)
        uv = np.array([[1, 2], [3, 4]], dtype=np.float32)
        collector(uv)
        uv[:] = 0
        # Should have a copy, not affected by mutation
        assert collector._snapshots[0][0, 0] == 1.0

    def test_save_empty_raises(self):
        collector = SnapshotCollector(every_n=1)
        verts = np.zeros((2, 3), dtype=np.float32)
        faces = np.array([[0, 1, 0]], dtype=np.int32)
        orig = np.array([0, 1], dtype=np.int32)
        with pytest.raises(ValueError, match="No snapshots"):
            collector.save("/tmp/empty.npz", verts, faces, orig)


# ---------------------------------------------------------------------------
# render_snapshot_frames validation
# ---------------------------------------------------------------------------


class TestRenderSnapshotFramesValidation:
    def test_invalid_npz_missing_keys(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            npz_path = os.path.join(tmpdir, "bad.npz")
            np.savez(npz_path, foo=np.array([1, 2, 3]))
            with pytest.raises(ValueError, match="missing required arrays"):
                render_snapshot_frames(npz_path, os.path.join(tmpdir, "frames"))


# ---------------------------------------------------------------------------
# _expand_frames_with_holds
# ---------------------------------------------------------------------------


class TestExpandFramesWithHolds:
    def test_empty_indices(self):
        result = _expand_frames_with_holds(np.array([], dtype=int), None, 10, 1, 0.5, 1)
        assert len(result) == 0

    def test_hold_start_and_end(self):
        indices = np.array([0, 1, 2])
        result = _expand_frames_with_holds(indices, None, 10, 1.0, 0, 2.0)
        # 10 hold start + 3 original + 20 hold end = 33
        assert len(result) == 33
        assert result[0] == 0
        assert result[-1] == 2

    def test_phase_transition_holds(self):
        indices = np.array([0, 1, 2, 3])
        metadata = [
            {"phase": "initial"},
            {"phase": "initial"},
            {"phase": "epoch_1"},
            {"phase": "epoch_1"},
        ]
        result = _expand_frames_with_holds(indices, metadata, 10, 0, 1.0, 0)
        # Original 4 + 10 hold at transition (index 1→2) = 14
        assert len(result) == 14
        # The hold frames should be copies of index 1 (last of outgoing phase)
        assert np.all(result[2:12] == 1)

    def test_no_holds(self):
        indices = np.array([0, 1, 2])
        result = _expand_frames_with_holds(indices, None, 10, 0, 0, 0)
        np.testing.assert_array_equal(result, indices)

    def test_no_metadata_skips_phase_holds(self):
        indices = np.array([0, 1, 2])
        result = _expand_frames_with_holds(indices, None, 10, 0, 1.0, 0)
        # No metadata → no phase transitions detected → no holds added
        np.testing.assert_array_equal(result, indices)


# ---------------------------------------------------------------------------
# _compute_face_areas_3d
# ---------------------------------------------------------------------------


class TestComputeFaceAreas3d:
    def test_unit_triangle(self):
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        faces = np.array([[0, 1, 2]])
        areas = _compute_face_areas_3d(verts, faces)
        np.testing.assert_allclose(areas, [0.5], atol=1e-6)

    def test_multiple_faces(self):
        verts, _, faces = _make_quad_mesh()
        areas = _compute_face_areas_3d(verts, faces)
        assert areas.shape == (2,)
        np.testing.assert_allclose(areas, [0.5, 0.5], atol=1e-6)


# ---------------------------------------------------------------------------
# _compute_distortion_colors
# ---------------------------------------------------------------------------


class TestComputeDistortionColors:
    def test_no_distortion(self):
        """When 2D and 3D areas match, log ratio should be ~0 (middle of cmap)."""
        verts_3d, uv, faces = _make_quad_mesh()
        areas_3d = _compute_face_areas_3d(verts_3d, faces)
        colors, flipped = _compute_distortion_colors(uv, faces, areas_3d)
        assert colors.shape == (2, 4)
        assert flipped.shape == (2,)
        assert not np.any(flipped)

    def test_flipped_detected(self):
        """Clockwise winding should be flagged as flipped."""
        _, uv, faces = _make_quad_mesh()
        # Flip winding of first face
        faces_flipped = faces.copy()
        faces_flipped[0] = faces_flipped[0][::-1]
        areas_3d = np.array([0.5, 0.5])
        colors, flipped = _compute_distortion_colors(uv, faces_flipped, areas_3d)
        assert flipped[0] == True
        assert flipped[1] == False

    def test_output_rgba(self):
        _, uv, faces = _make_quad_mesh()
        areas_3d = np.array([0.5, 0.5])
        colors, _ = _compute_distortion_colors(uv, faces, areas_3d)
        # RGBA values in [0, 1]
        assert np.all(colors >= 0) and np.all(colors <= 1)


# ---------------------------------------------------------------------------
# _draw_flipped_triangles
# ---------------------------------------------------------------------------


class TestDrawFlippedTriangles:
    def test_draws_without_error(self):
        import matplotlib.pyplot as plt

        _, uv, faces = _make_quad_mesh()
        flipped_mask = np.array([True, False])
        fig, ax = plt.subplots()
        _draw_flipped_triangles(ax, uv, faces, flipped_mask)
        plt.close(fig)

    def test_no_flipped(self):
        import matplotlib.pyplot as plt

        _, uv, faces = _make_quad_mesh()
        flipped_mask = np.array([False, False])
        fig, ax = plt.subplots()
        # Should not raise even with no flipped triangles
        # (caller checks np.any before calling, but function should handle it)
        _draw_flipped_triangles(ax, uv, faces, flipped_mask)
        plt.close(fig)


# ---------------------------------------------------------------------------
# _draw_label
# ---------------------------------------------------------------------------


class TestDrawLabel:
    def test_with_phase(self):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        _draw_label(
            ax, {"phase": "epoch_1", "J_d": 0.1234, "J_a": 0.5678, "n_flipped": 10}
        )
        title = ax.get_title()
        assert "Epoch 1" in title
        assert "0.1234" in title
        assert "10" in title
        plt.close(fig)

    def test_unknown_phase(self):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        _draw_label(ax, {"phase": "custom_phase"})
        title = ax.get_title()
        assert "Custom Phase" in title
        plt.close(fig)

    def test_empty_metadata(self):
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        _draw_label(ax, {})
        plt.close(fig)


# ---------------------------------------------------------------------------
# _load_face_colors
# ---------------------------------------------------------------------------


class TestLoadFaceColors:
    def test_no_curv(self):
        faces = np.array([[0, 1, 2], [0, 2, 3]])
        orig_indices = np.arange(4)
        colors = _load_face_colors(None, None, orig_indices, faces)
        assert colors.shape == (2, 4)
        # Uniform gray
        np.testing.assert_allclose(colors[:, :3], 0.6)

    def test_nonexistent_curv_path(self):
        faces = np.array([[0, 1, 2]])
        orig_indices = np.arange(3)
        colors = _load_face_colors("/nonexistent/path.curv", None, orig_indices, faces)
        # Falls back to uniform gray
        assert colors.shape == (1, 4)


# ---------------------------------------------------------------------------
# _add_distortion_colorbar
# ---------------------------------------------------------------------------


class TestAddDistortionColorbar:
    def test_adds_colorbar(self):
        from autoflatten.animation import _add_distortion_colorbar
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        n_axes_before = len(fig.axes)
        _add_distortion_colorbar(fig, ax)
        assert len(fig.axes) > n_axes_before
        plt.close(fig)


# ---------------------------------------------------------------------------
# render_snapshot_frames (integration)
# ---------------------------------------------------------------------------


class TestRenderSnapshotFrames:
    def test_basic_rendering(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            npz_path = os.path.join(tmpdir, "snaps.npz")
            _make_snapshot_npz(npz_path, n_snapshots=3)
            out_dir = os.path.join(tmpdir, "frames")

            paths = render_snapshot_frames(
                npz_path,
                out_dir,
                n_frames=3,
                fps=2,
                hold_start=0,
                hold_phase_transition=0,
                hold_end=0,
            )
            assert len(paths) == 3
            for p in paths:
                assert os.path.exists(p)

    def test_distortion_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            npz_path = os.path.join(tmpdir, "snaps.npz")
            _make_snapshot_npz(npz_path, n_snapshots=2)
            out_dir = os.path.join(tmpdir, "frames")

            paths = render_snapshot_frames(
                npz_path,
                out_dir,
                n_frames=2,
                color_mode="distortion",
                fps=2,
                hold_start=0,
                hold_phase_transition=0,
                hold_end=0,
            )
            assert len(paths) == 2

    def test_hold_frames_produce_copies(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            npz_path = os.path.join(tmpdir, "snaps.npz")
            metadata = [
                {"phase": "initial"},
                {"phase": "epoch_1"},
                {"phase": "epoch_1"},
            ]
            _make_snapshot_npz(npz_path, n_snapshots=3, metadata=metadata)
            out_dir = os.path.join(tmpdir, "frames")

            paths = render_snapshot_frames(
                npz_path,
                out_dir,
                n_frames=3,
                fps=2,
                hold_start=1.0,
                hold_phase_transition=0.5,
                hold_end=1.0,
            )
            # Should have more frames than original due to holds
            assert len(paths) > 3
            for p in paths:
                assert os.path.exists(p)

    def test_subsampling(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            npz_path = os.path.join(tmpdir, "snaps.npz")
            _make_snapshot_npz(npz_path, n_snapshots=20)
            out_dir = os.path.join(tmpdir, "frames")

            paths = render_snapshot_frames(
                npz_path,
                out_dir,
                n_frames=5,
                fps=2,
                hold_start=0,
                hold_phase_transition=0,
                hold_end=0,
            )
            assert len(paths) == 5
