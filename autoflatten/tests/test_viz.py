"""
Tests for the viz module (matplotlib-based visualization).
"""

import os
import tempfile

import numpy as np

from autoflatten.viz import (
    compute_kring_distortion,
    compute_triangle_areas,
    parse_log_file,
    load_curvature,
    _get_view_angles,
    plot_projection,
)


class TestComputeTriangleAreas:
    """Tests for compute_triangle_areas function."""

    def test_single_triangle_positive(self):
        """Test area computation for single counter-clockwise triangle."""
        vertices_2d = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, 1.0],
            ]
        )
        faces = np.array([[0, 1, 2]])
        areas = compute_triangle_areas(vertices_2d, faces)
        assert areas.shape == (1,)
        assert areas[0] > 0  # Counter-clockwise = positive

    def test_single_triangle_negative(self):
        """Test area computation for single clockwise (flipped) triangle."""
        vertices_2d = np.array(
            [
                [0.0, 0.0],
                [0.5, 1.0],
                [1.0, 0.0],
            ]
        )
        faces = np.array([[0, 1, 2]])
        areas = compute_triangle_areas(vertices_2d, faces)
        assert areas.shape == (1,)
        assert areas[0] < 0  # Clockwise = negative (flipped)

    def test_right_triangle_area_value(self):
        """Test that area value is correct for known triangle."""
        vertices_2d = np.array(
            [
                [0.0, 0.0],
                [2.0, 0.0],
                [0.0, 2.0],
            ]
        )
        faces = np.array([[0, 1, 2]])
        areas = compute_triangle_areas(vertices_2d, faces)
        # Right triangle with legs 2, 2 has area 2.0
        assert np.abs(areas[0] - 2.0) < 1e-10

    def test_multiple_triangles(self):
        """Test area computation for multiple triangles."""
        vertices_2d = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
            ]
        )
        # Two triangles forming a square
        faces = np.array(
            [
                [0, 1, 2],
                [0, 2, 3],
            ]
        )
        areas = compute_triangle_areas(vertices_2d, faces)
        assert areas.shape == (2,)
        # Each triangle should have area 0.5
        assert np.allclose(areas, [0.5, 0.5])

    def test_mixed_orientations(self):
        """Test triangles with mixed orientations."""
        vertices_2d = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, 0.5],
                [0.5, -0.5],
            ]
        )
        # One counter-clockwise, one clockwise
        faces = np.array(
            [
                [0, 1, 2],  # Counter-clockwise
                [0, 2, 1],  # Clockwise (reversed)
            ]
        )
        areas = compute_triangle_areas(vertices_2d, faces)
        assert areas[0] > 0
        assert areas[1] < 0


class TestParseLogFile:
    """Tests for parse_log_file function."""

    def test_parse_empty_result_for_missing_file(self):
        """Test that missing file returns empty dict."""
        result = parse_log_file("/nonexistent/path/to/log.log")
        assert result == {}

    def test_parse_log_with_final_result(self):
        """Test parsing log file with final result section."""
        # Note: parse_log_file extracts parent directory name as "subject",
        # so for /data/sub-01/lh.patch.3d it gets "sub-01"
        log_content = """
Autoflatten Log
===============
Input patch: /data/sub-01/lh.autoflatten.patch.3d

FINAL RESULT
Flipped triangles: 100 -> 5
Mean % distance error: 2.35%
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.write(log_content)
            temp_path = f.name

        try:
            result = parse_log_file(temp_path)
            assert result.get("flipped") == 5
            assert result.get("distance_error") == 2.35
            assert result.get("subject") == "sub-01"
            assert result.get("hemisphere") == "lh"
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_parse_log_rh_hemisphere(self):
        """Test parsing log file for right hemisphere."""
        log_content = """
Input patch: /data/sub-02/rh.autoflatten.patch.3d

FINAL RESULT
Flipped triangles: 50 -> 2
Mean % distance error: 1.85%
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.write(log_content)
            temp_path = f.name

        try:
            result = parse_log_file(temp_path)
            assert result.get("hemisphere") == "rh"
            assert result.get("subject") == "sub-02"
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_parse_log_partial_info(self):
        """Test parsing log file with only partial information."""
        log_content = """
Some other content
Input patch: /data/sub-03/lh.autoflatten.patch.3d
More content
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.write(log_content)
            temp_path = f.name

        try:
            result = parse_log_file(temp_path)
            # Should have subject/hemi but not flipped/distance_error
            assert result.get("subject") == "sub-03"
            assert result.get("hemisphere") == "lh"
            assert "flipped" not in result
            assert "distance_error" not in result
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_parse_log_no_match(self):
        """Test parsing log file with no matching patterns."""
        log_content = """
Random content
Nothing useful here
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.write(log_content)
            temp_path = f.name

        try:
            result = parse_log_file(temp_path)
            assert result == {}
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


class TestComputeKringDistortion:
    """Tests for compute_kring_distortion function."""

    def _make_simple_mesh(self):
        """Create a simple 2D mesh for testing (square with 4 vertices, 2 triangles)."""
        # 3D vertices (flat in z plane for simplicity)
        base_vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        )
        # Two triangles forming a square
        base_faces = np.array(
            [
                [0, 1, 2],
                [0, 2, 3],
            ]
        )
        # All vertices are in the patch
        orig_indices = np.array([0, 1, 2, 3])
        return base_vertices, base_faces, orig_indices

    def test_basic_distortion_flat_surface(self):
        """Test distortion computation on a flat surface."""
        base_vertices, base_faces, orig_indices = self._make_simple_mesh()
        # 2D coordinates that exactly match the 3D layout
        xy = base_vertices[:, :2]

        vertex_dist, mean_dist = compute_kring_distortion(
            xy,
            base_vertices,
            base_faces,
            orig_indices,
            k=1,
            n_samples_per_ring=None,
            verbose=False,
        )

        # Note: Even for perfectly flat surfaces, there's some distortion
        # due to the graph distance correction factor used in geodesic computation
        # (graph distances along mesh edges differ from Euclidean distances)
        assert vertex_dist.shape == (4,)
        assert np.all(np.isfinite(vertex_dist))
        assert np.isfinite(mean_dist)
        # All vertices should have similar distortion for a symmetric mesh
        assert np.std(vertex_dist) < 5.0  # Low variance across vertices

    def test_distortion_with_scaling(self):
        """Test that scaled 2D coords produce different distortion than unscaled."""
        base_vertices, base_faces, orig_indices = self._make_simple_mesh()

        # Baseline distortion with matching coordinates
        xy_baseline = base_vertices[:, :2]
        _, mean_dist_baseline = compute_kring_distortion(
            xy_baseline,
            base_vertices,
            base_faces,
            orig_indices,
            k=1,
            n_samples_per_ring=None,
            verbose=False,
        )

        # 2D coordinates scaled by 2x - distances should be doubled
        xy_scaled = base_vertices[:, :2] * 2.0
        vertex_dist, mean_dist_scaled = compute_kring_distortion(
            xy_scaled,
            base_vertices,
            base_faces,
            orig_indices,
            k=1,
            n_samples_per_ring=None,
            verbose=False,
        )

        # Scaling should change the distortion significantly
        assert vertex_dist.shape == (4,)
        assert np.all(np.isfinite(vertex_dist))
        # Scaled distortion should differ from baseline
        assert abs(mean_dist_scaled - mean_dist_baseline) > 10.0

    def test_different_k_values(self):
        """Test that different k values produce valid results."""
        base_vertices, base_faces, orig_indices = self._make_simple_mesh()
        xy = base_vertices[:, :2]

        # k=1
        vertex_dist_k1, mean_dist_k1 = compute_kring_distortion(
            xy,
            base_vertices,
            base_faces,
            orig_indices,
            k=1,
            n_samples_per_ring=None,
            verbose=False,
        )

        # k=2
        vertex_dist_k2, mean_dist_k2 = compute_kring_distortion(
            xy,
            base_vertices,
            base_faces,
            orig_indices,
            k=2,
            n_samples_per_ring=None,
            verbose=False,
        )

        # Both should produce valid arrays
        assert vertex_dist_k1.shape == (4,)
        assert vertex_dist_k2.shape == (4,)
        # k=2 includes more neighbors, so may differ from k=1
        # Just check they are valid (not NaN/Inf)
        assert np.all(np.isfinite(vertex_dist_k1))
        assert np.all(np.isfinite(vertex_dist_k2))

    def test_angular_sampling(self):
        """Test that angular sampling parameter works."""
        base_vertices, base_faces, orig_indices = self._make_simple_mesh()
        xy = base_vertices[:, :2]

        # With angular sampling
        vertex_dist, mean_dist = compute_kring_distortion(
            xy,
            base_vertices,
            base_faces,
            orig_indices,
            k=2,
            n_samples_per_ring=4,
            verbose=False,
        )

        assert vertex_dist.shape == (4,)
        assert np.all(np.isfinite(vertex_dist))

    def test_isolated_vertex(self):
        """Test handling of isolated vertices with no neighbors."""
        # Create a mesh where one vertex is disconnected
        base_vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.5, 0.5, 0.0],
                [10.0, 10.0, 0.0],  # Isolated vertex (far away, not in any face)
            ]
        )
        # Only one triangle, vertex 3 is not connected
        base_faces = np.array([[0, 1, 2]])
        # Include all vertices in the patch
        orig_indices = np.array([0, 1, 2, 3])
        xy = base_vertices[:, :2]

        vertex_dist, mean_dist = compute_kring_distortion(
            xy,
            base_vertices,
            base_faces,
            orig_indices,
            k=1,
            n_samples_per_ring=None,
            verbose=False,
        )

        # Isolated vertex should have 0 distortion (no neighbors)
        assert vertex_dist[3] == 0.0

    def test_zero_target_distances(self):
        """Test handling of zero target distances (division by zero protection)."""
        # Create a degenerate case where vertices are at the same position
        base_vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],  # Same position as vertex 0
                [0.0, 0.0, 0.0],  # Same position as vertex 0
            ]
        )
        base_faces = np.array([[0, 1, 2]])
        orig_indices = np.array([0, 1, 2])
        xy = np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        )

        # Should not raise division by zero
        vertex_dist, mean_dist = compute_kring_distortion(
            xy,
            base_vertices,
            base_faces,
            orig_indices,
            k=1,
            n_samples_per_ring=None,
            verbose=False,
        )

        # All should be finite (no NaN from division by zero)
        assert np.all(np.isfinite(vertex_dist))
        assert np.isfinite(mean_dist)

    def test_output_shapes(self):
        """Test that output shapes are correct."""
        base_vertices, base_faces, orig_indices = self._make_simple_mesh()
        xy = base_vertices[:, :2]

        vertex_dist, mean_dist = compute_kring_distortion(
            xy,
            base_vertices,
            base_faces,
            orig_indices,
            k=1,
            n_samples_per_ring=None,
            verbose=False,
        )

        assert vertex_dist.shape == (len(orig_indices),)
        assert isinstance(mean_dist, float)


class TestLoadCurvature:
    """Tests for load_curvature function."""

    def test_load_curvature_returns_array(self):
        """Test that load_curvature returns a numpy array."""
        import nibabel
        import struct

        # Create a minimal curv file in FreeSurfer format
        # FreeSurfer curv format: 3 bytes magic, 4 bytes n_vertices, 4 bytes n_faces,
        # 4 bytes vals_per_vertex, then n_vertices floats
        n_vertices = 10
        curv_values = np.random.randn(n_vertices).astype(np.float32)

        with tempfile.NamedTemporaryFile(suffix=".curv", delete=False) as f:
            temp_path = f.name
            # Write new-style curv file
            # Magic number (3 bytes): 0xff 0xff 0xff
            f.write(b"\xff\xff\xff")
            # Number of vertices (4 bytes big-endian)
            f.write(struct.pack(">i", n_vertices))
            # Number of faces (4 bytes big-endian)
            f.write(struct.pack(">i", 0))
            # Values per vertex (4 bytes big-endian)
            f.write(struct.pack(">i", 1))
            # Curvature values (big-endian floats)
            for v in curv_values:
                f.write(struct.pack(">f", v))

        try:
            result = load_curvature(temp_path)
            assert isinstance(result, np.ndarray)
            assert result.shape == (n_vertices,)
            assert np.allclose(result, curv_values, atol=1e-6)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


class TestGetViewAngles:
    """Tests for _get_view_angles function."""

    def test_medial_view_lh(self):
        """Test medial view angles for left hemisphere."""
        elev, azim = _get_view_angles("lh", "medial")
        assert elev == 0
        assert azim == 0

    def test_medial_view_rh(self):
        """Test medial view angles for right hemisphere."""
        elev, azim = _get_view_angles("rh", "medial")
        assert elev == 0
        assert azim == 180

    def test_ventral_view_lh(self):
        """Test ventral view angles for left hemisphere."""
        elev, azim = _get_view_angles("lh", "ventral")
        assert elev == -90
        assert azim == 180

    def test_ventral_view_rh(self):
        """Test ventral view angles for right hemisphere."""
        elev, azim = _get_view_angles("rh", "ventral")
        assert elev == -90
        assert azim == 180

    def test_frontal_view_lh(self):
        """Test frontal view angles for left hemisphere."""
        elev, azim = _get_view_angles("lh", "frontal")
        assert elev == 0
        assert azim == -90

    def test_frontal_view_rh(self):
        """Test frontal view angles for right hemisphere."""
        elev, azim = _get_view_angles("rh", "frontal")
        assert elev == 0
        assert azim == -90

    def test_invalid_view_raises(self):
        """Test that invalid view type raises ValueError."""
        import pytest

        with pytest.raises(ValueError, match="Unknown view type"):
            _get_view_angles("lh", "invalid")


class TestPlotProjection:
    """Tests for plot_projection function."""

    @staticmethod
    def _create_mock_subject_dir(tmp_path, hemi="lh"):
        """Create a mock subject directory with minimal surface data.

        Creates a simple triangular mesh with 4 vertices and 2 faces.
        """
        import struct
        import nibabel.freesurfer.io as fsio

        from autoflatten.freesurfer import write_patch

        subject_dir = tmp_path / "test_subject"
        surf_dir = subject_dir / "surf"
        surf_dir.mkdir(parents=True)

        # Create a simple mesh: 4 vertices forming a square, split into 2 triangles
        vertices = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float32,
        )
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)

        # Write inflated surface using nibabel
        inflated_path = surf_dir / f"{hemi}.inflated"
        fsio.write_geometry(str(inflated_path), vertices, faces)

        # Write curvature file (binary format)
        curv_path = surf_dir / f"{hemi}.curv"
        n_vertices = len(vertices)
        curv_values = np.array([0.1, -0.1, 0.2, -0.2], dtype=np.float32)
        with open(curv_path, "wb") as f:
            # Magic number (3 bytes)
            f.write(b"\xff\xff\xff")
            # Number of vertices (4 bytes big-endian)
            f.write(struct.pack(">i", n_vertices))
            # Number of faces (4 bytes big-endian)
            f.write(struct.pack(">i", 0))
            # Values per vertex (4 bytes big-endian)
            f.write(struct.pack(">i", 1))
            # Curvature values (big-endian floats)
            for v in curv_values:
                f.write(struct.pack(">f", v))

        # Create a patch file with only 3 of the 4 vertices (simulating a cut)
        patch_path = surf_dir / f"{hemi}.autoflatten.patch.3d"
        patch_vertices = vertices[:3]  # First 3 vertices
        patch_indices = np.array([0, 1, 2], dtype=np.int32)
        write_patch(str(patch_path), patch_vertices, patch_indices)

        return subject_dir, patch_path

    def test_plot_projection_returns_figure(self, tmp_path):
        """Test that plot_projection returns a matplotlib figure when no output path."""
        import matplotlib.pyplot as plt

        subject_dir, patch_path = self._create_mock_subject_dir(tmp_path, hemi="lh")

        fig = plot_projection(
            patch_path=str(patch_path),
            subject_dir=str(subject_dir),
            output_path=None,
        )

        assert hasattr(fig, "savefig")  # It's a matplotlib figure
        plt.close(fig)

    def test_plot_projection_saves_file(self, tmp_path):
        """Test that plot_projection saves file when output_path is given."""
        import matplotlib

        matplotlib.use("Agg")  # Use non-interactive backend

        subject_dir, patch_path = self._create_mock_subject_dir(tmp_path, hemi="lh")
        output_path = tmp_path / "test_output.png"

        result = plot_projection(
            patch_path=str(patch_path),
            subject_dir=str(subject_dir),
            output_path=str(output_path),
        )

        assert result == str(output_path)
        assert output_path.exists()

    def test_plot_projection_detects_hemisphere_lh(self, tmp_path):
        """Test that left hemisphere is correctly detected from filename."""
        import matplotlib.pyplot as plt

        subject_dir, patch_path = self._create_mock_subject_dir(tmp_path, hemi="lh")

        # Should not raise - hemisphere detected from "lh." prefix
        fig = plot_projection(
            patch_path=str(patch_path),
            subject_dir=str(subject_dir),
        )
        plt.close(fig)

    def test_plot_projection_detects_hemisphere_rh(self, tmp_path):
        """Test that right hemisphere is correctly detected from filename."""
        import matplotlib.pyplot as plt

        subject_dir, patch_path = self._create_mock_subject_dir(tmp_path, hemi="rh")

        # Should not raise - hemisphere detected from "rh." prefix
        fig = plot_projection(
            patch_path=str(patch_path),
            subject_dir=str(subject_dir),
        )
        plt.close(fig)

    def test_plot_projection_invalid_hemisphere_raises(self, tmp_path):
        """Test that invalid hemisphere prefix raises ValueError."""
        import pytest

        subject_dir, patch_path = self._create_mock_subject_dir(tmp_path, hemi="lh")

        # Rename patch to have invalid prefix
        invalid_patch = tmp_path / "invalid.patch.3d"
        patch_path.rename(invalid_patch)

        with pytest.raises(ValueError, match="Cannot determine hemisphere"):
            plot_projection(
                patch_path=str(invalid_patch),
                subject_dir=str(subject_dir),
            )

    def test_plot_projection_missing_inflated_raises(self, tmp_path):
        """Test that missing inflated surface raises FileNotFoundError."""
        import pytest

        subject_dir, patch_path = self._create_mock_subject_dir(tmp_path, hemi="lh")

        # Remove the inflated surface
        inflated_path = subject_dir / "surf" / "lh.inflated"
        inflated_path.unlink()

        with pytest.raises(FileNotFoundError, match="Inflated surface not found"):
            plot_projection(
                patch_path=str(patch_path),
                subject_dir=str(subject_dir),
            )

    def test_plot_projection_auto_detect_subject_dir(self, tmp_path):
        """Test that subject_dir is auto-detected when patch is in surf/ directory."""
        import matplotlib.pyplot as plt

        subject_dir, patch_path = self._create_mock_subject_dir(tmp_path, hemi="lh")

        # Don't pass subject_dir - should auto-detect from patch location
        fig = plot_projection(
            patch_path=str(patch_path),
            subject_dir=None,  # Should auto-detect
        )
        plt.close(fig)

    def test_plot_projection_auto_detect_fails_for_non_surf_path(self, tmp_path):
        """Test that auto-detect fails when patch is not in surf/ directory."""
        import pytest

        subject_dir, patch_path = self._create_mock_subject_dir(tmp_path, hemi="lh")

        # Move patch to a non-surf location
        new_patch_path = tmp_path / "lh.autoflatten.patch.3d"
        patch_path.rename(new_patch_path)

        with pytest.raises(ValueError, match="Cannot auto-detect subject directory"):
            plot_projection(
                patch_path=str(new_patch_path),
                subject_dir=None,
            )

    def test_plot_projection_overwrite_false_skips(self, tmp_path, capsys):
        """Test that existing output is skipped when overwrite=False."""
        subject_dir, patch_path = self._create_mock_subject_dir(tmp_path, hemi="lh")
        output_path = tmp_path / "existing_output.png"

        # Create an existing file
        output_path.write_text("existing content")

        result = plot_projection(
            patch_path=str(patch_path),
            subject_dir=str(subject_dir),
            output_path=str(output_path),
            overwrite=False,
        )

        assert result == str(output_path)
        # Check that file was not modified
        assert output_path.read_text() == "existing content"
        # Check that message was printed
        captured = capsys.readouterr()
        assert "already exists" in captured.out

    def test_plot_projection_overwrite_true_regenerates(self, tmp_path):
        """Test that existing output is regenerated when overwrite=True."""
        import matplotlib

        matplotlib.use("Agg")

        subject_dir, patch_path = self._create_mock_subject_dir(tmp_path, hemi="lh")
        output_path = tmp_path / "existing_output.png"

        # Create an existing file
        output_path.write_text("existing content")
        original_size = output_path.stat().st_size

        result = plot_projection(
            patch_path=str(patch_path),
            subject_dir=str(subject_dir),
            output_path=str(output_path),
            overwrite=True,
        )

        assert result == str(output_path)
        # Check that file was modified (should be a real PNG now)
        assert output_path.stat().st_size != original_size

    def test_plot_projection_custom_title(self, tmp_path):
        """Test that custom title is used."""
        import matplotlib.pyplot as plt

        subject_dir, patch_path = self._create_mock_subject_dir(tmp_path, hemi="lh")

        fig = plot_projection(
            patch_path=str(patch_path),
            subject_dir=str(subject_dir),
            title="Custom Title",
        )

        # Check that the figure has our custom title
        assert fig._suptitle.get_text() == "Custom Title"
        plt.close(fig)

    def test_plot_projection_missing_curvature_uses_fallback(self, tmp_path, capsys):
        """Test that missing curvature file uses uniform gray fallback."""
        import matplotlib.pyplot as plt

        subject_dir, patch_path = self._create_mock_subject_dir(tmp_path, hemi="lh")

        # Remove curvature file
        curv_path = subject_dir / "surf" / "lh.curv"
        curv_path.unlink()

        fig = plot_projection(
            patch_path=str(patch_path),
            subject_dir=str(subject_dir),
        )

        plt.close(fig)

        # Check warning was printed
        captured = capsys.readouterr()
        assert "Curvature file not found" in captured.out
        assert "uniform gray" in captured.out
