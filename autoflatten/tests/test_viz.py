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
