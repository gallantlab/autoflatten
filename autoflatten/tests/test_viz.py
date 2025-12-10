"""
Tests for the viz module (matplotlib-based visualization).
"""

import os
import tempfile

import numpy as np

from autoflatten.viz import compute_triangle_areas, parse_log_file


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
