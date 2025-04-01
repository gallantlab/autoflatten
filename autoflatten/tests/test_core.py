"""
Tests for the core module.
"""

import os
import subprocess
import tempfile
from unittest.mock import MagicMock, call, patch

import networkx as nx
import numpy as np
import pytest

from autoflatten.core import ensure_continuous_cuts, map_cuts_to_subject
from autoflatten.freesurfer import is_freesurfer_available

# Mark tests to skip if FreeSurfer is not available
requires_freesurfer = pytest.mark.skipif(
    not is_freesurfer_available(), reason="FreeSurfer is not installed or not in PATH"
)


@pytest.fixture
def mock_surface_data():
    """
    Create mock surface data for testing.

    Returns
    -------
    dict
        Dictionary containing vertices, faces, and vertex dictionary
    """
    # Create a small set of vertices and faces for testing
    # Using a simple mesh structure with two disconnected components
    vertices_inflated = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0],  # First component
            [3, 0, 0],
            [4, 0, 0],
            [3, 1, 0],
            [4, 1, 0],  # Second component
        ]
    )

    vertices_fiducial = vertices_inflated.copy()  # Same for simplicity

    # Faces connecting vertices (triangles)
    faces = np.array(
        [
            [0, 1, 2],
            [1, 3, 2],  # First component
            [4, 5, 6],
            [5, 7, 6],  # Second component
        ]
    )

    # Create vertex dictionary with disconnected cuts
    vertex_dict = {
        "mwall": np.array([]),  # Empty medial wall for simplicity
        "cut1": np.array([1, 3]),  # First component vertices
        "cut2": np.array([5, 7]),  # Second component vertices
        "cut3": np.array([]),  # Empty cut
        "cut4": np.array([]),  # Empty cut
        "cut5": np.array([]),  # Empty cut
    }

    return {
        "vertices_inflated": vertices_inflated,
        "vertices_fiducial": vertices_fiducial,
        "faces": faces,
        "vertex_dict": vertex_dict,
    }


def test_ensure_continuous_cuts(mock_surface_data, monkeypatch):
    """
    Test ensure_continuous_cuts function.

    This test verifies that the function correctly identifies
    disconnected cuts and handles them appropriately.
    """

    # Mock cortex.db.get_surf function
    def mock_get_surf(subject, surf_type, hemisphere=None):
        if surf_type == "inflated":
            return mock_surface_data["vertices_inflated"], mock_surface_data["faces"]
        elif surf_type == "fiducial":
            return mock_surface_data["vertices_fiducial"], mock_surface_data["faces"]
        else:
            raise ValueError(f"Unexpected surface type: {surf_type}")

    # Apply the monkeypatch
    monkeypatch.setattr("cortex.db.get_surf", mock_get_surf)

    # Run the function with the mock data
    vertex_dict = mock_surface_data["vertex_dict"].copy()
    result = ensure_continuous_cuts(vertex_dict, "test_subject", "lh")

    # Verify that the resulting dictionary contains the expected keys
    assert "cut1" in result
    assert "cut2" in result
    assert "cut3" in result
    assert "cut4" in result
    assert "cut5" in result

    # The function identifies each cut as continuous since they are within their
    # own connected components. Therefore, we expect the cuts to stay separate
    # and maintain their original vertices.
    assert np.array_equal(result["cut1"], vertex_dict["cut1"])
    assert np.array_equal(result["cut2"], vertex_dict["cut2"])


@pytest.fixture
def mock_surface_data_with_disconnected_cut():
    """
    Create mock surface data with a single connected component
    and a disconnected cut within it.

    Returns
    -------
    dict
        Dictionary containing vertices, faces, and vertex dictionary
    """
    # Create a connected set of vertices in a 3x3 grid
    vertices_inflated = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [2, 0, 0],  # Row 1
            [0, 1, 0],
            [1, 1, 0],
            [2, 1, 0],  # Row 2
            [0, 2, 0],
            [1, 2, 0],
            [2, 2, 0],  # Row 3
        ]
    )

    vertices_fiducial = vertices_inflated.copy()

    # Faces connecting vertices to form a grid
    faces = np.array(
        [
            # Bottom row of squares, each split into two triangles
            [0, 1, 3],
            [1, 4, 3],
            [1, 2, 4],
            [2, 5, 4],
            # Middle row of squares
            [3, 4, 6],
            [4, 7, 6],
            [4, 5, 7],
            [5, 8, 7],
        ]
    )

    # Create a disconnected cut within the same component
    # Vertices 0 and 8 are part of cut1 but have no direct connection
    vertex_dict = {
        "mwall": np.array([]),
        "cut1": np.array([0, 8]),  # Disconnected vertices in the same component
        "cut2": np.array([]),
        "cut3": np.array([]),
        "cut4": np.array([]),
        "cut5": np.array([]),
    }

    return {
        "vertices_inflated": vertices_inflated,
        "vertices_fiducial": vertices_fiducial,
        "faces": faces,
        "vertex_dict": vertex_dict,
    }


def test_ensure_continuous_cuts_with_disconnected_cut(
    mock_surface_data_with_disconnected_cut, monkeypatch
):
    """
    Test ensure_continuous_cuts function with a disconnected cut in a single component.

    This test verifies that the function correctly adds vertices to make disconnected cuts continuous.
    """

    # Mock cortex.db.get_surf function
    def mock_get_surf(subject, surf_type, hemisphere=None):
        if surf_type == "inflated":
            return mock_surface_data_with_disconnected_cut[
                "vertices_inflated"
            ], mock_surface_data_with_disconnected_cut["faces"]
        elif surf_type == "fiducial":
            return mock_surface_data_with_disconnected_cut[
                "vertices_fiducial"
            ], mock_surface_data_with_disconnected_cut["faces"]
        else:
            raise ValueError(f"Unexpected surface type: {surf_type}")

    # Apply the monkeypatch
    monkeypatch.setattr("cortex.db.get_surf", mock_get_surf)

    # Run the function with the mock data
    vertex_dict = mock_surface_data_with_disconnected_cut["vertex_dict"].copy()
    original_vertices = len(vertex_dict["cut1"])
    result = ensure_continuous_cuts(vertex_dict, "test_subject", "lh")

    # Verify the result
    assert "cut1" in result

    # Check that originally disconnected cut is now connected
    # The resulting cut should include the original vertices plus connecting vertices
    assert len(result["cut1"]) > original_vertices

    # Create a graph from the mock surface to verify connectivity
    G = nx.Graph()
    vertices_fiducial = mock_surface_data_with_disconnected_cut["vertices_fiducial"]
    faces = mock_surface_data_with_disconnected_cut["faces"]

    # Add edges from faces
    for face in faces:
        for i in range(3):
            v1 = face[i]
            for j in range(i + 1, 3):
                v2 = face[j]
                weight = np.linalg.norm(vertices_fiducial[v1] - vertices_fiducial[v2])
                G.add_edge(v1, v2, weight=weight)

    # Extract the subgraph for the resulting cut1
    subgraph = G.subgraph(result["cut1"])

    # Check that the original disconnected vertices are now connected
    if len(vertex_dict["cut1"]) >= 2:
        v1 = vertex_dict["cut1"][0]
        v2 = vertex_dict["cut1"][1]

        # Both vertices should be in the resulting cut
        assert v1 in result["cut1"]
        assert v2 in result["cut1"]

        # There should be a path between them in the resulting subgraph
        assert nx.has_path(subgraph, v1, v2)


@requires_freesurfer
def test_map_cuts_to_subject_with_freesurfer():
    """
    Test map_cuts_to_subject with actual FreeSurfer.

    This test is skipped if FreeSurfer is not available.
    """
    # Create test vertex dictionary
    vertex_dict = {
        "mwall": np.array([0, 1, 2]),
        "cut1": np.array([3, 4, 5]),
    }

    # Run with actual FreeSurfer (this will be skipped if FreeSurfer is not available)
    result = map_cuts_to_subject(vertex_dict, "fsaverage", "lh")

    # Basic validation
    assert isinstance(result, dict)
    assert "mwall" in result
    assert "cut1" in result


def test_map_cuts_to_subject_mocked():
    """
    Test map_cuts_to_subject with mocked FreeSurfer commands.

    This test mocks the subprocess and FreeSurfer label functions to verify
    that map_cuts_to_subject correctly interacts with FreeSurfer commands.
    """
    # Create test vertex dictionary
    vertex_dict = {
        "mwall": np.array([0, 1, 2]),
        "cut1": np.array([3, 4, 5]),
        "empty_cut": np.array([]),
    }

    # Mock the functions that interact with FreeSurfer
    with (
        patch("autoflatten.core.create_label_file") as mock_create_label,
        patch("autoflatten.core.read_freesurfer_label") as mock_read_label,
        patch("subprocess.run") as mock_subprocess,
    ):
        # Configure mocks
        mock_create_label.return_value = "/tmp/temp_label.label"
        mock_read_label.return_value = np.array([10, 11, 12])
        mock_subprocess.return_value = MagicMock(returncode=0)

        # Run the function
        result = map_cuts_to_subject(vertex_dict, "test_subject", "lh", "fsaverage")

        # Verify the results
        assert isinstance(result, dict)
        assert "mwall" in result
        assert "cut1" in result
        assert "empty_cut" in result

        # Check mock calls
        assert (
            mock_create_label.call_count == 2
        )  # Called for mwall and cut1, not for empty_cut
        assert mock_read_label.call_count == 2
        assert mock_subprocess.call_count == 2

        # Verify first call to mri_label2label (for mwall)
        first_call = mock_subprocess.call_args_list[0]
        cmd = first_call[0][0]
        assert "mri_label2label" in cmd
        assert "--srcsubject" in cmd
        assert "fsaverage" in cmd
        assert "--trgsubject" in cmd
        assert "test_subject" in cmd
        assert "--hemi" in cmd
        assert "lh" in cmd

        # Check the mapped vertices for mwall
        assert np.array_equal(result["mwall"], np.array([10, 11, 12]))

        # Check that empty_cut is still empty
        assert len(result["empty_cut"]) == 0


def test_map_cuts_to_subject_error_handling():
    """
    Test error handling in map_cuts_to_subject.

    This test verifies that the function properly handles errors from
    subprocess calls and file operations.
    """
    # Create test vertex dictionary
    vertex_dict = {
        "mwall": np.array([0, 1, 2]),
        "cut1": np.array([3, 4, 5]),
    }

    # Test subprocess error
    with (
        patch("autoflatten.core.create_label_file") as mock_create_label,
        patch("subprocess.run") as mock_subprocess,
    ):
        # Configure mocks
        mock_create_label.return_value = "/tmp/temp_label.label"
        mock_subprocess.side_effect = subprocess.CalledProcessError(
            cmd=["mri_label2label"], returncode=1, output=b"", stderr=b"Test error"
        )

        # Run the function
        result = map_cuts_to_subject(vertex_dict, "test_subject", "lh", "fsaverage")

        # Verify that we get empty arrays for the cuts due to the error
        assert isinstance(result, dict)
        assert "mwall" in result
        assert "cut1" in result
        assert len(result["mwall"]) == 0
        assert len(result["cut1"]) == 0
