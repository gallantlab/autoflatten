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

from autoflatten.core import (
    _find_trapped_vertices,
    ensure_continuous_cuts,
    fill_holes_in_patch,
    map_cuts_to_subject,
    refine_cuts_with_geodesic,
)
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

    # Mock load_surface function directly at the module level where it's imported
    def mock_load_surface(subject, surf_type, hemi, subjects_dir=None):
        if surf_type == "inflated":
            return mock_surface_data["vertices_inflated"], mock_surface_data["faces"]
        elif surf_type == "fiducial":
            return mock_surface_data["vertices_fiducial"], mock_surface_data["faces"]
        else:
            raise ValueError(f"Unexpected surface type: {surf_type}")

    # Apply the monkeypatch to the imported function in core.py
    monkeypatch.setattr("autoflatten.core.load_surface", mock_load_surface)

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
    # Vertices 0 and 8 are part of calcarine but have no direct connection
    vertex_dict = {
        "mwall": np.array([]),
        "calcarine": np.array([0, 8]),  # Disconnected vertices in the same component
        "medial1": np.array([]),
        "medial2": np.array([]),
        "medial3": np.array([]),
        "temporal": np.array([]),
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

    # Mock load_surface function directly at the module level where it's imported
    def mock_load_surface(subject, surf_type, hemi, subjects_dir=None):
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

    # Apply the monkeypatch to the imported function in core.py
    monkeypatch.setattr("autoflatten.core.load_surface", mock_load_surface)

    # Run the function with the mock data
    vertex_dict = mock_surface_data_with_disconnected_cut["vertex_dict"].copy()
    original_vertices = len(vertex_dict["calcarine"])
    result = ensure_continuous_cuts(vertex_dict, "test_subject", "lh")

    # Verify the result
    assert "calcarine" in result

    # Check that originally disconnected cut is now connected
    # The resulting cut should include the original vertices plus connecting vertices
    assert len(result["calcarine"]) > original_vertices

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

    # Extract the subgraph for the resulting calcarine cut
    subgraph = G.subgraph(result["calcarine"])

    # Check that the original disconnected vertices are now connected
    if len(vertex_dict["calcarine"]) >= 2:
        v1 = vertex_dict["calcarine"][0]
        v2 = vertex_dict["calcarine"][1]

        # Both vertices should be in the resulting cut
        assert v1 in result["calcarine"]
        assert v2 in result["calcarine"]

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


@pytest.fixture
def mock_grid_surface():
    """
    Create a 5x5 grid surface for testing geodesic refinement.

    Grid layout (vertex indices):
        20 21 22 23 24
        15 16 17 18 19
        10 11 12 13 14
         5  6  7  8  9
         0  1  2  3  4

    Each unit is 1mm apart.
    """
    # Create 5x5 grid of vertices
    vertices = np.array([[x, y, 0] for y in range(5) for x in range(5)], dtype=float)

    # Create triangular faces for the grid
    faces = []
    for y in range(4):
        for x in range(4):
            v0 = y * 5 + x
            v1 = v0 + 1
            v2 = v0 + 5
            v3 = v2 + 1
            faces.append([v0, v1, v2])
            faces.append([v1, v3, v2])
    faces = np.array(faces)

    return {"vertices": vertices, "faces": faces}


def test_refine_cuts_degenerate_endpoints(mock_grid_surface, monkeypatch):
    """
    Test that geodesic refinement handles degenerate cases where
    medial wall border endpoints are too close (<5mm).

    When the two mwall border endpoints are adjacent, the function should
    fall back to using one mwall border endpoint + farthest cut vertex.
    """
    vertices = mock_grid_surface["vertices"]
    faces = mock_grid_surface["faces"]

    # Mock load_surface to return our grid
    def mock_load_surface(subject, surf_type, hemi, subjects_dir=None):
        return vertices, faces

    monkeypatch.setattr("autoflatten.core.load_surface", mock_load_surface)

    # Create a cut that has two adjacent mwall border endpoints
    # mwall is the bottom row (0-4), cut touches mwall at vertices 1 and 2 (adjacent)
    # but extends up to vertex 17
    vertex_dict = {
        "mwall": np.array([0, 1, 2, 3, 4]),  # Bottom row
        "calcarine": np.array([6, 7, 11, 12, 17]),  # Cut: vertices 6,7 border mwall
    }

    result = refine_cuts_with_geodesic(
        vertex_dict, "test_subject", "lh", medial_wall_vertices=vertex_dict["mwall"]
    )

    # The refined cut should have more than 2 vertices
    # (if it had only 2, it means we used the degenerate adjacent endpoints)
    assert len(result["calcarine"]) > 2, (
        f"Refined cut has only {len(result['calcarine'])} vertices - "
        "degenerate endpoints not handled correctly"
    )

    # The refined cut should span a reasonable distance
    cut_vertices = result["calcarine"]
    cut_positions = vertices[cut_vertices]
    max_span = np.max(np.linalg.norm(cut_positions - cut_positions[0], axis=1))
    assert max_span > 2.0, f"Cut span {max_span} is too small"


def test_refine_cuts_no_mwall_border(mock_grid_surface, monkeypatch):
    """
    Test that geodesic refinement handles cuts with no medial wall border vertices.

    When a cut has no vertices adjacent to the medial wall, the function should
    extend the cut to connect to the nearest medial wall border.
    """
    vertices = mock_grid_surface["vertices"]
    faces = mock_grid_surface["faces"]

    def mock_load_surface(subject, surf_type, hemi, subjects_dir=None):
        return vertices, faces

    monkeypatch.setattr("autoflatten.core.load_surface", mock_load_surface)

    # Create a cut that is completely disconnected from mwall
    # mwall is the bottom row (0-4), cut is in the middle of the grid
    vertex_dict = {
        "mwall": np.array([0, 1, 2, 3, 4]),  # Bottom row
        "medial1": np.array([12, 13, 17, 18]),  # Cut in the middle, not touching mwall
    }

    result = refine_cuts_with_geodesic(
        vertex_dict, "test_subject", "lh", medial_wall_vertices=vertex_dict["mwall"]
    )

    # The refined cut should extend to connect to the medial wall
    cut_vertices = result["medial1"]

    # Check that the path includes a vertex near the medial wall border
    # The mwall border vertices are 5, 6, 7, 8, 9 (row above mwall)
    mwall_border = {5, 6, 7, 8, 9}
    has_mwall_connection = any(v in mwall_border for v in cut_vertices)

    assert has_mwall_connection, (
        f"Refined cut {list(cut_vertices)} does not connect to medial wall border"
    )


def test_refine_cuts_does_not_modify_input(mock_grid_surface, monkeypatch):
    """
    Test that refine_cuts_with_geodesic does not modify the input dict.
    """
    vertices = mock_grid_surface["vertices"]
    faces = mock_grid_surface["faces"]

    def mock_load_surface(subject, surf_type, hemi, subjects_dir=None):
        return vertices, faces

    monkeypatch.setattr("autoflatten.core.load_surface", mock_load_surface)

    vertex_dict = {
        "mwall": np.array([0, 1, 2, 3, 4]),
        "calcarine": np.array([6, 7, 11, 12]),
    }
    original_calcarine = vertex_dict["calcarine"].copy()

    result = refine_cuts_with_geodesic(
        vertex_dict, "test_subject", "lh", medial_wall_vertices=vertex_dict["mwall"]
    )

    # Input should be unchanged
    assert np.array_equal(vertex_dict["calcarine"], original_calcarine), (
        "Input vertex_dict was modified in-place"
    )

    # Result should be a different object
    assert result is not vertex_dict


# Tests for fill_holes_in_patch


def test_fill_holes_in_patch_no_holes():
    """Test fill_holes_in_patch returns empty set when no holes exist."""
    # Create a simple triangular mesh (disk topology, one boundary loop)
    # Vertices: 0-4 form outer boundary in a pentagon, 5 is center
    #       0
    #      /|\
    #     / 5 \
    #    4     1
    #     \   /
    #      3-2
    # Fan triangulation: [0,1,5], [1,2,5], [2,3,5], [3,4,5], [4,0,5]
    faces = np.array(
        [
            [0, 1, 5],
            [1, 2, 5],
            [2, 3, 5],
            [3, 4, 5],
            [4, 0, 5],
        ]
    )
    excluded_vertices = set()

    result = fill_holes_in_patch(faces, excluded_vertices)

    assert result == set(), "Expected empty set when no holes exist"


def test_fill_holes_in_patch_with_hole():
    """Test fill_holes_in_patch detects and fills a hole."""
    # Create a mesh with a hole (annulus topology, two boundary loops)
    # Outer vertices: 0-5, Inner vertices: 6-8 (the hole)
    faces = np.array(
        [
            # Outer ring connected to inner ring
            [0, 1, 6],
            [1, 7, 6],
            [1, 2, 7],
            [2, 8, 7],
            [2, 3, 8],
            [3, 6, 8],
            [3, 4, 6],
            [4, 0, 6],
        ]
    )
    excluded_vertices = set()

    result = fill_holes_in_patch(faces, excluded_vertices)

    # The inner loop (vertices 6, 7, 8) should be detected as a hole
    # and its boundary vertices should be returned
    assert len(result) > 0, "Expected hole vertices to be filled"


def test_fill_holes_in_patch_all_vertices_excluded():
    """Test fill_holes_in_patch handles case where all vertices are excluded."""
    # Create simple faces
    faces = np.array([[0, 1, 2], [1, 2, 3]])
    # Exclude all vertices - no patch faces remain
    excluded_vertices = {0, 1, 2, 3}

    result = fill_holes_in_patch(faces, excluded_vertices)

    # Should return empty set when no patch faces exist
    assert result == set()


# Tests for _find_trapped_vertices


def test_find_trapped_vertices_no_trapped():
    """Test _find_trapped_vertices returns empty when neighbors are well-connected."""
    # Create a large graph where anchor's neighbors connect to >100 vertices
    G = nx.Graph()
    # Create a 20x20 grid graph (400 vertices)
    for i in range(400):
        row, col = i // 20, i % 20
        if col < 19:
            G.add_edge(i, i + 1)
        if row < 19:
            G.add_edge(i, i + 20)

    # Exclude bottom row (0-19) and anchor at vertex 20
    excluded = set(range(21))  # 0-20 excluded
    mwall_set = set(range(20))  # Bottom row is mwall
    anchor = 20  # First vertex of row 2

    result = _find_trapped_vertices(G, excluded, mwall_set, anchor)

    # Vertex 21 (neighbor of anchor) can reach >100 vertices, so not trapped
    assert result == [], f"Expected no trapped vertices, got {result}"


def test_find_trapped_vertices_with_trapped():
    """Test _find_trapped_vertices identifies trapped vertices correctly."""
    # Create a graph where vertex 10 is nearly isolated
    # (only connected to the mesh through the excluded set)
    G = nx.Graph()
    # Create a small connected region
    G.add_edge(0, 1)
    G.add_edge(1, 2)
    G.add_edge(2, 3)
    G.add_edge(3, 0)  # Square of vertices 0-3
    # Add anchor vertex 4 connected to vertex 1
    G.add_edge(1, 4)
    # Add isolated vertex 10 only connected to anchor
    G.add_edge(4, 10)

    excluded = {0, 1, 2, 3, 4}
    mwall_set = {0, 1, 2, 3}
    anchor = 4

    result = _find_trapped_vertices(G, excluded, mwall_set, anchor)

    # Vertex 10 should be identified as trapped since it's only
    # connected through excluded vertices
    assert 10 in result, f"Expected vertex 10 to be trapped, got {result}"


def test_find_trapped_vertices_no_neighbors():
    """Test _find_trapped_vertices handles anchor with no non-excluded neighbors."""
    G = nx.Graph()
    G.add_edge(0, 1)
    G.add_edge(1, 2)

    excluded = {0, 1, 2}
    mwall_set = {0, 2}
    anchor = 1

    result = _find_trapped_vertices(G, excluded, mwall_set, anchor)

    # No neighbors outside excluded set, should return empty
    assert result == []


def test_fill_holes_in_patch_with_tjunction():
    """Test fill_holes_in_patch handles T-junctions (holes touching main boundary).

    A T-junction occurs when a boundary vertex has >2 neighbors, which happens
    when multiple boundary paths meet at a single vertex. This test creates a mesh
    where the boundary structure naturally has T-junctions due to how the triangles
    connect, and verifies that fill_holes_in_patch correctly identifies and excludes
    these T-junction vertices.

    The H-shaped mesh below creates T-junctions at vertices 4, 5, 6, 8, 9 where
    the horizontal bar meets the vertical bars.
    """
    # Create an H-shaped mesh that naturally has T-junctions in its boundary
    faces = np.array(
        [
            # Left vertical bar
            [0, 1, 4],
            [0, 4, 7],
            [4, 7, 8],
            [1, 4, 8],
            # Right vertical bar
            [2, 3, 6],
            [2, 6, 9],
            [6, 9, 10],
            [3, 6, 10],
            # Horizontal bar connecting left and right
            [4, 5, 8],
            [5, 6, 9],
            [4, 5, 6],
            [5, 8, 9],
        ]
    )

    excluded_vertices = set()

    result = fill_holes_in_patch(faces, excluded_vertices)

    # This mesh has T-junctions at vertices where the horizontal bar meets
    # the vertical bars. The T-junction detection should find and exclude them.
    # Key assertion: T-junction vertices should be detected
    assert len(result) > 0, (
        "Expected T-junction vertices to be detected in H-shaped mesh"
    )
    # The T-junctions are at the connection points
    possible_tjunctions = {4, 5, 6, 8, 9}
    assert result & possible_tjunctions, (
        f"Expected some T-junction vertices from {possible_tjunctions}, got {result}"
    )


def test_fill_holes_in_patch_simple_hole():
    """Test that fill_holes_in_patch detects simple holes without T-junctions.

    This creates a mesh with a simple internal hole (annulus topology) and verifies
    that the hole boundary vertices are correctly identified.
    """
    # Create an annulus mesh (ring shape with hole in center)
    # Outer ring: vertices 0-5
    # Inner ring (hole): vertices 6-8
    faces = np.array(
        [
            # Connect outer to inner ring
            [0, 1, 6],
            [1, 7, 6],
            [1, 2, 7],
            [2, 8, 7],
            [2, 3, 8],
            [3, 6, 8],
            [3, 4, 6],
            [4, 0, 6],
        ]
    )

    excluded_vertices = set()

    result = fill_holes_in_patch(faces, excluded_vertices)

    # The inner ring (vertices 6, 7, 8) should be detected as a hole
    inner_ring = {6, 7, 8}
    assert len(result) > 0, "Expected hole to be detected in annulus mesh"
    # The result should include at least some inner ring vertices
    assert result & inner_ring, (
        f"Expected inner ring vertices {inner_ring} to be in result, got {result}"
    )
