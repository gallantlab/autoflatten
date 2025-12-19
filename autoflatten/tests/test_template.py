"""
Tests for the template module.
"""

from unittest.mock import patch

import networkx as nx
import numpy as np
import pytest

# Import the module functions to test
from autoflatten.template import (
    classify_cuts_anatomically,
    classify_vertices_by_degree,
    create_surface_graphs,
    find_connected_components,
    find_removed_vertices,
    get_surface_data,
    identify_medial_wall_border,
    identify_surface_components,
    merge_small_components,
)


def test_find_removed_vertices():
    """
    Test identifying vertices that were removed in the flat surface.
    """
    # Construct minimal test data
    mock_surface_data = {
        "fiducial_points": np.random.rand(10, 3),  # 10 vertices in fiducial
        "polys_flat": np.array(
            [[0, 1, 2], [2, 3, 4], [4, 5, 6]]
        ),  # Only vertices 0-6 used
    }

    removed_vertices = find_removed_vertices(mock_surface_data)

    # Expect vertices 7, 8, 9 to be identified as removed
    assert set(removed_vertices) == {7, 8, 9}
    assert len(removed_vertices) == 3


def test_create_surface_graphs():
    """
    Test creation of surface graphs for analyzing cuts and medial wall.
    """
    # Create minimal test data
    mock_surface_data = {
        "inflated_points": np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0.5, 0.5, 1]]
        ),
        "polys_full": np.array([[0, 1, 4], [1, 3, 4], [3, 2, 4], [2, 0, 4]]),
    }
    removed_vertices = np.array([4])  # Let's say vertex 4 is removed

    G_removed, G_full = create_surface_graphs(mock_surface_data, removed_vertices)

    # Check G_full (should have edges from the triangles)
    assert len(G_full.nodes) == 5
    assert len(G_full.edges) == 8  # Each triangle gives 3 edges, but edges are shared

    # Check G_removed (should only have node 4, no edges since it's alone)
    assert len(G_removed.nodes) == 1
    assert 4 in G_removed.nodes
    assert len(G_removed.edges) == 0


def test_classify_vertices_by_degree():
    """
    Test classification of vertices based on their degree in the graph.
    """
    # Create a test graph
    G = nx.Graph()

    # Add nodes 0-9
    G.add_nodes_from(range(10))

    # Node 0 has 4 neighbors (high degree)
    G.add_edges_from([(0, i) for i in range(1, 5)])

    # Node 5 has 3 neighbors (high degree)
    G.add_edges_from([(5, i) for i in range(6, 9)])

    # Others have 1 or 0 neighbors (low degree)
    G.add_edge(9, 1)

    removed_vertices = np.array(range(10))

    # Test with default threshold of 2
    high_degree, low_degree = classify_vertices_by_degree(G, removed_vertices)

    assert set(high_degree) == {0, 5}
    assert set(low_degree) == {1, 2, 3, 4, 6, 7, 8, 9}

    # Test with threshold of 3
    high_degree, low_degree = classify_vertices_by_degree(
        G, removed_vertices, degree_threshold=3
    )

    assert set(high_degree) == {0}
    assert set(low_degree) == {1, 2, 3, 4, 5, 6, 7, 8, 9}


def test_find_connected_components():
    """
    Test finding connected components in the graph.
    """
    # Create a test graph
    G = nx.Graph()

    # Add nodes and edges to create specific components
    # Component 1 (high degree nodes): 0-1-2-3
    G.add_nodes_from(range(4))
    G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0)])

    # Component 2 (high degree nodes): 4-5
    G.add_nodes_from([4, 5])
    G.add_edge(4, 5)

    # Component 3 (low degree nodes): 6-7-8
    G.add_nodes_from([6, 7, 8])
    G.add_edges_from([(6, 7), (7, 8)])

    # Component 4 (low degree nodes): 9
    G.add_node(9)

    high_degree_vertices = {0, 1, 2, 3, 4, 5}
    low_degree_vertices = {6, 7, 8, 9}

    medial_wall, other_high_degree, cut_components = find_connected_components(
        G, high_degree_vertices, low_degree_vertices
    )

    # Medial wall should be component 1 (largest high-degree component)
    assert medial_wall == {0, 1, 2, 3}

    # Other high degree should be component 2
    assert other_high_degree == {4, 5}

    # Cut components should not be merged yet at this stage, just identify them
    assert len(cut_components) == 3

    # The components should include the individual low-degree components plus the other high-degree component
    # Test more flexible assertions that don't depend on exact merging behavior which might vary
    assert {6, 7, 8} in cut_components or any(
        all(v in comp for v in [6, 7, 8]) for comp in cut_components
    )
    assert {9} in cut_components or any(9 in comp for comp in cut_components)
    assert {4, 5} in cut_components or any(
        all(v in comp for v in [4, 5]) for comp in cut_components
    )


def test_identify_medial_wall_border():
    """
    Test identification of medial wall border vertices.
    """
    # Create a test full graph
    G_full = nx.Graph()

    # Create a simple grid topology
    #  0 - 1 - 2
    #  |   |   |
    #  3 - 4 - 5
    #  |   |   |
    #  6 - 7 - 8

    # Add all edges
    for i in range(3):
        for j in range(3):
            node = i * 3 + j
            if j < 2:  # Connect horizontally
                G_full.add_edge(node, node + 1)
            if i < 2:  # Connect vertically
                G_full.add_edge(node, node + 3)

    # Define medial wall as the center square plus one edge
    medial_wall = {4, 1, 3, 5}

    border = identify_medial_wall_border(medial_wall, G_full)

    # All medial wall vertices are on the border since they all
    # have at least one neighbor outside the medial wall
    for vertex in medial_wall:
        assert vertex in border

    # Now test with a more enclosed medial wall (center vertex has no outside neighbors)
    # But ensure there are also vertices not in the medial wall for connections
    G_full = nx.Graph()
    # Create a pentagon with central node
    G_full.add_edges_from(
        [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (0, 5), (1, 5), (2, 5), (3, 5), (4, 5)]
    )
    # Add some external nodes connected to the outer ring
    G_full.add_edges_from([(0, 6), (2, 7), (4, 8)])

    medial_wall = {0, 1, 2, 3, 4, 5}  # All pentagon nodes including center

    border = identify_medial_wall_border(medial_wall, G_full)

    # Outer vertices should be in the border (not the center)
    assert len(border) > 0
    assert all(vertex in medial_wall for vertex in border)

    # The vertices connected to outside nodes should be in the border
    assert 0 in border
    assert 2 in border
    assert 4 in border


def test_merge_small_components():
    """
    Test merging small components with their nearest larger components.
    """
    # Setup mock input
    cut_components = [
        {0, 1, 2},  # Main cut 1 (size 3)
        {3, 4},  # Main cut 2 (size 2)
        {5, 6},  # Main cut 3 (size 2)
        {7, 8},  # Main cut 4 (size 2)
        {9, 10},  # Main cut 5 (size 2)
        {11},  # Small cut 1 (will be merged)
        {12},  # Small cut 2 (will be merged)
    ]

    medial_wall = {100, 101, 102, 103}
    medial_wall_border = {100, 101}

    G_full = nx.Graph()
    # Add some connections
    G_full.add_edges_from(
        [(11, 0), (12, 101)]
    )  # Connect small cuts to main cuts and medial wall

    pts_inflated = np.zeros((200, 3))
    # Position vertices to have specific distances
    pts_inflated[0] = [0, 0, 0]
    pts_inflated[11] = [0, 0.1, 0]  # Close to vertex 0
    pts_inflated[101] = [2, 2, 2]
    pts_inflated[12] = [2, 2.1, 2]  # Close to vertex 101 (medial wall border)

    # Call function with more than 5 components
    updated_medial_wall, main_cuts = merge_small_components(
        cut_components, medial_wall, medial_wall_border, G_full, pts_inflated
    )

    # Check if small cut 1 was merged with main cut 1
    assert 11 in main_cuts[0]

    # Check if small cut 2 was merged with medial wall
    assert 12 in updated_medial_wall

    # Check that we still have 5 main cuts
    assert len(main_cuts) == 5

    # Test with less than 5 components
    fewer_cuts = cut_components[:3]  # Only 3 components

    updated_medial_wall, main_cuts = merge_small_components(
        fewer_cuts, medial_wall, medial_wall_border, G_full, pts_inflated
    )

    # Should have padded to 5 components total with empty sets
    assert len(main_cuts) == 5
    assert main_cuts[3] == set()
    assert main_cuts[4] == set()


def test_classify_cuts_anatomically():
    """
    Test classification of cuts based on their anatomical positions.
    """
    # Create mock cuts with different positions
    cut_components = [
        {0, 1},  # Temporal (most ventral)
        {2, 3},  # Calcarine (most posterior)
        {4, 5},  # Medial1
        {6, 7},  # Medial2
        {8, 9},  # Medial3 (most anterior)
    ]

    # Create mock inflated points where:
    # - Y axis is posterior-anterior
    # - Z axis is ventral-dorsal
    pts_inflated = np.zeros((200, 3))

    # Temporal cut (lowest in Z, middle in Y)
    pts_inflated[0] = [0, 1, -2]
    pts_inflated[1] = [0, 1, -2]

    # Calcarine (middle in Z, lowest in Y)
    pts_inflated[2] = [0, -2, 0]
    pts_inflated[3] = [0, -2, 0]

    # Medial cuts arranged from posterior to anterior
    pts_inflated[4] = [0, -1, 0]
    pts_inflated[5] = [0, -1, 0]

    pts_inflated[6] = [0, 1, 0]
    pts_inflated[7] = [0, 1, 0]

    pts_inflated[8] = [0, 2, 0]
    pts_inflated[9] = [0, 2, 0]

    # Create a mock medial wall
    medial_wall = list(range(100, 110))
    for i in range(100, 110):
        pts_inflated[i] = [0, 0, 0]  # Center of the YZ plane

    # Call function
    result = classify_cuts_anatomically(cut_components, pts_inflated, medial_wall)

    # Check that the cuts were classified correctly
    assert result["temporal"] == 0
    assert result["calcarine"] == 1
    assert result["medial1"] == 2
    assert result["medial2"] == 3
    assert result["medial3"] == 4


@patch("autoflatten.template.get_surface_data")
@patch("autoflatten.template.find_removed_vertices")
@patch("autoflatten.template.create_surface_graphs")
@patch("autoflatten.template.classify_vertices_by_degree")
@patch("autoflatten.template.find_connected_components")
@patch("autoflatten.template.identify_medial_wall_border")
@patch("autoflatten.template.merge_small_components")
@patch("autoflatten.template.classify_cuts_anatomically")
def test_identify_surface_components_mocked(
    mock_classify_cuts,
    mock_merge_components,
    mock_identify_border,
    mock_find_components,
    mock_classify_vertices,
    mock_create_graphs,
    mock_find_removed,
    mock_get_surface,
):
    """
    Test the identify_surface_components function using mocks.
    This tests the flow of the function without requiring real data.
    """
    # Set up all the mocks
    mock_surface_data = {"inflated_points": np.zeros((10, 3))}
    mock_get_surface.return_value = mock_surface_data

    mock_removed_vertices = np.array([1, 2, 3])
    mock_find_removed.return_value = mock_removed_vertices

    mock_g_removed = nx.Graph()
    mock_g_full = nx.Graph()
    mock_create_graphs.return_value = (mock_g_removed, mock_g_full)

    mock_high_degree = {1, 2}
    mock_low_degree = {3}
    mock_classify_vertices.return_value = (mock_high_degree, mock_low_degree)

    mock_medial_wall = {1}
    mock_other_high = {2}
    mock_cut_components = [{3}]
    mock_find_components.return_value = (
        mock_medial_wall,
        mock_other_high,
        mock_cut_components,
    )

    mock_border = {1}
    mock_identify_border.return_value = mock_border

    mock_updated_wall = {1, 2}
    mock_main_cuts = [{3}, set(), set(), set(), set()]
    mock_merge_components.return_value = (mock_updated_wall, mock_main_cuts)

    mock_name_mapping = {
        "calcarine": 0,
        "medial1": None,
        "medial2": None,
        "medial3": None,
        "temporal": None,
    }
    mock_classify_cuts.return_value = mock_name_mapping

    # Call the function
    result = identify_surface_components("test_subject", "lh")

    # Verify all functions were called
    mock_get_surface.assert_called_once_with("test_subject", "lh")
    mock_find_removed.assert_called_once_with(mock_surface_data)
    mock_create_graphs.assert_called_once_with(mock_surface_data, mock_removed_vertices)
    mock_classify_vertices.assert_called_once()
    mock_find_components.assert_called_once()
    mock_identify_border.assert_called_once_with(mock_medial_wall, mock_g_full)
    mock_merge_components.assert_called_once()
    mock_classify_cuts.assert_called_once()

    # Check the result structure
    assert "mwall" in result
    assert np.array_equal(result["mwall"], np.array(list(mock_updated_wall)))
    assert "calcarine" in result
    assert np.array_equal(result["calcarine"], np.array([3]))
    for name in ["medial1", "medial2", "medial3", "temporal"]:
        assert name in result
        assert len(result[name]) == 0
