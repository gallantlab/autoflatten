"""Functions to derive a template for cuts to be used in autoflatten."""

import networkx as nx
import numpy as np
from sklearn.decomposition import PCA

from .freesurfer import load_surface


def get_surface_data(subject, hemi):
    """
    Load surface data for a given subject and hemisphere.

    Parameters
    ----------
    subject : str
        Pycortex subject identifier.
    hemi : str
        Hemisphere identifier ('lh' or 'rh').

    Returns
    -------
    dict
        Dictionary containing surface data with keys:
        - 'fiducial_points': 3D coordinates of fiducial surface vertices
        - 'flat_points': 3D coordinates of flat surface vertices
        - 'inflated_points': 3D coordinates of inflated surface vertices
        - 'polys_full': Triangles of the full surface
        - 'polys_flat': Triangles of the flat surface
    """
    pts_fiducial, polys_full = load_surface(subject, "fiducial", hemi)
    pts_flat, polys_flat = load_surface(subject, "flat", hemi)
    pts_inflated, _ = load_surface(subject, "inflated", hemi)

    return {
        "fiducial_points": pts_fiducial,
        "flat_points": pts_flat,
        "inflated_points": pts_inflated,
        "polys_full": polys_full,
        "polys_flat": polys_flat,
    }


def find_removed_vertices(surface_data):
    """
    Find vertices that are present in the fiducial surface but removed from the flat surface.

    Parameters
    ----------
    surface_data : dict
        Dictionary containing surface data from get_surface_data().

    Returns
    -------
    numpy.ndarray
        Array of vertex indices that were removed in the flat surface.
    """
    pts_fiducial = surface_data["fiducial_points"]
    polys_flat = surface_data["polys_flat"]

    # Find all vertices that are in fiducial but not in flat
    removed_vertices = np.array(
        list(set(range(len(pts_fiducial))) - set(np.unique(polys_flat)))
    )

    return removed_vertices


def create_surface_graphs(surface_data, removed_vertices):
    """
    Create graph representations of the surface for analysis.

    Parameters
    ----------
    surface_data : dict
        Dictionary containing surface data from get_surface_data().
    removed_vertices : numpy.ndarray
        Array of removed vertex indices from find_removed_vertices().

    Returns
    -------
    tuple
        (G_removed, G_full) where:
        - G_removed: NetworkX graph of only the removed vertices
        - G_full: NetworkX graph of the entire surface
    """
    pts_inflated = surface_data["inflated_points"]
    polys_full = surface_data["polys_full"]

    # Create graph of removed vertices
    G_removed = nx.Graph()
    G_removed.add_nodes_from(removed_vertices)

    # Create full graph for finding borders
    G_full = nx.Graph()
    G_full.add_nodes_from(range(len(pts_inflated)))

    for triangle in polys_full:
        for i in range(3):
            v1 = triangle[i]
            for j in range(i + 1, 3):
                v2 = triangle[j]
                # Add edge to full graph
                G_full.add_edge(v1, v2)

                # Add edge to removed vertices graph if both vertices are removed
                if v1 in removed_vertices and v2 in removed_vertices:
                    weight = np.linalg.norm(pts_inflated[v1] - pts_inflated[v2])
                    G_removed.add_edge(v1, v2, weight=weight)

    return G_removed, G_full


def classify_vertices_by_degree(G_removed, removed_vertices, degree_threshold=2):
    """
    Separate vertices by their degree in the removed graph.

    Parameters
    ----------
    G_removed : networkx.Graph
        Graph of removed vertices.
    removed_vertices : numpy.ndarray
        Array of removed vertex indices.
    degree_threshold : int, optional
        Minimum degree threshold for high-degree vertices (default: 2).

    Returns
    -------
    tuple
        (high_degree_vertices, low_degree_vertices) where each is a set of vertex indices.
    """
    high_degree_vertices = set()
    low_degree_vertices = set()

    for v in removed_vertices:
        if G_removed.degree(v) > degree_threshold:
            high_degree_vertices.add(v)
        else:
            low_degree_vertices.add(v)

    print(
        f"Found {len(high_degree_vertices)} high-degree vertices and "
        f"{len(low_degree_vertices)} low-degree vertices"
    )

    return high_degree_vertices, low_degree_vertices


def find_connected_components(G_removed, high_degree_vertices, low_degree_vertices):
    """
    Find connected components in high-degree and low-degree vertex subgraphs.

    Parameters
    ----------
    G_removed : networkx.Graph
        Graph of removed vertices.
    high_degree_vertices : set
        Set of high-degree vertex indices.
    low_degree_vertices : set
        Set of low-degree vertex indices.

    Returns
    -------
    tuple
        (medial_wall, other_high_degree, cut_components) where:
        - medial_wall: Set of vertices in the largest high-degree component
        - other_high_degree: Set of vertices in smaller high-degree components
        - cut_components: List of connected components in low-degree vertices
    """
    # Find connected components in high-degree vertices (potential medial wall)
    G_mwall = G_removed.subgraph(high_degree_vertices).copy()
    mwall_components = list(nx.connected_components(G_mwall))

    # Take the largest component as the definite medial wall
    if mwall_components:
        mwall_components.sort(key=len, reverse=True)
        medial_wall = mwall_components[0]

        # If there are more high-degree components, they might be misclassified
        other_high_degree = set()
        for comp in mwall_components[1:]:
            other_high_degree.update(comp)
    else:
        medial_wall = set()
        other_high_degree = set()

    # Find connected components in low-degree vertices (potential cuts)
    G_cuts = G_removed.subgraph(low_degree_vertices | other_high_degree).copy()
    cut_components = list(nx.connected_components(G_cuts))

    # Sort by size
    cut_components.sort(key=len, reverse=True)

    print(
        f"Found {len(cut_components)} cut components with sizes: "
        f"{[len(c) for c in cut_components]}"
    )

    return medial_wall, other_high_degree, cut_components


def identify_medial_wall_border(medial_wall, G_full):
    """
    Identify border vertices of the medial wall.

    Parameters
    ----------
    medial_wall : set
        Set of medial wall vertex indices.
    G_full : networkx.Graph
        Full graph of the surface.

    Returns
    -------
    set
        Set of vertex indices that form the border of the medial wall.
    """
    medial_wall_border = set()

    for v in medial_wall:
        for neighbor in G_full.neighbors(v):
            if neighbor not in medial_wall:
                medial_wall_border.add(v)
                break

    print(f"Identified {len(medial_wall_border)} border vertices for the medial wall")

    return medial_wall_border


def merge_small_components(
    cut_components, medial_wall, medial_wall_border, G_full, pts_inflated
):
    """
    Merge small components with nearest large components.

    Parameters
    ----------
    cut_components : list
        List of connected components (sets of vertices) for potential cuts.
    medial_wall : set
        Set of medial wall vertex indices.
    medial_wall_border : set
        Set of border vertices of the medial wall.
    G_full : networkx.Graph
        Full graph of the surface.
    pts_inflated : numpy.ndarray
        3D coordinates of inflated surface vertices.

    Returns
    -------
    tuple
        (updated_medial_wall, main_cuts) where:
        - updated_medial_wall: Set of medial wall vertices after merging
        - main_cuts: List of cut components after merging (max 5)
    """
    updated_medial_wall = medial_wall.copy()

    # If we have too many cut components, merge smaller ones
    if len(cut_components) > 5:
        print(f"Found {len(cut_components)} cut components, merging smaller components")

        # Take the 5 largest as our main cuts
        main_cuts = cut_components[:5]
        small_cuts = cut_components[5:]

        # For each small component, find which main component it's closest to
        for small_comp in small_cuts:
            small_comp_list = list(small_comp)

            # Compute nearest-neighbor distance to each main cut
            closest_idx = -1
            min_distance = float("inf")
            assign_to_medial_wall = False

            # First, check distance to each main cut
            for i, main_comp in enumerate(main_cuts):
                main_comp_list = list(main_comp)

                # Find minimum distance between any vertex in small_comp and any vertex in main_comp
                for v1 in small_comp_list:
                    for v2 in main_comp_list:
                        dist = np.linalg.norm(pts_inflated[v1] - pts_inflated[v2])
                        if dist < min_distance:
                            min_distance = dist
                            closest_idx = i
                            assign_to_medial_wall = False

            # Now, check distance to medial wall BORDER
            if medial_wall_border:
                medial_wall_border_list = list(medial_wall_border)

                for v1 in small_comp_list:
                    for v2 in medial_wall_border_list:
                        dist = np.linalg.norm(pts_inflated[v1] - pts_inflated[v2])
                        if dist < min_distance:
                            min_distance = dist
                            assign_to_medial_wall = True

            # Assign the small component based on nearest-neighbor distance
            if assign_to_medial_wall:
                # Add to medial wall
                updated_medial_wall.update(small_comp)
            else:
                # Add to closest cut
                main_cuts[closest_idx].update(small_comp)
    else:
        main_cuts = cut_components

    # If we don't have enough cut components, add empty components
    while len(main_cuts) < 5:
        print(
            f"Warning: Found only {len(main_cuts)} cut components, adding empty component"
        )
        main_cuts.append(set())

    return updated_medial_wall, main_cuts


def classify_cuts_anatomically(cut_components, pts_inflated, medial_wall):
    """
    Classify cuts based on their anatomical positions after projecting onto YZ plane
    and normalizing orientation.

    Parameters
    ----------
    cut_components : list
        List of connected components (sets of vertices) for cuts.
    pts_inflated : numpy.ndarray
        3D coordinates of inflated surface vertices.
    medial_wall : array
        Array of medial wall vertex indices.

    Returns
    -------
    dict
        Dictionary mapping anatomical names to cut indices in consistent order:
        'calcarine', 'medial1', 'medial2', 'medial3', 'temporal'
    """

    # Initialize result dictionary with consistent order
    result_dict = {
        "calcarine": None,
        "medial1": None,
        "medial2": None,
        "medial3": None,
        "temporal": None,
    }

    # Collect all vertices from all cuts
    all_cut_vertices = []
    for cut in cut_components:
        if cut:
            all_cut_vertices.extend(list(cut))

    if not all_cut_vertices:
        return result_dict  # Return early if no cuts found

    # Step 1: Project all medial wall vertices onto YZ plane
    yz_points = pts_inflated[medial_wall][:, 1:3]  # Take only Y and Z coordinates

    # Step 2: Perform PCA on YZ points to normalize orientation
    pca = PCA(n_components=2)
    pca.fit(yz_points)

    # Get transformation matrix
    transform_matrix = pca.components_

    # Ensure consistent orientation:
    # - First principal component should align with dorsal-ventral axis
    # - Second principal component should align with posterior-anterior axis

    # We want PC1 (dorsal-ventral) to have higher loading on Z than Y
    if abs(transform_matrix[0, 0]) > abs(transform_matrix[0, 1]):
        # Swap PCs to ensure PC1 aligns more with Z
        transform_matrix[[0, 1]] = transform_matrix[[1, 0]]

    # Ensure PC1 (dorsal-ventral) points dorsally (Z+)
    if transform_matrix[0, 1] < 0:
        transform_matrix[0, :] = -transform_matrix[0, :]

    # Ensure PC2 (posterior-anterior) points anteriorly (Y+)
    if transform_matrix[1, 0] < 0:
        transform_matrix[1, :] = -transform_matrix[1, :]

    print("YZ Plane Transformation Matrix:")
    print(transform_matrix)

    # Calculate information for each cut in normalized space
    cut_info = []
    for i, cut in enumerate(cut_components):
        if not cut:  # Skip empty components
            continue

        # Get YZ coordinates for this cut
        cut_yz = pts_inflated[[v for v in cut]][:, 1:3]

        # Transform to normalized space
        normalized_yz = np.dot(cut_yz, transform_matrix.T)

        # Calculate centroid in normalized space
        norm_centroid = np.mean(normalized_yz, axis=0)

        cut_info.append(
            {
                "index": i,
                "pc1": norm_centroid[0],  # Dorsal-ventral axis
                "pc2": norm_centroid[1],  # Posterior-anterior axis
                "size": len(cut),
            }
        )

    # Debug print
    print("\nNormalized coordinates for cuts (PC space):")
    for info in cut_info:
        print(
            f"Cut {info['index']}: PC1 (dorsal-ventral)={info['pc1']:.2f}, "
            f"PC2 (posterior-anterior)={info['pc2']:.2f}, size={info['size']}"
        )

    # Step 3: Identify temporal cut (most ventral - lowest on dorsal-ventral axis)
    cut_info.sort(key=lambda p: p["pc1"])
    result_dict["temporal"] = cut_info[0]["index"]

    # Remove temporal from consideration
    remaining_cuts = [p for p in cut_info if p["index"] != result_dict["temporal"]]

    # If we've run out of cuts, return what we have
    if not remaining_cuts:
        return result_dict

    # Step 4: Sort remaining cuts by position on posterior-anterior axis
    remaining_cuts.sort(key=lambda p: p["pc2"])

    # Assign cuts in order: calcarine, medial1, medial2, medial3
    cut_labels = ["calcarine", "medial1", "medial2", "medial3"]
    for i, label in enumerate(cut_labels):
        if i < len(remaining_cuts):
            result_dict[label] = remaining_cuts[i]["index"]

    # Debug information
    print("\nCut classification results:")
    print(f"Temporal cut (most ventral): index={result_dict['temporal']}")
    for label in cut_labels:
        if result_dict[label] is not None:
            cut = next((p for p in cut_info if p["index"] == result_dict[label]), None)
            if cut:
                print(
                    f"{label} cut: index={result_dict[label]}, "
                    f"PC2 (posterior-anterior)={cut['pc2']:.2f}"
                )

    return result_dict


def identify_surface_components(subject, hemi):
    """
    Main function to identify medial wall and cuts on a subject's surface.

    Parameters
    ----------
    subject : str
        Pycortex subject identifier.
    hemi : str
        Hemisphere identifier ('lh' or 'rh').

    Returns
    -------
    vertex_dict : dict
        Dictionary containing the medial wall vertices and anatomically named cut vertices.
        Keys are "mwall", "calcarine", "medial1", "medial2", "medial3", and "temporal".
    """
    # Step 1: Load surface data
    surface_data = get_surface_data(subject, hemi)

    # Step 2: Find removed vertices
    removed_vertices = find_removed_vertices(surface_data)

    # Step 3: Create surface graphs
    G_removed, G_full = create_surface_graphs(surface_data, removed_vertices)

    # Step 4: Classify vertices by degree
    high_degree_vertices, low_degree_vertices = classify_vertices_by_degree(
        G_removed, removed_vertices
    )

    # Step 5: Find connected components
    medial_wall, other_high_degree, cut_components = find_connected_components(
        G_removed, high_degree_vertices, low_degree_vertices
    )

    # Step 6: Identify medial wall border
    medial_wall_border = identify_medial_wall_border(medial_wall, G_full)

    # Step 7: Merge small components
    medial_wall, main_cuts = merge_small_components(
        cut_components,
        medial_wall,
        medial_wall_border,
        G_full,
        surface_data["inflated_points"],
    )

    # Step 8: Classify cuts anatomically
    name_mapping = classify_cuts_anatomically(
        main_cuts, surface_data["inflated_points"], list(medial_wall)
    )

    # Step 9: Create final vertex dictionary
    vertex_dict = {
        "mwall": np.array(list(medial_wall)),
    }

    # Assign cut vertices to the named cuts
    for name, idx in name_mapping.items():
        if idx is not None and idx < len(main_cuts):
            vertex_dict[name] = np.array(list(main_cuts[idx]))
        else:
            # Handle empty components
            vertex_dict[name] = np.array([], dtype=int)

    # Print final information
    print(f"Final medial wall size: {len(medial_wall)}")
    for name in ["calcarine", "medial1", "medial2", "medial3", "temporal"]:
        print(f"Final {name} cut size: {len(vertex_dict[name])}")

    return vertex_dict
