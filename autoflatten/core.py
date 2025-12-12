"""Core functions for autoflatten."""

import os
import shutil
import subprocess
import tempfile
import warnings
from collections import defaultdict, deque

import networkx as nx
import numpy as np
from scipy.spatial.distance import cdist

from .freesurfer import create_label_file, load_surface, read_freesurfer_label
from .flatten.algorithm import count_boundary_loops

# Distance thresholds for geodesic refinement anchor selection (mm)
# NEAR_MWALL_THRESHOLD_MM: Cut vertices within this distance are considered "near" medial wall
# ANCHOR_SEARCH_RADIUS_MM: Search for candidate anchors within this distance of the cut
NEAR_MWALL_THRESHOLD_MM = 10.0
ANCHOR_SEARCH_RADIUS_MM = 15.0

# Thresholds for trapped vertex detection in _find_trapped_vertices
# A vertex is considered trapped if BFS can only reach fewer than MIN_REACHABLE vertices
# We limit BFS to MAX_BFS_VERTICES to avoid expensive traversals of well-connected regions
TRAPPED_VERTEX_MIN_REACHABLE = 100
TRAPPED_VERTEX_MAX_BFS = 200

# Maximum iterations for hole filling to prevent infinite loops
# Typically 1-2 iterations suffice; 10 provides ample margin for complex cases
HOLE_FILL_MAX_ITERATIONS = 10


def _find_geometric_endpoints(cut_vertices, pts):
    """Find the two most geometrically distant vertices in a cut.

    Parameters
    ----------
    cut_vertices : array-like
        Vertex indices of the cut.
    pts : ndarray of shape (V, 3)
        Vertex coordinates.

    Returns
    -------
    start : int
        First endpoint vertex index.
    end : int
        Second endpoint vertex index.
    max_dist : float
        Euclidean distance between the endpoints.
    """
    max_dist = 0
    start, end = cut_vertices[0], cut_vertices[0]
    for idx1, v1 in enumerate(cut_vertices):
        pos1 = pts[v1]
        for v2 in cut_vertices[idx1 + 1 :]:
            dist = np.linalg.norm(pos1 - pts[v2])
            if dist > max_dist:
                max_dist = dist
                start, end = v1, v2
    return start, end, max_dist


def ensure_continuous_cuts(vertex_dict, subject, hemi):
    """
    Make cuts continuous using Euclidean distances on the inflated surface for speed.

    Parameters
    ----------
    vertex_dict : dict
        Dictionary containing medial wall and cut vertices.
    subject : str
        Subject identifier.
    hemi : str
        Hemisphere identifier ('lh' or 'rh').

    Returns
    -------
    vertex_dict : dict
        Updated dictionary with continuous cuts.
    """
    # Get INFLATED surface geometry instead of fiducial
    print("Loading inflated surface...")
    pts_inflated, polys = load_surface(subject, "inflated", hemi)

    # Also get fiducial for accurate path finding
    try:
        pts_fiducial, _ = load_surface(subject, "fiducial", hemi)
    except FileNotFoundError:
        print("Fiducial surface not found, computing it from smoothwm and pial.")
        # Need to compute it from smoothwm and pial
        pts_wm, _ = load_surface(subject, "smoothwm", hemi)
        pts_pial, _ = load_surface(subject, "pial", hemi)
        pts_fiducial = (pts_wm + pts_pial) / 2.0

    # Create surface graph for path finding (using fiducial)
    print("Creating surface graph...")
    G = nx.Graph()
    G.add_nodes_from(range(len(pts_fiducial)))

    for triangle in polys:
        for i in range(3):
            v1 = triangle[i]
            for j in range(i + 1, 3):
                v2 = triangle[j]
                # Use fiducial for edge weights to get accurate paths
                weight = np.linalg.norm(pts_fiducial[v1] - pts_fiducial[v2])
                G.add_edge(v1, v2, weight=weight)

    # Process each cut (using anatomical names from template)
    cut_names = ["calcarine", "medial1", "medial2", "medial3", "temporal"]
    for cut_key in cut_names:
        if cut_key not in vertex_dict or len(vertex_dict[cut_key]) == 0:
            continue

        print(f"Processing {cut_key}...")
        cut_vertices = list(vertex_dict[cut_key])

        # Step 1: Find connected components
        G_cut = G.subgraph(cut_vertices).copy()
        components = list(nx.connected_components(G_cut))

        if len(components) == 1:
            print(f"{cut_key} is already continuous.")
            continue

        print(f"{cut_key} has {len(components)} disconnected components")

        # Step 2: Find endpoints of each component using Euclidean distances
        component_endpoints = []

        for comp in components:
            # Convert to list for indexing
            comp_list = list(comp)
            if len(comp_list) == 1:
                # Single vertex component
                component_endpoints.append((comp_list[0], comp_list[0]))
                continue

            # Create subgraph for topological analysis
            comp_graph = G_cut.subgraph(comp).copy()

            # Find degree-1 vertices (natural endpoints)
            deg1_vertices = [v for v in comp_graph.nodes() if comp_graph.degree(v) == 1]

            if deg1_vertices:
                if len(deg1_vertices) == 1:
                    # Find most distant vertex from the degree-1 vertex
                    start = deg1_vertices[0]
                    max_dist = 0
                    end = start

                    # Use Euclidean distance on inflated surface
                    start_pos = pts_inflated[start]
                    for v in comp:
                        dist = np.linalg.norm(start_pos - pts_inflated[v])
                        if dist > max_dist:
                            max_dist = dist
                            end = v

                    component_endpoints.append((start, end))
                else:
                    # Find most distant pair among degree-1 vertices
                    max_dist = 0
                    best_pair = (deg1_vertices[0], deg1_vertices[0])

                    # Use Euclidean distance on inflated surface
                    for idx1, v1 in enumerate(deg1_vertices):
                        pos1 = pts_inflated[v1]
                        for v2 in deg1_vertices[idx1 + 1 :]:
                            dist = np.linalg.norm(pos1 - pts_inflated[v2])
                            if dist > max_dist:
                                max_dist = dist
                                best_pair = (v1, v2)

                    component_endpoints.append(best_pair)
            else:
                # No degree-1 vertices - find diameter using Euclidean distances
                # Two-pass approach for finding diameter
                start = comp_list[0]
                max_dist = 0
                far_vertex = start

                # First pass - find furthest vertex from arbitrary start
                start_pos = pts_inflated[start]
                for v in comp:
                    dist = np.linalg.norm(start_pos - pts_inflated[v])
                    if dist > max_dist:
                        max_dist = dist
                        far_vertex = v

                # Second pass - find furthest vertex from far_vertex
                max_dist = 0
                end = far_vertex
                far_pos = pts_inflated[far_vertex]

                for v in comp:
                    dist = np.linalg.norm(far_pos - pts_inflated[v])
                    if dist > max_dist:
                        max_dist = dist
                        end = v

                component_endpoints.append((far_vertex, end))

        # Step 3: Find global start and end points using Euclidean distances
        flat_endpoints = [
            (v, comp_idx)
            for comp_idx, (start, end) in enumerate(component_endpoints)
            for v in (start, end)
        ]

        max_dist = 0
        global_start_idx, global_end_idx = 0, 1  # Default to first two endpoints

        # Find the most distant pair using Euclidean distance
        for i in range(len(flat_endpoints) - 1):
            v1, comp1 = flat_endpoints[i]
            pos1 = pts_inflated[v1]

            for j in range(i + 1, len(flat_endpoints)):
                v2, comp2 = flat_endpoints[j]
                # Skip pairs from same component
                if comp1 == comp2:
                    continue

                # Use Euclidean distance on inflated surface
                dist = np.linalg.norm(pos1 - pts_inflated[v2])
                if dist > max_dist:
                    max_dist = dist
                    global_start_idx, global_end_idx = i, j

        # Get global endpoints
        global_start, start_comp = flat_endpoints[global_start_idx]
        global_end, end_comp = flat_endpoints[global_end_idx]

        print(f"Global start: vertex {global_start} in component {start_comp}")
        print(f"Global end: vertex {global_end} in component {end_comp}")

        # Step 4: Connect components from start to end
        # Initialize with original vertices
        final_vertices = set(cut_vertices)

        # Track connected and remaining components
        component_list = list(components)
        connected = {start_comp}
        remaining = set(range(len(component_list))) - connected

        # If start and end are in same component, no need to connect
        if start_comp == end_comp:
            print("Start and end are in same component, no need to connect.")
            continue

        # Connect components in sequence
        while remaining and end_comp in remaining:
            # Find closest remaining component to any connected component
            min_dist = float("inf")
            best_conn = None
            best_remain = None
            closest_v1 = None
            closest_v2 = None

            # Use Euclidean distance to find closest components
            for conn_idx in connected:
                conn_comp = component_list[conn_idx]

                for remain_idx in remaining:
                    remain_comp = component_list[remain_idx]

                    # Find closest vertices between components using Euclidean distance
                    for v1 in conn_comp:
                        pos1 = pts_inflated[v1]

                        for v2 in remain_comp:
                            dist = np.linalg.norm(pos1 - pts_inflated[v2])
                            if dist < min_dist:
                                min_dist = dist
                                best_conn = conn_idx
                                best_remain = remain_idx
                                closest_v1 = v1
                                closest_v2 = v2

            if closest_v1 is not None and closest_v2 is not None:
                # Use surface graph to find shortest path between closest vertices
                try:
                    path = nx.shortest_path(G, closest_v1, closest_v2, weight="weight")
                    # Add path to final vertices
                    final_vertices.update(path)
                    connected.add(best_remain)
                    remaining.remove(best_remain)
                    print(
                        f"Connected component {best_remain} with path of length {len(path)}"
                    )

                    # If we've reached the end component, we're done
                    if best_remain == end_comp:
                        break
                except nx.NetworkXNoPath:
                    print(
                        f"Warning: No path between closest vertices ({closest_v1}, {closest_v2})"
                    )
                    # Remove this pair from consideration
                    min_dist = float("inf")
            else:
                print("Warning: Could not connect all components")
                break

        # Step 5: Ensure global start and end points are connected
        G_final = G.subgraph(final_vertices).copy()

        try:
            path = nx.shortest_path(G_final, global_start, global_end, weight="weight")
            print(f"Verified connection from start to end: path length {len(path)}")
        except nx.NetworkXNoPath:
            # Add direct path if not connected
            try:
                direct_path = nx.shortest_path(
                    G, global_start, global_end, weight="weight"
                )
                final_vertices.update(direct_path)
                print(f"Added direct path from start to end: length {len(direct_path)}")
            except nx.NetworkXNoPath:
                print("Warning: Could not connect global endpoints")

        # Update the vertex dictionary
        vertex_dict[cut_key] = np.array(sorted(list(final_vertices)))
        print(
            f"Final continuous cut has {len(final_vertices)} vertices "
            f"(originally {len(cut_vertices)})"
        )

    return vertex_dict


def _find_trapped_vertices(G, excluded, mwall_set, anchor):
    """Find vertices that would become isolated if excluded vertices are removed.

    A vertex is "trapped" if its only connections to the main patch go through
    the excluded set (mwall + cut). These trapped vertices would form holes in
    the patch and should be added to the cut.

    Parameters
    ----------
    G : networkx.Graph
        Surface graph.
    excluded : set
        Vertices that are excluded (mwall + geodesic path).
    mwall_set : set
        Medial wall vertices.
    anchor : int
        The anchor vertex (end of geodesic path, adjacent to mwall).

    Returns
    -------
    list
        Trapped vertices that should be added to the cut.
    """
    # Find all non-mwall, non-cut neighbors of the anchor
    potential_trapped = set()
    for neighbor in G.neighbors(anchor):
        if neighbor not in excluded:
            potential_trapped.add(neighbor)

    if not potential_trapped:
        return []

    # For each potential trapped vertex, check if it can reach a "safe" vertex
    # (one that's far from the excluded set) without going through excluded
    trapped = set()
    for v in potential_trapped:
        # Use BFS to check connectivity to the rest of the mesh
        # If we can reach many vertices without going through excluded,
        # then v is connected to the main patch and not trapped
        visited = {v}
        queue = deque([v])

        while queue and len(visited) < TRAPPED_VERTEX_MAX_BFS:
            current = queue.popleft()
            for neighbor in G.neighbors(current):
                if neighbor not in visited and neighbor not in excluded:
                    visited.add(neighbor)
                    queue.append(neighbor)

        # If we could only reach a small number of vertices, it's trapped
        if len(visited) < TRAPPED_VERTEX_MIN_REACHABLE:
            trapped.add(v)
            # Also add any other vertices in this small isolated region
            trapped.update(visited)

    return list(trapped)


def fill_holes_in_patch(faces, excluded_vertices):
    """Fill holes in a patch by excluding hole boundary vertices.

    A patch should have exactly one boundary loop (the outer boundary). Multiple
    boundary loops indicate holes in the patch. This function detects holes and
    returns the additional vertices that should be excluded to fill them.

    This function also handles T-junctions, where a hole boundary shares a vertex
    with the main boundary, creating vertices with >2 boundary neighbors. Such
    vertices are treated as hole vertices and excluded to allow proper loop detection.

    Parameters
    ----------
    faces : ndarray of shape (F, 3)
        Triangle face indices for the full surface.
    excluded_vertices : set
        Set of vertex indices to exclude (medial wall + cuts).

    Returns
    -------
    hole_vertices : set
        Additional vertices to exclude to fill holes. Empty set if no holes.
    """
    # Handle empty faces array
    if faces.size == 0:
        return set()

    all_hole_vertices = set()

    for iteration in range(HOLE_FILL_MAX_ITERATIONS):
        # Build patch faces (faces where all 3 vertices are in patch)
        patch_vertex_set = (
            set(range(faces.max() + 1)) - excluded_vertices - all_hole_vertices
        )
        patch_faces = []
        for face in faces:
            if all(int(v) in patch_vertex_set for v in face):
                patch_faces.append([int(v) for v in face])

        if len(patch_faces) == 0:
            break

        patch_faces = np.array(patch_faces)

        # Build boundary adjacency to detect T-junctions
        # Count how many faces each edge belongs to
        edge_face_count = defaultdict(int)
        for face in patch_faces:
            for i in range(3):
                edge = tuple(sorted([int(face[i]), int(face[(i + 1) % 3])]))
                edge_face_count[edge] += 1

        # Boundary edges appear in exactly 1 face
        boundary_edges = {e for e, count in edge_face_count.items() if count == 1}

        if not boundary_edges:
            break

        # Build adjacency from boundary edges
        boundary_adj = defaultdict(set)
        for v1, v2 in boundary_edges:
            boundary_adj[v1].add(v2)
            boundary_adj[v2].add(v1)

        # Detect T-junction vertices (boundary vertices with >2 neighbors)
        # These occur when a hole boundary meets the main boundary at a single vertex
        tjunction_vertices = {
            v for v, neighbors in boundary_adj.items() if len(neighbors) > 2
        }

        if tjunction_vertices:
            # T-junctions found: mark them as hole vertices and continue
            # This "breaks" the junction, allowing proper loop detection next iteration
            print(
                f"  Iteration {iteration + 1}: Found {len(tjunction_vertices)} "
                f"T-junction vertices, excluding them"
            )
            all_hole_vertices.update(tjunction_vertices)
            continue

        # No T-junctions: count boundary loops normally
        n_loops, loops = count_boundary_loops(patch_faces)

        if n_loops <= 1:
            # No holes (or no boundary - shouldn't happen)
            break

        # Find the largest loop (main boundary) and mark smaller ones as holes
        loops_sorted = sorted(loops, key=len, reverse=True)
        hole_loops = loops_sorted[1:]

        # Collect all vertices in hole boundary loops
        new_hole_vertices = set()
        for loop in hole_loops:
            new_hole_vertices.update(int(v) for v in loop)

        if not new_hole_vertices:
            break

        print(
            f"  Iteration {iteration + 1}: Found {len(hole_loops)} hole(s) with "
            f"{len(new_hole_vertices)} boundary vertices"
        )
        all_hole_vertices.update(new_hole_vertices)
    else:
        # Loop completed without break - max iterations reached
        warnings.warn(
            f"Hole filling reached maximum iterations ({HOLE_FILL_MAX_ITERATIONS}). "
            "This may indicate unexpected mesh topology. "
            f"Total hole vertices found: {len(all_hole_vertices)}"
        )

    return all_hole_vertices


def refine_cuts_with_geodesic(vertex_dict, subject, hemi, medial_wall_vertices=None):
    """
    Refine cuts by replacing them with geodesic shortest paths between endpoints.

    This function takes projected cuts (which may have some wiggling from registration)
    and replaces them with the shortest geodesic path on the target surface between
    the cut endpoints. This should produce more anatomically direct cuts and reduce
    distortion during flattening.

    Parameters
    ----------
    vertex_dict : dict
        Dictionary containing medial wall and cut vertices.
    subject : str
        Subject identifier.
    hemi : str
        Hemisphere identifier ('lh' or 'rh').
    medial_wall_vertices : array-like, optional
        Vertices of the medial wall. If provided, endpoints will be chosen from
        vertices that border the medial wall. If None, uses geometric endpoints.

    Returns
    -------
    vertex_dict : dict
        Updated dictionary with geodesically refined cuts.
    """
    print("\n=== Refining cuts with geodesic shortest paths ===")

    # Create a copy to avoid modifying the input dict in-place
    vertex_dict = {k: np.array(v) for k, v in vertex_dict.items()}

    # Load fiducial surface for accurate geodesic distances
    print("Loading fiducial surface...")
    try:
        pts_fiducial, polys = load_surface(subject, "fiducial", hemi)
    except FileNotFoundError:
        print("Fiducial surface not found, computing it from smoothwm and pial.")
        pts_wm, _ = load_surface(subject, "smoothwm", hemi)
        pts_pial, _ = load_surface(subject, "pial", hemi)
        pts_fiducial = (pts_wm + pts_pial) / 2.0
        _, polys = load_surface(subject, "smoothwm", hemi)

    # Also load inflated for reference
    pts_inflated, _ = load_surface(subject, "inflated", hemi)

    # Create surface graph with geodesic weights
    print("Creating surface graph with geodesic weights...")
    G = nx.Graph()
    G.add_nodes_from(range(len(pts_fiducial)))

    for triangle in polys:
        for i in range(3):
            v1 = triangle[i]
            for j in range(i + 1, 3):
                v2 = triangle[j]
                # Use fiducial surface for accurate geodesic distances
                weight = np.linalg.norm(pts_fiducial[v1] - pts_fiducial[v2])
                G.add_edge(v1, v2, weight=weight)

    # Convert medial wall to set for fast lookup
    mwall_set = set(medial_wall_vertices) if medial_wall_vertices is not None else set()

    # Process each cut in the vertex_dict (excluding medial wall)
    for cut_key in vertex_dict.keys():
        if cut_key == "mwall":  # Skip medial wall
            continue
        if len(vertex_dict[cut_key]) == 0:
            continue

        print(f"\nRefining {cut_key}...")
        cut_vertices = vertex_dict[cut_key]

        if len(cut_vertices) < 2:
            print(f"  {cut_key} has fewer than 2 vertices, skipping")
            continue

        # Step 1: Find endpoints of the cut
        # Strategy:
        # - Start point: cut vertex FARTHEST from medial wall (in the cortex)
        # - End point: optimal anchor on medial wall border that creates a path
        #   with maximum clearance from the medial wall
        if mwall_set:
            mwall_list = list(mwall_set)
            mwall_coords = pts_inflated[mwall_list]
            cut_coords = pts_inflated[cut_vertices]

            # For each cut vertex, find minimum Euclidean distance to any medial wall
            # vertex. We use Euclidean rather than geodesic distance here for
            # efficiency, since we only need a rough proximity measure to identify
            # which cut vertices are "near" the medial wall.
            dist_to_mwall = cdist(cut_coords, mwall_coords).min(axis=1)

            # Start point: cut vertex FARTHEST from medial wall
            farthest_idx = np.argmax(dist_to_mwall)
            start = cut_vertices[farthest_idx]
            start_dist_to_mwall = dist_to_mwall[farthest_idx]

            print(
                f"  Start (farthest from mwall): vertex {start} "
                f"(dist to mwall: {start_dist_to_mwall:.2f}mm)"
            )

            # Build set of medial wall border vertices (non-mwall vertices adjacent
            # to mwall). These are potential anchor points for cuts.
            mwall_border_set = set()
            for mw_v in mwall_set:
                for neighbor in G.neighbors(mw_v):
                    if neighbor not in mwall_set:
                        mwall_border_set.add(neighbor)

            # Find cut vertices that are near the medial wall (potential anchor region).
            # These define the region where we should look for anchor points.
            cut_near_mwall_mask = dist_to_mwall < NEAR_MWALL_THRESHOLD_MM
            if not cut_near_mwall_mask.any():
                # If no cut vertices are near mwall, use the closest one
                cut_near_mwall_mask = np.zeros(len(cut_vertices), dtype=bool)
                cut_near_mwall_mask[np.argmin(dist_to_mwall)] = True

            cut_near_mwall = cut_vertices[cut_near_mwall_mask]
            cut_near_mwall_coords = pts_inflated[cut_near_mwall]

            # Find candidate anchor points: mwall border vertices near the cut's
            # mwall-adjacent region.
            candidate_anchors = []
            for border_v in mwall_border_set:
                border_coord = pts_inflated[border_v]
                dist_to_cut = np.linalg.norm(
                    cut_near_mwall_coords - border_coord, axis=1
                ).min()
                if dist_to_cut < ANCHOR_SEARCH_RADIUS_MM:
                    candidate_anchors.append(border_v)

            if not candidate_anchors:
                # Fallback: use nearest border vertex to closest cut vertex
                closest_cut_v = cut_vertices[np.argmin(dist_to_mwall)]
                min_dist = float("inf")
                for border_v in mwall_border_set:
                    dist = np.linalg.norm(
                        pts_inflated[border_v] - pts_inflated[closest_cut_v]
                    )
                    if dist < min_dist:
                        min_dist = dist
                        candidate_anchors = [border_v]

            # Check if we have any candidate anchors (mwall_border_set might be empty)
            if not candidate_anchors:
                print(
                    "  WARNING: No candidate anchors found on medial wall border. "
                    "Using geometric endpoints."
                )
                start, end, _ = _find_geometric_endpoints(cut_vertices, pts_inflated)
            else:
                print(
                    f"  Found {len(candidate_anchors)} candidate anchor points on mwall"
                )

                # Evaluate each candidate anchor: find the one whose geodesic path
                # to start has the maximum minimum distance from the medial wall
                best_anchor = candidate_anchors[0]
                best_min_clearance = -1
                found_valid_path = False

                for anchor in candidate_anchors:
                    try:
                        path = nx.shortest_path(G, start, anchor, weight="weight")

                        # Compute minimum distance to mwall along this path
                        path_coords = pts_inflated[path]
                        path_to_mwall_dists = cdist(path_coords, mwall_coords).min(
                            axis=1
                        )
                        min_clearance = path_to_mwall_dists.min()

                        if min_clearance > best_min_clearance:
                            best_min_clearance = min_clearance
                            best_anchor = anchor
                            found_valid_path = True

                    except nx.NetworkXNoPath:
                        continue

                if found_valid_path:
                    end = best_anchor
                    print(
                        f"  Best anchor: vertex {end} "
                        f"(path min clearance: {best_min_clearance:.2f}mm)"
                    )
                else:
                    # No valid geodesic path to any anchor - use geometric fallback
                    print(
                        "  WARNING: No valid geodesic path to any anchor. "
                        "Using geometric endpoints."
                    )
                    start, end, _ = _find_geometric_endpoints(
                        cut_vertices, pts_inflated
                    )

        else:
            # No medial wall provided - use geometric endpoints
            start, end, max_cut_dist = _find_geometric_endpoints(
                cut_vertices, pts_inflated
            )
            print(
                f"  No medial wall provided, using geometric endpoints: "
                f"{start}, {end} (dist: {max_cut_dist:.2f}mm)"
            )

        # Step 2: Compute geodesic shortest path between endpoints
        try:
            geodesic_path = nx.shortest_path(G, start, end, weight="weight")

            # Calculate path length
            path_length = sum(
                G[geodesic_path[i]][geodesic_path[i + 1]]["weight"]
                for i in range(len(geodesic_path) - 1)
            )

            print(f"  Original cut: {len(cut_vertices)} vertices")
            print(
                f"  Geodesic path: {len(geodesic_path)} vertices, length: {path_length:.2f}"
            )

            # Step 3: Find and include any trapped vertices
            # A vertex is "trapped" if excluding mwall + geodesic_path would
            # isolate it from the rest of the patch (creating a hole)
            geodesic_set = set(geodesic_path)
            excluded = mwall_set | geodesic_set

            # Find vertices that would be trapped by this cut
            trapped_vertices = _find_trapped_vertices(
                G,
                excluded,
                mwall_set,
                geodesic_path[-1],  # anchor is last vertex
            )

            if trapped_vertices:
                print(
                    f"  Found {len(trapped_vertices)} trapped vertices, adding to cut"
                )
                geodesic_path = list(geodesic_path) + list(trapped_vertices)

            # Step 4: Replace cut with geodesic path + trapped vertices
            vertex_dict[cut_key] = np.array(geodesic_path)

            # Calculate reduction
            original_len = len(cut_vertices)
            new_len = len(geodesic_path)
            reduction_pct = 100 * (1 - new_len / original_len)
            print(f"  Reduced by {reduction_pct:.1f}%")

        except nx.NetworkXNoPath:
            print(
                f"  WARNING: No path found between {start} and {end}, keeping original cut"
            )
        except Exception as e:
            print(f"  ERROR: Failed to compute geodesic path: {e}")
            print(f"  Keeping original cut")

    # Post-processing: find and absorb any isolated small regions that would
    # create holes in the patch
    all_vertices = set(G.nodes())
    excluded_final = set()

    # Collect all excluded vertices
    if mwall_set:
        excluded_final.update(mwall_set)
    for key in vertex_dict.keys():
        if key == "mwall":
            continue
        if len(vertex_dict[key]) > 0:
            excluded_final.update(int(v) for v in vertex_dict[key])

    # Find connected components of patch vertices
    patch_vertices = all_vertices - excluded_final
    patch_subgraph = G.subgraph(patch_vertices)
    components = list(nx.connected_components(patch_subgraph))

    if len(components) > 1:
        # Find the largest component (main patch)
        components.sort(key=len, reverse=True)
        isolated_regions = components[1:]

        total_isolated = sum(len(c) for c in isolated_regions)
        print(
            f"\nPost-processing: Found {len(isolated_regions)} isolated regions "
            f"({total_isolated} vertices total)"
        )

        # Add isolated vertices to the nearest cut
        for region in isolated_regions:
            region_vertices = list(region)
            # Find which cut is nearest to this region
            best_cut = None
            best_dist = float("inf")

            for key in vertex_dict.keys():
                if key == "mwall":
                    continue
                if len(vertex_dict[key]) > 0:
                    cut_verts = vertex_dict[key]
                    cut_coords = pts_fiducial[cut_verts]
                    region_coords = pts_fiducial[region_vertices]
                    dists = cdist(region_coords, cut_coords)
                    min_dist = dists.min()
                    if min_dist < best_dist:
                        best_dist = min_dist
                        best_cut = key

            if best_cut is not None:
                print(f"  Adding {len(region_vertices)} vertices to {best_cut}")
                vertex_dict[best_cut] = np.concatenate(
                    [vertex_dict[best_cut], np.array(region_vertices)]
                )

    return vertex_dict


def map_cuts_to_subject(vertex_dict, target_subject, hemi, source_subject="fsaverage"):
    """
    Map cutting vertices from a source subject to a target subject using FreeSurfer's
    mri_label2label.

    Parameters:
    -----------
    vertex_dict : dict
        Dictionary with keys 'mwall', 'cut1', 'cut2', 'cut3', 'cut4', 'cut5'
        Each key contains a list/array of vertex IDs from the source subject
    target_subject : str
        Subject ID for the target subject
    hemi : str
        Hemisphere ('lh' or 'rh')
    source_subject : str
        Source subject ID (default: "fsaverage")

    Returns:
    --------
    mapped_cuts : dict
        Dictionary with the same keys as vertex_dict, but with vertex IDs
        mapped to the target subject's surface
    """
    mapped_cuts = {}

    # Create a temporary directory for intermediate files
    temp_dir = tempfile.mkdtemp()
    try:
        # Process each cut
        for cut_name, vertices in vertex_dict.items():
            if not isinstance(vertices, (list, np.ndarray)) or len(vertices) == 0:
                print(f"Warning: No vertices for {cut_name}, skipping")
                mapped_cuts[cut_name] = []
                continue

            # Convert vertices to array if needed
            if isinstance(vertices, list):
                vertices = np.array(vertices)

            # Create source label file in temp directory
            source_label = os.path.join(temp_dir, f"{cut_name}_{hemi}.label")
            create_label_file(vertices, source_subject, hemi, source_label)

            # Create target label filename
            target_label = os.path.join(
                temp_dir, f"{cut_name}_{hemi}_{target_subject}.label"
            )

            # Map label from source to target using mri_label2label
            cmd = [
                "mri_label2label",
                "--srcsubject",
                source_subject,
                "--trgsubject",
                target_subject,
                "--srclabel",
                source_label,
                "--trglabel",
                target_label,
                "--hemi",
                hemi,
                "--regmethod",
                "surface",
            ]

            # Run the command
            try:
                subprocess.run(
                    cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )
            except subprocess.CalledProcessError as e:
                print(f"Error mapping {cut_name}: {e}")
                print(f"Command: {' '.join(cmd)}")
                print(f"Stderr: {e.stderr.decode()}")
                mapped_cuts[cut_name] = []
                continue

            # Read the target label file to get mapped vertices
            try:
                mapped_vertices = read_freesurfer_label(target_label)
                mapped_cuts[cut_name] = mapped_vertices
                print(
                    f"Successfully mapped {len(vertices)} source vertices to "
                    f"{len(mapped_vertices)} target vertices for {cut_name}"
                )
            except Exception as e:
                print(f"Error reading mapped label for {cut_name}: {e}")
                mapped_cuts[cut_name] = []

    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir)

    return mapped_cuts
