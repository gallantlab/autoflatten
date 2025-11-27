"""Core functions for autoflatten."""

import os
import shutil
import subprocess
import tempfile

import networkx as nx
import numpy as np

from .freesurfer import create_label_file, load_surface, read_freesurfer_label


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

    # Process each cut
    for i in range(1, 6):
        cut_key = f"cut{i}"
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
        # If medial wall is provided, find vertices that border it
        if mwall_set:
            # Find cut vertices that are neighbors of medial wall vertices
            endpoints = []
            for v in cut_vertices:
                for neighbor in G.neighbors(v):
                    if neighbor in mwall_set:
                        endpoints.append(v)
                        break

            # If we found exactly 2 endpoints, great!
            if len(endpoints) == 2:
                start, end = endpoints[0], endpoints[1]
                print(f"  Found endpoints at medial wall border: {start}, {end}")
            else:
                print(
                    f"  Found {len(endpoints)} medial wall border vertices, using geometric endpoints instead"
                )
                endpoints = None
        else:
            endpoints = None

        # Fallback: use geometric endpoints (most distant vertices)
        if endpoints is None or len(endpoints) != 2:
            # Find the two most distant vertices in the cut using inflated surface
            max_dist = 0
            start, end = cut_vertices[0], cut_vertices[0]

            for idx1, v1 in enumerate(cut_vertices):
                pos1 = pts_inflated[v1]
                for v2 in cut_vertices[idx1 + 1 :]:
                    dist = np.linalg.norm(pos1 - pts_inflated[v2])
                    if dist > max_dist:
                        max_dist = dist
                        start, end = v1, v2

            print(
                f"  Using geometric endpoints: {start}, {end} (distance: {max_dist:.2f})"
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

            # Step 3: Replace cut with geodesic path
            vertex_dict[cut_key] = np.array(geodesic_path)

            # Calculate reduction
            reduction_pct = 100 * (1 - len(geodesic_path) / len(cut_vertices))
            print(f"  Reduced by {reduction_pct:.1f}%")

        except nx.NetworkXNoPath:
            print(
                f"  WARNING: No path found between {start} and {end}, keeping original cut"
            )
        except Exception as e:
            print(f"  ERROR: Failed to compute geodesic path: {e}")
            print(f"  Keeping original cut")

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
