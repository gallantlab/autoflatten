import os
import shutil
import struct
import subprocess
import tempfile

import cortex
import networkx as nx
import nibabel as nib
import numpy as np


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
    pts_inflated, polys = cortex.db.get_surf(subject, "inflated", hemisphere=hemi)

    # Also get fiducial for accurate path finding
    pts_fiducial, _ = cortex.db.get_surf(subject, "fiducial", hemisphere=hemi)

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
            f"Final continuous cut has {len(final_vertices)} vertices (originally {len(cut_vertices)})"
        )

    return vertex_dict


def create_patch_file(filename, vertices, faces, vertex_dict, coords=None):
    """
    Create a FreeSurfer patch file based on vertex and face information

    Parameters
    ----------
    filename : str
        Output filename for the patch
    vertices : array-like
        Array of vertex coordinates with shape (n_vertices, 3)
    faces : array-like
        Array of face indices with shape (n_faces, 3)
    vertex_dict : dict
        Dictionary containing lists of vertex indices for:
        - 'mwall': medial wall vertices to exclude
        - 'cut1' through 'cut5': vertices for the five cuts to exclude
    coords : array-like, optional
        Alternative coordinates to use (e.g., inflated). If None, uses vertices.

    Returns
    -------
    filename : str
        The filename of the created patch file
    patch_vertices : list
        List of vertices included in the patch file
    """
    if coords is None:
        coords = vertices

    # Collect all vertices to exclude (medial wall and all cuts)
    excluded_vertices = set(vertex_dict["mwall"])
    for i in range(1, 6):
        cut_key = f"cut{i}"
        if cut_key in vertex_dict:
            excluded_vertices.update(vertex_dict[cut_key])

    # Find vertices that are adjacent to cuts (border vertices)
    border_vertices = set()

    # Create a vertex adjacency list
    adjacency = [set() for _ in range(len(vertices))]
    for face in faces:
        for i in range(3):
            adjacency[face[i]].update([face[j] for j in range(3) if j != i])

    # Find vertices adjacent to cuts but not in cuts themselves
    for cut_key in [f"cut{i}" for i in range(1, 6) if f"cut{i}" in vertex_dict]:
        for v in vertex_dict[cut_key]:
            for neighbor in adjacency[v]:
                if neighbor not in excluded_vertices:
                    border_vertices.add(neighbor)

    # Collect vertices used in faces (excluding the excluded vertices)
    included_vertices = set()
    for face in faces:
        # Skip faces if any of its vertices are excluded
        if all(v not in excluded_vertices for v in face):
            included_vertices.update(face)

    # Create list of vertices to include in the patch file
    patch_vertices = []
    for v in included_vertices:
        patch_vertices.append((v, coords[v]))

    # Write the patch file
    with open(filename, "wb") as fp:
        fp.write(struct.pack(">2i", -1, len(patch_vertices)))

        for idx, coord in patch_vertices:
            if idx in border_vertices:
                # Border vertices get positive indices
                fp.write(struct.pack(">i3f", idx + 1, *coord))
            else:
                # Interior vertices get negative indices
                fp.write(struct.pack(">i3f", -(idx + 1), *coord))

    print(f"Created patch file {filename} with {len(patch_vertices)} vertices")
    print(f"Excluded {len(excluded_vertices)} vertices (medial wall and cuts)")
    print(
        f"Marked {len(border_vertices & included_vertices)} vertices as border vertices"
    )

    return filename, patch_vertices


def setup_freesurfer(freesurfer_home=None, subjects_dir=None):
    """
    Set up FreeSurfer environment variables within a Jupyter notebook

    Parameters:
    -----------
    freesurfer_home : str, optional
        Path to FreeSurfer installation directory.
        If None, will try to use existing FREESURFER_HOME or default locations
    subjects_dir : str, optional
        Path to subjects directory. If None, will use existing SUBJECTS_DIR
        or default to $FREESURFER_HOME/subjects

    Returns:
    --------
    bool
        True if setup was successful, False otherwise
    """
    # Try to find FreeSurfer home directory
    if freesurfer_home is None:
        # Check if already set
        freesurfer_home = os.environ.get("FREESURFER_HOME")

        # If not set, try common locations
        if not freesurfer_home:
            common_locations = [
                "/usr/local/freesurfer",
                "/opt/freesurfer",
                "/Applications/freesurfer",
                os.path.expanduser("~/freesurfer"),
            ]

            for loc in common_locations:
                if os.path.exists(loc):
                    freesurfer_home = loc
                    break

    if not freesurfer_home or not os.path.exists(freesurfer_home):
        print(
            "FreeSurfer installation not found. Please specify the path to FreeSurfer."
        )
        return False

    # Set essential FreeSurfer environment variables
    os.environ["FREESURFER_HOME"] = freesurfer_home

    # Handle subjects directory
    if subjects_dir is None:
        # Keep existing SUBJECTS_DIR if set
        subjects_dir = os.environ.get("SUBJECTS_DIR")
        if not subjects_dir:
            # Default to $FREESURFER_HOME/subjects
            subjects_dir = os.path.join(freesurfer_home, "subjects")

    # Ensure the subjects directory exists
    if not os.path.exists(subjects_dir):
        print(
            f"Warning: Subjects directory {subjects_dir} does not exist. Creating it..."
        )
        try:
            os.makedirs(subjects_dir, exist_ok=True)
        except Exception as e:
            print(f"Failed to create subjects directory: {e}")

    os.environ["SUBJECTS_DIR"] = subjects_dir
    print(f"Using subjects directory: {subjects_dir}")

    # Set PATH to include FreeSurfer binaries
    fs_bin = os.path.join(freesurfer_home, "bin")
    current_path = os.environ.get("PATH", "")
    if fs_bin not in current_path:
        os.environ["PATH"] = f"{fs_bin}:{current_path}"

    # FreeSurfer configuration file
    fs_setup = os.path.join(freesurfer_home, "SetUpFreeSurfer.sh")
    if os.path.exists(fs_setup):
        # Get environment variables from the setup script
        try:
            # This command sources the FreeSurfer setup and prints all environment variables
            cmd = f"source {fs_setup} && env"
            output = subprocess.check_output(cmd, shell=True, executable="/bin/bash")
            output = output.decode("utf-8")

            # Parse and set environment variables
            for line in output.split("\n"):
                if "=" in line:
                    key, value = line.split("=", 1)
                    # Don't overwrite SUBJECTS_DIR with the one from the script
                    if key != "SUBJECTS_DIR":
                        os.environ[key] = value

            print(f"FreeSurfer environment set up successfully from {fs_setup}")
        except subprocess.CalledProcessError:
            print(f"Failed to source {fs_setup}, continuing with basic setup")

    # Verify setup
    try:
        version = subprocess.check_output(
            ["mri_info", "--version"], stderr=subprocess.STDOUT
        )
        print(f"FreeSurfer setup successful: {version.decode('utf-8').strip()}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("FreeSurfer setup incomplete. Tools may not work correctly.")
        return False


def get_medial_wall_verts(subject, hemis=None):
    """
    Get the medial wall of a subject's surface.

    Parameters
    ----------
    subject : str
        Pycortex subject identifier.

    Returns
    -------
    medial_wall_vertices: list or array
        List of vertices that are part of the medial wall.
    """
    # Run for each hemisphere separately
    if hemis is None:
        hemis = ["lh", "rh"]
    elif isinstance(hemis, str):
        hemis = [hemis]
    medial_wall_vertices = []
    n_vertices = 0.0
    for hemi in hemis:
        pts_full, polys_full, potential_medial_wall = get_removed_vertices(
            subject, hemi
        )

        G = make_graph_from_poly(pts_full, polys_full)
        G_mwall = G.subgraph(potential_medial_wall)
        # filter the medial wall graph to get only the medial wall
        G_mwall, largest_cc = filter_medial_wall_graph(G_mwall)
        # Get the medial wall vertices
        medial_wall_vertices.append(np.array(sorted(list(largest_cc))) + n_vertices)
        n_vertices += len(pts_full)
    # Concatenate the medial wall vertices from both hemispheres
    medial_wall_vertices = np.concatenate(medial_wall_vertices)
    return medial_wall_vertices.astype(int)


def get_removed_vertices(subject, hemi):
    pts_full, polys_full = cortex.db.get_surf(subject, "fiducial", hemisphere=hemi)
    pts_flat, polys_flat = cortex.db.get_surf(subject, "flat", hemisphere=hemi)
    potential_medial_wall = np.array(
        list(set(np.arange(len(pts_flat))) - set(np.unique(polys_flat)))
    )

    return pts_full, polys_full, potential_medial_wall


def make_graph_from_poly(pts, polys):
    """
    Create a graph from a set of points and polygons.

    Parameters
    ----------
    pts : numpy.ndarray
        Array of points (vertices).
    polys : numpy.ndarray
        Array of polygons (triangles).

    Returns
    -------
    G : networkx.Graph
        Graph representation of the surface.
    """

    # Create a graph
    G = nx.Graph()

    # Add nodes for each vertex
    G.add_nodes_from(range(len(pts)))

    # Add edges for each polygon (triangle)
    for triangle in polys:
        for i in range(3):
            v1 = triangle[i]
            for j in range(i + 1, 3):
                v2 = triangle[j]
                G.add_edge(v1, v2)

    return G


def filter_medial_wall_graph(G_mwall, min_degree=2, max_iterations=10):
    """
    Iteratively filter a medial wall graph by retaining vertices with degree > min_degree
    and extracting the largest connected component in each iteration.

    Parameters
    ----------
    G_mwall : networkx.Graph
        The initial medial wall graph
    min_degree : int, optional
        Minimum degree threshold for vertices to keep (default: 2)
    max_iterations : int, optional
        Maximum number of filtering iterations to perform (default: 10)

    Returns
    -------
    G_filtered : networkx.Graph
        The filtered graph after all iterations
    largest_cc : set
        The largest connected component in the final graph
    """

    G_current = G_mwall
    prev_size = 0

    for i in range(max_iterations):
        # Find nodes with degree > min_degree
        potential_nodes = []
        for node in G_current.nodes():
            if len(list(G_current.neighbors(node))) > min_degree:
                potential_nodes.append(node)

        # Create subgraph of high-degree nodes
        G_current = G_current.subgraph(potential_nodes)

        # Get connected components
        components = list(nx.connected_components(G_current))

        # Print component sizes
        cc_sizes = [len(c) for c in sorted(components, key=len, reverse=True)]
        print(f"Iteration {i + 1} component sizes: {cc_sizes}")

        # Extract largest connected component
        if len(components) > 0:
            largest_cc = max(components, key=len)
            G_current = G_current.subgraph(largest_cc)
            current_size = len(largest_cc)

            # Check if we have only one component and its size hasn't changed over two iterations
            if len(components) == 1 and current_size == prev_size:
                print(
                    f"Stopping early at iteration {i + 1}: single component with stable size {current_size}"
                )
                break

            # Update size history
            prev_size = current_size
        else:
            print("No connected components found")
            break

    return G_current, set(G_current.nodes())


def get_mwall_and_cuts_verts(subject, hemi):
    """
    Get the medial wall vertices and cut vertices of a subject's surface.

    Parameters
    ----------
    subject : str
        Pycortex subject identifier.
    hemi : str
        Hemisphere identifier ('lh' or 'rh').

    Returns
    -------
    vertex_dict : dict
        Dictionary containing the medial wall vertices and cut vertices.
        Keys are "mwall" and "cut1", "cut2", etc.
    """
    # Get all vertices, polygons, and vertices removed from the flat surface
    pts_full, polys_full, removed_vertices = get_removed_vertices(subject, hemi)
    # Identify vertices that belong to the medial wall
    mwall_vertices = get_medial_wall_verts(subject, hemi)
    # Calculate which removed vertices are cuts
    # vertices removed but not part of medial wall)
    cuts_vertices = np.setdiff1d(removed_vertices, mwall_vertices)
    # Create a graph representation of the surface
    G = make_graph_from_poly(pts_full, polys_full)
    # Create a subgraph containing only the cut vertices
    G_cuts = G.subgraph(cuts_vertices)
    # Find connected components in the cuts subgraph
    # Each component should be a separate cut in the surface
    cut_components = list(nx.connected_components(G_cuts))
    assert len(cut_components) == 5, f"Expected 5 cuts, found {len(cut_components)}"
    # Convert each connected component to a list of vertices
    cut_vertices = [list(cc) for cc in cut_components]
    # Verify that all removed vertices are accounted for
    # (either as medial wall or as cuts)
    n_removed_vertices = len(removed_vertices)
    n_mwall_vertices = len(mwall_vertices)
    n_cut_vertices = sum([len(v) for v in cut_vertices])
    assert (
        n_removed_vertices == n_mwall_vertices + n_cut_vertices
    ), "The number of removed vertices does not match the sum of mwall and cut vertices."
    # Store into a dict
    vertex_dict = {
        "mwall": np.array(sorted(mwall_vertices)),
    }
    for i, cv in enumerate(cut_vertices):
        vertex_dict[f"cut{i + 1}"] = np.array(sorted(cv))

    return vertex_dict


def create_label_file(vertex_ids, subject, hemi, output_file):
    """
    Create a FreeSurfer label file from a list of vertex IDs

    Parameters:
    -----------
    vertex_ids : list or array
        List of vertex IDs to include in the label
    subject : str
        Subject ID (needed for the header)
    hemi : str
        Hemisphere ('lh' or 'rh')
    output_file : str
        Path to output label file
    """
    # Header information
    header = f"#!ascii label  , from subject {subject} {hemi}"

    # FreeSurfer label files contain 5 columns:
    # vertex_id  x  y  z  value
    # We need to get the coordinates for each vertex

    # Get the surface coordinates from the subject's surface file
    # You can use nibabel for this, or call a FreeSurfer command

    coords, polys = cortex.db.get_surf(subject, "inflated", hemi)

    # Create the label data
    n_vertices = len(vertex_ids)
    label_data = np.zeros((n_vertices, 5))

    for i, vid in enumerate(vertex_ids):
        label_data[i, 0] = vid
        label_data[i, 1:4] = coords[vid]  # x, y, z coordinates
        label_data[i, 4] = 1.0  # Value (typically 1.0)

    # Write the file
    with open(output_file, "w") as f:
        f.write(header + "\n")
        f.write(str(n_vertices) + "\n")
        np.savetxt(f, label_data, fmt="%d %.6f %.6f %.6f %.6f")

    return output_file


def read_freesurfer_label(label_file):
    """
    Parse a FreeSurfer label file and extract vertex IDs.

    Parameters:
    -----------
    label_file : str
        Path to the FreeSurfer label file

    Returns:
    --------
    vertices : list
        List of vertex IDs (integers) in the label
    """
    vertices = []

    with open(label_file, "r") as f:
        lines = f.readlines()

        # Skip header lines (first line is a comment, second is vertex count)
        # Header format: #!ascii label, from subject subject_name hemi
        # Second line: number of vertices
        header_line_count = 2

        # Get number of vertices from the second line
        num_vertices = int(lines[1].strip())

        # Parse vertex IDs (first column in the data)
        for i in range(header_line_count, len(lines)):
            line = lines[i].strip()
            if line and not line.startswith("#"):  # Skip empty lines and comments
                # Format: vertex_id x y z value
                parts = line.split()
                if len(parts) >= 5:  # Ensure line has all components
                    vertex_id = int(float(parts[0]))
                    vertices.append(vertex_id)

    # Verify we got the expected number of vertices
    if len(vertices) != num_vertices:
        print(f"Warning: Expected {num_vertices} vertices but found {len(vertices)}")

    return np.array(sorted(vertices))


def map_cuts_to_subject(vertex_dict, target_subject, hemi, source_subject="fsaverage"):
    """
    Map cutting vertices from a source subject to a target subject using FreeSurfer's mri_label2label.

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
