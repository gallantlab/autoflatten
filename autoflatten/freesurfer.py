"""Utility functions to interface with FreeSurfer."""

import os
import shutil
import struct
import subprocess

import cortex
import numpy as np


def is_freesurfer_available():
    """
    Check if FreeSurfer is installed and accessible.

    Returns
    -------
    bool
        True if FreeSurfer is available, False otherwise
    """
    try:
        # Try to run a simple FreeSurfer command
        subprocess.run(
            ["mri_info", "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False


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
        - 'calcarine', 'medial1', 'medial2', 'medial3', 'temporal': vertices for cuts to exclude
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

    # Add vertices from each cut type
    cut_keys = ["calcarine", "medial1", "medial2", "medial3", "temporal"]
    for cut_key in cut_keys:
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
    for cut_key in cut_keys:
        if cut_key in vertex_dict:
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


def run_mris_flatten(
    subject, hemi, patch_file, output_dir, iterations=None, overwrite=False
):
    """
    Run mris_flatten on a patch file to create a flattened surface.

    Parameters
    ----------
    subject : str
        FreeSurfer subject identifier
    hemi : str
        Hemisphere ('lh' or 'rh')
    patch_file : str
        Path to the patch file
    output_dir : str
        Directory to save output files
    iterations : int or None, optional
        Save intermediate flattening patches after this many iterations.
        If None, only the final flat patch is saved.
    overwrite : bool, optional
        Whether to overwrite existing files

    Returns
    -------
    str
        Path to the output flat surface
    """
    # Construct output path
    flat_file = os.path.join(output_dir, f"{hemi}.autoflatten.flat.patch.3d")

    # Check if output file exists and whether to overwrite
    if os.path.exists(flat_file) and not overwrite:
        print(
            f"Flat patch file {flat_file} already exists, skipping (use --overwrite to force)"
        )
        return flat_file

    # Get the subject's surf directory from SUBJECTS_DIR
    subjects_dir = os.environ.get("SUBJECTS_DIR")
    subject_surf_dir = os.path.join(subjects_dir, subject, "surf")

    # If output_dir is not the subject's surf directory, we need to ensure mris_flatten can find
    # the necessary surface files by running it from the surf directory
    if os.path.normpath(output_dir) != os.path.normpath(subject_surf_dir):
        # Copy the patch file to the subject's surf directory
        temp_patch_file = os.path.join(subject_surf_dir, os.path.basename(patch_file))
        temp_flat_file = os.path.join(subject_surf_dir, os.path.basename(flat_file))

        print(f"Copying patch file to subject's surf directory: {temp_patch_file}")
        shutil.copy2(patch_file, temp_patch_file)

        # Change to the subject's surf directory
        original_dir = os.getcwd()
        os.chdir(subject_surf_dir)

        try:
            # Construct mris_flatten command
            cmd = ["mris_flatten"]

            # Add iterations parameter only if specified
            if iterations is not None:
                cmd.extend(["-w", str(iterations)])

            # Add input and output files (just the basenames since we're in the surf directory)
            cmd.extend(
                [os.path.basename(temp_patch_file), os.path.basename(temp_flat_file)]
            )

            print(f"Running from {subject_surf_dir}: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)

            # Copy the flat file back to the output directory
            print(f"Copying flat file back to output directory: {flat_file}")
            shutil.copy2(temp_flat_file, flat_file)

            # Clean up the temporary files
            os.remove(temp_patch_file)
            os.remove(temp_flat_file)

        finally:
            # Change back to the original directory
            os.chdir(original_dir)
    else:
        # We're already in the subject's surf directory, so we can run mris_flatten directly
        cmd = ["mris_flatten"]

        # Add iterations parameter only if specified
        if iterations is not None:
            cmd.extend(["-w", str(iterations)])

        # Add input and output files
        cmd.extend([patch_file, flat_file])

        print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

    return flat_file
