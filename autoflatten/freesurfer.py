"""Utility functions to interface with FreeSurfer."""

import os
import shutil
import struct
import subprocess

import nibabel as nib
import numpy as np


def load_surface(subject, type, hemi, subjects_dir=None):
    """Load FreeSurfer surface information.

    Parameters
    ----------
    subject : str
        PyCortex or FreeSurfer subject identifier
    type : str
        Type of surface ('white', 'pial', 'inflated', etc.)
    hemi : str
        Hemisphere ('lh' or 'rh')
    subjects_dir : str, optional
        Path to the FreeSurfer subjects directory. If None, uses the
        SUBJECTS_DIR environment variable.

    Returns
    -------
    coords : ndarray
        Array of vertex coordinates with shape (n_vertices, 3)
    faces : ndarray
        Array of face indices with shape (n_faces, 3)
    """
    try:
        # Try to load from PyCortex database first for backward compatibility
        import cortex

        coords, faces = cortex.db.get_surf(subject, type, hemi)
    except KeyError:
        # We don't have the subject in the database,
        # so we need to load it from FreeSurfer
        # Get the subject's surf directory from SUBJECTS_DIR
        subjects_dir = os.environ.get("SUBJECTS_DIR", subjects_dir)
        if subjects_dir is None:
            raise ValueError("SUBJECTS_DIR environment variable not set")
        subject_surf_dir = os.path.join(subjects_dir, subject, "surf")
        # Construct the file path
        surf_file = os.path.join(subject_surf_dir, f"{hemi}.{type}")
        if not os.path.exists(surf_file):
            raise FileNotFoundError(f"Surface file {surf_file} not found")
        # Load the surface using nibabel
        surf_data = nib.freesurfer.read_geometry(surf_file)
        coords, faces = surf_data
    return coords, faces


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
    coords, polys = load_surface(subject, "inflated", hemi)

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


def _build_mris_flatten_cmd(
    norand=None,
    seed=None,
    threads=None,
    distances=None,
    n=None,
    dilate=None,
    extra_params=None,
):
    """
    Build the command list for mris_flatten.

    Parameters
    ----------
    norand : bool
        Whether to use the -norand flag
    seed : int
        Random seed value to use with -seed flag
    threads : int
        Number of threads to use
    distances : tuple of int
        Distance parameters as a tuple (distance1, distance2)
    n : int
        Maximum number of iterations to run, used with -n flag
    dilate : int
        Number of dilations to perform, used with -dilate flag
    extra_params : dict, optional
        Dictionary of additional parameters to pass to mris_flatten as -key value pairs

    Returns
    -------
    list
        List of command line arguments for mris_flatten
    """
    cmd = ["mris_flatten"]

    # Add mandatory parameters
    if norand is not None and norand:
        cmd.append("-norand")
    if seed is not None:
        cmd.extend(["-seed", str(seed)])
    if threads is not None:
        cmd.extend(["-threads", str(threads)])
    if distances is not None:
        cmd.extend(["-distances", str(distances[0]), str(distances[1])])
    if n is not None:
        cmd.extend(["-n", str(n)])
    if dilate is not None:
        cmd.extend(["-dilate", str(dilate)])

    # Add any extra parameters
    if extra_params:
        for key, value in extra_params.items():
            if value is None:
                cmd.append(f"-{key}")
            elif isinstance(value, bool):
                if value:  # Only add flag if True
                    cmd.append(f"-{key}")
            elif isinstance(value, (list, tuple)):
                cmd.append(f"-{key}")
                cmd.extend([str(v) for v in value])
            else:
                cmd.append(f"-{key}")
                cmd.append(str(value))

    return cmd


def _resolve_subject_dir(subject, subjects_dir=None):
    """
    Resolve the subject's surf directory.

    Parameters
    ----------
    subject : str
        FreeSurfer subject identifier
    subjects_dir : str, optional
        Path to the FreeSurfer subjects directory. If None, uses the
        SUBJECTS_DIR environment variable.

    Returns
    -------
    str
        Path to the subject's surf directory.

    Raises
    ------
    ValueError
        If the SUBJECTS_DIR environment variable is not set.
    FileNotFoundError
        If the subject's surf directory does not exist.
    """
    subjects_dir = os.environ.get("SUBJECTS_DIR", subjects_dir)
    if not subjects_dir:
        raise ValueError("SUBJECTS_DIR environment variable not set")
    surf_dir = os.path.join(subjects_dir, subject, "surf")
    if not os.path.isdir(surf_dir):
        raise FileNotFoundError(f"Subject surf directory not found: {surf_dir}")
    return surf_dir


def _stage_patch_file(patch_file, surf_dir):
    """
    Stage the patch file in the subject's surf directory.

    Parameters
    ----------
    patch_file : str
        Path to the input patch file.
    surf_dir : str
        Path to the subject's surf directory.

    Returns
    -------
    tuple
        A tuple containing the basename of the patch file, the staged file path,
        and a boolean indicating whether the file was copied.
    """
    basename = os.path.basename(patch_file)
    dest = os.path.join(surf_dir, basename)
    copied = False
    if os.path.abspath(patch_file) != os.path.abspath(dest):
        shutil.copy2(patch_file, dest)
        copied = True
    return basename, dest, copied


def _run_command(cmd, cwd, log_path):
    """
    Run a command and log its output.

    Parameters
    ----------
    cmd : list
        Command to execute as a list of arguments.
    cwd : str
        Working directory to execute the command in.
    log_path : str
        Path to the log file to write stdout and stderr.

    Returns
    -------
    int
        Return code of the command.
    """
    print(f"Running command: {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    with open(log_path, "w") as f:
        f.write(proc.stdout)
        f.write(proc.stderr)
    return proc.returncode


def run_mris_flatten(
    subject,
    hemi,
    patch_file,
    output_dir,
    output_name=None,
    norand=True,
    seed=0,
    threads=16,
    distances=(15, 80),
    n=80,
    dilate=1,
    extra_params=None,
    overwrite=False,
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
        Path to the input patch file. Must exist.
    output_dir : str
        Directory to save the final output flat patch file and its log.
    output_name : str, optional
        Base name for the output flat patch file (e.g., 'lh.myflat.patch.3d').
        If None, a default name based on parameters will be generated.
    norand : bool, optional
        Whether to use the -norand flag (default True).
    seed : int, optional
        Random seed value to use with -seed flag (default 0).
    threads : int, optional
        Number of threads to use (default 16).
    distances : tuple of int, optional
        Distance parameters as a tuple (distance1, distance2) (default: (15, 80)).
    n : int, optional
        Maximum number of iterations to run, used with -n flag (default 80).
    dilate : int, optional
        Number of dilations to perform, used with -dilate flag (default 1).
    extra_params : dict, optional
        Dictionary of additional parameters to pass to mris_flatten as -key value pairs.
    overwrite : bool, optional
        Whether to overwrite existing output files (default False).

    Returns
    -------
    str
        Path to the final output flat surface file in `output_dir`.

    Raises
    ------
    FileNotFoundError
        If the input `patch_file` does not exist or the subject's surf directory cannot be found.
    ValueError
        If the `SUBJECTS_DIR` environment variable is not set.
    RuntimeError
        If the `mris_flatten` command fails.
    """
    # validate input patch
    if not os.path.isfile(patch_file):
        raise FileNotFoundError(f"Input patch file not found: {patch_file}")

    surf_dir = _resolve_subject_dir(subject)
    os.makedirs(output_dir, exist_ok=True)

    # Generate output filename if not provided
    if output_name is None:
        distances_str = f"distances{distances[0]:02d}{distances[1]:02d}"
        output_name = f"{hemi}.autoflatten_{distances_str}_n{n}_dilate{dilate}"
        if extra_params:
            passes = extra_params.get("p", 1)
            if passes > 1:
                output_name += f"_passes{passes}"
        output_name += f"_seed{seed}"
        output_name += ".flat.patch.3d"

    final_flat_file = os.path.join(output_dir, output_name)
    final_log_file = os.path.splitext(final_flat_file)[0] + ".log"
    final_out_file = final_flat_file + ".out"

    # skip or remove existing outputs
    if os.path.exists(final_flat_file):
        if not overwrite:
            print(f"{final_flat_file} exists, skipping (overwrite=False)")
            return final_flat_file
        for f in (final_flat_file, final_log_file, final_out_file):
            if os.path.exists(f):
                os.remove(f)

    # prepare basenames and temp paths
    flat_basename = os.path.basename(final_flat_file)
    temp_log = os.path.splitext(os.path.join(surf_dir, flat_basename))[0] + ".log"
    temp_out = os.path.join(surf_dir, flat_basename + ".out")

    # stage patch
    patch_basename, temp_patch, copied = _stage_patch_file(patch_file, surf_dir)

    # build and run
    cmd = _build_mris_flatten_cmd(
        norand, seed, threads, distances, n, dilate, extra_params
    )
    cmd += [patch_basename, flat_basename]
    ret = _run_command(cmd, cwd=surf_dir, log_path=temp_log)

    # on failure: copy logs back, clean staged patch, then error
    if ret != 0:
        # copy stderr/stdout log
        if os.path.abspath(temp_log) != os.path.abspath(final_log_file):
            shutil.copy2(temp_log, final_log_file)
        # copy mris_flatten .out if exists
        if os.path.exists(temp_out):
            if os.path.abspath(temp_out) != os.path.abspath(final_out_file):
                shutil.copy2(temp_out, final_out_file)
        if copied:
            os.remove(temp_patch)
        raise RuntimeError(f"mris_flatten failed (see {final_log_file})")

    # on success: copy outputs
    src_flat = os.path.join(surf_dir, flat_basename)
    if os.path.abspath(src_flat) != os.path.abspath(final_flat_file):
        shutil.copy2(src_flat, final_flat_file)
    src_log = temp_log
    if os.path.abspath(src_log) != os.path.abspath(final_log_file):
        shutil.copy2(src_log, final_log_file)
    if os.path.exists(temp_out):
        if os.path.abspath(temp_out) != os.path.abspath(final_out_file):
            shutil.copy2(temp_out, final_out_file)

    # cleanup only if patch was copied and surf_dir != output_dir
    if copied and os.path.abspath(surf_dir) != os.path.abspath(output_dir):
        os.remove(temp_patch)
        os.remove(os.path.join(surf_dir, flat_basename))
        os.remove(temp_log)
        if os.path.exists(temp_out):
            os.remove(temp_out)

    return final_flat_file
