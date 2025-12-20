"""Utility functions to interface with FreeSurfer."""

import os
import shutil
import struct
import subprocess
import tempfile

import nibabel as nib
import numpy as np


def read_surface(filepath):
    """Read FreeSurfer surface file.

    Low-level function to read a surface file directly by path.

    Parameters
    ----------
    filepath : str
        Path to FreeSurfer surface file (.white, .pial, .inflated, etc.)

    Returns
    -------
    vertices : ndarray of shape (N, 3)
        Vertex coordinates.
    faces : ndarray of shape (F, 3)
        Triangle indices.
    """
    vertices, faces = nib.freesurfer.read_geometry(filepath)
    return vertices, faces


def load_surface(subject, type, hemi, subjects_dir=None):
    """Load FreeSurfer surface information.

    Parameters
    ----------
    subject : str
        FreeSurfer subject identifier
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

    # Collect all vertices to exclude (all keys in vertex_dict: medial wall, cuts, holes, etc.)
    excluded_vertices = set()
    for key, verts in vertex_dict.items():
        excluded_vertices.update(int(v) for v in verts)

    # Find vertices that are adjacent to cuts (border vertices)
    border_vertices = set()

    # Create a vertex adjacency list
    adjacency = [set() for _ in range(len(vertices))]
    for face in faces:
        for i in range(3):
            adjacency[face[i]].update([face[j] for j in range(3) if j != i])

    # Find vertices adjacent to excluded vertices (cuts, holes, medial wall)
    # These form the border of the patch
    for key, verts in vertex_dict.items():
        for v in verts:
            v_int = int(v)
            if v_int < len(adjacency):
                for neighbor in adjacency[v_int]:
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
            # Convert to Python int to avoid unsigned integer overflow
            # (numpy uint32 indices would wrap around when negated)
            idx_int = int(idx)
            # FreeSurfer convention: negative = border, positive = interior
            if idx in border_vertices:
                # Border vertices get negative indices -(idx + 1)
                fp.write(struct.pack(">i3f", -(idx_int + 1), *coord))
            else:
                # Interior vertices get positive indices (idx + 1)
                fp.write(struct.pack(">i3f", idx_int + 1, *coord))

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
    seed=None,
    threads=None,
    distances=None,
    n=None,
    dilate=None,
    passes=None,
    tol=None,
    extra_params=None,
):
    """
    Build the command list for mris_flatten.

    Parameters
    ----------
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
    passes : int
        Number of passes to perform, used with -p flag
    tol : float
        Tolerance, used with -tol flag
    extra_params : dict, optional
        Dictionary of additional parameters to pass to mris_flatten as -key value pairs

    Returns
    -------
    list
        List of command line arguments for mris_flatten
    """
    cmd = ["mris_flatten"]

    # Add mandatory parameters
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
    if passes is not None:
        cmd.extend(["-p", str(passes)])
    if tol is not None:
        cmd.extend(["-tol", str(tol)])

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


def _create_temp_surf_directory(subject, surf_dir, temp_root):
    """
    Create a temporary surf directory with symlinks to original FreeSurfer files.

    This allows mris_flatten to run in an isolated environment without modifying
    the original subject directory. Creates the structure:
    {temp_root}/{subject}/surf/ with symlinks to all surface files.

    Parameters
    ----------
    subject : str
        FreeSurfer subject identifier
    surf_dir : str
        Original subject's surf directory (absolute path)
    temp_root : str
        Temporary directory root (from tempfile.mkdtemp)

    Returns
    -------
    str
        Path to temporary surf directory where mris_flatten should execute

    Raises
    ------
    OSError
        If temporary directory creation fails
    """
    import glob

    # Create temp structure: {temp_root}/{subject}/surf/
    temp_surf_dir = os.path.join(temp_root, subject, "surf")
    os.makedirs(temp_surf_dir, exist_ok=True)

    # Surface file patterns that mris_flatten might need
    # These are common FreeSurfer surface files that should be accessible
    surface_patterns = [
        "*.white",  # White matter surface
        "*.pial",  # Pial surface
        "*.inflated",  # Inflated surface
        "*.sphere",  # Spherical surface
        "*.sphere.reg",  # Registered sphere
        "*.smoothwm",  # Smoothed white matter
        "*.orig",  # Original surface
        "*.jacobian_white",  # Jacobian
        "*.sulc",  # Sulcal depth
        "*.curv",  # Curvature
        "*.thickness",  # Cortical thickness
        "*.area",  # Surface area
        "*.volume",  # Volume
        "*.avg_curv",  # Average curvature
    ]

    # Create symlinks for all existing surface files
    linked_count = 0
    for pattern in surface_patterns:
        for orig_file in glob.glob(os.path.join(surf_dir, pattern)):
            basename = os.path.basename(orig_file)
            link_path = os.path.join(temp_surf_dir, basename)
            if not os.path.exists(link_path):
                try:
                    os.symlink(orig_file, link_path)
                    linked_count += 1
                except OSError as e:
                    print(f"Warning: Failed to create symlink for {basename}: {e}")

    print(f"Created temporary surf directory with {linked_count} symlinked files")
    return temp_surf_dir


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


def _run_command_with_env(cmd, cwd, log_path, env=None):
    """
    Run a command with custom environment and log its output.

    This is similar to _run_command but allows overriding environment variables,
    particularly SUBJECTS_DIR to point to a temporary location.

    Parameters
    ----------
    cmd : list
        Command to execute as a list of arguments.
    cwd : str
        Working directory to execute the command in.
    log_path : str
        Path to the log file to write stdout and stderr.
    env : dict, optional
        Environment variables for the subprocess. If None, inherits current environment.

    Returns
    -------
    int
        Return code of the command.
    """
    print(f"Running command: {' '.join(cmd)}")
    print(f"Working directory: {cwd}")
    if env and "SUBJECTS_DIR" in env:
        print(f"Using temporary SUBJECTS_DIR: {env['SUBJECTS_DIR']}")

    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, env=env)

    # Write combined output to log file
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
    seed=0,
    threads=16,
    distances=(15, 80),
    n=200,
    dilate=1,
    passes=1,
    tol=0.005,
    extra_params=None,
    overwrite=False,
    debug=False,
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
    seed : int, optional
        Random seed value to use with -seed flag (default 0).
    threads : int, optional
        Number of threads to use (default 16).
    distances : tuple of int, optional
        Distance parameters as a tuple (distance1, distance2) (default: (15, 80)).
    n : int, optional
        Maximum number of iterations to run, used with -n flag (default 200).
    dilate : int, optional
        Number of dilations to perform, used with -dilate flag (default 1).
    passes : int, optional
        Number of passes to perform, used with -p flag (default 1).
    tol : float, optional
        Tolerance passed with -tol flag (default 0.005).
    extra_params : dict, optional
        Dictionary of additional parameters to pass to mris_flatten as -key value pairs.
    overwrite : bool, optional
        Whether to overwrite existing output files (default False).
    debug : bool, optional
        If True, preserve the temporary directory for debugging (default False).

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
    # Validate input patch
    if not os.path.isfile(patch_file):
        raise FileNotFoundError(f"Input patch file not found: {patch_file}")

    # Resolve original surf directory (for reading reference files only)
    original_surf_dir = _resolve_subject_dir(subject)
    os.makedirs(output_dir, exist_ok=True)

    # Generate output filename if not provided
    if output_name is None:
        distances_str = f"distances{distances[0]:02d}{distances[1]:02d}"
        output_name = f"{hemi}.autoflatten_{distances_str}_n{n}_dilate{dilate}"
        if passes > 1:
            output_name += f"_passes{passes}"
        output_name += f"_seed{seed}"
        output_name += ".flat.patch.3d"

    final_flat_file = os.path.join(output_dir, output_name)
    final_log_file = os.path.splitext(final_flat_file)[0] + ".log"
    final_out_file = final_flat_file + ".out"

    # Skip or remove existing outputs
    if os.path.exists(final_flat_file):
        if not overwrite:
            print(f"{final_flat_file} exists, skipping (overwrite=False)")
            return final_flat_file
        for f in (final_flat_file, final_log_file, final_out_file):
            if os.path.exists(f):
                os.remove(f)

    # Create temporary directory for isolated execution
    temp_root = tempfile.mkdtemp(prefix="autoflatten_")
    try:
        # Create temporary surf directory with symlinks to original files
        temp_surf_dir = _create_temp_surf_directory(
            subject, original_surf_dir, temp_root
        )

        # Copy patch file to temp surf directory
        patch_basename = os.path.basename(patch_file)
        temp_patch = os.path.join(temp_surf_dir, patch_basename)
        shutil.copy2(patch_file, temp_patch)
        print(f"Copied patch file to temporary location: {temp_patch}")

        # Prepare output file paths in temp directory
        flat_basename = os.path.basename(final_flat_file)
        temp_log = (
            os.path.splitext(os.path.join(temp_surf_dir, flat_basename))[0] + ".log"
        )
        temp_out = os.path.join(temp_surf_dir, flat_basename + ".out")

        # Build mris_flatten command
        cmd = _build_mris_flatten_cmd(
            seed, threads, distances, n, dilate, passes, tol, extra_params
        )
        cmd += [patch_basename, flat_basename]

        # Create custom environment with temporary SUBJECTS_DIR
        env = os.environ.copy()
        env["SUBJECTS_DIR"] = temp_root

        # Run mris_flatten from temp surf directory
        ret = _run_command_with_env(cmd, cwd=temp_surf_dir, log_path=temp_log, env=env)

        # Handle failure
        if ret != 0:
            # Copy logs to final output directory for debugging
            if os.path.exists(temp_log):
                shutil.copy2(temp_log, final_log_file)
            if os.path.exists(temp_out):
                shutil.copy2(temp_out, final_out_file)
            raise RuntimeError(f"mris_flatten failed (see {final_log_file})")

        # On success: copy all outputs to final location
        src_flat = os.path.join(temp_surf_dir, flat_basename)
        if not os.path.exists(src_flat):
            raise RuntimeError(
                f"mris_flatten succeeded but output file not found: {src_flat}"
            )

        shutil.copy2(src_flat, final_flat_file)
        shutil.copy2(temp_log, final_log_file)
        if os.path.exists(temp_out):
            shutil.copy2(temp_out, final_out_file)

        print(f"Flattening completed successfully: {final_flat_file}")
        return final_flat_file

    finally:
        # Clean up temporary directory unless debug mode
        if os.path.exists(temp_root):
            if debug:
                print(f"Debug mode: Preserving temporary directory at {temp_root}")
            else:
                try:
                    shutil.rmtree(temp_root)
                    print(f"Cleaned up temporary directory: {temp_root}")
                except Exception as e:
                    print(f"Warning: Failed to clean up {temp_root}: {e}")
                    print("You may need to manually remove it.")


# =============================================================================
# Patch file I/O functions (for pyflatten integration)
# =============================================================================


def read_patch(filepath):
    """
    Read FreeSurfer binary patch file.

    FreeSurfer patch files store vertices with their original surface indices.
    Border vertices (on the cut boundary) have positive indices, interior
    vertices have negative indices.

    Parameters
    ----------
    filepath : str
        Path to patch file (e.g., 'lh.cortex.patch.3d').

    Returns
    -------
    vertices : ndarray of shape (N, 3)
        Vertex coordinates.
    original_indices : ndarray of shape (N,)
        Original vertex indices in the full surface (0-based).
    is_border : ndarray of shape (N,) of bool
        True for border vertices (on the patch boundary).

    Notes
    -----
    FreeSurfer patch binary format:
    - Header: 2 big-endian int32 [-1, n_vertices]
    - Per vertex: 1 int32 (signed index) + 3 float32 (x, y, z)
    - Border vertices: negative index (-(idx + 1))
    - Interior vertices: positive index (idx + 1)

    The patch file does NOT contain face information. To get faces,
    use `extract_patch_faces` with the original surface.

    Examples
    --------
    >>> vertices, orig_idx, is_border = read_patch('lh.cortex.patch.3d')
    >>> # Get faces from original surface
    >>> _, orig_faces = load_surface(subject, 'white', 'lh')
    >>> faces = extract_patch_faces(orig_faces, orig_idx)
    """
    with open(filepath, "rb") as fp:
        # Read header
        header = struct.unpack(">2i", fp.read(8))
        if header[0] != -1:
            raise ValueError(f"Invalid patch file header: expected -1, got {header[0]}")
        n_vertices = header[1]

        # Read vertex data
        vertices = np.zeros((n_vertices, 3), dtype=np.float64)
        original_indices = np.zeros(n_vertices, dtype=np.int32)
        is_border = np.zeros(n_vertices, dtype=bool)

        for i in range(n_vertices):
            data = struct.unpack(">i3f", fp.read(16))
            raw_idx = data[0]

            # Decode index: negative = border, positive = interior
            # (FreeSurfer convention: border vertices use -(vno+1))
            if raw_idx < 0:
                original_indices[i] = -raw_idx - 1  # Convert to 0-based
                is_border[i] = True
            else:
                original_indices[i] = raw_idx - 1  # Convert to 0-based
                is_border[i] = False

            vertices[i] = data[1:4]

    return vertices, original_indices, is_border


def write_patch(filepath, vertices, original_indices, is_border=None):
    """
    Write FreeSurfer binary patch file.

    Parameters
    ----------
    filepath : str
        Output path (e.g., 'lh.cortex.flat.patch.3d').
    vertices : ndarray of shape (N, 2) or (N, 3)
        Vertex coordinates. If 2D, z=0 is appended.
    original_indices : ndarray of shape (N,)
        Original vertex indices in the full surface (0-based).
    is_border : ndarray of shape (N,) of bool, optional
        True for border vertices. If None, all vertices are marked as interior.

    Notes
    -----
    FreeSurfer patch binary format:
    - Header: 2 big-endian int32 [-1, n_vertices]
    - Per vertex: 1 int32 (signed index) + 3 float32 (x, y, z)
    - Border vertices: negative index (-(idx + 1))
    - Interior vertices: positive index (idx + 1)

    Examples
    --------
    >>> write_patch('lh.flat.patch.3d', xy_flat, orig_idx, is_border)
    """
    vertices = np.asarray(vertices, dtype=np.float32)
    original_indices = np.asarray(original_indices, dtype=np.int32)

    # Handle 2D input
    if vertices.ndim == 2 and vertices.shape[1] == 2:
        vertices = np.column_stack(
            [vertices, np.zeros(len(vertices), dtype=np.float32)]
        )

    n_vertices = len(vertices)

    if is_border is None:
        is_border = np.zeros(n_vertices, dtype=bool)

    with open(filepath, "wb") as fp:
        # Write header
        fp.write(struct.pack(">2i", -1, n_vertices))

        # Write vertex data
        for i in range(n_vertices):
            # Convert to Python int to avoid unsigned integer overflow
            # (numpy uint32 indices would wrap around when negated)
            idx = int(original_indices[i])
            # FreeSurfer convention: negative = border, positive = interior
            if is_border[i]:
                raw_idx = -(idx + 1)  # Negative for border
            else:
                raw_idx = idx + 1  # Positive for interior

            fp.write(struct.pack(">i3f", raw_idx, *vertices[i]))


def extract_patch_faces(faces, patch_indices):
    """
    Extract faces for a patch from the original surface.

    Given the original surface faces and the vertex indices in the patch,
    returns faces that have all three vertices in the patch, re-indexed
    to patch-local indices.

    Parameters
    ----------
    faces : ndarray of shape (F, 3)
        Original surface faces.
    patch_indices : ndarray of shape (N,)
        Original vertex indices that are in the patch.

    Returns
    -------
    patch_faces : ndarray of shape (G, 3)
        Faces with vertices re-indexed to patch-local indices (0 to N-1).

    Examples
    --------
    >>> vertices, orig_idx, is_border = read_patch('lh.cortex.patch.3d')
    >>> _, orig_faces = load_surface(subject, 'white', 'lh')
    >>> faces = extract_patch_faces(orig_faces, orig_idx)
    """
    # Create mapping from original index to patch index
    idx_set = set(patch_indices)
    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(patch_indices)}

    # Filter faces where all vertices are in the patch
    patch_faces = []
    for face in faces:
        if all(v in idx_set for v in face):
            patch_faces.append([old_to_new[v] for v in face])

    if len(patch_faces) == 0:
        return np.zeros((0, 3), dtype=np.int32)

    return np.array(patch_faces, dtype=np.int32)
