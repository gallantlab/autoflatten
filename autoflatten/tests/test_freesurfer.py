"""
Tests for the freesurfer module.
"""

import os
import struct
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from autoflatten.freesurfer import (
    create_label_file,
    create_patch_file,
    is_freesurfer_available,
    read_freesurfer_label,
    run_mris_flatten,
)

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
        Dictionary containing vertices, faces, and vertex dictionaries
    """
    # Create a small set of vertices and faces (simple tetrahedron)
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0.5, 0.866, 0], [0.5, 0.289, 0.816]])

    faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])

    # Create vertex dictionary with mock medial wall and cuts
    vertex_dict = {
        "mwall": [0],  # Vertex 0 is in the medial wall
        "calcarine": [1],  # Vertex 1 is in calcarine cut
    }

    return {"vertices": vertices, "faces": faces, "vertex_dict": vertex_dict}


def test_create_patch_file(mock_surface_data):
    """
    Test creating a FreeSurfer patch file.

    This test verifies that the create_patch_file function correctly creates
    a patch file with the expected format and content.
    """
    vertices = mock_surface_data["vertices"]
    faces = mock_surface_data["faces"]
    vertex_dict = mock_surface_data["vertex_dict"]

    with tempfile.TemporaryDirectory() as temp_dir:
        patch_file = os.path.join(temp_dir, "test.patch")

        # Create the patch file
        filename, patch_vertices = create_patch_file(
            patch_file, vertices, faces, vertex_dict
        )

        # Check that the file was created
        assert os.path.exists(patch_file)

        # Verify the file content
        with open(patch_file, "rb") as fp:
            # Read header: -1 and number of vertices
            header = struct.unpack(">2i", fp.read(8))
            assert header[0] == -1

            # Verify number of vertices in patch
            assert header[1] == len(patch_vertices)

            # For each vertex, read vertex index and coordinates
            vertex_data = []
            for _ in range(header[1]):
                data = struct.unpack(">i3f", fp.read(16))
                # Check that vertex indices are either positive (border) or negative (interior)
                assert data[0] != 0
                vertex_data.append(data)

            # Verify the content matches the expected values
            for i, (idx, coord) in enumerate(patch_vertices):
                # Get the stored data for this vertex
                stored_idx, x, y, z = vertex_data[i]

                # Check if this is a border vertex (positive index) or interior vertex (negative index)
                if stored_idx > 0:
                    # Border vertices have positive indices (1-based)
                    assert stored_idx == idx + 1
                else:
                    # Interior vertices have negative indices (1-based, but negative)
                    assert stored_idx == -(idx + 1)

                # Verify the coordinates match (within floating point precision)
                np.testing.assert_allclose([x, y, z], coord, rtol=1e-5)


def test_read_freesurfer_label():
    """
    Test reading a FreeSurfer label file.

    This test creates a mock FreeSurfer label file and verifies that
    read_freesurfer_label correctly parses it.
    """
    # Create test vertex IDs
    test_vertices = [10, 20, 30, 40, 50]

    with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
        # Write mock FreeSurfer label file content
        temp_file.write("#!ascii label, from subject test lh\n")
        temp_file.write(f"{len(test_vertices)}\n")

        # Write mock vertex data (ID, x, y, z, value)
        for vid in test_vertices:
            temp_file.write(f"{vid} 0.1 0.2 0.3 1.0\n")

        temp_filename = temp_file.name

    try:
        # Read the label file
        vertices = read_freesurfer_label(temp_filename)

        # Verify the parsed vertices
        assert np.array_equal(vertices, np.array([10, 20, 30, 40, 50]))
        assert len(vertices) == len(test_vertices)

    finally:
        # Clean up
        if os.path.exists(temp_filename):
            os.remove(temp_filename)


@requires_freesurfer
def test_create_label_file(monkeypatch):
    """
    Test creating a FreeSurfer label file.

    This test mocks the autoflatten.freesurfer.load_surface function and verifies that
    create_label_file correctly creates a label file.
    """

    # Mock load_surface function
    def mock_load_surface(subject, surface, hemi, subjects_dir=None):
        # Return mock coords and polys
        coords = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
        polys = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
        return coords, polys

    # Apply the monkeypatch
    monkeypatch.setattr("autoflatten.freesurfer.load_surface", mock_load_surface)

    # Test vertex IDs
    vertex_ids = [0, 2, 3]
    with tempfile.TemporaryDirectory() as temp_dir:
        label_file = os.path.join(temp_dir, "test.label")

        # Create the label file
        output_file = create_label_file(vertex_ids, "test_subject", "lh", label_file)

        # Check that the file was created
        assert os.path.exists(output_file)

        # Read the label file and verify its content
        vertices = read_freesurfer_label(output_file)
        assert np.array_equal(vertices, np.array(vertex_ids))


def test_run_mris_flatten(monkeypatch):
    """
    Test running mris_flatten.

    This test mocks the subprocess.run function and verifies that
    run_mris_flatten correctly calls mris_flatten with the expected arguments.
    """
    # Mock subprocess.run
    mock_run_calls = []

    def mock_run(cmd, check=False):
        mock_run_calls.append(cmd)
        return subprocess.CompletedProcess(cmd, returncode=0)

    # Apply the monkeypatch
    monkeypatch.setattr("subprocess.run", mock_run)

    # Mock os.path.exists to always return False (forcing the command to run)
    monkeypatch.setattr("os.path.exists", lambda x: False)

    # Create temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        patch_file = os.path.join(temp_dir, "test.patch")
        # Make sure the patch file exists
        Path(patch_file).touch()

        # Set up environment for testing
        os.environ["SUBJECTS_DIR"] = temp_dir
        subject = "test_subject"
        hemi = "lh"

        # Create subject directories
        subject_dir = os.path.join(temp_dir, subject)
        surf_dir = os.path.join(subject_dir, "surf")
        os.makedirs(surf_dir, exist_ok=True)

        # Run the function with default parameters
        flat_file = run_mris_flatten(subject, hemi, patch_file, surf_dir)

        # Check that mris_flatten was called with the correct arguments
        assert len(mock_run_calls) > 0
        last_call = mock_run_calls[-1]
        assert "mris_flatten" == last_call[0]
        assert "-norand" in last_call
        assert "-seed" in last_call
        assert "0" in last_call
        assert "-threads" in last_call
        assert "16" in last_call
        assert "-distances" in last_call
        assert "30" in last_call
        assert "-n" in last_call
        assert "80" in last_call
        assert "-dilate" in last_call
        assert "1" in last_call
        assert patch_file == last_call[-2]
        assert flat_file == last_call[-1]

        # Test with custom parameters
        mock_run_calls.clear()
        extra_params = {"test_key": "test_value", "flag": True}
        flat_file = run_mris_flatten(
            subject,
            hemi,
            patch_file,
            surf_dir,
            norand=False,
            seed=42,
            threads=8,
            distances=(20, 25),
            n=100,
            dilate=2,
            extra_params=extra_params,
        )

        # Check the command was constructed correctly
        last_call = mock_run_calls[-1]
        assert "mris_flatten" == last_call[0]
        assert "-norand" not in last_call
        assert "-seed" in last_call
        assert "42" in last_call
        assert "-threads" in last_call
        assert "8" in last_call
        assert "-distances" in last_call
        assert "20" in last_call
        assert "25" in last_call
        assert "-n" in last_call
        assert "100" in last_call
        assert "-dilate" in last_call
        assert "2" in last_call
        assert "-test_key" in last_call
        assert "test_value" in last_call
        assert "-flag" in last_call
        assert patch_file == last_call[-2]
        assert flat_file == last_call[-1]
