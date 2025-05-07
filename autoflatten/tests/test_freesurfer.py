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

import autoflatten.freesurfer as fs
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


class TestRunMrisFlatten:
    """
    Tests for the run_mris_flatten function.
    """

    def test_input_patch_not_exists(self, tmp_path, monkeypatch):
        subject = "subj"
        hemi = "lh"
        surf_dir = tmp_path / "subjects" / subject / "surf"
        surf_dir.mkdir(parents=True)
        monkeypatch.setattr(fs, "_resolve_subject_dir", lambda subj: str(surf_dir))

        with pytest.raises(FileNotFoundError):
            run_mris_flatten(
                subject, hemi, str(tmp_path / "no.patch"), str(tmp_path / "outdir")
            )

    def test_output_exists_no_overwrite(self, tmp_path, monkeypatch):
        subject = "subj"
        hemi = "lh"
        surf_dir = tmp_path / "subjects" / subject / "surf"
        surf_dir.mkdir(parents=True)
        patch_file = tmp_path / "in.patch"
        patch_file.write_text("patch")
        output_dir = tmp_path / "outdir"
        output_dir.mkdir()
        output_name = "out.patch"
        existing_output = output_dir / output_name
        existing_output.write_text("old")

        monkeypatch.setattr(fs, "_resolve_subject_dir", lambda subj: str(surf_dir))
        monkeypatch.setattr(
            fs,
            "_run_command",
            lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("_run_command should not be called")
            ),
        )

        result = run_mris_flatten(
            subject,
            hemi,
            str(patch_file),
            str(output_dir),
            output_name=output_name,
            overwrite=False,
        )
        assert result == str(existing_output)
        assert existing_output.read_text() == "old"

    def test_overwrite_true_creates_and_cleans(self, tmp_path, monkeypatch):
        subject = "subj"
        hemi = "lh"
        subjects_root = tmp_path / "subjects"
        surf_dir = subjects_root / subject / "surf"
        surf_dir.mkdir(parents=True)
        patch_file = tmp_path / "in.patch"
        patch_file.write_text("patch")
        output_dir = tmp_path / "outdir"
        output_dir.mkdir()
        output_name = "out.patch"
        existing_output = output_dir / output_name
        existing_output.write_text("old")

        monkeypatch.setattr(fs, "_resolve_subject_dir", lambda subj: str(surf_dir))

        def fake_run_command(cmd, cwd, log_path):
            flat = cmd[-1]
            with open(os.path.join(cwd, flat), "w") as f:
                f.write("new")
            with open(os.path.join(cwd, flat + ".out"), "w") as f:
                f.write("out")
            with open(log_path, "w") as f:
                f.write("log")
            return 0

        monkeypatch.setattr(fs, "_run_command", fake_run_command)

        result = run_mris_flatten(
            subject,
            hemi,
            str(patch_file),
            str(output_dir),
            output_name=output_name,
            overwrite=True,
        )
        assert result == str(existing_output)
        assert existing_output.read_text() == "new"
        assert not (surf_dir / "in.patch").exists()
        assert not (surf_dir / output_name).exists()
        assert not (surf_dir / (output_name + ".log")).exists()
        assert not (surf_dir / (output_name + ".out")).exists()

    def test_command_failure_raises(self, tmp_path, monkeypatch):
        subject = "subj"
        hemi = "lh"
        surf_dir = tmp_path / "subjects" / subject / "surf"
        surf_dir.mkdir(parents=True)
        patch_file = tmp_path / "in.patch"
        patch_file.write_text("patch")
        output_dir = tmp_path / "outdir"
        output_dir.mkdir()

        monkeypatch.setattr(fs, "_resolve_subject_dir", lambda subj: str(surf_dir))

        def fake_fail(cmd, cwd, log_path):
            with open(log_path, "w") as f:
                f.write("error")
            return 1

        monkeypatch.setattr(fs, "_run_command", fake_fail)

        with pytest.raises(RuntimeError):
            run_mris_flatten(
                subject, hemi, str(patch_file), str(output_dir), overwrite=True
            )

    def test_default_output_name(self, tmp_path, monkeypatch):
        subject = "subj"
        hemi = "rh"
        surf_dir = tmp_path / "subjects" / subject / "surf"
        surf_dir.mkdir(parents=True)
        patch_file = tmp_path / "in.patch"
        patch_file.write_text("patch")
        output_dir = tmp_path / "outdir"
        output_dir.mkdir()

        monkeypatch.setattr(fs, "_resolve_subject_dir", lambda subj: str(surf_dir))

        def fake_run(cmd, cwd, log_path):
            flat = cmd[-1]
            with open(os.path.join(cwd, flat), "w") as f:
                f.write("ok")
            with open(log_path, "w") as f:
                f.write("log")
            return 0

        monkeypatch.setattr(fs, "_run_command", fake_run)
        result = run_mris_flatten(subject, hemi, str(patch_file), str(output_dir))
        fname = os.path.basename(result)
        assert fname.startswith("rh.autoflatten") and fname.endswith(".flat.patch.3d")
        assert os.path.isfile(result)
