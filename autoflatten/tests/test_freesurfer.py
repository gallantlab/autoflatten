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


@pytest.fixture
def mock_surface_data_uint32():
    """
    Create mock surface data with uint32 faces for testing unsigned integer handling.

    This fixture specifically uses uint32 dtype for faces to test that the patch
    file creation correctly handles unsigned integer indices without overflow.

    Returns
    -------
    dict
        Dictionary containing vertices, faces (uint32), and vertex dictionaries
    """
    # Create a larger mesh to have more vertex indices
    # Simple grid of 10x10 = 100 vertices
    n = 10
    vertices = []
    for i in range(n):
        for j in range(n):
            vertices.append([i, j, 0])
    vertices = np.array(vertices, dtype=np.float32)

    # Create triangular faces (as uint32 - the problematic dtype)
    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            v0 = i * n + j
            v1 = i * n + j + 1
            v2 = (i + 1) * n + j
            v3 = (i + 1) * n + j + 1
            faces.append([v0, v1, v2])
            faces.append([v1, v3, v2])
    faces = np.array(faces, dtype=np.uint32)  # Use uint32 explicitly

    # Create vertex dictionary with mock medial wall (first column)
    vertex_dict = {
        "mwall": list(range(0, n * n, n)),  # First column: 0, 10, 20, ...
        "calcarine": [1, 11],  # A couple of cut vertices
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


def test_create_patch_file_uint32_faces(mock_surface_data_uint32):
    """
    Test creating a patch file with uint32 face indices.

    This test specifically verifies that create_patch_file correctly handles
    surfaces where face indices have dtype uint32 (unsigned 32-bit integers).
    Without proper handling, negating unsigned integers causes overflow:
    -(np.uint32(0) + 1) = 4294967295 instead of -1.

    This regression test ensures the fix for the unsigned integer overflow bug
    remains in place.
    """
    vertices = mock_surface_data_uint32["vertices"]
    faces = mock_surface_data_uint32["faces"]
    vertex_dict = mock_surface_data_uint32["vertex_dict"]

    # Verify we're actually testing with uint32 faces
    assert faces.dtype == np.uint32, "Test requires uint32 faces"

    with tempfile.TemporaryDirectory() as temp_dir:
        patch_file = os.path.join(temp_dir, "test_uint32.patch")

        # Create the patch file - this should not raise an error
        filename, patch_vertices = create_patch_file(
            patch_file, vertices, faces, vertex_dict
        )

        # Check that the file was created with correct size
        # (header 8 bytes + 16 bytes per vertex)
        expected_size = 8 + len(patch_vertices) * 16
        actual_size = os.path.getsize(patch_file)
        assert actual_size == expected_size, (
            f"File size mismatch: expected {expected_size}, got {actual_size}. "
            "This may indicate unsigned integer overflow in vertex index handling."
        )

        # Verify the file content is valid
        with open(patch_file, "rb") as fp:
            header = struct.unpack(">2i", fp.read(8))
            assert header[0] == -1, "Invalid magic number"
            assert header[1] == len(patch_vertices), "Vertex count mismatch"

            # Read and verify each vertex
            for i in range(header[1]):
                data = struct.unpack(">i3f", fp.read(16))
                stored_idx = data[0]

                # Critical check: indices must be in valid int32 range
                # Overflow would result in large positive values like 4294967295
                assert -2147483648 <= stored_idx <= 2147483647, (
                    f"Vertex index {stored_idx} out of int32 range - "
                    "possible unsigned integer overflow"
                )

                # Interior vertices have negative indices, border have positive
                # Neither should be zero
                assert stored_idx != 0, "Invalid vertex index (zero)"

                # Check that negative indices are actually negative (not wrapped)
                # Interior vertices have negative stored_idx = -(original_idx + 1)
                # This verifies no unsigned integer overflow occurred
                if stored_idx < 0:
                    # Verify the stored index decodes correctly back to original
                    decoded_idx = -(stored_idx + 1)
                    assert decoded_idx >= 0, (
                        f"Decoded index should be non-negative, got {decoded_idx}"
                    )


def test_write_patch_uint32_indices():
    """
    Test write_patch with uint32 original_indices.

    This test verifies that write_patch correctly handles original_indices
    arrays with uint32 dtype without unsigned integer overflow.
    """
    from autoflatten.freesurfer import write_patch, read_patch

    # Create test data with uint32 indices (the problematic dtype)
    n_vertices = 50
    vertices = np.random.rand(n_vertices, 3).astype(np.float32)
    original_indices = np.arange(n_vertices, dtype=np.uint32)
    is_border = np.zeros(n_vertices, dtype=bool)
    is_border[:5] = True  # First 5 vertices are border

    with tempfile.TemporaryDirectory() as temp_dir:
        patch_file = os.path.join(temp_dir, "test_write_uint32.patch")

        # Write the patch file - should not raise an error
        write_patch(patch_file, vertices, original_indices, is_border)

        # Verify file size is correct
        expected_size = 8 + n_vertices * 16
        actual_size = os.path.getsize(patch_file)
        assert actual_size == expected_size, (
            f"File size mismatch: expected {expected_size}, got {actual_size}"
        )

        # Read back and verify
        read_vertices, read_indices, read_is_border = read_patch(patch_file)

        # Verify all indices were correctly round-tripped
        np.testing.assert_array_equal(read_indices, original_indices)
        np.testing.assert_array_equal(read_is_border, is_border)
        np.testing.assert_allclose(read_vertices, vertices, rtol=1e-5)


def test_patch_follows_freesurfer_convention():
    """
    Test that patch files follow FreeSurfer's border vertex convention.

    FreeSurfer convention (from mrisurf.c):
    - Border vertices: negative index -(vno+1)
    - Interior vertices: positive index (vno+1)

    This test ensures we don't regress to the inverted convention.
    """
    import struct
    from autoflatten.freesurfer import write_patch

    n_vertices = 10
    vertices = np.random.rand(n_vertices, 3).astype(np.float32)
    original_indices = np.arange(n_vertices, dtype=np.int32)
    is_border = np.zeros(n_vertices, dtype=bool)
    is_border[0] = True  # First vertex is border
    is_border[5] = True  # Sixth vertex is border

    with tempfile.TemporaryDirectory() as temp_dir:
        patch_file = os.path.join(temp_dir, "test_convention.patch")
        write_patch(patch_file, vertices, original_indices, is_border)

        # Read raw bytes and verify FreeSurfer convention
        with open(patch_file, "rb") as fp:
            header = struct.unpack(">2i", fp.read(8))
            assert header[0] == -1, "Invalid header"
            assert header[1] == n_vertices, "Wrong vertex count"

            for i in range(n_vertices):
                raw_idx = struct.unpack(">i", fp.read(4))[0]
                fp.read(12)  # skip xyz

                expected_vno = original_indices[i]
                if is_border[i]:
                    # Border vertices should have NEGATIVE index -(vno+1)
                    expected_raw = -(expected_vno + 1)
                    assert raw_idx < 0, (
                        f"Border vertex {i} should have negative index, got {raw_idx}"
                    )
                else:
                    # Interior vertices should have POSITIVE index (vno+1)
                    expected_raw = expected_vno + 1
                    assert raw_idx > 0, (
                        f"Interior vertex {i} should have positive index, got {raw_idx}"
                    )

                assert raw_idx == expected_raw, (
                    f"Vertex {i}: expected raw index {expected_raw}, got {raw_idx}"
                )


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
        # Create some dummy surface files for symlinking
        (surf_dir / "lh.white").write_text("white")
        (surf_dir / "lh.pial").write_text("pial")
        patch_file = tmp_path / "in.patch"
        patch_file.write_text("patch")
        output_dir = tmp_path / "outdir"
        output_dir.mkdir()
        output_name = "out.patch"
        existing_output = output_dir / output_name
        existing_output.write_text("old")

        monkeypatch.setattr(fs, "_resolve_subject_dir", lambda subj: str(surf_dir))

        def fake_run_command(cmd, cwd, log_path, env=None):
            flat = cmd[-1]
            with open(os.path.join(cwd, flat), "w") as f:
                f.write("new")
            with open(os.path.join(cwd, flat + ".out"), "w") as f:
                f.write("out")
            with open(log_path, "w") as f:
                f.write("log")
            return 0

        monkeypatch.setattr(fs, "_run_command_with_env", fake_run_command)

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
        # Original surf directory should not be modified
        assert not (surf_dir / "in.patch").exists()
        assert not (surf_dir / output_name).exists()

    def test_command_failure_raises(self, tmp_path, monkeypatch):
        subject = "subj"
        hemi = "lh"
        surf_dir = tmp_path / "subjects" / subject / "surf"
        surf_dir.mkdir(parents=True)
        # Create some dummy surface files for symlinking
        (surf_dir / "lh.white").write_text("white")
        (surf_dir / "lh.pial").write_text("pial")
        patch_file = tmp_path / "in.patch"
        patch_file.write_text("patch")
        output_dir = tmp_path / "outdir"
        output_dir.mkdir()

        monkeypatch.setattr(fs, "_resolve_subject_dir", lambda subj: str(surf_dir))

        def fake_fail(cmd, cwd, log_path, env=None):
            with open(log_path, "w") as f:
                f.write("error")
            return 1

        monkeypatch.setattr(fs, "_run_command_with_env", fake_fail)

        with pytest.raises(RuntimeError):
            run_mris_flatten(
                subject, hemi, str(patch_file), str(output_dir), overwrite=True
            )

    def test_default_output_name(self, tmp_path, monkeypatch):
        subject = "subj"
        hemi = "rh"
        surf_dir = tmp_path / "subjects" / subject / "surf"
        surf_dir.mkdir(parents=True)
        # Create some dummy surface files for symlinking
        (surf_dir / "rh.white").write_text("white")
        (surf_dir / "rh.pial").write_text("pial")
        patch_file = tmp_path / "in.patch"
        patch_file.write_text("patch")
        output_dir = tmp_path / "outdir"
        output_dir.mkdir()

        monkeypatch.setattr(fs, "_resolve_subject_dir", lambda subj: str(surf_dir))

        def fake_run(cmd, cwd, log_path, env=None):
            flat = cmd[-1]
            with open(os.path.join(cwd, flat), "w") as f:
                f.write("ok")
            with open(log_path, "w") as f:
                f.write("log")
            return 0

        monkeypatch.setattr(fs, "_run_command_with_env", fake_run)
        result = run_mris_flatten(subject, hemi, str(patch_file), str(output_dir))
        fname = os.path.basename(result)
        assert fname.startswith("rh.autoflatten") and fname.endswith(".flat.patch.3d")
        assert os.path.isfile(result)

    def test_create_temp_surf_directory(self, tmp_path):
        """
        Test creating temporary surf directory with symlinks.

        Verifies that the helper function correctly creates the directory structure
        and symlinks to original surface files.
        """
        # Create mock original surf directory with surface files
        subject = "test_subj"
        original_surf = tmp_path / "subjects" / subject / "surf"
        original_surf.mkdir(parents=True)

        # Create mock surface files
        (original_surf / "lh.white").write_text("white_surface_data")
        (original_surf / "lh.pial").write_text("pial_surface_data")
        (original_surf / "lh.inflated").write_text("inflated_surface_data")
        (original_surf / "lh.sphere").write_text("sphere_surface_data")
        (original_surf / "rh.white").write_text("rh_white_data")

        # Create temp root
        temp_root = tmp_path / "temp"
        temp_root.mkdir()

        # Create temp surf directory with symlinks
        temp_surf = fs._create_temp_surf_directory(
            subject, str(original_surf), str(temp_root)
        )

        # Verify directory structure created
        assert os.path.isdir(temp_surf)
        assert temp_surf == str(temp_root / subject / "surf")

        # Verify symlinks created
        assert os.path.islink(os.path.join(temp_surf, "lh.white"))
        assert os.path.islink(os.path.join(temp_surf, "lh.pial"))
        assert os.path.islink(os.path.join(temp_surf, "lh.inflated"))
        assert os.path.islink(os.path.join(temp_surf, "lh.sphere"))
        assert os.path.islink(os.path.join(temp_surf, "rh.white"))

        # Verify symlinks point to correct original files
        assert os.readlink(os.path.join(temp_surf, "lh.white")) == str(
            original_surf / "lh.white"
        )
        assert os.readlink(os.path.join(temp_surf, "lh.pial")) == str(
            original_surf / "lh.pial"
        )

        # Verify we can read through symlinks (read-only access)
        with open(os.path.join(temp_surf, "lh.white")) as f:
            assert f.read() == "white_surface_data"

    def test_run_mris_flatten_preserves_original_surf_dir(self, tmp_path, monkeypatch):
        """
        Test that run_mris_flatten never modifies original surf directory.

        This is the critical test ensuring our implementation meets the requirement
        to never touch the original FreeSurfer subject directory.
        """
        subject = "test_subj"
        hemi = "lh"

        # Create original surf directory with surface files
        surf_dir = tmp_path / "subjects" / subject / "surf"
        surf_dir.mkdir(parents=True)

        # Create mock surface files
        (surf_dir / f"{hemi}.white").write_text("original_white")
        (surf_dir / f"{hemi}.pial").write_text("original_pial")
        (surf_dir / f"{hemi}.inflated").write_text("original_inflated")

        # Record initial state of surf directory
        initial_files = sorted(os.listdir(surf_dir))
        initial_mtimes = {f: os.path.getmtime(surf_dir / f) for f in initial_files}

        # Create patch file in different location
        patch_file = tmp_path / "input" / "test.patch"
        patch_file.parent.mkdir()
        patch_file.write_text("patch_data_content")

        # Output to yet another different directory
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Mock _resolve_subject_dir to return our test surf_dir
        monkeypatch.setattr(fs, "_resolve_subject_dir", lambda subj: str(surf_dir))

        # Mock _run_command_with_env to simulate successful mris_flatten
        def fake_run_command(cmd, cwd, log_path, env=None):
            # Verify it's running in a temp directory, not original
            assert str(surf_dir) not in cwd, "Running in original surf directory!"
            assert "autoflatten_" in cwd, "Not running in expected temp directory"

            # Verify SUBJECTS_DIR override
            assert env is not None
            assert "SUBJECTS_DIR" in env
            assert str(surf_dir) not in env["SUBJECTS_DIR"]

            # Simulate mris_flatten creating output files
            flat_file = os.path.join(cwd, cmd[-1])
            with open(flat_file, "w") as f:
                f.write("flattened_surface_data")
            with open(log_path, "w") as f:
                f.write("mris_flatten log output")
            with open(flat_file + ".out", "w") as f:
                f.write("mris_flatten out file")
            return 0

        monkeypatch.setattr(fs, "_run_command_with_env", fake_run_command)

        # Run mris_flatten
        result = run_mris_flatten(
            subject, hemi, str(patch_file), str(output_dir), overwrite=True
        )

        # ===== CRITICAL VERIFICATION =====
        # Verify original surf directory completely unchanged
        final_files = sorted(os.listdir(surf_dir))
        assert initial_files == final_files, (
            f"Original surf directory was modified! "
            f"Before: {initial_files}, After: {final_files}"
        )

        # Verify no files were modified (check mtimes)
        for filename in initial_files:
            original_mtime = initial_mtimes[filename]
            current_mtime = os.path.getmtime(surf_dir / filename)
            assert original_mtime == current_mtime, (
                f"File {filename} was modified (mtime changed)"
            )

        # Verify output files exist in output_dir (not surf_dir)
        assert os.path.exists(result)
        assert str(output_dir) in result
        # Log file has pattern: base.flat.patch.log (not base.log)
        log_file = os.path.splitext(result)[0] + ".log"
        assert os.path.exists(log_file), f"Log file not found: {log_file}"

        # Verify output content correct
        with open(result) as f:
            assert f.read() == "flattened_surface_data"

    def test_debug_flag_preserves_temp_directory(self, tmp_path, monkeypatch):
        """
        Test that debug=True preserves temporary directory.

        Verifies that when debug mode is enabled, the temporary directory
        is not deleted after execution.
        """
        subject = "test_subj"
        hemi = "lh"

        # Create original surf directory with surface files
        surf_dir = tmp_path / "subjects" / subject / "surf"
        surf_dir.mkdir(parents=True)
        (surf_dir / f"{hemi}.white").write_text("white")
        (surf_dir / f"{hemi}.pial").write_text("pial")

        # Create patch file
        patch_file = tmp_path / "test.patch"
        patch_file.write_text("patch")

        # Output directory
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Track the temp directory created
        temp_dirs = []

        # Mock _resolve_subject_dir
        monkeypatch.setattr(fs, "_resolve_subject_dir", lambda subj: str(surf_dir))

        # Mock _run_command_with_env and capture temp directory
        def fake_run_command(cmd, cwd, log_path, env=None):
            # Track the working directory (which is in the temp directory)
            temp_dirs.append(cwd)
            flat_file = os.path.join(cwd, cmd[-1])
            with open(flat_file, "w") as f:
                f.write("data")
            with open(log_path, "w") as f:
                f.write("log")
            return 0

        monkeypatch.setattr(fs, "_run_command_with_env", fake_run_command)

        # Run with debug=True
        result = run_mris_flatten(
            subject, hemi, str(patch_file), str(output_dir), debug=True
        )

        # Verify temporary directory was preserved
        assert len(temp_dirs) == 1
        temp_dir = temp_dirs[0]

        # The temp directory should still exist
        # Extract temp root from the temp_surf_dir path
        temp_root = os.path.dirname(os.path.dirname(temp_dir))
        assert os.path.exists(temp_root), (
            "Temporary directory should be preserved in debug mode"
        )
        assert "autoflatten_" in temp_root

        # Clean up manually (since debug mode doesn't clean up)
        import shutil

        if os.path.exists(temp_root):
            shutil.rmtree(temp_root)
