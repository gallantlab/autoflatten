"""Shared pytest fixtures for autoflatten tests."""

import os
from unittest.mock import patch

import numpy as np
import pytest


@pytest.fixture
def mock_freesurfer_env(tmp_path):
    """Mock FreeSurfer environment variables.

    Creates temporary directories and patches environment variables
    for FREESURFER_HOME and SUBJECTS_DIR.
    """
    freesurfer_home = tmp_path / "freesurfer"
    freesurfer_home.mkdir()

    subjects_dir = tmp_path / "subjects"
    subjects_dir.mkdir()

    env_patch = {
        "FREESURFER_HOME": str(freesurfer_home),
        "SUBJECTS_DIR": str(subjects_dir),
    }

    with patch.dict(os.environ, env_patch):
        yield {
            "freesurfer_home": freesurfer_home,
            "subjects_dir": subjects_dir,
        }


@pytest.fixture
def temp_subject_dir(tmp_path):
    """Create a temporary subject directory structure.

    Creates the basic FreeSurfer subject directory structure:
        subject/
            surf/
            label/
            mri/
    """
    subject_dir = tmp_path / "test_subject"
    subject_dir.mkdir()

    # Create standard FreeSurfer subdirectories
    (subject_dir / "surf").mkdir()
    (subject_dir / "label").mkdir()
    (subject_dir / "mri").mkdir()

    return subject_dir


@pytest.fixture
def no_freesurfer_env():
    """Remove FreeSurfer environment variables for testing error handling."""
    env_without_fs = {k: v for k, v in os.environ.items()}
    env_without_fs.pop("FREESURFER_HOME", None)
    env_without_fs.pop("SUBJECTS_DIR", None)

    with patch.dict(os.environ, env_without_fs, clear=True):
        yield


# =============================================================================
# Synthetic Mesh Fixtures for Flatten Module Tests
# =============================================================================


@pytest.fixture
def simple_quad_mesh():
    r"""4-vertex quad mesh (2 triangles) for basic tests.

    Layout::

        2---3
        |\  |
        | \ |
        |  \|
        0---1

    Diagonal runs from vertex 1 to vertex 2 (bottom-right to top-left).
    Both triangles are CCW when viewed from +Z.
    """
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int32)
    return vertices, faces


@pytest.fixture
def simple_quad_uv():
    """2D UV coordinates for the simple quad mesh."""
    return np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)


@pytest.fixture
def triangle_strip_mesh():
    r"""6-vertex triangle strip for boundary/topology tests.

    Layout (rectangular grid with diagonal edges)::

        4---5
        |\  |
        | \ |
        2---3
        |\  |
        | \ |
        0---1

    Vertices form a 2x3 rectangular grid. Diagonal edges run from
    bottom-right to top-left within each cell (1->2, 3->4).
    All triangles are CCW when viewed from +Z.
    """
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 2.0, 0.0],
            [1.0, 2.0, 0.0],
        ],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2], [1, 3, 2], [2, 3, 4], [3, 5, 4]], dtype=np.int32)
    return vertices, faces


@pytest.fixture
def mesh_with_flipped_triangle():
    """Quad mesh where one triangle is flipped (CW instead of CCW).

    Same layout as simple_quad_mesh but second triangle has flipped winding.
    """
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    # First triangle CCW, second triangle CW (flipped)
    faces = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int32)
    return vertices, faces


@pytest.fixture
def mesh_with_isolated_vertex():
    """Mesh with one vertex not connected to any face."""
    vertices = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 1.0, 0.0],
            [5.0, 5.0, 0.0],  # Isolated vertex
        ],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    return vertices, faces
