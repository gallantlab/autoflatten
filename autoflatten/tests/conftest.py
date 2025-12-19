"""Shared pytest fixtures for autoflatten tests."""

import os
from unittest.mock import patch

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
