"""
Tests for the backends module.
"""

import os
import tempfile

import pytest

from autoflatten.backends import (
    get_backend,
    get_default_backend,
    available_backends,
    find_base_surface,
    DEFAULT_BACKEND,
)
from autoflatten.backends.base import FlattenBackend
from autoflatten.backends.pyflatten import PyflattenBackend
from autoflatten.backends.freesurfer import FreeSurferBackend


class TestBackendRegistry:
    """Tests for backend registry functions."""

    def test_default_backend_is_pyflatten(self):
        """Test that the default backend is pyflatten."""
        assert DEFAULT_BACKEND == "pyflatten"

    def test_available_backends_returns_list(self):
        """Test that available_backends returns a list."""
        backends = available_backends()
        assert isinstance(backends, list)
        # At minimum, pyflatten should be available (it's always installed now)
        assert "pyflatten" in backends

    def test_get_backend_pyflatten(self):
        """Test getting the pyflatten backend."""
        backend = get_backend("pyflatten")
        assert isinstance(backend, PyflattenBackend)
        assert backend.name == "pyflatten"

    def test_get_backend_freesurfer(self):
        """Test getting the freesurfer backend."""
        # FreeSurfer may not be available, so check first
        fs_backend = FreeSurferBackend()
        if not fs_backend.is_available():
            pytest.skip("FreeSurfer not available")
        backend = get_backend("freesurfer")
        assert isinstance(backend, FreeSurferBackend)
        assert backend.name == "freesurfer"

    def test_get_backend_none_returns_default(self):
        """Test that get_backend(None) returns the default backend."""
        backend = get_backend(None)
        assert backend.name == DEFAULT_BACKEND

    def test_get_backend_invalid_raises_error(self):
        """Test that get_backend raises ValueError for invalid backend."""
        with pytest.raises(ValueError, match="Unknown backend"):
            get_backend("invalid_backend")

    def test_get_default_backend(self):
        """Test get_default_backend returns a FlattenBackend."""
        backend = get_default_backend()
        assert isinstance(backend, FlattenBackend)
        # Should be pyflatten since it's always installed
        assert backend.name == "pyflatten"


class TestPyflattenBackend:
    """Tests for the PyflattenBackend class."""

    def test_name(self):
        """Test backend name is 'pyflatten'."""
        backend = PyflattenBackend()
        assert backend.name == "pyflatten"

    def test_is_available(self):
        """Test that pyflatten is available (deps are core now)."""
        backend = PyflattenBackend()
        assert backend.is_available() is True

    def test_get_install_instructions(self):
        """Test install instructions are returned."""
        backend = PyflattenBackend()
        instructions = backend.get_install_instructions()
        assert isinstance(instructions, str)
        assert "pip install" in instructions.lower() or "jax" in instructions.lower()


class TestFreeSurferBackend:
    """Tests for the FreeSurferBackend class."""

    def test_name(self):
        """Test backend name is 'freesurfer'."""
        backend = FreeSurferBackend()
        assert backend.name == "freesurfer"

    def test_is_available_returns_bool(self):
        """Test that is_available returns a boolean."""
        backend = FreeSurferBackend()
        result = backend.is_available()
        assert isinstance(result, bool)

    def test_get_install_instructions(self):
        """Test install instructions are returned."""
        backend = FreeSurferBackend()
        instructions = backend.get_install_instructions()
        assert isinstance(instructions, str)
        assert "freesurfer" in instructions.lower()


class TestFindBaseSurface:
    """Tests for the find_base_surface utility function."""

    def test_find_base_surface_with_fiducial(self):
        """Test finding base surface when fiducial exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock FreeSurfer structure
            surf_dir = os.path.join(tmpdir, "sub-01", "surf")
            os.makedirs(surf_dir)

            # Create mock patch file
            patch_path = os.path.join(surf_dir, "lh.autoflatten.patch.3d")
            with open(patch_path, "wb") as f:
                f.write(b"\x00" * 10)

            # Create mock fiducial surface
            fiducial_path = os.path.join(surf_dir, "lh.fiducial")
            with open(fiducial_path, "wb") as f:
                f.write(b"\x00" * 10)

            result = find_base_surface(patch_path)
            assert result == fiducial_path

    def test_find_base_surface_with_smoothwm_fallback(self):
        """Test finding base surface falling back to smoothwm."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock FreeSurfer structure
            surf_dir = os.path.join(tmpdir, "sub-01", "surf")
            os.makedirs(surf_dir)

            # Create mock patch file
            patch_path = os.path.join(surf_dir, "lh.autoflatten.patch.3d")
            with open(patch_path, "wb") as f:
                f.write(b"\x00" * 10)

            # Create only smoothwm surface (no fiducial)
            smoothwm_path = os.path.join(surf_dir, "lh.smoothwm")
            with open(smoothwm_path, "wb") as f:
                f.write(b"\x00" * 10)

            result = find_base_surface(patch_path)
            assert result == smoothwm_path

    def test_find_base_surface_rh_hemisphere(self):
        """Test finding base surface for right hemisphere."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock FreeSurfer structure
            surf_dir = os.path.join(tmpdir, "sub-01", "surf")
            os.makedirs(surf_dir)

            # Create mock patch file
            patch_path = os.path.join(surf_dir, "rh.autoflatten.patch.3d")
            with open(patch_path, "wb") as f:
                f.write(b"\x00" * 10)

            # Create mock fiducial surface
            fiducial_path = os.path.join(surf_dir, "rh.fiducial")
            with open(fiducial_path, "wb") as f:
                f.write(b"\x00" * 10)

            result = find_base_surface(patch_path)
            assert result == fiducial_path

    def test_find_base_surface_not_found(self):
        """Test find_base_surface returns None when no surface found."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock patch file in non-FreeSurfer directory
            patch_path = os.path.join(tmpdir, "lh.patch.3d")
            with open(patch_path, "wb") as f:
                f.write(b"\x00" * 10)

            result = find_base_surface(patch_path)
            assert result is None


# =============================================================================
# Tier 2: Additional Backend Tests
# =============================================================================


class TestCheckPyflattenAvailable:
    """Tests for _check_pyflatten_available function."""

    def test_returns_true_when_deps_available(self):
        """Test that function returns True when all deps are installed."""
        from autoflatten.backends.pyflatten import _check_pyflatten_available

        # These are core dependencies now, should always be available
        result = _check_pyflatten_available()
        assert result is True

    def test_check_requires_jax_igl_numba(self):
        """Test that the function checks for jax, igl, and numba.

        Note: This test cannot reliably verify import failure detection because
        Python's import system caches modules. Once jax/igl/numba are imported,
        they remain cached for the entire test session.
        """
        pytest.skip(
            "Cannot test import failure detection due to Python import caching. "
            "The _check_pyflatten_available function is validated by the "
            "test_returns_true_when_deps_available test when deps are present."
        )


class TestHemisphereDetection:
    """Tests for hemisphere detection from patch file paths."""

    def test_find_base_surface_extracts_correct_hemisphere(self):
        """Test that find_base_surface uses correct hemisphere prefix."""
        with tempfile.TemporaryDirectory() as tmpdir:
            surf_dir = os.path.join(tmpdir, "sub-01", "surf")
            os.makedirs(surf_dir)

            # Create lh and rh fiducial files
            lh_fiducial = os.path.join(surf_dir, "lh.fiducial")
            rh_fiducial = os.path.join(surf_dir, "rh.fiducial")
            for f in [lh_fiducial, rh_fiducial]:
                with open(f, "wb") as fh:
                    fh.write(b"\x00" * 10)

            # Create lh patch
            lh_patch = os.path.join(surf_dir, "lh.patch.3d")
            with open(lh_patch, "wb") as f:
                f.write(b"\x00" * 10)

            # Should find lh.fiducial, not rh.fiducial
            result = find_base_surface(lh_patch)
            assert result == lh_fiducial


class TestBackendBaseClass:
    """Tests for the FlattenBackend abstract base class."""

    def test_backend_is_abstract(self):
        """Test that FlattenBackend cannot be instantiated directly."""
        # FlattenBackend has abstract methods, so direct instantiation should fail
        with pytest.raises(
            TypeError, match=r"Can't instantiate abstract class.*FlattenBackend"
        ):
            FlattenBackend()

    def test_pyflatten_inherits_from_base(self):
        """Test that PyflattenBackend inherits from FlattenBackend."""
        assert issubclass(PyflattenBackend, FlattenBackend)

    def test_freesurfer_inherits_from_base(self):
        """Test that FreeSurferBackend inherits from FlattenBackend."""
        assert issubclass(FreeSurferBackend, FlattenBackend)
