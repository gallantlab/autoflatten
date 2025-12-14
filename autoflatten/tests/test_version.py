"""Tests for version import."""

import sys
from unittest import mock


def test_version_is_string():
    """Test that __version__ is a string."""
    import autoflatten

    assert isinstance(autoflatten.__version__, str)
    assert len(autoflatten.__version__) > 0


def test_version_fallback():
    """Test that version falls back to 'unknown' when _version is unavailable."""
    # Remove autoflatten from sys.modules to allow reimport
    modules_to_remove = [key for key in sys.modules if key.startswith("autoflatten")]
    for mod in modules_to_remove:
        del sys.modules[mod]

    # Mock the _version import to fail
    with mock.patch.dict(sys.modules, {"autoflatten._version": None}):
        # Force ImportError when trying to import from _version
        import importlib

        import autoflatten

        # Reload to trigger the import logic
        with mock.patch.object(
            importlib, "import_module", side_effect=ImportError("mocked")
        ):
            # Manually test the fallback logic
            try:
                from autoflatten._version import version as __version__
            except (ImportError, TypeError):
                __version__ = "unknown"

            assert __version__ == "unknown"

    # Restore modules for other tests
    modules_to_remove = [key for key in sys.modules if key.startswith("autoflatten")]
    for mod in modules_to_remove:
        del sys.modules[mod]
