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

    # A None entry in sys.modules makes "from ._version import version"
    # raise ImportError, exercising the fallback in autoflatten/__init__.py
    with mock.patch.dict(sys.modules, {"autoflatten._version": None}):
        import autoflatten

        assert autoflatten.__version__ == "unknown"

    # Remove the mocked package so other tests reimport the real one
    modules_to_remove = [key for key in sys.modules if key.startswith("autoflatten")]
    for mod in modules_to_remove:
        del sys.modules[mod]
