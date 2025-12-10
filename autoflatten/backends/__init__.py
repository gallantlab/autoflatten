"""Backend abstraction for surface flattening.

This module provides a unified interface for different flattening backends,
allowing users to choose between FreeSurfer's mris_flatten and pyflatten.

Available backends:
    - pyflatten: JAX-accelerated flattening (default, recommended)
    - freesurfer: FreeSurfer mris_flatten wrapper

Example:
    >>> from autoflatten.backends import get_backend, available_backends
    >>> print(available_backends())  # ['pyflatten', 'freesurfer']
    >>> backend = get_backend('pyflatten')
    >>> backend.flatten(patch_path, surface_path, output_path)
"""

from typing import Optional

from .base import FlattenBackend, find_base_surface
from .pyflatten import PyflattenBackend
from .freesurfer import FreeSurferBackend


# Registry of available backends
_BACKENDS = {
    "pyflatten": PyflattenBackend,
    "freesurfer": FreeSurferBackend,
}

# Default backend
DEFAULT_BACKEND = "pyflatten"


def available_backends() -> list[str]:
    """Return list of available backend names.

    Returns
    -------
    list of str
        Names of available backends (those with dependencies installed)
    """
    available = []
    for name, backend_cls in _BACKENDS.items():
        backend = backend_cls()
        if backend.is_available():
            available.append(name)
    return available


def get_backend(name: Optional[str] = None) -> FlattenBackend:
    """Get a flattening backend by name.

    Parameters
    ----------
    name : str, optional
        Backend name ('pyflatten' or 'freesurfer').
        If None, returns the default backend (pyflatten).

    Returns
    -------
    FlattenBackend
        The requested backend instance

    Raises
    ------
    ValueError
        If the backend name is not recognized
    RuntimeError
        If the requested backend is not available
    """
    if name is None:
        name = DEFAULT_BACKEND

    if name not in _BACKENDS:
        raise ValueError(
            f"Unknown backend: {name}. Available backends: {list(_BACKENDS.keys())}"
        )

    backend_cls = _BACKENDS[name]
    backend = backend_cls()

    if not backend.is_available():
        raise RuntimeError(
            f"Backend '{name}' is not available.\n{backend.get_install_instructions()}"
        )

    return backend


def get_default_backend() -> FlattenBackend:
    """Get the default flattening backend.

    Returns pyflatten if available, otherwise falls back to freesurfer.

    Returns
    -------
    FlattenBackend
        The default available backend

    Raises
    ------
    RuntimeError
        If no backends are available
    """
    # Try pyflatten first (default)
    pyflatten = PyflattenBackend()
    if pyflatten.is_available():
        return pyflatten

    # Fall back to FreeSurfer
    freesurfer = FreeSurferBackend()
    if freesurfer.is_available():
        print(
            "Warning: pyflatten backend not available, falling back to FreeSurfer.\n"
            f"{pyflatten.get_install_instructions()}"
        )
        return freesurfer

    # No backends available
    raise RuntimeError(
        "No flattening backends available.\n"
        "Install pyflatten dependencies: pip install autoflatten\n"
        "Or install FreeSurfer: https://surfer.nmr.mgh.harvard.edu/"
    )


__all__ = [
    "FlattenBackend",
    "PyflattenBackend",
    "FreeSurferBackend",
    "get_backend",
    "get_default_backend",
    "available_backends",
    "find_base_surface",
    "DEFAULT_BACKEND",
]
