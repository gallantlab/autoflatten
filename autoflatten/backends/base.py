"""Abstract base class for flattening backends.

This module defines the interface that all flattening backends must implement.
"""

from abc import ABC, abstractmethod
from typing import Optional
from pathlib import Path


class FlattenBackend(ABC):
    """Abstract base class for surface flattening backends.

    All flattening backends must implement this interface to be usable
    with autoflatten's CLI and pipeline.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the backend name."""
        pass

    @abstractmethod
    def flatten(
        self,
        patch_path: str,
        surface_path: str,
        output_path: str,
        verbose: bool = True,
        **kwargs,
    ) -> str:
        """Flatten a cortical surface patch.

        Parameters
        ----------
        patch_path : str
            Path to the input patch file (e.g., lh.autoflatten.patch.3d)
        surface_path : str
            Path to the base surface file (e.g., lh.fiducial or lh.white)
        output_path : str
            Path for the output flat patch file
        verbose : bool
            Whether to print progress messages
        **kwargs
            Backend-specific options

        Returns
        -------
        str
            Path to the output flat patch file
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend is available (dependencies installed).

        Returns
        -------
        bool
            True if the backend can be used
        """
        pass

    def get_install_instructions(self) -> str:
        """Return instructions for installing this backend's dependencies.

        Returns
        -------
        str
            Installation instructions
        """
        return f"Backend '{self.name}' is not available."


def find_base_surface(patch_path: str) -> Optional[str]:
    """Auto-detect the base surface from a patch file path.

    If the patch is in a FreeSurfer directory structure, attempts to find
    the corresponding fiducial or white surface.

    Parameters
    ----------
    patch_path : str
        Path to the patch file

    Returns
    -------
    str or None
        Path to the base surface, or None if not found
    """
    patch_path = Path(patch_path)

    # Determine hemisphere from patch filename
    hemi = None
    name = patch_path.name
    if name.startswith("lh."):
        hemi = "lh"
    elif name.startswith("rh."):
        hemi = "rh"

    if hemi is None:
        return None

    # Check for surfaces in the same directory (typical FreeSurfer surf/ location)
    surf_dir = patch_path.parent

    # Try fiducial first (preferred for pyflatten)
    fiducial = surf_dir / f"{hemi}.fiducial"
    if fiducial.exists():
        return str(fiducial)

    # Fall back to smoothwm (standard FreeSurfer surface)
    smoothwm = surf_dir / f"{hemi}.smoothwm"
    if smoothwm.exists():
        return str(smoothwm)

    return None
