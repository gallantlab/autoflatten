"""Automatic Surface Flattening Pipeline"""

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

# Export key functions for programmatic use
from .core import (
    ensure_continuous_cuts,
    fill_holes_in_patch,
    map_cuts_to_subject,
    refine_cuts_with_geodesic,
    validate_patch_topology,
)
from .template import identify_surface_components

__all__ = [
    "__version__",
    "ensure_continuous_cuts",
    "fill_holes_in_patch",
    "map_cuts_to_subject",
    "refine_cuts_with_geodesic",
    "validate_patch_topology",
    "identify_surface_components",
]
