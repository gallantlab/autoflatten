"""Surface flattening module.

This module provides JAX-accelerated cortical surface flattening,
implementing FreeSurfer-style optimization with modern automatic
differentiation and vectorized computation.

Main classes and functions:
    - SurfaceFlattener: Main class for running flattening optimization
    - FlattenConfig: Configuration dataclass for flattening parameters
    - setup_logging, restore_logging: Logging utilities

Example:
    >>> from autoflatten.flatten import SurfaceFlattener, FlattenConfig
    >>> config = FlattenConfig()
    >>> flattener = SurfaceFlattener(config)
    >>> flattener.load_data("lh.cortex.patch.3d", "lh.fiducial")
    >>> flattener.compute_kring_distances()
    >>> flattener.prepare_optimization()
    >>> uv = flattener.run()
    >>> flattener.save_result(uv, "lh.flat.patch.3d")
"""

from .algorithm import (
    SurfaceFlattener,
    TopologyError,
    count_flipped_triangles,
    freesurfer_projection,
    remove_isolated_vertices,
    validate_topology,
)

# Import logging utilities from the shared module
from ..logging import restore_logging, setup_logging

from .config import (
    FlattenConfig,
    KRingConfig,
    ConvergenceConfig,
    LineSearchConfig,
    PhaseConfig,
    NegativeAreaRemovalConfig,
    SpringSmoothingConfig,
    get_kring_cache_filename,
)

from .distance import (
    compute_kring_geodesic_distances,
    compute_kring_geodesic_distances_angular,
    build_mesh_graph,
    get_k_ring,
    get_k_ring_fast,
    GRAPH_DISTANCE_CORRECTION,
)

from .energy import (
    compute_2d_areas,
    compute_3d_surface_area,
    compute_metric_energy,
    compute_area_energy,
    compute_area_energy_fs_v6,
    prepare_metric_data,
    prepare_edge_list,
    prepare_smoothing_data,
    smooth_gradient,
    compute_spring_displacement,
    get_vertices_with_negative_area,
)

__all__ = [
    # Main class
    "SurfaceFlattener",
    "TopologyError",
    # Configuration
    "FlattenConfig",
    "KRingConfig",
    "ConvergenceConfig",
    "LineSearchConfig",
    "PhaseConfig",
    "NegativeAreaRemovalConfig",
    "SpringSmoothingConfig",
    "get_kring_cache_filename",
    # Logging
    "setup_logging",
    "restore_logging",
    # Mesh utilities
    "count_flipped_triangles",
    "freesurfer_projection",
    "remove_isolated_vertices",
    "validate_topology",
    # Distance computation
    "compute_kring_geodesic_distances",
    "compute_kring_geodesic_distances_angular",
    "build_mesh_graph",
    "get_k_ring",
    "get_k_ring_fast",
    "GRAPH_DISTANCE_CORRECTION",
    # Energy functions
    "compute_2d_areas",
    "compute_3d_surface_area",
    "compute_metric_energy",
    "compute_area_energy",
    "compute_area_energy_fs_v6",
    "prepare_metric_data",
    "prepare_edge_list",
    "prepare_smoothing_data",
    "smooth_gradient",
    "compute_spring_displacement",
    "get_vertices_with_negative_area",
]
