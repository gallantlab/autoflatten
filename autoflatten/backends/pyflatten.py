"""Pyflatten backend (JAX-accelerated flattening).

This module provides the pyflatten backend for cortical surface flattening,
using JAX-accelerated gradient descent optimization.
"""

from typing import Optional

from .base import FlattenBackend

# Import threading configuration (does NOT import JAX)
from ..flatten.threading import configure_threading


def _check_pyflatten_available() -> bool:
    """Check if pyflatten dependencies are available."""
    import logging

    logger = logging.getLogger(__name__)

    missing = []
    for module in ["jax", "igl", "numba"]:
        try:
            __import__(module)
        except ImportError as e:
            missing.append(f"{module}: {e}")

    if missing:
        logger.debug(f"Pyflatten dependencies unavailable: {'; '.join(missing)}")
        return False
    return True


class PyflattenBackend(FlattenBackend):
    """Pyflatten backend using JAX-accelerated optimization.

    This backend uses the autoflatten.flatten module for cortical surface
    flattening with JAX-accelerated gradient descent and vectorized line search.
    """

    @property
    def name(self) -> str:
        return "pyflatten"

    def is_available(self) -> bool:
        """Check if pyflatten dependencies are available."""
        return _check_pyflatten_available()

    def get_install_instructions(self) -> str:
        return (
            "Pyflatten backend requires JAX, libigl, and numba.\n"
            "Install with: pip install autoflatten\n"
            "For GPU acceleration: pip install autoflatten[cuda]"
        )

    def flatten(
        self,
        patch_path: str,
        surface_path: str,
        output_path: str,
        verbose: bool = True,
        k_ring: int = 7,
        n_neighbors_per_ring: Optional[int] = 12,
        skip_phases: Optional[list] = None,
        skip_spring_smoothing: bool = False,
        skip_neg_area: bool = False,
        config_path: Optional[str] = None,
        n_jobs: int = -1,
        cache_distances: bool = False,
        tqdm_position: int = 0,
        print_every: int = 1,
        **kwargs,
    ) -> str:
        """Flatten a cortical surface patch using pyflatten.

        Parameters
        ----------
        patch_path : str
            Path to the input patch file
        surface_path : str
            Path to the base surface file (fiducial or white)
        output_path : str
            Path for the output flat patch file
        verbose : bool
            Whether to print progress messages
        k_ring : int
            Number of neighborhood rings for distance computation
        n_neighbors_per_ring : int or None
            Number of angular samples per ring (None = use all neighbors)
        skip_phases : list of str, optional
            Phase names to skip (e.g., ['epoch_3'])
        skip_spring_smoothing : bool
            Whether to skip final spring smoothing
        skip_neg_area : bool
            Whether to skip negative area removal
        config_path : str, optional
            Path to JSON config file for custom configuration
        n_jobs : int
            Number of parallel jobs (-1 = use all CPUs)
        cache_distances : bool
            Whether to cache computed k-ring distances
        tqdm_position : int
            Position of tqdm progress bar (for stacking bars in parallel execution)
        print_every : int
            Print progress every N iterations (default: 1 = every iteration)
        **kwargs
            Additional arguments (ignored)

        Returns
        -------
        str
            Path to the output flat patch file
        """
        # Configure threading BEFORE importing JAX-dependent modules
        configure_threading(n_jobs)

        # Import here to allow lazy loading and graceful fallback
        from ..flatten import (
            SurfaceFlattener,
            FlattenConfig,
            setup_logging,
            restore_logging,
            get_kring_cache_filename,
        )
        from ..flatten.config import KRingConfig

        # Load or create configuration
        if config_path is not None:
            config = FlattenConfig.from_json_file(config_path)
        else:
            config = FlattenConfig()

        # Apply CLI overrides
        config.kring.k_ring = k_ring
        config.kring.n_neighbors_per_ring = n_neighbors_per_ring
        config.verbose = verbose
        config.n_jobs = n_jobs
        config.print_every = print_every

        if skip_spring_smoothing:
            config.spring_smoothing.enabled = False

        if skip_neg_area:
            config.negative_area_removal.enabled = False

        if skip_phases:
            for phase in config.phases:
                if phase.name in skip_phases:
                    phase.enabled = False

        # Setup logging to output_path.log
        original_stdout, log_file = setup_logging(output_path, verbose=verbose)

        try:
            if verbose:
                print(f"Running pyflatten backend")
                print(f"  Input patch: {patch_path}")
                print(f"  Base surface: {surface_path}")
                print(f"  Output: {output_path}")
                print(f"  K-ring: {k_ring}, neighbors/ring: {n_neighbors_per_ring}")

            # Create flattener
            flattener = SurfaceFlattener(config)

            # Load data
            flattener.load_data(patch_path, surface_path)

            # Compute or load cached k-ring distances
            cache_path = None
            if cache_distances:
                cache_path = get_kring_cache_filename(output_path, config.kring)

            flattener.compute_kring_distances(
                cache_path=cache_path, tqdm_position=tqdm_position
            )

            # Prepare optimization
            flattener.prepare_optimization()

            # Run optimization
            uv = flattener.run()

            # Save result
            flattener.save_result(uv, output_path)

            return output_path

        finally:
            restore_logging(original_stdout, log_file)
