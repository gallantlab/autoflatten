"""Thread configuration for parallel backends.

This module provides utilities for controlling thread counts across
multiple parallelization backends used by pyflatten:
- JAX/XLA (CPU operations)
- Numba (k-ring distance computation)
- OpenMP, MKL, OpenBLAS (underlying BLAS/LAPACK)

IMPORTANT: configure_threading() must be called BEFORE importing JAX
for XLA thread limits to take effect.
"""

import os
from typing import Optional

_THREADING_CONFIGURED = False


def configure_threading(n_threads: Optional[int] = None) -> None:
    """Configure thread counts for all parallel backends.

    Sets environment variables and runtime settings to limit CPU
    parallelism across JAX, Numba, OpenMP, and BLAS libraries.

    IMPORTANT: Must be called BEFORE importing JAX for XLA settings
    to take effect. Numba settings can be applied at any time.

    Parameters
    ----------
    n_threads : int, optional
        Number of threads to use. If None or -1, uses all available CPUs.
        If 0, also uses all available CPUs (no limit).

    Notes
    -----
    Environment variables set:
    - XLA_FLAGS: JAX/XLA CPU thread count and Eigen thread-pool control
        * --xla_force_host_platform_device_count
        * --xla_cpu_multi_thread_eigen
        * --xla_cpu_multi_thread_eigen_thread_count
    - OMP_NUM_THREADS: OpenMP parallelism
    - MKL_NUM_THREADS: Intel MKL
    - OPENBLAS_NUM_THREADS: OpenBLAS
    - VECLIB_MAXIMUM_THREADS: Apple Accelerate
    - NUMEXPR_NUM_THREADS: NumExpr

    Examples
    --------
    >>> from autoflatten.flatten.threading import configure_threading
    >>> configure_threading(4)  # Limit to 4 threads
    >>> # Now import JAX and other libraries
    >>> import jax
    """
    global _THREADING_CONFIGURED

    # None or -1 or 0 means use all CPUs (no limit)
    if n_threads is None or n_threads <= 0:
        _THREADING_CONFIGURED = True
        return

    def _append_xla_flag(current: str, flag: str) -> str:
        return f"{current} {flag}".strip()

    def _has_flag(env: str, prefix: str) -> bool:
        return any(part.startswith(prefix) for part in env.split())

    n_str = str(n_threads)

    # JAX/XLA - controls intra-op parallelism
    # This splits CPU into N "devices" limiting total thread usage
    existing_xla = os.environ.get("XLA_FLAGS", "")
    updated_xla = existing_xla
    if not _has_flag(existing_xla, "--xla_force_host_platform_device_count"):
        xla_flag = f"--xla_force_host_platform_device_count={n_threads}"
        updated_xla = _append_xla_flag(updated_xla, xla_flag)
    if not _has_flag(existing_xla, "--xla_cpu_multi_thread_eigen"):
        updated_xla = _append_xla_flag(updated_xla, "--xla_cpu_multi_thread_eigen=true")

    # Limit Eigen thread pool used by XLA CPU backend
    if not _has_flag(existing_xla, "--xla_cpu_multi_thread_eigen_thread_count"):
        eigen_flag = f"--xla_cpu_multi_thread_eigen_thread_count={n_threads}"
        updated_xla = _append_xla_flag(updated_xla, eigen_flag)

    if updated_xla != existing_xla:
        os.environ["XLA_FLAGS"] = updated_xla

    # OpenMP - used by many numerical libraries
    if "OMP_NUM_THREADS" not in os.environ:
        os.environ["OMP_NUM_THREADS"] = n_str

    # Intel MKL
    if "MKL_NUM_THREADS" not in os.environ:
        os.environ["MKL_NUM_THREADS"] = n_str

    # OpenBLAS
    if "OPENBLAS_NUM_THREADS" not in os.environ:
        os.environ["OPENBLAS_NUM_THREADS"] = n_str

    # Apple Accelerate (macOS)
    if "VECLIB_MAXIMUM_THREADS" not in os.environ:
        os.environ["VECLIB_MAXIMUM_THREADS"] = n_str

    # NumExpr
    if "NUMEXPR_NUM_THREADS" not in os.environ:
        os.environ["NUMEXPR_NUM_THREADS"] = n_str

    # Numba - can be set at runtime (unlike env vars which need early setting)
    # Note: numba.set_num_threads raises ValueError if n > available cores
    try:
        import numba

        max_threads = numba.get_num_threads()
        numba.set_num_threads(min(n_threads, max_threads))
    except ImportError:
        pass  # Numba not installed
    except ValueError:
        pass  # Requested threads exceed available cores

    _THREADING_CONFIGURED = True


def is_configured() -> bool:
    """Check if threading has been configured.

    Returns
    -------
    bool
        True if configure_threading() has been called.
    """
    return _THREADING_CONFIGURED


def get_effective_threads() -> dict:
    """Get the current effective thread settings.

    Returns
    -------
    dict
        Dictionary with thread counts for each backend.
    """
    result = {
        "XLA_FLAGS": os.environ.get("XLA_FLAGS", "(not set)"),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "(not set)"),
        "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS", "(not set)"),
        "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS", "(not set)"),
        "VECLIB_MAXIMUM_THREADS": os.environ.get("VECLIB_MAXIMUM_THREADS", "(not set)"),
        "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS", "(not set)"),
    }

    # Numba threads (if available)
    try:
        import numba

        result["numba_threads"] = numba.get_num_threads()
    except ImportError:
        result["numba_threads"] = "(numba not installed)"

    return result
