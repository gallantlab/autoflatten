"""Surface flattening optimization.

This module provides the SurfaceFlattener class for cortical surface flattening
using FreeSurfer-style gradient descent with vectorized line search.
"""

import os
import time
import warnings
from typing import Optional

import igl
import jax
import jax.numpy as jnp
import numpy as np


# =============================================================================
# Exceptions
# =============================================================================


class TopologyError(Exception):
    """Raised when surface topology is incompatible with flattening."""

    pass


from .config import (
    FlattenConfig,
    NegativeAreaRemovalConfig,
    SpringSmoothingConfig,
)
from .distance import (
    compute_kring_geodesic_distances,
    compute_kring_geodesic_distances_angular,
)
from .energy import (
    compute_2d_areas,
    compute_3d_surface_area,
    compute_area_energy_fs_v6,
    compute_metric_energy,
    compute_spring_displacement,
    get_vertices_with_negative_area,
    prepare_metric_data,
    prepare_smoothing_data,
    smooth_gradient,
)

# Import I/O functions from autoflatten.freesurfer
from ..freesurfer import extract_patch_faces, read_patch, read_surface, write_patch


# =============================================================================
# Mesh utilities
# =============================================================================


def remove_isolated_vertices(
    vertices: np.ndarray, faces: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Remove vertices not referenced by any face, return reindexed mesh.

    Args:
        vertices: (V, 3) vertex coordinates
        faces: (F, 3) face indices

    Returns:
        Tuple of (new_vertices, new_faces, used_vertex_indices)
    """
    used = np.unique(faces)
    old_to_new = np.full(len(vertices), -1)
    old_to_new[used] = np.arange(len(used))
    return vertices[used], old_to_new[faces], used


def remove_small_components(
    vertices: np.ndarray,
    faces: np.ndarray,
    max_small_component_size: int = 20,
    warn_medium_threshold: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Remove small disconnected components, keeping only the largest.

    This function identifies connected components in the mesh and removes
    any components with at most max_small_component_size vertices. This is
    useful for cleaning up small isolated triangles that can cause topology
    errors (Euler characteristic != 1).

    The function is conservative: it only removes very small components
    (default <= 20 vertices) to avoid masking more serious topology issues.
    If medium-sized disconnected components are found, a warning is raised.

    Parameters
    ----------
    vertices : ndarray of shape (V, 3)
        Vertex coordinates.
    faces : ndarray of shape (F, 3)
        Face indices.
    max_small_component_size : int, default=20
        Maximum size of components to automatically remove. Components larger
        than this will not be removed.
    warn_medium_threshold : int, default=100
        If a secondary component is larger than max_small_component_size but
        smaller than this threshold, a warning is logged.

    Returns
    -------
    new_vertices : ndarray of shape (V', 3)
        Vertices after removing small components.
    new_faces : ndarray of shape (F', 3)
        Faces after removing small components.
    used_vertex_indices : ndarray of shape (V',)
        Original indices of the kept vertices.

    Raises
    ------
    TopologyError
        If a secondary component is found that is larger than
        warn_medium_threshold, indicating a potentially serious issue.
    """
    import logging
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    logger = logging.getLogger(__name__)
    n_verts = len(vertices)

    # Build adjacency matrix
    row = np.concatenate(
        [faces[:, 0], faces[:, 1], faces[:, 2], faces[:, 1], faces[:, 2], faces[:, 0]]
    )
    col = np.concatenate(
        [faces[:, 1], faces[:, 2], faces[:, 0], faces[:, 0], faces[:, 1], faces[:, 2]]
    )
    data = np.ones(len(row), dtype=np.int8)
    adj = csr_matrix((data, (row, col)), shape=(n_verts, n_verts))

    # Find connected components
    n_components, labels = connected_components(adj, directed=False)

    if n_components == 1:
        # Already single component, nothing to remove
        return vertices, faces, np.arange(n_verts)

    # Find component sizes
    component_sizes = np.bincount(labels, minlength=n_components)

    # Find largest component
    largest_idx = np.argmax(component_sizes)
    largest_size = component_sizes[largest_idx]

    # Analyze secondary components
    secondary_sizes = np.delete(component_sizes, largest_idx)
    max_secondary_size = secondary_sizes.max()

    # Check for problematic medium-sized components
    if max_secondary_size > warn_medium_threshold:
        raise TopologyError(
            f"Mesh has {n_components} disconnected components. "
            f"Largest: {largest_size:,} vertices, "
            f"secondary: {max_secondary_size:,} vertices. "
            f"The secondary component is too large (>{warn_medium_threshold}) "
            f"to be an isolated triangle artifact. "
            f"This indicates a fundamental issue with the patch creation."
        )

    if max_secondary_size > max_small_component_size:
        logger.warning(
            f"Mesh has {n_components} disconnected components. "
            f"Largest: {largest_size:,} vertices, "
            f"secondary: {max_secondary_size:,} vertices. "
            f"Secondary component is larger than {max_small_component_size} vertices."
        )

    # Remove only small components (<= max_small_component_size), but always
    # keep the largest component regardless of its size
    small_component_mask = component_sizes <= max_small_component_size
    small_component_mask[largest_idx] = False  # Never remove the largest component
    small_component_labels = np.where(small_component_mask)[0]

    # Build mask of vertices to remove
    remove_mask = np.isin(labels, small_component_labels)

    if not remove_mask.any():
        # Nothing small enough to remove
        return vertices, faces, np.arange(n_verts)

    # Log what we're removing
    removed_sizes = component_sizes[small_component_labels]
    n_removed_verts = remove_mask.sum()
    logger.info(
        f"Removing {len(small_component_labels)} small disconnected component(s) "
        f"with sizes {np.sort(removed_sizes).tolist()} ({n_removed_verts} total vertices)"
    )

    # Get faces that are entirely within kept vertices (vectorized)
    keep_mask = ~remove_mask
    kept_indices = np.where(keep_mask)[0]
    face_mask = np.all(np.isin(faces, kept_indices), axis=1)
    new_faces = faces[face_mask]

    # Reindex to remove gaps
    return remove_isolated_vertices(vertices, new_faces)


def validate_topology(
    vertices: np.ndarray, faces: np.ndarray, strict: bool = True
) -> int:
    """Validate that surface has disk topology suitable for flattening.

    A surface patch must have Euler characteristic χ = 1 to be homeomorphic
    to a disk and thus flattenable to a plane without cuts or self-intersections.

    χ = V - E + F where V=vertices, E=edges, F=faces

    Args:
        vertices: (V, 3) vertex coordinates
        faces: (F, 3) face indices
        strict: If True, raise TopologyError for χ ≠ 1. If False, just warn.

    Returns:
        Euler characteristic

    Raises:
        TopologyError: If strict=True and χ ≠ 1
    """
    n_vertices = len(vertices)
    n_edges = len(igl.edges(faces))
    n_faces = len(faces)
    euler = n_vertices - n_edges + n_faces

    if euler != 1:
        msg = (
            f"Surface has Euler characteristic χ = {euler} (expected 1 for disk topology).\n"
            f"  Vertices: {n_vertices:,}, Edges: {n_edges:,}, Faces: {n_faces:,}\n"
            f"  This indicates the surface has topological defects "
            f"(holes, handles, or disconnected components).\n"
            f"  A surface with χ ≠ 1 cannot be flattened to a plane without "
            f"self-intersections.\n"
            f"  Consider using mris_cut to create a disk-like patch, or check "
            f"the surface for defects."
        )
        if strict:
            raise TopologyError(msg)
        else:
            print(f"WARNING: {msg}")

    return euler


def count_boundary_loops(faces: np.ndarray) -> tuple[int, list[np.ndarray]]:
    """Count all boundary loops in a mesh.

    A mesh with disk topology should have exactly one boundary loop.
    Multiple loops indicate holes or disconnected boundaries.

    Uses edge-face counting to find boundary edges (edges appearing in
    exactly one face), then traces connected loops.

    Parameters
    ----------
    faces : ndarray of shape (F, 3)
        Triangle face indices.

    Returns
    -------
    n_loops : int
        Number of boundary loops.
    loops : list of ndarray
        List of arrays, each containing vertex indices for one loop.
    """
    from collections import defaultdict

    # Count how many faces each edge belongs to
    edge_face_count = defaultdict(int)
    for face in faces:
        for i in range(3):
            edge = tuple(sorted([int(face[i]), int(face[(i + 1) % 3])]))
            edge_face_count[edge] += 1

    # Boundary edges appear in exactly 1 face
    boundary_edges = {e for e, count in edge_face_count.items() if count == 1}

    if not boundary_edges:
        return 0, []

    # Build adjacency from boundary edges
    boundary_adj = defaultdict(set)
    for v1, v2 in boundary_edges:
        boundary_adj[v1].add(v2)
        boundary_adj[v2].add(v1)

    # Validate boundary structure: each boundary vertex should have exactly 2 neighbors
    # (one on each side along the boundary). Vertices with != 2 neighbors indicate
    # T-junctions or endpoints, which shouldn't occur in a valid mesh boundary.
    for v, neighbors in boundary_adj.items():
        if len(neighbors) != 2:
            warnings.warn(
                f"Boundary vertex {v} has {len(neighbors)} neighbors (expected 2). "
                "This may indicate mesh topology issues."
            )

    # Trace connected loops
    loops = []
    visited = set()

    for start in boundary_adj:
        if start in visited:
            continue

        loop = [start]
        visited.add(start)
        current = start

        while True:
            neighbors = boundary_adj[current] - visited
            if not neighbors:
                break
            current = next(iter(neighbors))
            loop.append(current)
            visited.add(current)

        # Validate loop closure: the start vertex should be a neighbor of the last vertex
        if len(loop) > 1 and start not in boundary_adj[loop[-1]]:
            warnings.warn(
                f"Loop starting at vertex {start} may not be properly closed. "
                f"Last vertex {loop[-1]} is not connected back to start."
            )

        loops.append(np.array(loop))

    return len(loops), loops


def freesurfer_projection(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Project onto plane perpendicular to average surface normal.

    This implements FreeSurfer's MRISflattenPatch initial projection strategy.
    The algorithm:
    1. Computes vertex normals (average of adjacent face normals per vertex)
    2. Sums vertex normals to get average normal direction
    3. Centers vertices at origin
    4. Rotates mesh so average normal aligns with +Z axis
    5. Takes XY coordinates as the 2D projection

    The rotation uses FreeSurfer's exact transform() function formulation.
    No hemisphere-specific logic is used - the geometry naturally determines
    the orientation.

    Args:
        vertices: (V, 3) vertex coordinates
        faces: (F, 3) face indices

    Returns:
        (V, 2) UV coordinates
    """
    # Step 1: Compute vertex normals
    # igl.per_vertex_normals returns area-weighted average of adjacent face normals
    vertex_normals = igl.per_vertex_normals(vertices, faces)

    # Step 2: Compute average normal (sum of vertex normals; direction matters for rotation,
    # and the vector is normalized before use)
    avg_normal = vertex_normals.sum(axis=0)

    # Step 3 & 4: Center vertices at origin
    centroid = vertices.mean(axis=0)
    centered = vertices - centroid

    # Step 5: Normalize the average normal
    avg_normal = avg_normal / np.linalg.norm(avg_normal)
    nx, ny, nz = avg_normal

    # Step 6: Compute d = sqrt(nx² + ny²)
    d = np.sqrt(nx * nx + ny * ny)

    # Step 7: Apply rotation transform if patch isn't already in xy-plane
    if d > 1e-6:
        # FreeSurfer's transform() rotation matrix
        # Maps average normal (nx, ny, nz) to z-axis (0, 0, 1)
        R = np.array(
            [
                [nx * nz / d, ny * nz / d, -d],
                [-ny / d, nx / d, 0.0],
                [nx, ny, nz],
            ]
        )
        rotated = centered @ R.T
    else:
        # Average normal already points along z-axis, no rotation needed
        rotated = centered

    # Step 8: Take XY coordinates (projection onto xy-plane)
    uv = rotated[:, :2]

    return uv


@jax.jit
def count_flipped_triangles(uv: jnp.ndarray, faces: jnp.ndarray) -> jnp.ndarray:
    """Count triangles with negative (flipped) area.

    Args:
        uv: (V, 2) UV coordinates
        faces: (F, 3) face indices

    Returns:
        Number of flipped triangles (scalar)
    """
    v0, v1, v2 = uv[faces[:, 0]], uv[faces[:, 1]], uv[faces[:, 2]]
    areas = 0.5 * (
        (v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1])
        - (v2[:, 0] - v0[:, 0]) * (v1[:, 1] - v0[:, 1])
    )
    return jnp.sum(areas <= 0)


@jax.jit
def _compute_distance_error_jit(
    uv: jnp.ndarray,
    neighbors: jnp.ndarray,
    targets: jnp.ndarray,
    mask: jnp.ndarray,
) -> jnp.ndarray:
    """JIT-compiled core of distance error computation.

    Uses FreeSurfer's formula:
        pct = 100 * mean(|dist - odist|) / mean(odist)

    Args:
        uv: (V, 2) UV coordinates
        neighbors: (V, max_neighbors) neighbor indices
        targets: (V, max_neighbors) target distances
        mask: (V, max_neighbors) validity mask

    Returns:
        Percentage distance error (scalar)
    """
    uv_neighbors = uv[neighbors]
    uv_diffs = uv_neighbors - uv[:, None, :]
    uv_distances = jnp.sqrt(jnp.sum(uv_diffs**2, axis=-1))

    valid = mask & (targets > 0)

    abs_errors = jnp.where(valid, jnp.abs(uv_distances - targets), 0.0)
    targets_valid = jnp.where(valid, targets, 0.0)
    n_valid = jnp.sum(valid)

    mean_error = jnp.sum(abs_errors) / n_valid
    mean_odist = jnp.sum(targets_valid) / n_valid

    return 100.0 * mean_error / mean_odist


# =============================================================================
# Energy and gradient functions
# =============================================================================


def make_energy_functions(
    neighbors_jax: jnp.ndarray,
    targets_jax: jnp.ndarray,
    mask_jax: jnp.ndarray,
    faces_jax: jnp.ndarray,
):
    """Create JIT-compiled energy and gradient functions.

    Args:
        neighbors_jax: (V, max_neighbors) neighbor indices
        targets_jax: (V, max_neighbors) target distances
        mask_jax: (V, max_neighbors) validity mask
        faces_jax: (F, 3) face indices

    Returns:
        Tuple of (compute_energies, grad_J_d, grad_J_a)
    """

    @jax.jit
    def compute_energies(uv):
        """Compute both energy components."""
        J_d = compute_metric_energy(uv, neighbors_jax, targets_jax, mask_jax)
        J_a = compute_area_energy_fs_v6(uv, faces_jax)
        return J_d, J_a

    @jax.jit
    def grad_J_d(uv):
        return jax.grad(
            lambda u: compute_metric_energy(u, neighbors_jax, targets_jax, mask_jax)
        )(uv)

    @jax.jit
    def grad_J_a(uv):
        return jax.grad(lambda u: compute_area_energy_fs_v6(u, faces_jax))(uv)

    return compute_energies, grad_J_d, grad_J_a


def compute_normalized_lambdas(
    J_d: float, J_a: float, ratio: float
) -> tuple[float, float]:
    """Compute lambda weights so effective contributions have desired ratio.

    Args:
        J_d: Current distance energy
        J_a: Current area energy
        ratio: Desired area/distance energy ratio

    Returns:
        Tuple of (lambda_d, lambda_a)
    """
    lambda_d = 1.0
    lambda_a = ratio * J_d / J_a
    return lambda_d, lambda_a


def make_energy_fn(
    lambda_d: float,
    lambda_a: float,
    neighbors_jax: jnp.ndarray,
    targets_jax: jnp.ndarray,
    mask_jax: jnp.ndarray,
    faces_jax: jnp.ndarray,
):
    """Create energy function with given lambda weights.

    Args:
        lambda_d: Weight for distance energy
        lambda_a: Weight for area energy
        neighbors_jax: (V, max_neighbors) neighbor indices
        targets_jax: (V, max_neighbors) target distances
        mask_jax: (V, max_neighbors) validity mask
        faces_jax: (F, 3) face indices

    Returns:
        JIT-compiled energy function: uv -> (total_energy, (J_d, J_a))
    """

    @jax.jit
    def energy_fn(uv):
        J_d = compute_metric_energy(uv, neighbors_jax, targets_jax, mask_jax)
        J_a = compute_area_energy_fs_v6(uv, faces_jax)
        return lambda_d * J_d + lambda_a * J_a, (J_d, J_a)

    return energy_fn


def make_weighted_gradient_fn(
    grad_J_d_fn,
    grad_J_a_fn,
    avg_nbrs: float,
):
    """Create function to compute weighted gradient with FreeSurfer-style normalization.

    FreeSurfer normalizes the distance gradient by 1/avg_nbrs to balance the
    gradient magnitudes between distance and area terms.

    Args:
        grad_J_d_fn: JIT-compiled distance energy gradient function
        grad_J_a_fn: JIT-compiled area energy gradient function
        avg_nbrs: Average number of neighbors per vertex (for gradient normalization)

    Returns:
        Function: (uv, l_dist, l_nlarea) -> gradient
    """

    def compute_weighted_gradient(uv, l_dist, l_nlarea):
        g_d = grad_J_d_fn(uv)
        g_a = grad_J_a_fn(uv)

        # FreeSurfer-style: normalize distance gradient by avg_nbrs
        # This balances the gradient magnitudes between distance and area terms
        g_d_normalized = g_d / avg_nbrs

        # Weighted combination: l_dist * g_d_normalized + l_nlarea * g_a
        return l_dist * g_d_normalized + l_nlarea * g_a

    return compute_weighted_gradient


# =============================================================================
# Vectorized line search
# =============================================================================


def make_vectorized_line_search(energy_fn, n_coarse_steps: int = 15):
    """Create a vectorized line search function with quadratic refinement.

    This implements FreeSurfer's mrisLineMinimize approach:
    1. Evaluate energy at log-spaced step sizes (vectorized)
    2. Find the best coarse step
    3. Fit a quadratic through 3 points around the best step
    4. Return the minimum of the quadratic (if valid) or the best evaluated point

    Args:
        energy_fn: function(uv) -> (energy, aux)
        n_coarse_steps: number of coarse step sizes to try

    Returns:
        JIT-compiled line search function
    """

    @jax.jit
    def _eval_energy_at_alpha(uv, grad_normalized, alpha):
        """Evaluate energy at uv - alpha * grad_normalized."""
        uv_new = uv - alpha * grad_normalized
        energy, aux = energy_fn(uv_new)
        return energy, uv_new, aux

    # Vectorize over alpha
    _eval_energies_batch = jax.vmap(
        lambda uv, grad_normalized, alpha: _eval_energy_at_alpha(
            uv, grad_normalized, alpha
        ),
        in_axes=(None, None, 0),
    )

    @jax.jit
    def _fit_quadratic_minimum(x0, x1, x2, y0, y1, y2):
        """Fit a quadratic through 3 points and find its minimum."""
        denom0 = (x0 - x1) * (x0 - x2)
        denom1 = (x1 - x0) * (x1 - x2)
        denom2 = (x2 - x0) * (x2 - x1)

        a = y0 / denom0 + y1 / denom1 + y2 / denom2
        b = -y0 * (x1 + x2) / denom0 - y1 * (x0 + x2) / denom1 - y2 * (x0 + x1) / denom2

        x_min = -b / (2 * a + 1e-20)
        valid = (a > 1e-12) & (x_min > x0) & (x_min < x2)

        return jnp.where(valid, x_min, x1)

    @jax.jit
    def vectorized_line_search(uv, grad, max_mm=1000.0, min_mm=0.001):
        """Vectorized line search with quadratic refinement.

        Args:
            uv: current UV coordinates
            grad: gradient direction
            max_mm: maximum displacement in coordinate units
            min_mm: minimum displacement in coordinate units

        Returns:
            (new_uv, new_energy, best_alpha, aux)
        """
        grad_normalized = grad / (jnp.linalg.norm(grad) + 1e-20)

        max_grad = jnp.max(jnp.abs(grad_normalized))
        mean_grad = jnp.mean(jnp.abs(grad_normalized))

        max_dt = max_mm / (max_grad + 1e-20)
        min_dt = min_mm / (mean_grad + 1e-20)
        min_dt = jnp.minimum(min_dt, max_dt / 1000)

        log_min = jnp.log10(min_dt)
        log_max = jnp.log10(max_dt)
        alphas = jnp.logspace(log_min, log_max, n_coarse_steps)

        energies, uvs, auxs = _eval_energies_batch(uv, grad_normalized, alphas)

        energy_0, aux_0 = energy_fn(uv)

        best_coarse_idx = jnp.argmin(energies)
        best_coarse_energy = energies[best_coarse_idx]
        best_coarse_alpha = alphas[best_coarse_idx]

        idx_left = jnp.maximum(best_coarse_idx - 1, 0)
        idx_right = jnp.minimum(best_coarse_idx + 1, n_coarse_steps - 1)

        x0, x1, x2 = alphas[idx_left], alphas[best_coarse_idx], alphas[idx_right]
        y0, y1, y2 = energies[idx_left], energies[best_coarse_idx], energies[idx_right]

        alpha_quad = _fit_quadratic_minimum(x0, x1, x2, y0, y1, y2)

        energy_quad, uv_quad, aux_quad = _eval_energy_at_alpha(
            uv, grad_normalized, alpha_quad
        )

        use_quad = energy_quad < best_coarse_energy
        best_from_search_energy = jnp.where(use_quad, energy_quad, best_coarse_energy)
        best_from_search_alpha = jnp.where(use_quad, alpha_quad, best_coarse_alpha)
        best_from_search_uv = jax.lax.cond(
            use_quad, lambda: uv_quad, lambda: uvs[best_coarse_idx]
        )
        best_from_search_aux = jax.lax.cond(
            use_quad,
            lambda: aux_quad,
            lambda: (auxs[0][best_coarse_idx], auxs[1][best_coarse_idx]),
        )

        use_start = energy_0 <= best_from_search_energy
        final_energy = jnp.where(use_start, energy_0, best_from_search_energy)
        final_alpha = jnp.where(use_start, 0.0, best_from_search_alpha)
        final_uv = jax.lax.cond(use_start, lambda: uv, lambda: best_from_search_uv)
        final_aux = jax.lax.cond(use_start, lambda: aux_0, lambda: best_from_search_aux)

        return final_uv, final_energy, final_alpha, final_aux

    return vectorized_line_search


# =============================================================================
# Optimization functions
# =============================================================================


def run_smoothed_optimization(
    uv_init: np.ndarray,
    lambda_d: float,
    lambda_a: float,
    smoothing_schedule: list[int],
    neighbors_jax: jnp.ndarray,
    targets_jax: jnp.ndarray,
    mask_jax: jnp.ndarray,
    faces_jax: jnp.ndarray,
    smooth_neighbors_jax: jnp.ndarray,
    smooth_mask_jax: jnp.ndarray,
    smooth_counts_jax: jnp.ndarray,
    avg_nbrs: float,
    iters_per_level: int = 50,
    print_every: int = 10,
    verbose: bool = True,
    base_tol: float = 0.2,
    max_small: int = 50000,
    total_small_limit: int = 15000,
    n_coarse_steps: int = 15,
) -> np.ndarray:
    """Run gradient descent with FreeSurfer-style gradient smoothing.

    Uses vectorized line search with quadratic refinement.
    Convergence is based on relative SSE change (FreeSurfer v6.0.0 style).

    Args:
        uv_init: Initial UV coordinates
        lambda_d: Weight for distance energy
        lambda_a: Weight for area energy
        smoothing_schedule: List of gradient averaging counts
        neighbors_jax: (V, max_neighbors) neighbor indices
        targets_jax: (V, max_neighbors) target distances
        mask_jax: (V, max_neighbors) validity mask
        faces_jax: (F, 3) face indices
        smooth_neighbors_jax: Smoothing adjacency
        smooth_mask_jax: Smoothing validity mask
        smooth_counts_jax: Smoothing neighbor counts
        avg_nbrs: Average number of neighbors per vertex (for gradient normalization)
        iters_per_level: Max iterations per smoothing level
        print_every: Print progress every N iterations
        verbose: Print progress messages
        base_tol: Base convergence tolerance
        max_small: Max consecutive small steps
        total_small_limit: Max total small steps
        n_coarse_steps: Number of line search steps

    Returns:
        Optimized UV coordinates
    """
    energy_fn = make_energy_fn(
        lambda_d, lambda_a, neighbors_jax, targets_jax, mask_jax, faces_jax
    )

    # Create separate gradient functions for FreeSurfer-style normalization
    @jax.jit
    def grad_J_d_fn(uv):
        return jax.grad(
            lambda u: compute_metric_energy(u, neighbors_jax, targets_jax, mask_jax)
        )(uv)

    @jax.jit
    def grad_J_a_fn(uv):
        return jax.grad(lambda u: compute_area_energy_fs_v6(u, faces_jax))(uv)

    # FreeSurfer-style gradient: lambda_d * (g_d / avg_nbrs) + lambda_a * g_a
    compute_weighted_gradient = make_weighted_gradient_fn(
        grad_J_d_fn, grad_J_a_fn, avg_nbrs
    )

    line_search_fn = make_vectorized_line_search(
        energy_fn, n_coarse_steps=n_coarse_steps
    )

    uv = jnp.asarray(uv_init)

    iteration = 0
    total_small = 0.0
    start_time = time.time()

    if verbose:
        print(
            f"\n{'Iter':>5} {'n_avg':>6} {'Energy':>12} {'J_d':>10} {'J_a':>10} "
            f"{'relΔSSE':>10} {'alpha':>10} {'Flipped':>8} {'%err':>7}"
        )
        print("-" * 100)

    for n_avg in smoothing_schedule:
        # FreeSurfer-style tolerance scaling (tighter at low n_avg)
        scaled_tol = base_tol * np.sqrt((n_avg + 1.0) / 1024.0)

        nsmall = 0
        old_sse = None

        for i in range(iters_per_level):
            # FreeSurfer-style gradient: lambda_d * (g_d / avg_nbrs) + lambda_a * g_a
            grad = compute_weighted_gradient(uv, lambda_d, lambda_a)

            if n_avg > 0:
                grad_smooth = smooth_gradient(
                    grad,
                    smooth_neighbors_jax,
                    smooth_mask_jax,
                    smooth_counts_jax,
                    n_avg,
                )
            else:
                grad_smooth = grad

            uv, energy, alpha, (J_d, J_a) = line_search_fn(uv, grad_smooth)

            iteration += 1
            current_sse = float(energy)
            alpha_val = float(alpha)

            # FreeSurfer convergence: 100 * rel_change < tol, i.e., rel_change < tol/100
            rel_change = 0.0
            if old_sse is not None and old_sse > 0:
                rel_change = (old_sse - current_sse) / old_sse

                # Divide scaled_tol by 100 to match FreeSurfer's formula
                if rel_change < scaled_tol / 100.0:
                    nsmall += 1
                    total_small += 1
                else:
                    total_small = max(0, total_small - 0.25)
                    nsmall = 0

            should_print = verbose and (iteration % print_every == 0 or i == 0)
            if should_print:
                n_flipped = int(count_flipped_triangles(uv, faces_jax))
                pct_err = float(
                    _compute_distance_error_jit(
                        uv, neighbors_jax, targets_jax, mask_jax
                    )
                )
                print(
                    f"{iteration:5d} {n_avg:6d} {current_sse:12.4f} {float(J_d):10.4f} "
                    f"{float(J_a):10.4f} {rel_change:10.2e} {alpha_val:10.2e} "
                    f"{n_flipped:8d} {pct_err:6.1f}%"
                )

            if nsmall > max_small:
                if verbose:
                    print(
                        f"  -> Converged at n_avg={n_avg}: {nsmall} consecutive "
                        f"small steps (>{max_small})"
                    )
                break

            if total_small > total_small_limit:
                if verbose:
                    print(
                        f"  -> Converged: total_small={total_small:.0f} > "
                        f"{total_small_limit}"
                    )
                break

            if old_sse is not None and current_sse > 0:
                if 100 * (old_sse - current_sse) / current_sse < scaled_tol:
                    if verbose:
                        print(
                            f"  -> Converged at n_avg={n_avg}: relative change < "
                            f"{scaled_tol:.2e}"
                        )
                    break

            if alpha_val == 0:
                if verbose:
                    print(f"  -> At minimum (alpha=0) at n_avg={n_avg}")
                break

            old_sse = current_sse

        if verbose:
            n_flipped = int(count_flipped_triangles(uv, faces_jax))
            pct_err = float(
                _compute_distance_error_jit(uv, neighbors_jax, targets_jax, mask_jax)
            )
            print(
                f"  -> Level n_avg={n_avg} done: {n_flipped} flipped, {pct_err:.1f}% err, "
                f"nsmall={nsmall}, total_small={total_small:.0f}"
            )

    elapsed = time.time() - start_time
    if verbose:
        print(f"\nTotal: {iteration} iterations in {elapsed:.1f}s")

    return np.array(uv)


def run_adaptive_optimization(
    uv_init: np.ndarray,
    lambda_d: float,
    lambda_a: float,
    smoothing_schedule: list[int],
    neighbors_jax: jnp.ndarray,
    targets_jax: jnp.ndarray,
    mask_jax: jnp.ndarray,
    faces_jax: jnp.ndarray,
    smooth_neighbors_jax: jnp.ndarray,
    smooth_mask_jax: jnp.ndarray,
    smooth_counts_jax: jnp.ndarray,
    compute_energies_fn,
    avg_nbrs: float,
    iters_per_level: int = 50,
    print_every: int = 10,
    verbose: bool = True,
    base_tol: float = 0.2,
    max_small: int = 50000,
    total_small_limit: int = 15000,
    n_coarse_steps: int = 15,
    flipped_threshold_factor: float = 20.0,
    recovery_area_ratio: float = 0.5,
    recovery_iterations: int = 50,
) -> np.ndarray:
    """Run adaptive gradient descent with flipped-triangle recovery.

    This is similar to run_smoothed_optimization but monitors for flipped
    triangle explosions and triggers recovery when needed. When the flipped
    count exceeds a threshold, the optimizer temporarily increases the area
    weight to fix the flipped triangles before resuming normal optimization.

    Args:
        uv_init: Initial UV coordinates
        lambda_d: Weight for distance energy
        lambda_a: Weight for area energy
        smoothing_schedule: List of gradient averaging counts
        neighbors_jax: (V, max_neighbors) neighbor indices
        targets_jax: (V, max_neighbors) target distances
        mask_jax: (V, max_neighbors) validity mask
        faces_jax: (F, 3) face indices
        smooth_neighbors_jax: Smoothing adjacency
        smooth_mask_jax: Smoothing validity mask
        smooth_counts_jax: Smoothing neighbor counts
        compute_energies_fn: Function to compute (J_d, J_a) for energy tracking
        avg_nbrs: Average number of neighbors per vertex (for gradient normalization)
        iters_per_level: Max iterations per smoothing level
        print_every: Print progress every N iterations
        verbose: Print progress messages
        base_tol: Base convergence tolerance
        max_small: Max consecutive small steps
        total_small_limit: Max total small steps
        n_coarse_steps: Number of line search steps
        flipped_threshold_factor: Trigger recovery when flipped > initial * factor
        recovery_area_ratio: Area/distance ratio during recovery (higher = more area weight)
        recovery_iterations: Number of recovery iterations

    Returns:
        Optimized UV coordinates
    """
    energy_fn = make_energy_fn(
        lambda_d, lambda_a, neighbors_jax, targets_jax, mask_jax, faces_jax
    )

    # Create separate gradient functions for FreeSurfer-style normalization
    @jax.jit
    def grad_J_d_fn(uv):
        return jax.grad(
            lambda u: compute_metric_energy(u, neighbors_jax, targets_jax, mask_jax)
        )(uv)

    @jax.jit
    def grad_J_a_fn(uv):
        return jax.grad(lambda u: compute_area_energy_fs_v6(u, faces_jax))(uv)

    # FreeSurfer-style gradient: lambda_d * (g_d / avg_nbrs) + lambda_a * g_a
    compute_weighted_gradient = make_weighted_gradient_fn(
        grad_J_d_fn, grad_J_a_fn, avg_nbrs
    )

    line_search_fn = make_vectorized_line_search(
        energy_fn, n_coarse_steps=n_coarse_steps
    )

    uv = jnp.asarray(uv_init)

    # Track initial flipped count for adaptive recovery threshold
    initial_flipped = int(count_flipped_triangles(uv, faces_jax))
    flipped_threshold = max(initial_flipped * flipped_threshold_factor, 500)
    best_uv = uv
    best_flipped = initial_flipped
    recovery_count = 0
    max_recoveries = 3  # Limit recovery attempts

    iteration = 0
    total_small = 0.0
    start_time = time.time()

    if verbose:
        print(
            f"\n{'Iter':>5} {'n_avg':>6} {'Energy':>12} {'J_d':>10} {'J_a':>10} "
            f"{'relΔSSE':>10} {'alpha':>10} {'Flipped':>8} {'%err':>7}"
        )
        print("-" * 100)

    for n_avg in smoothing_schedule:
        # FreeSurfer-style tolerance scaling (tighter at low n_avg)
        scaled_tol = base_tol * np.sqrt((n_avg + 1.0) / 1024.0)

        nsmall = 0
        old_sse = None

        for i in range(iters_per_level):
            # FreeSurfer-style gradient: lambda_d * (g_d / avg_nbrs) + lambda_a * g_a
            grad = compute_weighted_gradient(uv, lambda_d, lambda_a)

            if n_avg > 0:
                grad_smooth = smooth_gradient(
                    grad,
                    smooth_neighbors_jax,
                    smooth_mask_jax,
                    smooth_counts_jax,
                    n_avg,
                )
            else:
                grad_smooth = grad

            uv, energy, alpha, (J_d, J_a) = line_search_fn(uv, grad_smooth)

            iteration += 1
            current_sse = float(energy)
            alpha_val = float(alpha)

            # Check flipped count and trigger recovery if needed
            n_flipped = int(count_flipped_triangles(uv, faces_jax))

            # Track best state
            if n_flipped < best_flipped:
                best_flipped = n_flipped
                best_uv = uv

            # Adaptive recovery: if flipped explodes, run recovery phase
            if n_flipped > flipped_threshold and recovery_count < max_recoveries:
                if verbose:
                    print(
                        f"  -> RECOVERY TRIGGERED: {n_flipped} flipped > "
                        f"threshold {flipped_threshold:.0f}"
                    )

                # Run recovery with higher area weight
                J_d_curr, J_a_curr = compute_energies_fn(uv)
                recovery_lambda_d, recovery_lambda_a = compute_normalized_lambdas(
                    float(J_d_curr), float(J_a_curr), ratio=recovery_area_ratio
                )

                recovery_energy_fn = make_energy_fn(
                    recovery_lambda_d,
                    recovery_lambda_a,
                    neighbors_jax,
                    targets_jax,
                    mask_jax,
                    faces_jax,
                )
                recovery_grad_fn = jax.jit(jax.grad(lambda u: recovery_energy_fn(u)[0]))
                recovery_line_search = make_vectorized_line_search(
                    recovery_energy_fn, n_coarse_steps=n_coarse_steps
                )

                for r_iter in range(recovery_iterations):
                    r_grad = recovery_grad_fn(uv)
                    if n_avg > 0:
                        r_grad = smooth_gradient(
                            r_grad,
                            smooth_neighbors_jax,
                            smooth_mask_jax,
                            smooth_counts_jax,
                            min(n_avg, 64),
                        )
                    uv, _, _, _ = recovery_line_search(uv, r_grad)

                n_flipped_after = int(count_flipped_triangles(uv, faces_jax))
                recovery_count += 1

                if verbose:
                    print(
                        f"  -> RECOVERY COMPLETE: {n_flipped} -> {n_flipped_after} flipped "
                        f"(attempt {recovery_count}/{max_recoveries})"
                    )

                # Update threshold to prevent repeated triggers
                flipped_threshold = max(n_flipped_after * 5, flipped_threshold)

            # FreeSurfer convergence: 100 * rel_change < tol, i.e., rel_change < tol/100
            rel_change = 0.0
            if old_sse is not None and old_sse > 0:
                rel_change = (old_sse - current_sse) / old_sse

                # Divide scaled_tol by 100 to match FreeSurfer's formula
                if rel_change < scaled_tol / 100.0:
                    nsmall += 1
                    total_small += 1
                else:
                    total_small = max(0, total_small - 0.25)
                    nsmall = 0

            should_print = verbose and (iteration % print_every == 0 or i == 0)
            if should_print:
                pct_err = float(
                    _compute_distance_error_jit(
                        uv, neighbors_jax, targets_jax, mask_jax
                    )
                )
                print(
                    f"{iteration:5d} {n_avg:6d} {current_sse:12.4f} {float(J_d):10.4f} "
                    f"{float(J_a):10.4f} {rel_change:10.2e} {alpha_val:10.2e} "
                    f"{n_flipped:8d} {pct_err:6.1f}%"
                )

            if nsmall > max_small:
                if verbose:
                    print(
                        f"  -> Converged at n_avg={n_avg}: {nsmall} consecutive "
                        f"small steps (>{max_small})"
                    )
                break

            if total_small > total_small_limit:
                if verbose:
                    print(
                        f"  -> Converged: total_small={total_small:.0f} > "
                        f"{total_small_limit}"
                    )
                break

            if old_sse is not None and current_sse > 0:
                if 100 * (old_sse - current_sse) / current_sse < scaled_tol:
                    if verbose:
                        print(
                            f"  -> Converged at n_avg={n_avg}: relative change < "
                            f"{scaled_tol:.2e}"
                        )
                    break

            if alpha_val == 0:
                if verbose:
                    print(f"  -> At minimum (alpha=0) at n_avg={n_avg}")
                break

            old_sse = current_sse

        if verbose:
            n_flipped = int(count_flipped_triangles(uv, faces_jax))
            pct_err = float(
                _compute_distance_error_jit(uv, neighbors_jax, targets_jax, mask_jax)
            )
            print(
                f"  -> Level n_avg={n_avg} done: {n_flipped} flipped, {pct_err:.1f}% err, "
                f"nsmall={nsmall}, total_small={total_small:.0f}"
            )

    elapsed = time.time() - start_time

    # Use best state if final state has significantly more flipped triangles
    final_flipped = int(count_flipped_triangles(uv, faces_jax))
    if best_flipped < final_flipped * 0.5:
        if verbose:
            print(
                f"  -> Using best state: {best_flipped} flipped "
                f"(vs final {final_flipped})"
            )
        uv = best_uv

    if verbose:
        print(f"\nTotal: {iteration} iterations in {elapsed:.1f}s")

    return np.array(uv)


@jax.jit
def _apply_area_preserving_scale(
    uv: jnp.ndarray, faces: jnp.ndarray, orig_area: float
) -> jnp.ndarray:
    """Apply area-preserving scaling to UV coordinates (JIT-compiled).

    Implements FreeSurfer's area-preserving scaling formula from mrisurf.c:6260-6264:
        scale = sqrt(orig_area / (total_area + neg_area))

    This scaling maintains the original surface area during optimization,
    preventing progressive shrinkage that would destabilize the optimization.

    Parameters
    ----------
    uv : ndarray of shape (V, 2)
        Current 2D vertex positions
    faces : ndarray of shape (F, 3)
        Triangle face indices
    orig_area : float
        Original 3D surface area to maintain

    Returns
    -------
    ndarray of shape (V, 2)
        Scaled UV coordinates
    """
    total_area, neg_area = compute_2d_areas(uv, faces)
    # Protect against division by zero in degenerate cases
    epsilon = 1e-8
    scale = jnp.sqrt(orig_area / jnp.maximum(total_area + neg_area, epsilon))
    centroid = jnp.mean(uv, axis=0)
    return (uv - centroid) * scale + centroid


def remove_negative_area(
    uv_init: np.ndarray,
    neighbors_jax: jnp.ndarray,
    targets_jax: jnp.ndarray,
    mask_jax: jnp.ndarray,
    faces_jax: jnp.ndarray,
    smooth_neighbors_jax: jnp.ndarray,
    smooth_mask_jax: jnp.ndarray,
    smooth_counts_jax: jnp.ndarray,
    compute_energies_fn,
    grad_J_d_fn,
    grad_J_a_fn,
    avg_nbrs: float,
    config: NegativeAreaRemovalConfig,
    convergence_max_small: int = 50000,
    convergence_total_small: int = 15000,
    n_coarse_steps: int = 15,
    print_every: int = 20,
    verbose: bool = True,
    orig_area: Optional[float] = None,
) -> np.ndarray:
    """FreeSurfer-style negative area removal with vectorized line search.

    Args:
        uv_init: Initial UV coordinates
        neighbors_jax: (V, max_neighbors) neighbor indices
        targets_jax: (V, max_neighbors) target distances
        mask_jax: (V, max_neighbors) validity mask
        faces_jax: (F, 3) face indices
        smooth_neighbors_jax: Smoothing adjacency
        smooth_mask_jax: Smoothing validity mask
        smooth_counts_jax: Smoothing neighbor counts
        compute_energies_fn: Function to compute (J_d, J_a)
        grad_J_d_fn: JIT-compiled distance energy gradient
        grad_J_a_fn: JIT-compiled area energy gradient
        avg_nbrs: Average number of neighbors per vertex (for gradient normalization)
        config: Negative area removal configuration
        convergence_max_small: Max consecutive small steps
        convergence_total_small: Max total small steps
        n_coarse_steps: Number of line search steps
        print_every: Print progress every N iterations
        verbose: Print progress messages
        orig_area: Original 3D surface area for area-preserving scaling.
            If provided, rescales UV coordinates at each iteration to maintain
            this area (FreeSurfer mrisurf.c:6260-6264).

    Returns:
        UV coordinates with reduced negative area
    """
    compute_weighted_gradient = make_weighted_gradient_fn(
        grad_J_d_fn, grad_J_a_fn, avg_nbrs
    )

    def make_schedule(base):
        schedule = []
        n = base
        while n >= 1:
            schedule.append(n)
            n //= 4
        schedule.append(0)
        return schedule

    uv = jnp.asarray(uv_init)
    total_iterations = 0
    total_small = 0.0
    start_time = time.time()

    if verbose:
        print(f"\n{'=' * 85}")
        print(
            "NEGATIVE AREA REMOVAL (Vectorized Quadratic Line Search, "
            "FreeSurfer v6.0.0 convergence)"
        )
        print(f"{'=' * 85}")

    # Check if we can skip NAR entirely (FreeSurfer only checks before starting)
    n_flipped = int(count_flipped_triangles(uv, faces_jax))
    total_faces = len(faces_jax)
    pct_neg = 100.0 * n_flipped / total_faces
    if pct_neg <= config.min_area_pct:
        if verbose:
            print(f"\nSkipping NAR: {pct_neg:.2f}% <= {config.min_area_pct}% target")
        return np.array(uv)

    # Run through ALL l_dist_ratios (FreeSurfer always runs all 5, no early stopping)
    n_ratios = len(config.l_dist_ratios)
    for pass_idx in range(n_ratios):
        l_dist = config.l_dist_ratios[pass_idx]

        n_flipped = int(count_flipped_triangles(uv, faces_jax))
        total_faces = len(faces_jax)
        pct_neg = 100.0 * n_flipped / total_faces

        if verbose:
            print(
                f"\n--- Pass {pass_idx + 1}/{n_ratios}: "
                f"l_nlarea={config.l_nlarea}, l_dist={l_dist:.0e}, "
                f"flipped={n_flipped} ({pct_neg:.2f}%) ---"
            )

        J_d_curr, J_a_curr = compute_energies_fn(uv)
        J_d_curr, J_a_curr = float(J_d_curr), float(J_a_curr)

        smoothing_schedule = make_schedule(config.base_averages)

        # Use fixed FreeSurfer-style weights directly
        lambda_d = l_dist
        lambda_a = config.l_nlarea
        combined_energy_fn = make_energy_fn(
            lambda_d, lambda_a, neighbors_jax, targets_jax, mask_jax, faces_jax
        )

        line_search_fn = make_vectorized_line_search(
            combined_energy_fn, n_coarse_steps=n_coarse_steps
        )

        if verbose:
            print(
                f"  J_d={J_d_curr:.4f}, J_a={J_a_curr:.6f}, schedule={smoothing_schedule}"
            )
            print(
                f"  {'Iter':>5} {'n_avg':>6} {'J_d':>10} {'J_a':>10} {'relΔSSE':>10} "
                f"{'alpha':>10} {'Flipped':>8} {'%err':>7}"
            )
            print("  " + "-" * 88)

        iteration = 0

        for n_avg in smoothing_schedule:
            # FreeSurfer-style tolerance scaling (tighter at low n_avg)
            scaled_tol = config.base_tol * np.sqrt((n_avg + 1.0) / 1024.0)
            nsmall = 0
            old_sse = None

            for i in range(config.iters_per_level):
                # Area-preserving scaling (FreeSurfer mrisurf.c:6260-6264)
                # Maintains original patch area to prevent shrinkage and keep
                # spring forces balanced
                if orig_area is not None:
                    uv = _apply_area_preserving_scale(uv, faces_jax, orig_area)

                # FreeSurfer-style gradient: l_dist * (g_d / avg_nbrs) + l_nlarea * g_a
                grad = compute_weighted_gradient(uv, l_dist, config.l_nlarea)

                if n_avg > 0:
                    grad_smooth = smooth_gradient(
                        grad,
                        smooth_neighbors_jax,
                        smooth_mask_jax,
                        smooth_counts_jax,
                        n_avg,
                    )
                else:
                    grad_smooth = grad

                uv, energy, alpha, (J_d, J_a) = line_search_fn(uv, grad_smooth)

                iteration += 1
                total_iterations += 1
                current_sse = float(energy)
                alpha_val = float(alpha)

                # FreeSurfer convergence: 100 * rel_change < tol, i.e., rel_change < tol/100
                rel_change = 0.0
                if old_sse is not None and old_sse > 0:
                    rel_change = (old_sse - current_sse) / old_sse

                    # Divide scaled_tol by 100 to match FreeSurfer's formula
                    if rel_change < scaled_tol / 100.0:
                        nsmall += 1
                        total_small += 1
                    else:
                        total_small = max(0, total_small - 0.25)
                        nsmall = 0

                if verbose and iteration % print_every == 0:
                    n_flipped = int(count_flipped_triangles(uv, faces_jax))
                    pct_err = float(
                        _compute_distance_error_jit(
                            uv, neighbors_jax, targets_jax, mask_jax
                        )
                    )
                    print(
                        f"  {iteration:5d} {n_avg:6d} {float(J_d):10.4f} "
                        f"{float(J_a):10.4f} {rel_change:10.2e} {alpha_val:10.2e} "
                        f"{n_flipped:8d} {pct_err:6.1f}%"
                    )

                if (
                    nsmall > convergence_max_small
                    or total_small > convergence_total_small
                ):
                    break

                if old_sse is not None and current_sse > 0:
                    if 100 * (old_sse - current_sse) / current_sse < scaled_tol:
                        break

                if alpha_val == 0:
                    break

                old_sse = current_sse

        if verbose:
            n_flipped = int(count_flipped_triangles(uv, faces_jax))
            pct_err = float(
                _compute_distance_error_jit(uv, neighbors_jax, targets_jax, mask_jax)
            )
            print(
                f"  Pass {pass_idx + 1} done: {n_flipped} flipped, {pct_err:.1f}% err, "
                f"total_small={total_small:.0f}"
            )

    elapsed = time.time() - start_time
    if verbose:
        n_flipped_final = int(count_flipped_triangles(uv, faces_jax))
        pct_err_final = float(
            _compute_distance_error_jit(uv, neighbors_jax, targets_jax, mask_jax)
        )
        print(
            f"\nNegative area removal complete: {total_iterations} iterations "
            f"in {elapsed:.1f}s"
        )
        print(f"Final: {n_flipped_final} flipped, {pct_err_final:.1f}% err")

    return np.array(uv)


# =============================================================================
# Final Spring Smoothing
# =============================================================================


def final_spring_smoothing(
    uv_init: np.ndarray,
    faces_jax: jnp.ndarray,
    smooth_neighbors_jax: jnp.ndarray,
    smooth_mask_jax: jnp.ndarray,
    smooth_counts_jax: jnp.ndarray,
    is_boundary_jax: jnp.ndarray,
    neighbors_jax: jnp.ndarray,
    targets_jax: jnp.ndarray,
    mask_jax: jnp.ndarray,
    config: SpringSmoothingConfig,
    verbose: bool = True,
) -> np.ndarray:
    """FreeSurfer-style final spring smoothing.

    Uses direct displacement toward neighbor centroids (NOT energy gradient).
    This creates a Laplacian smoothing effect that regularizes triangle shapes,
    producing visually smoother flatmaps at the cost of slightly higher distance
    error (similar to FreeSurfer's ~13% -> ~15% after smoothing).

    FreeSurfer boundary handling (mrisurf.c:22094):
        if (v->border && !v->neg) continue;
    Only skip boundary vertices that are NOT involved in flipped triangles.
    This allows fixing flipped triangles at the boundary.

    Reference: FreeSurfer mrisurf.c:7904-7928, mrisurf.c:22059

    Args:
        uv_init: Initial UV coordinates
        faces_jax: (F, 3) face indices
        smooth_neighbors_jax: Smoothing adjacency
        smooth_mask_jax: Smoothing validity mask
        smooth_counts_jax: Smoothing neighbor counts
        is_boundary_jax: (V,) boolean mask of boundary vertices
        neighbors_jax: (V, max_neighbors) neighbor indices for distance error
        targets_jax: (V, max_neighbors) target distances
        mask_jax: (V, max_neighbors) validity mask
        config: Spring smoothing configuration
        verbose: Print progress messages

    Returns:
        Smoothed UV coordinates
    """
    uv = jnp.asarray(uv_init)

    if verbose:
        print(f"\n{'=' * 85}")
        print("FINAL SPRING SMOOTHING (FreeSurfer-style)")
        print(f"{'=' * 85}")
        print(
            f"  n_iterations={config.n_iterations}, dt={config.dt}, "
            f"max_step={config.max_step_mm}"
        )

    pct_err_before = float(
        _compute_distance_error_jit(uv, neighbors_jax, targets_jax, mask_jax)
    )
    n_flipped_before = int(count_flipped_triangles(uv, faces_jax))
    if verbose:
        print(
            f"  Before: {n_flipped_before} flipped, {pct_err_before:.2f}% distance error"
        )

    for i in range(config.n_iterations):
        # FreeSurfer: skip boundary vertices UNLESS they have negative area
        # Condition: if (v->border && !v->neg) continue;
        has_neg = get_vertices_with_negative_area(uv, faces_jax)
        skip_mask = is_boundary_jax & ~has_neg  # Skip only healthy boundary vertices

        # Compute spring displacement (pulls toward neighbor centroids)
        displacement = compute_spring_displacement(
            uv,
            smooth_neighbors_jax,
            smooth_mask_jax,
            smooth_counts_jax,
            skip_mask=skip_mask,
        )

        # Scale by dt (l_spring = 1.0 is implicit)
        step = config.dt * displacement

        # Clip step magnitude per vertex (FreeSurfer's MAX_MOMENTUM_MM)
        step_mag = jnp.linalg.norm(step, axis=1, keepdims=True)
        step = jnp.where(
            step_mag > config.max_step_mm,
            step * config.max_step_mm / (step_mag + 1e-12),
            step,
        )

        # Apply step (displacement points toward centroid, so we add)
        uv = uv + step

        if verbose:
            n_flipped = int(count_flipped_triangles(uv, faces_jax))
            pct_err = float(
                _compute_distance_error_jit(uv, neighbors_jax, targets_jax, mask_jax)
            )
            print(f"  Iter {i + 1}: {n_flipped} flipped, {pct_err:.2f}% err")

    pct_err_after = float(
        _compute_distance_error_jit(uv, neighbors_jax, targets_jax, mask_jax)
    )
    n_flipped_after = int(count_flipped_triangles(uv, faces_jax))
    if verbose:
        print(
            f"  After: {n_flipped_after} flipped, {pct_err_after:.2f}% distance error"
        )

    return np.array(uv)


# =============================================================================
# SurfaceFlattener class
# =============================================================================


class SurfaceFlattener:
    """Orchestrates cortical surface flattening optimization.

    This class manages the state and orchestrates the multi-phase optimization
    for flattening cortical surface patches. It follows the FreeSurfer approach
    with JAX-accelerated energy computation and gradient descent.

    Example:
        >>> config = FlattenConfig()
        >>> flattener = SurfaceFlattener(config)
        >>> flattener.load_data("lh.cortex.patch.3d", "lh.fiducial")
        >>> flattener.compute_kring_distances()
        >>> flattener.prepare_optimization()
        >>> uv = flattener.run()
        >>> flattener.save_result(uv, "lh.flat.patch.3d")
    """

    def __init__(self, config: FlattenConfig):
        """Initialize flattener with configuration.

        Args:
            config: Flattening configuration
        """
        self.config = config

        # Mesh data (set by load_data)
        self.vertices: Optional[np.ndarray] = None
        self.faces: Optional[np.ndarray] = None
        self.fiducial_vertices: Optional[np.ndarray] = None
        self.orig_indices: Optional[np.ndarray] = None
        self.orig_area: Optional[float] = None  # Original 3D patch surface area

        # K-ring data (set by compute_kring_distances)
        self.k_rings: Optional[list] = None
        self.target_distances: Optional[list] = None
        self.avg_neighbors: Optional[float] = (
            None  # For FreeSurfer-style gradient normalization
        )

        # JAX arrays (set by prepare_optimization)
        self.neighbors_jax: Optional[jnp.ndarray] = None
        self.targets_jax: Optional[jnp.ndarray] = None
        self.mask_jax: Optional[jnp.ndarray] = None
        self.faces_jax: Optional[jnp.ndarray] = None
        self.smooth_neighbors_jax: Optional[jnp.ndarray] = None
        self.smooth_mask_jax: Optional[jnp.ndarray] = None
        self.smooth_counts_jax: Optional[jnp.ndarray] = None
        self.is_boundary_jax: Optional[jnp.ndarray] = None

        # JIT-compiled functions (set by prepare_optimization)
        self._compute_energies = None
        self._grad_J_d = None
        self._grad_J_a = None

    def load_data(self, patch_path: str, surface_path: str) -> None:
        """Load patch and surface data.

        Args:
            patch_path: Path to FreeSurfer patch file
            surface_path: Path to FreeSurfer surface file (fiducial)
        """
        if self.config.verbose:
            print("Loading cortical surface patch...")

        orig_vertices, orig_faces = read_surface(surface_path)
        patch_vertices, orig_idx, is_border = read_patch(patch_path)
        patch_faces = extract_patch_faces(orig_faces, orig_idx)

        # Remove small disconnected components (e.g., isolated triangles)
        # before removing isolated vertices
        patch_vertices_clean, patch_faces_clean, used_after_clean = (
            remove_small_components(
                patch_vertices.astype(np.float64),
                patch_faces,
                max_small_component_size=20,
                warn_medium_threshold=100,
            )
        )
        # Update orig_idx and is_border to reflect removed vertices
        orig_idx = orig_idx[used_after_clean]
        is_border = is_border[used_after_clean]

        # Use patch vertices for mesh, fiducial for distances
        self.vertices, self.faces, used = remove_isolated_vertices(
            patch_vertices_clean, patch_faces_clean
        )

        fiducial_patch_vertices = orig_vertices[orig_idx].astype(np.float64)
        self.fiducial_vertices, _, _ = remove_isolated_vertices(
            fiducial_patch_vertices, patch_faces_clean
        )

        self.orig_indices = orig_idx[used]
        self.is_border = is_border[used]

        if self.config.verbose:
            print(f"Mesh: {len(self.vertices):,} vertices, {len(self.faces):,} faces")

        # Validate topology before proceeding
        euler = validate_topology(
            self.vertices, self.faces, strict=self.config.strict_topology
        )
        boundary = igl.boundary_loop(self.faces)

        # Validate single boundary loop (no holes)
        n_loops, loops = count_boundary_loops(self.faces)
        if n_loops != 1:
            loop_sizes = sorted([len(loop) for loop in loops], reverse=True)
            msg = (
                f"Patch has {n_loops} boundary loops (expected 1 for disk topology).\n"
                f"  Loop sizes: {loop_sizes}\n"
                f"  This indicates the patch has holes or disconnected boundaries.\n"
                f"  Check the cut projection for gaps between cuts and medial wall."
            )
            if self.config.strict_topology:
                raise TopologyError(msg)
            else:
                print(f"WARNING: {msg}")

        if self.config.verbose:
            print(f"Euler characteristic: {euler}, Boundary vertices: {len(boundary)}")

        # Compute original 3D surface area for area-preserving scaling
        # IMPORTANT: Use reduced surface (self.vertices, self.faces) which has
        # cuts and medial wall already removed - this is the actual patch area
        self.orig_area = compute_3d_surface_area(self.vertices, self.faces)
        if self.orig_area <= 0:
            raise ValueError(
                f"Surface has non-positive area ({self.orig_area:.6f}). "
                "This indicates a degenerate mesh that cannot be flattened."
            )
        if self.config.verbose:
            print(f"Original 3D patch surface area: {self.orig_area:.2f}")

    def compute_kring_distances(
        self, cache_path: Optional[str] = None, tqdm_position: int = 0
    ) -> None:
        """Compute or load cached k-ring geodesic distances.

        Args:
            cache_path: Path to cache file. If None, no caching is performed.
            tqdm_position: Position of tqdm progress bar (for stacking bars in parallel execution).
        """
        if self.fiducial_vertices is None:
            raise RuntimeError("Must call load_data before compute_kring_distances")

        kring_config = self.config.kring

        if kring_config.n_neighbors_per_ring is None:
            # Use all neighbors
            if cache_path and os.path.exists(cache_path):
                if self.config.verbose:
                    print(f"Loading cached k-ring distances from {cache_path}...")
                cached = np.load(cache_path, allow_pickle=True)
                self.k_rings = list(cached["k_rings"])
                self.target_distances = list(cached["target_distances"])
            else:
                if self.config.verbose:
                    print(
                        f"Computing {kring_config.k_ring}-ring geodesic distances "
                        f"(all neighbors)..."
                    )
                self.k_rings, self.target_distances = compute_kring_geodesic_distances(
                    self.fiducial_vertices,
                    self.faces,
                    kring_config.k_ring,
                    tqdm_position=tqdm_position,
                )
                if cache_path:
                    if self.config.verbose:
                        print(f"Caching to {cache_path}...")
                    cache_dir = os.path.dirname(cache_path)
                    if cache_dir:
                        os.makedirs(cache_dir, exist_ok=True)
                    np.savez(
                        cache_path,
                        k_rings=np.array(self.k_rings, dtype=object),
                        target_distances=np.array(self.target_distances, dtype=object),
                    )
        else:
            # Angular sampling
            if cache_path and os.path.exists(cache_path):
                if self.config.verbose:
                    print(f"Loading cached k-ring distances from {cache_path}...")
                cached = np.load(cache_path, allow_pickle=True)
                self.k_rings = list(cached["k_rings"])
                self.target_distances = list(cached["target_distances"])
            else:
                if self.config.verbose:
                    print(
                        f"Computing {kring_config.k_ring}-ring geodesic distances "
                        f"with angular sampling ({kring_config.n_neighbors_per_ring}/ring)..."
                    )
                self.k_rings, self.target_distances = (
                    compute_kring_geodesic_distances_angular(
                        self.fiducial_vertices,
                        self.faces,
                        kring_config.k_ring,
                        n_samples_per_ring=kring_config.n_neighbors_per_ring,
                        tqdm_position=tqdm_position,
                    )
                )
                if cache_path:
                    if self.config.verbose:
                        print(f"Caching to {cache_path}...")
                    cache_dir = os.path.dirname(cache_path)
                    if cache_dir:
                        os.makedirs(cache_dir, exist_ok=True)
                    np.savez(
                        cache_path,
                        k_rings=np.array(self.k_rings, dtype=object),
                        target_distances=np.array(self.target_distances, dtype=object),
                    )

        # Compute and store average neighbors (used for FreeSurfer-style gradient normalization)
        total_neighbors = sum(len(n) for n in self.k_rings)
        self.avg_neighbors = total_neighbors / len(self.k_rings)

        if self.config.verbose:
            print(
                f"Distance constraints: {total_neighbors:,} edges, "
                f"avg {self.avg_neighbors:.1f} neighbors/vertex"
            )

    def prepare_optimization(self) -> None:
        """Prepare JAX arrays and JIT-compiled functions.

        Must be called after load_data and compute_kring_distances.
        """
        if self.k_rings is None:
            raise RuntimeError(
                "Must call compute_kring_distances before prepare_optimization"
            )

        # Prepare metric data
        neighbors, targets, mask = prepare_metric_data(
            self.k_rings, self.target_distances
        )
        self.neighbors_jax = jnp.asarray(neighbors)
        self.targets_jax = jnp.asarray(targets)
        self.mask_jax = jnp.asarray(mask)

        # Prepare smoothing data
        if self.config.verbose:
            print("Preparing smoothing data...")
        smooth_neighbors, smooth_mask, smooth_counts = prepare_smoothing_data(
            self.faces, len(self.vertices)
        )
        self.smooth_neighbors_jax = jnp.asarray(smooth_neighbors)
        self.smooth_mask_jax = jnp.asarray(smooth_mask)
        self.smooth_counts_jax = jnp.asarray(smooth_counts.astype(np.float32))

        if self.config.verbose:
            print(f"Max vertex degree: {smooth_mask.shape[1]}")

        # Faces
        self.faces_jax = jnp.asarray(self.faces)

        # Boundary mask for spring smoothing
        boundary = igl.boundary_loop(self.faces)
        is_boundary = np.zeros(len(self.vertices), dtype=bool)
        is_boundary[boundary] = True
        self.is_boundary_jax = jnp.asarray(is_boundary)

        if self.config.verbose:
            print(f"Boundary vertices: {len(boundary)}")

        # Initialize energy functions
        if self.config.verbose:
            print("Initializing gradient functions...")
        self._compute_energies, self._grad_J_d, self._grad_J_a = make_energy_functions(
            self.neighbors_jax,
            self.targets_jax,
            self.mask_jax,
            self.faces_jax,
        )

    def initial_projection(self) -> np.ndarray:
        """Compute initial 2D projection with FreeSurfer-style scaling.

        Applies a global scale factor after projection to compensate for
        projection-induced shrinkage. FreeSurfer default is 3.0.

        Returns:
            (V, 2) initial UV coordinates
        """
        uv = freesurfer_projection(self.vertices, self.faces)

        # Apply initial global scale to compensate for projection shrinkage
        # (FreeSurfer default: 3.0, see mris_flatten.c:509)
        scale = self.config.initial_scale
        if scale != 1.0:
            centroid = np.mean(uv, axis=0)
            uv = (uv - centroid) * scale + centroid

        return uv

    def run(self) -> np.ndarray:
        """Run complete optimization pipeline.

        Returns:
            Optimized (V, 2) UV coordinates
        """
        if self._compute_energies is None:
            raise RuntimeError("Must call prepare_optimization before run")

        # Track total elapsed time
        total_start_time = time.time()

        config = self.config
        verbose = config.verbose

        if verbose:
            print("\n" + "=" * 85)
            print("FREESURFER-STYLE OPTIMIZATION (Vectorized Quadratic Line Search)")
            print("=" * 85)

        # Initial projection
        uv = self.initial_projection()
        uv_jax = jnp.asarray(uv)

        n_flipped_init = int(count_flipped_triangles(uv_jax, self.faces_jax))
        if verbose:
            print(f"Initial projection: {n_flipped_init} flipped triangles")

        J_d_init, J_a_init = self._compute_energies(uv_jax)
        if verbose:
            print(
                f"Initial energies: J_d={float(J_d_init):.4f}, "
                f"J_a={float(J_a_init):.4f}"
            )
            print(f"Raw ratio J_d/J_a = {float(J_d_init) / float(J_a_init):.2e}")

        # Phase 0: Negative area removal
        if config.negative_area_removal.enabled:
            # Only apply area-preserving scaling if enabled in config
            # FreeSurfer has this step commented out, so it's disabled by default
            scale_orig_area = (
                self.orig_area if config.negative_area_removal.scale_area else None
            )
            uv = remove_negative_area(
                uv,
                self.neighbors_jax,
                self.targets_jax,
                self.mask_jax,
                self.faces_jax,
                self.smooth_neighbors_jax,
                self.smooth_mask_jax,
                self.smooth_counts_jax,
                self._compute_energies,
                self._grad_J_d,
                self._grad_J_a,
                self.avg_neighbors,
                config.negative_area_removal,
                convergence_max_small=config.convergence.max_small,
                convergence_total_small=config.convergence.total_small,
                n_coarse_steps=config.line_search.n_coarse_steps,
                print_every=config.print_every,
                verbose=verbose,
                orig_area=scale_orig_area,
            )

        # Main optimization phases
        for phase_idx, phase in enumerate(config.phases):
            if not phase.enabled:
                continue

            if verbose:
                print("\n" + "=" * 85)
                print(
                    f"MAIN OPTIMIZATION: Phase {phase_idx + 1} - {phase.name} "
                    f"(l_nlarea={phase.l_nlarea}, l_dist={phase.l_dist})"
                )
                print("=" * 85)

            J_d_curr, J_a_curr = self._compute_energies(jnp.asarray(uv))
            # Use fixed FreeSurfer-style weights directly
            lambda_d = phase.l_dist
            lambda_a = phase.l_nlarea

            if verbose:
                print(f"  J_d={float(J_d_curr):.4f}, J_a={float(J_a_curr):.6f}")
                print(f"  l_dist={lambda_d:.2e}, l_nlarea={lambda_a:.2e}")

            base_tol = (
                phase.base_tol
                if phase.base_tol is not None
                else config.convergence.base_tol
            )

            # Use adaptive optimization for the final epoch if enabled
            # (historically named "distance_refinement", now "epoch_3" by default)
            use_adaptive = config.adaptive_recovery and phase.name == "epoch_3"

            if use_adaptive:
                uv = run_adaptive_optimization(
                    uv,
                    lambda_d=lambda_d,
                    lambda_a=lambda_a,
                    smoothing_schedule=phase.smoothing_schedule,
                    neighbors_jax=self.neighbors_jax,
                    targets_jax=self.targets_jax,
                    mask_jax=self.mask_jax,
                    faces_jax=self.faces_jax,
                    smooth_neighbors_jax=self.smooth_neighbors_jax,
                    smooth_mask_jax=self.smooth_mask_jax,
                    smooth_counts_jax=self.smooth_counts_jax,
                    compute_energies_fn=self._compute_energies,
                    avg_nbrs=self.avg_neighbors,
                    iters_per_level=phase.iters_per_level,
                    print_every=config.print_every,
                    verbose=verbose,
                    base_tol=base_tol,
                    max_small=config.convergence.max_small,
                    total_small_limit=config.convergence.total_small,
                    n_coarse_steps=config.line_search.n_coarse_steps,
                )
            else:
                uv = run_smoothed_optimization(
                    uv,
                    lambda_d=lambda_d,
                    lambda_a=lambda_a,
                    smoothing_schedule=phase.smoothing_schedule,
                    neighbors_jax=self.neighbors_jax,
                    targets_jax=self.targets_jax,
                    mask_jax=self.mask_jax,
                    faces_jax=self.faces_jax,
                    smooth_neighbors_jax=self.smooth_neighbors_jax,
                    smooth_mask_jax=self.smooth_mask_jax,
                    smooth_counts_jax=self.smooth_counts_jax,
                    avg_nbrs=self.avg_neighbors,
                    iters_per_level=phase.iters_per_level,
                    print_every=config.print_every,
                    verbose=verbose,
                    base_tol=base_tol,
                    max_small=config.convergence.max_small,
                    total_small_limit=config.convergence.total_small,
                    n_coarse_steps=config.line_search.n_coarse_steps,
                )

        # Final negative area removal (FreeSurfer step 3)
        if config.final_negative_area_removal.enabled:
            final_nar = config.final_negative_area_removal
            if verbose:
                print("\n" + "=" * 85)
                print(
                    "FINAL NEGATIVE AREA REMOVAL "
                    f"(l_nlarea={final_nar.l_nlarea}, "
                    f"l_dist_ratios={final_nar.l_dist_ratios})"
                )
                print("=" * 85)

            # Create a NegativeAreaRemovalConfig using the full ratio schedule
            final_nar_config = NegativeAreaRemovalConfig(
                base_averages=final_nar.base_averages,
                min_area_pct=config.negative_area_removal.min_area_pct,
                l_nlarea=final_nar.l_nlarea,
                l_dist_ratios=final_nar.l_dist_ratios,
                iters_per_level=final_nar.iters_per_level,
                base_tol=final_nar.base_tol,
                enabled=True,
                scale_area=False,
            )

            uv = remove_negative_area(
                uv,
                neighbors_jax=self.neighbors_jax,
                targets_jax=self.targets_jax,
                mask_jax=self.mask_jax,
                faces_jax=self.faces_jax,
                smooth_neighbors_jax=self.smooth_neighbors_jax,
                smooth_mask_jax=self.smooth_mask_jax,
                smooth_counts_jax=self.smooth_counts_jax,
                compute_energies_fn=self._compute_energies,
                grad_J_d_fn=self._grad_J_d,
                grad_J_a_fn=self._grad_J_a,
                avg_nbrs=self.avg_neighbors,
                config=final_nar_config,
                convergence_max_small=config.convergence.max_small,
                convergence_total_small=config.convergence.total_small,
                n_coarse_steps=config.line_search.n_coarse_steps,
                print_every=config.print_every,
                verbose=verbose,
                orig_area=None,  # No area-preserving scaling for final NAR
            )

        # Final spring smoothing
        if config.spring_smoothing.enabled:
            uv = final_spring_smoothing(
                uv,
                faces_jax=self.faces_jax,
                smooth_neighbors_jax=self.smooth_neighbors_jax,
                smooth_mask_jax=self.smooth_mask_jax,
                smooth_counts_jax=self.smooth_counts_jax,
                is_boundary_jax=self.is_boundary_jax,
                neighbors_jax=self.neighbors_jax,
                targets_jax=self.targets_jax,
                mask_jax=self.mask_jax,
                config=config.spring_smoothing,
                verbose=verbose,
            )

        # Final stats
        uv_jax = jnp.asarray(uv)
        n_flipped_final = int(count_flipped_triangles(uv_jax, self.faces_jax))
        mean_pct_error = float(
            _compute_distance_error_jit(
                uv_jax, self.neighbors_jax, self.targets_jax, self.mask_jax
            )
        )

        # Calculate total elapsed time
        total_elapsed = time.time() - total_start_time

        if verbose:
            print(f"\n{'=' * 85}")
            print("FINAL RESULT")
            print(f"{'=' * 85}")
            print(f"Flipped triangles: {n_flipped_init} -> {n_flipped_final}")
            print(f"Mean % distance error: {mean_pct_error:.2f}%")
            # Format elapsed time
            hours = int(total_elapsed // 3600)
            minutes = int((total_elapsed % 3600) // 60)
            seconds = total_elapsed % 60
            if hours > 0:
                print(f"Total elapsed time: {hours}h {minutes}m {seconds:.1f}s")
            elif minutes > 0:
                print(f"Total elapsed time: {minutes}m {seconds:.1f}s")
            else:
                print(f"Total elapsed time: {seconds:.1f}s")

        return uv

    def save_result(self, uv: np.ndarray, output_path: str) -> None:
        """Save flattened patch to FreeSurfer format.

        Args:
            uv: (V, 2) UV coordinates
            output_path: Path to output patch file
        """
        if self.orig_indices is None:
            raise RuntimeError("Must call load_data before save_result")

        # Create output directory if needed
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Create 3D vertices with UV as XY and Z=0
        flat_vertices = np.column_stack([uv, np.zeros(len(uv))])

        # Write patch (preserve border vertex flags from input)
        write_patch(output_path, flat_vertices, self.orig_indices, self.is_border)

        if self.config.verbose:
            print(f"Saved flattened patch to {output_path}")

    def compute_distance_error(self, uv: np.ndarray) -> float:
        """Compute average % distance error for UV coordinates.

        Args:
            uv: (V, 2) UV coordinates

        Returns:
            Percentage distance error
        """
        uv_jax = jnp.asarray(uv)
        return float(
            _compute_distance_error_jit(
                uv_jax, self.neighbors_jax, self.targets_jax, self.mask_jax
            )
        )

    def count_flipped(self, uv: np.ndarray) -> int:
        """Count flipped triangles.

        Args:
            uv: (V, 2) UV coordinates

        Returns:
            Number of flipped triangles
        """
        uv_jax = jnp.asarray(uv)
        return int(count_flipped_triangles(uv_jax, self.faces_jax))
