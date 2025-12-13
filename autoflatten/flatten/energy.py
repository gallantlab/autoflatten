"""Energy functions for surface flattening optimization.

Implements the FreeSurfer energy functional with two components:
1. J_d: Metric distortion energy (preserve distances to neighbors)
2. J_a: Nonlinear area energy using sigmoid weighting (penalize flipped triangles)

The area energy uses FreeSurfer's sigmoid approach where:
- w_i = 1 / (1 + exp(k * ratio_i)) is high for flipped/small triangles
- Energy = sum(w_i * (area_i - original_area_i)^2)
"""

import jax
import jax.numpy as jnp
import numpy as np


def prepare_metric_data(k_rings, target_distances):
    """Prepare padded arrays for vectorized metric energy computation.

    Converts ragged k_rings and target_distances to rectangular arrays
    with masking. Call this once before optimization.

    Parameters
    ----------
    k_rings : list of ndarray
        k_rings[i] contains neighbor indices for vertex i
    target_distances : list of ndarray
        Target distances for each vertex's neighbors

    Returns
    -------
    neighbors_padded : ndarray of shape (V, max_neighbors)
        Int array of neighbor indices
    targets_padded : ndarray of shape (V, max_neighbors)
        Float array of target distances
    mask : ndarray of shape (V, max_neighbors)
        Bool array (True where valid)
    """
    n_vertices = len(k_rings)
    max_neighbors = max(len(kr) for kr in k_rings)

    # Initialize with zeros (index 0 is valid, will be masked)
    neighbors_padded = np.zeros((n_vertices, max_neighbors), dtype=np.int32)
    targets_padded = np.zeros((n_vertices, max_neighbors), dtype=np.float64)
    mask = np.zeros((n_vertices, max_neighbors), dtype=bool)

    for i, (kr, td) in enumerate(zip(k_rings, target_distances)):
        n = len(kr)
        neighbors_padded[i, :n] = kr
        targets_padded[i, :n] = td
        mask[i, :n] = True

    return neighbors_padded, targets_padded, mask


def prepare_edge_list(k_rings, target_distances):
    """Convert k-ring data to sorted edge list format for efficient computation.

    This format eliminates padding waste and enables cache-friendly memory access
    by sorting edges by source vertex.

    Parameters
    ----------
    k_rings : list of ndarray
        k_rings[i] contains neighbor indices for vertex i
    target_distances : list of ndarray
        Target distances for each vertex's neighbors

    Returns
    -------
    src : ndarray of shape (E,)
        Int array of source vertex indices (sorted)
    dst : ndarray of shape (E,)
        Int array of destination vertex indices
    targets : ndarray of shape (E,)
        Float array of target distances
    n_vertices : int
        Number of vertices
    """
    n_vertices = len(k_rings)

    # Count total edges for pre-allocation
    n_edges = sum(len(kr) for kr in k_rings)

    # Pre-allocate arrays
    src = np.empty(n_edges, dtype=np.int32)
    dst = np.empty(n_edges, dtype=np.int32)
    targets = np.empty(n_edges, dtype=np.float64)

    # Fill arrays (already sorted by src since we iterate in order)
    idx = 0
    for i, (neighbors, dists) in enumerate(zip(k_rings, target_distances)):
        n = len(neighbors)
        src[idx : idx + n] = i
        dst[idx : idx + n] = neighbors
        targets[idx : idx + n] = dists
        idx += n

    return src, dst, targets, n_vertices


@jax.jit
def compute_metric_energy_edges(uv, src, dst, targets, n_vertices):
    """Compute metric distortion energy using edge list format.

    This is more memory-efficient than the padded format:
    - No wasted computation on padding
    - src gather is sequential (cache-friendly)
    - All edges are independent (better parallelism)

    Returns raw sum without normalization (FreeSurfer-style).

    Parameters
    ----------
    uv : ndarray of shape (V, 2)
        Current 2D vertex positions
    src : ndarray of shape (E,)
        Source vertex indices
    dst : ndarray of shape (E,)
        Destination vertex indices
    targets : ndarray of shape (E,)
        Target distances
    n_vertices : int
        Number of vertices (kept for API compatibility, no longer used)

    Returns
    -------
    float
        Scalar energy value (raw sum)
    """
    # Gather source and destination positions
    # src is sorted, so uv[src] has good cache locality
    uv_src = uv[src]  # (E, 2)
    uv_dst = uv[dst]  # (E, 2)

    # Compute distances
    diff = uv_dst - uv_src
    current_dists = jnp.sqrt(jnp.sum(diff**2, axis=1) + 1e-12)

    # Squared errors (no masking needed - all edges are valid)
    errors = (current_dists - targets) ** 2

    # Sum without normalization (FreeSurfer-style raw sum)
    return jnp.sum(errors)


@jax.jit
def compute_both_energies_edges(
    uv, src, dst, edge_targets, n_vertices, faces, original_areas
):
    """Compute both energy components using edge list format.

    Parameters
    ----------
    uv : ndarray of shape (V, 2)
        Current 2D vertex positions
    src : ndarray of shape (E,)
        Source vertex indices
    dst : ndarray of shape (E,)
        Destination vertex indices
    edge_targets : ndarray of shape (E,)
        Target distances
    n_vertices : int
        Number of vertices
    faces : ndarray of shape (T, 3)
        Triangle indices
    original_areas : ndarray of shape (T,)
        Original 3D triangle areas

    Returns
    -------
    J_d : float
        Metric distortion energy
    J_a : float
        Area energy
    """
    J_d = compute_metric_energy_edges(uv, src, dst, edge_targets, n_vertices)
    J_a = compute_area_energy(uv, faces, original_areas)
    return J_d, J_a


@jax.jit
def compute_metric_energy(uv, neighbors, targets, mask):
    """Compute metric distortion energy J_d (vectorized, JIT-compiled).

    J_d = sum_i sum_{n in N(i)} (d_in^t - d_in^0)^2

    Note: Returns raw sum without normalization (FreeSurfer-style).

    Parameters
    ----------
    uv : ndarray of shape (V, 2)
        Current 2D vertex positions
    neighbors : ndarray of shape (V, max_neighbors)
        Neighbor indices (from prepare_metric_data)
    targets : ndarray of shape (V, max_neighbors)
        Target distances
    mask : ndarray of shape (V, max_neighbors)
        Bool array (True where valid)

    Returns
    -------
    float
        Scalar energy value
    """
    # Note: n_vertices was previously used for normalization but is no longer
    # needed with FreeSurfer-style raw sum. Line kept for clarity about what
    # changed from the original normalized version.

    # Get neighbor positions: (V, max_neighbors, 2)
    neighbor_pos = uv[neighbors]

    # Current vertex positions expanded: (V, 1, 2)
    vertex_pos = uv[:, None, :]

    # Compute distances: (V, max_neighbors)
    # Use safe norm to avoid NaN gradients when distance is 0
    diff = neighbor_pos - vertex_pos
    current_dists = jnp.sqrt(jnp.sum(diff**2, axis=2) + 1e-12)

    # Squared error with masking
    errors = jnp.where(mask, (current_dists - targets) ** 2, 0.0)

    # Sum without normalization (FreeSurfer-style raw sum)
    return jnp.sum(errors)


@jax.jit
def compute_area_energy(uv, faces, original_areas, neg_area_k=10.0):
    """Compute nonlinear area energy using FreeSurfer's sigmoid weighting.

    Uses sigmoid-weighted squared error to penalize flipped and shrunken triangles.
    The sigmoid weight is high for negative/small areas and low for positive areas,
    focusing the penalty on problematic triangles.

    J_a = (1/T) * sum_i w_i * (A_i - A_i^0)^2
    where w_i = 1 / (1 + exp(k * A_i / A_i^0))

    Parameters
    ----------
    uv : ndarray of shape (V, 2)
        Current 2D vertex positions
    faces : ndarray of shape (T, 3)
        Triangle indices
    original_areas : ndarray of shape (T,)
        Original 3D triangle areas (always positive)
    neg_area_k : float
        Steepness of sigmoid (default 10.0, from FreeSurfer)

    Returns
    -------
    float
        Scalar energy value
    """
    n_triangles = len(faces)

    # Compute signed areas of triangles in 2D
    # Single gather operation instead of 3 separate ones
    tri_verts = uv[faces]  # (T, 3, 2)
    v0, v1, v2 = tri_verts[:, 0], tri_verts[:, 1], tri_verts[:, 2]

    # Signed area = 0.5 * ((v1-v0) x (v2-v0))
    signed_areas = 0.5 * (
        (v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1])
        - (v2[:, 0] - v0[:, 0]) * (v1[:, 1] - v0[:, 1])
    )

    # Compute ratio and clamp to prevent numerical issues
    ratio = signed_areas / original_areas
    max_neg_ratio = 400.0 / neg_area_k  # = 40 with default k=10
    ratio = jnp.clip(ratio, -max_neg_ratio, max_neg_ratio)

    # Sigmoid weight: high for negative ratios, low for positive
    # When ratio < 0 (flipped): weight approaches 1
    # When ratio > 0 (good): weight approaches 0
    # Use jax.nn.sigmoid for numerical stability in gradients
    # Note: 1/(1+exp(x)) = sigmoid(-x)
    nlweight = jax.nn.sigmoid(-neg_area_k * ratio)

    # Squared area difference
    delta = signed_areas - original_areas

    # Normalize by T
    return jnp.sum(nlweight * delta * delta) / n_triangles


@jax.jit
def compute_area_energy_fs_v6(uv, faces, neg_area_k=10.0):
    """FreeSurfer v6.0.0 style log-softplus area energy.

    Unlike compute_area_energy(), this uses the exact FreeSurfer formula:
    - Uses raw signed area (not area/orig_area ratio)
    - Log-softplus penalty: log(1 + exp(k*A))/k - A
    - Approaches 0 for positive areas, grows linearly for negative

    Reference: mrisurf.c:10213-10273 (mrisComputeNonlinearAreaSSE)

    Parameters
    ----------
    uv : ndarray of shape (V, 2)
        Current 2D vertex positions
    faces : ndarray of shape (T, 3)
        Triangle indices
    neg_area_k : float
        Steepness of sigmoid (default 10.0, from FreeSurfer NEG_AREA_K)

    Returns
    -------
    float
        Scalar energy value (sum over all triangles, no normalization)
    """
    # Compute signed areas of triangles in 2D
    tri_verts = uv[faces]  # (T, 3, 2)
    v0, v1, v2 = tri_verts[:, 0], tri_verts[:, 1], tri_verts[:, 2]

    # Signed area = 0.5 * ((v1-v0) x (v2-v0))
    signed_areas = 0.5 * (
        (v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1])
        - (v2[:, 0] - v0[:, 0]) * (v1[:, 1] - v0[:, 1])
    )

    # FreeSurfer uses raw signed area as ratio (for patches, area_scale=1.0)
    ratio = signed_areas

    # Clamp to prevent numerical issues (MAX_NEG_RATIO = 400/k = 40 with k=10)
    max_neg_ratio = 400.0 / neg_area_k
    ratio = jnp.clip(ratio, -max_neg_ratio, max_neg_ratio)

    # Log-softplus penalty: log(1 + exp(k*x))/k - x
    # jax.nn.softplus(x) = log(1 + exp(x)) is numerically stable
    error = jax.nn.softplus(neg_area_k * ratio) / neg_area_k - ratio

    # FreeSurfer doesn't normalize by number of triangles
    return jnp.sum(error)


@jax.jit
def compute_log_barrier_area_energy(
    uv, faces, original_area_fracs, scale=10.0, barrier_weight=0.1
):
    """Hybrid barrier: FreeSurfer soft penalty + inverse barrier near zero.

    Combines two components:
    1. FreeSurfer's softplus penalty (linear for negative areas)
    2. Inverse barrier 1/A that provides strong repulsion as A -> 0

    Uses PROPORTIONAL normalization: ratio = (2D_area/total_2D) / (3D_area/total_3D)

    Parameters
    ----------
    uv : ndarray of shape (V, 2)
        Vertex positions
    faces : ndarray of shape (T, 3)
        Triangle indices
    original_area_fracs : ndarray of shape (T,)
        Original 3D triangle area FRACTIONS (each triangle's area / total 3D area)
    scale : float
        Steepness for FreeSurfer penalty (default 10.0)
    barrier_weight : float
        Weight for inverse barrier component (default 0.1)

    Returns
    -------
    float
        Scalar barrier energy (normalized by number of triangles)
    """
    n_triangles = len(faces)

    # Compute signed areas of triangles in 2D
    tri_verts = uv[faces]  # (T, 3, 2)
    v0, v1, v2 = tri_verts[:, 0], tri_verts[:, 1], tri_verts[:, 2]

    # Signed area = 0.5 * ((v1-v0) x (v2-v0))
    signed_areas = 0.5 * (
        (v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1])
        - (v2[:, 0] - v0[:, 0]) * (v1[:, 1] - v0[:, 1])
    )

    # Compute 2D area fractions (proportion of total 2D area)
    total_2d_area = jnp.sum(jnp.abs(signed_areas)) + 1e-12  # use abs for total
    area_fracs_2d = signed_areas / total_2d_area

    # Proportional ratio: (2D_frac) / (3D_frac)
    normalized_areas = area_fracs_2d / original_area_fracs

    # Component 1: FreeSurfer soft penalty (linear for negative areas)
    max_neg_ratio = 400.0 / scale
    ratio = jnp.clip(normalized_areas, -max_neg_ratio, max_neg_ratio)
    fs_penalty = jax.nn.softplus(scale * ratio) / scale - ratio

    # Component 2: Inverse barrier (1/A - 1, zero at A=1, +inf as A->0)
    min_normalized = 0.01
    safe_normalized = jnp.maximum(normalized_areas, min_normalized)

    # 1/A - 1: gives 0 at A=1 (original size), grows as A->0
    inverse_barrier = 1.0 / safe_normalized - 1.0
    inverse_barrier = jnp.maximum(inverse_barrier, 0.0)  # clamp to non-negative

    # Combined: FreeSurfer penalty + weighted inverse barrier
    total_error = fs_penalty + barrier_weight * inverse_barrier

    return jnp.sum(total_error) / n_triangles


# =============================================================================
# Surface Area Computation Functions
# =============================================================================


@jax.jit
def compute_3d_surface_area_jax(
    vertices: jnp.ndarray, faces: jnp.ndarray
) -> jnp.ndarray:
    """Compute total surface area of 3D mesh (JIT-compiled).

    Parameters
    ----------
    vertices : ndarray of shape (V, 3)
        3D vertex coordinates
    faces : ndarray of shape (F, 3)
        Triangle face indices

    Returns
    -------
    float
        Total surface area
    """
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    cross = jnp.cross(v1 - v0, v2 - v0)
    areas = 0.5 * jnp.linalg.norm(cross, axis=1)
    return jnp.sum(areas)


def compute_3d_surface_area(vertices: np.ndarray, faces: np.ndarray) -> float:
    """Compute total surface area of 3D mesh.

    Wrapper that calls JIT-compiled version and returns Python float.

    Parameters
    ----------
    vertices : ndarray of shape (V, 3)
        3D vertex coordinates
    faces : ndarray of shape (F, 3)
        Triangle face indices

    Returns
    -------
    float
        Total surface area
    """
    return float(compute_3d_surface_area_jax(jnp.asarray(vertices), jnp.asarray(faces)))


@jax.jit
def compute_2d_areas(uv: jnp.ndarray, faces: jnp.ndarray) -> tuple:
    """Compute 2D triangle areas for area-preserving scaling (JIT-compiled).

    This computes the values needed for FreeSurfer's area-preserving scaling
    formula: scale = sqrt(orig_area / (total_area + neg_area))

    Parameters
    ----------
    uv : ndarray of shape (V, 2)
        2D vertex positions
    faces : ndarray of shape (F, 3)
        Triangle face indices

    Returns
    -------
    total_area : float
        Sum of signed areas (can be < sum of |areas| if triangles are flipped)
    neg_area : float
        Sum of |signed_area| for flipped triangles only

    Note
    ----
    FreeSurfer uses total_area + neg_area = sum of |all areas|, which equals
    the sum of positive areas plus the absolute value of negative areas.
    """
    v0 = uv[faces[:, 0]]
    v1 = uv[faces[:, 1]]
    v2 = uv[faces[:, 2]]

    # Signed area: positive = CCW (correct), negative = CW (flipped)
    signed_areas = 0.5 * (
        (v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1])
        - (v2[:, 0] - v0[:, 0]) * (v1[:, 1] - v0[:, 1])
    )

    total_area = jnp.sum(signed_areas)  # sum of signed areas
    neg_area = jnp.sum(jnp.abs(jnp.minimum(signed_areas, 0.0)))  # |negative areas|

    return total_area, neg_area


@jax.jit
def compute_spring_energy(uv, neighbors, mask, counts):
    """Spring energy: pull vertices toward 1-ring centroid.

    E_spring = sum_i ||v_i - centroid(neighbors_i)||^2

    This creates a Laplacian smoothing effect that regularizes
    triangle shapes without explicitly preserving distances.

    Parameters
    ----------
    uv : ndarray of shape (V, 2)
        Vertex positions
    neighbors : ndarray of shape (V, max_degree)
        1-ring neighbor indices (from prepare_smoothing_data)
    mask : ndarray of shape (V, max_degree)
        Validity mask
    counts : ndarray of shape (V,)
        Number of neighbors + 1 (for normalization in smoothing)

    Returns
    -------
    float
        Scalar spring energy
    """
    neighbor_pos = uv[neighbors]
    masked_pos = jnp.where(mask[:, :, None], neighbor_pos, 0.0)

    # Centroid of neighbors (counts includes self, so use counts-1 for pure neighbors)
    n_neighbors = counts - 1
    centroids = jnp.sum(masked_pos, axis=1) / jnp.maximum(n_neighbors[:, None], 1)

    # Spring energy: distance from vertex to neighbor centroid
    delta = uv - centroids
    return jnp.sum(delta**2)


@jax.jit
def get_vertices_with_negative_area(uv, faces):
    """Get mask of vertices involved in flipped (negative area) triangles.

    This is used for FreeSurfer's boundary handling: v->border && !v->neg
    Only skip boundary vertices that are NOT involved in flipped triangles.

    Parameters
    ----------
    uv : ndarray of shape (V, 2)
        Vertex positions
    faces : ndarray of shape (F, 3)
        Face indices

    Returns
    -------
    ndarray of shape (V,)
        Boolean mask where True = vertex is in a flipped triangle
    """
    v0 = uv[faces[:, 0]]
    v1 = uv[faces[:, 1]]
    v2 = uv[faces[:, 2]]

    # Signed area (positive = CCW, negative = flipped)
    signed_areas = 0.5 * (
        (v1[:, 0] - v0[:, 0]) * (v2[:, 1] - v0[:, 1])
        - (v2[:, 0] - v0[:, 0]) * (v1[:, 1] - v0[:, 1])
    )

    # Find faces with negative area (as float for scatter_add)
    is_flipped = (signed_areas < 0).astype(jnp.float32)

    # Mark all vertices of flipped faces using scatter_add
    n_vertices = uv.shape[0]
    has_neg_count = jnp.zeros(n_vertices, dtype=jnp.float32)

    # Add flipped flag to each vertex of each face
    has_neg_count = has_neg_count.at[faces[:, 0]].add(is_flipped)
    has_neg_count = has_neg_count.at[faces[:, 1]].add(is_flipped)
    has_neg_count = has_neg_count.at[faces[:, 2]].add(is_flipped)

    # Convert to boolean: any count > 0 means vertex is in a flipped face
    return has_neg_count > 0


@jax.jit
def compute_spring_displacement(uv, neighbors, mask, counts, skip_mask=None):
    """Compute spring displacement toward 1-ring centroid (FreeSurfer-style).

    This computes the direct displacement vector, NOT an energy gradient.
    FreeSurfer's mrisComputeSpringTerm computes:
        displacement = (centroid - vertex) / n_neighbors

    Parameters
    ----------
    uv : ndarray of shape (V, 2)
        Vertex positions
    neighbors : ndarray of shape (V, max_degree)
        1-ring neighbor indices
    mask : ndarray of shape (V, max_degree)
        Validity mask
    counts : ndarray of shape (V,)
        Number of neighbors + 1
    skip_mask : ndarray of shape (V,), optional
        Boolean mask for vertices to skip.
        If provided, vertices with True get zero displacement.

    Returns
    -------
    ndarray of shape (V, 2)
        Displacement vectors pointing toward neighbor centroids
    """
    neighbor_pos = uv[neighbors]
    masked_pos = jnp.where(mask[:, :, None], neighbor_pos, 0.0)

    # Centroid of neighbors
    n_neighbors = counts - 1
    centroids = jnp.sum(masked_pos, axis=1) / jnp.maximum(n_neighbors[:, None], 1)

    # Displacement toward centroid (FreeSurfer direction)
    displacement = centroids - uv

    # Zero out skipped vertices
    if skip_mask is not None:
        displacement = jnp.where(skip_mask[:, None], 0.0, displacement)

    return displacement


@jax.jit
def compute_both_energies(uv, neighbors, targets, mask, faces, original_areas):
    """Compute both energy components in a single JIT-compiled call.

    This is more efficient than calling compute_metric_energy and
    compute_area_energy separately when both values are needed.

    Parameters
    ----------
    uv : ndarray of shape (V, 2)
        Current 2D vertex positions
    neighbors : ndarray of shape (V, max_neighbors)
        Padded neighbor indices
    targets : ndarray of shape (V, max_neighbors)
        Padded target distances
    mask : ndarray of shape (V, max_neighbors)
        Validity mask
    faces : ndarray of shape (T, 3)
        Triangle indices
    original_areas : ndarray of shape (T,)
        Original 3D triangle areas

    Returns
    -------
    J_d : float
        Metric distortion energy
    J_a : float
        Area energy
    """
    J_d = compute_metric_energy(uv, neighbors, targets, mask)
    J_a = compute_area_energy(uv, faces, original_areas)
    return J_d, J_a


def compute_total_energy(
    uv, neighbors, targets, mask, faces, original_areas, lambda_d=1.0, lambda_a=1.0
):
    """Compute total energy J = lambda_d * J_d + lambda_a * J_a.

    Parameters
    ----------
    uv : ndarray of shape (V, 2)
        Current 2D vertex positions
    neighbors : ndarray of shape (V, max_neighbors)
        Padded neighbor indices
    targets : ndarray of shape (V, max_neighbors)
        Padded target distances
    mask : ndarray of shape (V, max_neighbors)
        Validity mask
    faces : ndarray of shape (T, 3)
        Triangle indices
    original_areas : ndarray of shape (T,)
        Original 3D triangle areas
    lambda_d : float
        Weight for metric distortion energy
    lambda_a : float
        Weight for negative area energy

    Returns
    -------
    total_energy : float
        Weighted sum of energies
    J_d : float
        Metric distortion energy
    J_a : float
        Area energy
    """
    J_d, J_a = compute_both_energies(
        uv, neighbors, targets, mask, faces, original_areas
    )
    total = lambda_d * J_d + lambda_a * J_a
    return total, J_d, J_a


# =============================================================================
# Gradient Smoothing (FreeSurfer-style)
# =============================================================================


def prepare_smoothing_data(faces, n_vertices):
    """Build padded neighbor arrays for gradient smoothing.

    Constructs 1-ring adjacency from face connectivity and pads to rectangular
    arrays for efficient JAX operations.

    Parameters
    ----------
    faces : ndarray of shape (F, 3)
        Triangle indices
    n_vertices : int
        Number of vertices in the mesh

    Returns
    -------
    neighbors : ndarray of shape (V, max_degree)
        Int array of neighbor indices
    mask : ndarray of shape (V, max_degree)
        Bool mask for valid neighbors
    counts : ndarray of shape (V,)
        Normalization factors (degree + 1 for self)
    """
    from collections import defaultdict

    # Build adjacency list from faces
    adj = defaultdict(set)
    for f in faces:
        for i in range(3):
            adj[f[i]].add(f[(i + 1) % 3])
            adj[f[i]].add(f[(i + 2) % 3])

    # Find max degree for padding
    max_degree = max(len(adj[i]) for i in range(n_vertices))

    # Build padded arrays
    neighbors = np.zeros((n_vertices, max_degree), dtype=np.int32)
    mask = np.zeros((n_vertices, max_degree), dtype=bool)

    for i in range(n_vertices):
        nbrs = list(adj[i])
        n = len(nbrs)
        neighbors[i, :n] = nbrs
        mask[i, :n] = True

    # Counts include self: (degree + 1)
    counts = np.sum(mask, axis=1) + 1

    return neighbors, mask, counts


@jax.jit
def smooth_gradient_once(grad, neighbors, mask, counts):
    """One iteration of FreeSurfer-style gradient smoothing.

    Implements uniform neighbor averaging (Jacobi iteration):
        new_grad[i] = (grad[i] + sum(grad[neighbors])) / (1 + n_neighbors)

    Parameters
    ----------
    grad : ndarray of shape (V, 2)
        Gradient at each vertex
    neighbors : ndarray of shape (V, max_degree)
        Padded neighbor indices
    mask : ndarray of shape (V, max_degree)
        Validity mask
    counts : ndarray of shape (V,)
        Normalization factors

    Returns
    -------
    ndarray of shape (V, 2)
        Smoothed gradient
    """
    # Gather neighbor gradients: (V, max_degree, 2)
    neighbor_grads = grad[neighbors]

    # Zero out invalid neighbors
    masked_grads = jnp.where(mask[:, :, None], neighbor_grads, 0.0)

    # Sum of neighbor gradients: (V, 2)
    neighbor_sum = jnp.sum(masked_grads, axis=1)

    # Average: (self + neighbors) / (1 + n_neighbors)
    smoothed = (grad + neighbor_sum) / counts[:, None]

    return smoothed


def _make_smooth_gradient_n(n_iters):
    """Create a JIT-compiled function for n iterations of smoothing.

    Using a factory function ensures each n_iters value gets its own
    compiled function, avoiding recompilation overhead.
    """

    @jax.jit
    def smooth_n(grad, neighbors, mask, counts):
        def body_fn(_, g):
            return smooth_gradient_once(g, neighbors, mask, counts)

        return jax.lax.fori_loop(0, n_iters, body_fn, grad)

    return smooth_n


# Cache of compiled smoothing functions for common iteration counts
_smooth_gradient_cache = {}


def smooth_gradient(grad, neighbors, mask, counts, n_iters):
    """Apply n iterations of gradient smoothing (cached JIT compilation).

    Uses FreeSurfer-style uniform neighbor averaging. The effective smoothing
    radius grows as sqrt(n_iters) in terms of vertex hops.

    Typical FreeSurfer schedule: start with n_iters=1024, divide by 4 as
    optimization progresses (1024 -> 256 -> 64 -> 16 -> 4 -> 1 -> 0).

    Parameters
    ----------
    grad : ndarray of shape (V, 2)
        Gradient at each vertex (JAX array)
    neighbors : ndarray of shape (V, max_degree)
        Padded neighbor indices (JAX array)
    mask : ndarray of shape (V, max_degree)
        Validity mask (JAX array)
    counts : ndarray of shape (V,)
        Normalization factors (JAX array)
    n_iters : int
        Number of smoothing iterations

    Returns
    -------
    ndarray of shape (V, 2)
        Smoothed gradient
    """
    if n_iters <= 0:
        return grad

    # Get or create cached compiled function
    if n_iters not in _smooth_gradient_cache:
        _smooth_gradient_cache[n_iters] = _make_smooth_gradient_n(n_iters)

    return _smooth_gradient_cache[n_iters](grad, neighbors, mask, counts)
