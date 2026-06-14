"""Distance computation functions for surface meshes.

Computes geodesic distances with graph-based Dijkstra, which is fast and
accurate for local k-ring distances.

Includes Numba-accelerated implementations for significant speedups:
- K-ring computation: ~20x faster with parallel Numba
- Dijkstra: ~8x faster with Numba heap implementation

Thread control:
- Set NUMBA_NUM_THREADS environment variable before import, or
- Use numba.set_num_threads(n) at runtime
"""

import heapq

import igl
import numba
import numpy as np
from numba import njit, prange
from scipy import sparse
from tqdm import tqdm


# Correction factor for graph distances on triangulated surfaces (from FreeSurfer)
# Graph distances underestimate true geodesic distances; this corrects for that
GRAPH_DISTANCE_CORRECTION = (1 + np.sqrt(2)) / 2


# =============================================================================
# Graph-based Dijkstra (fast for local distances)
# =============================================================================


def build_mesh_graph(vertices, faces):
    """Build sparse adjacency matrix with edge lengths as weights.

    Parameters
    ----------
    vertices : ndarray of shape (N, 3)
        Vertex positions
    faces : ndarray of shape (F, 3)
        Face indices

    Returns
    -------
    sparse.csr_matrix
        (N, N) sparse matrix where entry (i,j) is the edge length
        between vertices i and j (0 if not connected)
    """
    edges = igl.edges(faces.astype(np.int64))
    n_vertices = len(vertices)

    # Compute edge lengths
    edge_lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)

    # Build symmetric sparse matrix
    row = np.concatenate([edges[:, 0], edges[:, 1]])
    col = np.concatenate([edges[:, 1], edges[:, 0]])
    data = np.concatenate([edge_lengths, edge_lengths])

    return sparse.csr_matrix((data, (row, col)), shape=(n_vertices, n_vertices))


def distance_optimal_scale(vertices, faces, uv, n_sources=200, seed=0, radius=None):
    """Global scale ``s*`` that makes a flat map metrically match true geodesic distances.

    Samples ``n_sources`` source vertices (deterministically), computes their true geodesic
    fields on the 3D patch with the heat method, and returns the single scale ``s`` that
    minimizes ``mean(|s*d_2d - d_geo| / d_geo)`` over all source->target pairs (optionally
    capped at ``radius`` mm). Used to replace the area-matched display scale with a
    distance-faithful one (the area-matched map is ~6% too small; ``s*`` ~ 1.06).

    Parameters
    ----------
    vertices : ndarray (V, 3)
        3D patch vertex positions (use the fiducial surface for anatomical distances).
    faces : ndarray (F, 3)
        Patch face indices.
    uv : ndarray (V, 2)
        2D flat-map coordinates (same vertex order as ``vertices``).
    n_sources : int
        Number of heat-geodesic sources to sample.
    seed : int
        RNG seed (keeps the result deterministic).
    radius : float or None
        If set, only score pairs within this geodesic distance (mm).

    Returns
    -------
    float
        Distance-optimal scale (1.0 if it cannot be computed).
    """
    v = np.ascontiguousarray(vertices, dtype=np.float64)
    f = np.ascontiguousarray(faces, dtype=np.int64)
    uv = np.ascontiguousarray(uv, dtype=np.float64)
    n_v = v.shape[0]
    if n_v == 0:
        return 1.0

    rng = np.random.default_rng(seed)
    srcs = np.sort(rng.choice(n_v, size=min(n_sources, n_v), replace=False))

    data = igl.HeatGeodesicsData()
    igl.heat_geodesics_precompute(v, f, data)

    d2_all, dg_all = [], []
    for s in srcs:
        geo = igl.heat_geodesics_solve(data, np.array([s], dtype=np.int64))
        mask = geo > 1e-6
        if radius is not None:
            mask &= geo <= radius
        if not np.any(mask):
            continue
        d2_all.append(np.linalg.norm(uv[mask] - uv[s], axis=1))
        dg_all.append(geo[mask])

    if not d2_all:
        return 1.0
    d2 = np.concatenate(d2_all)
    dg = np.concatenate(dg_all)
    scales = np.linspace(0.85, 1.20, 71)
    errs = np.array([np.mean(np.abs(sc * d2 - dg) / dg) for sc in scales])
    return float(scales[int(np.argmin(errs))])


def get_k_ring(faces, n_vertices, k):
    """Get k-ring neighbors for each vertex.

    Parameters
    ----------
    faces : ndarray of shape (F, 3)
        Face indices
    n_vertices : int
        Number of vertices
    k : int
        Number of rings to include

    Returns
    -------
    list of ndarray
        k_ring[i] contains indices of vertices within k edges of vertex i
    """
    # Build adjacency list (1-ring)
    adj = igl.adjacency_list(faces.astype(np.int64))

    # For each vertex, expand to k-ring using BFS
    k_rings = []
    for v in range(n_vertices):
        visited = {v}
        frontier = {v}
        for _ in range(k):
            new_frontier = set()
            for u in frontier:
                for neighbor in adj[u]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        new_frontier.add(neighbor)
            frontier = new_frontier
        # Exclude the vertex itself
        visited.discard(v)
        k_rings.append(np.array(sorted(visited), dtype=np.int64))

    return k_rings


@njit(parallel=True, cache=True)
def _get_k_rings_numba(adj_flat, adj_offsets, k, n_chunks):
    """Compute k-ring neighbors for all vertices in parallel using Numba.

    Parallelism is over **chunks**, not vertices: the vertices are split into ``n_chunks``
    contiguous blocks and the ``prange`` runs over chunks, so each iteration ``c`` owns its
    own scratch row (no races) and the scratch (visited / BFS levels / touched list) is
    allocated **once per chunk**, not per vertex. The previous version allocated three
    O(n_vertices) arrays *inside* a per-vertex ``prange`` and collected results with an
    O(n_vertices) scan per vertex; numba would not parallelize that (the per-iteration
    allocations defeat the parallel analysis), so a fresh compile ran serially and was
    O(n_vertices^2) -- ~16 min on a 193k-vertex mesh. This version is O(n_vertices * ring)
    and parallelizes. Output is identical: per-vertex neighbor indices sorted ascending,
    excluding the source.

    Parameters
    ----------
    adj_flat : ndarray
        Flattened adjacency list (concatenated neighbor arrays)
    adj_offsets : ndarray
        Offsets into adj_flat for each vertex (length n_vertices + 1)
    k : int
        Number of rings
    n_chunks : int
        Number of parallel chunks (typically the thread count).

    Returns
    -------
    k_rings_flat : ndarray
        Flattened k-ring results
    offsets : ndarray
        Offsets into k_rings_flat for each vertex
    """
    n_vertices = len(adj_offsets) - 1
    chunk_size = (n_vertices + n_chunks - 1) // n_chunks

    # per-chunk scratch, allocated ONCE (not per vertex)
    visited = np.zeros((n_chunks, n_vertices), dtype=np.bool_)
    cur = np.empty((n_chunks, n_vertices), dtype=np.int64)
    nxt = np.empty((n_chunks, n_vertices), dtype=np.int64)
    touched = np.empty((n_chunks, n_vertices), dtype=np.int64)

    sizes = np.zeros(n_vertices, dtype=np.int64)

    # First pass: compute sizes for each vertex
    for c in prange(n_chunks):
        vis = visited[c]
        tch = touched[c]
        v_start = c * chunk_size
        v_end = min(v_start + chunk_size, n_vertices)
        for v in range(v_start, v_end):
            cl = cur[c]
            nl = nxt[c]
            nt = 0
            vis[v] = True
            tch[nt] = v
            nt += 1
            cl[0] = v
            csz = 1
            for _ in range(k):
                nsz = 0
                for i in range(csz):
                    u = cl[i]
                    for j in range(adj_offsets[u], adj_offsets[u + 1]):
                        nb = adj_flat[j]
                        if not vis[nb]:
                            vis[nb] = True
                            tch[nt] = nb
                            nt += 1
                            nl[nsz] = nb
                            nsz += 1
                cl, nl = nl, cl
                csz = nsz
            sizes[v] = nt - 1  # exclude source
            for i in range(nt):
                vis[tch[i]] = False

    # Build offsets for flat output
    offsets = np.zeros(n_vertices + 1, dtype=np.int64)
    for i in range(n_vertices):
        offsets[i + 1] = offsets[i] + sizes[i]

    total_size = offsets[n_vertices]
    k_rings_flat = np.empty(total_size, dtype=np.int64)

    # Second pass: fill k-rings (collect from the touched list, exclude source, sort)
    for c in prange(n_chunks):
        vis = visited[c]
        tch = touched[c]
        v_start = c * chunk_size
        v_end = min(v_start + chunk_size, n_vertices)
        for v in range(v_start, v_end):
            cl = cur[c]
            nl = nxt[c]
            nt = 0
            vis[v] = True
            tch[nt] = v
            nt += 1
            cl[0] = v
            csz = 1
            for _ in range(k):
                nsz = 0
                for i in range(csz):
                    u = cl[i]
                    for j in range(adj_offsets[u], adj_offsets[u + 1]):
                        nb = adj_flat[j]
                        if not vis[nb]:
                            vis[nb] = True
                            tch[nt] = nb
                            nt += 1
                            nl[nsz] = nb
                            nsz += 1
                cl, nl = nl, cl
                csz = nsz
            out_start = offsets[v]
            idx = 0
            for i in range(nt):
                x = tch[i]
                if x != v:
                    k_rings_flat[out_start + idx] = x
                    idx += 1
            # sort ascending to match the reference (pure-Python) output order
            k_rings_flat[out_start : out_start + idx].sort()
            for i in range(nt):
                vis[tch[i]] = False

    return k_rings_flat, offsets


def get_k_ring_fast(faces, n_vertices, k):
    """Get k-ring neighbors for all vertices using Numba acceleration.

    This is ~20x faster than the pure Python version for large meshes.

    Parameters
    ----------
    faces : ndarray of shape (F, 3)
        Face indices
    n_vertices : int
        Number of vertices
    k : int
        Number of rings to include

    Returns
    -------
    list of ndarray
        k_ring[i] contains indices of vertices within k edges of vertex i
    """
    k_rings_flat, offsets = get_k_ring_fast_flat(faces, n_vertices, k)

    # Convert back to list of arrays
    k_rings = []
    for v in range(n_vertices):
        start = offsets[v]
        end = offsets[v + 1]
        k_rings.append(k_rings_flat[start:end])

    return k_rings


def get_k_ring_fast_flat(faces, n_vertices, k):
    """k-ring neighbors in flat (concatenated) form: ``(k_rings_flat, offsets)``.

    Same computation as :func:`get_k_ring_fast` but returns the flat arrays directly so
    callers that also compute per-vertex distances can avoid rebuilding the layout.
    """
    adj = igl.adjacency_list(faces.astype(np.int64))
    adj_flat = np.concatenate([np.array(a, dtype=np.int64) for a in adj])
    adj_offsets = np.zeros(n_vertices + 1, dtype=np.int64)
    for i, a in enumerate(adj):
        adj_offsets[i + 1] = adj_offsets[i] + len(a)

    # Compute k-rings in parallel (one scratch buffer per chunk)
    n_chunks = max(1, min(numba.get_num_threads(), max(1, n_vertices // 2000)))
    return _get_k_rings_numba(adj_flat, adj_offsets, k, n_chunks)


# =============================================================================
# Numba-accelerated Dijkstra (~8x faster)
# =============================================================================


@njit(cache=True)
def _limited_dijkstra_numba(indptr, indices, data, source, k_ring, correction):
    """Numba-accelerated limited Dijkstra with heap-like priority queue.

    Computes shortest path distances from source to k-ring neighbors only,
    with early termination once all targets are found.

    Parameters
    ----------
    indptr : ndarray
        CSR matrix indptr array
    indices : ndarray
        CSR matrix indices array
    data : ndarray
        CSR matrix data array (edge weights)
    source : int
        Source vertex index
    k_ring : ndarray
        Array of target vertex indices
    correction : float
        Correction factor to apply to distances

    Returns
    -------
    ndarray
        Distances to k_ring vertices (in same order as k_ring)
    """
    n_targets = len(k_ring)
    if n_targets == 0:
        return np.empty(0, dtype=np.float64)

    n_vertices = len(indptr) - 1

    # Initialize distances
    dist = np.full(n_vertices, np.inf, dtype=np.float64)
    dist[source] = 0.0
    visited = np.zeros(n_vertices, dtype=np.bool_)

    # Create target lookup
    is_target = np.zeros(n_vertices, dtype=np.bool_)
    for idx in k_ring:
        is_target[idx] = True

    found_count = 0

    # Priority queue as arrays (simple but effective for Numba)
    max_heap_size = n_vertices * 3
    heap_dists = np.empty(max_heap_size, dtype=np.float64)
    heap_verts = np.empty(max_heap_size, dtype=np.int64)
    heap_size = 1
    heap_dists[0] = 0.0
    heap_verts[0] = source

    while heap_size > 0 and found_count < n_targets:
        # Pop minimum (linear scan - fast enough for local neighborhoods)
        min_idx = 0
        min_dist = heap_dists[0]
        for i in range(1, heap_size):
            if heap_dists[i] < min_dist:
                min_dist = heap_dists[i]
                min_idx = i

        d = heap_dists[min_idx]
        u = heap_verts[min_idx]

        # Remove by swap with last element
        heap_size -= 1
        if min_idx < heap_size:
            heap_dists[min_idx] = heap_dists[heap_size]
            heap_verts[min_idx] = heap_verts[heap_size]

        if visited[u]:
            continue
        visited[u] = True

        if is_target[u]:
            found_count += 1

        # Relax edges
        for j in range(indptr[u], indptr[u + 1]):
            v = indices[j]
            w = data[j]
            if not visited[v]:
                new_dist = d + w
                if new_dist < dist[v]:
                    dist[v] = new_dist
                    # Add to heap (duplicates are filtered by visited check)
                    if heap_size < max_heap_size:
                        heap_dists[heap_size] = new_dist
                        heap_verts[heap_size] = v
                        heap_size += 1

    # Extract results in k_ring order with correction applied
    result = np.empty(n_targets, dtype=np.float64)
    for i, idx in enumerate(k_ring):
        result[i] = dist[idx] / correction

    return result


# =============================================================================
# Original Python Dijkstra (kept for reference/fallback)
# =============================================================================


def _limited_dijkstra(v, k_ring, graph, correction):
    """Compute graph-based distances from vertex v to its k-ring neighbors.

    Uses custom limited Dijkstra that stops once all k-ring neighbors are found.
    This is faster than scipy's dijkstra with limit for local neighborhoods.

    Parameters
    ----------
    v : int
        Source vertex index
    k_ring : ndarray
        Array of target vertex indices
    graph : sparse.csr_matrix
        Sparse CSR adjacency matrix with edge weights
    correction : float
        Correction factor to apply to graph distances

    Returns
    -------
    ndarray
        Distances to k_ring vertices (in same order as k_ring)
    """
    if len(k_ring) == 0:
        return np.array([])

    k_ring_set = set(k_ring)
    n_targets = len(k_ring_set)
    found = {}

    # Priority queue: (distance, vertex)
    pq = [(0.0, v)]
    visited = set()

    while pq and len(found) < n_targets:
        dist, u = heapq.heappop(pq)

        if u in visited:
            continue
        visited.add(u)

        if u in k_ring_set:
            found[u] = dist

        # Explore neighbors (graph is CSR format)
        neighbors = graph.indices[graph.indptr[u] : graph.indptr[u + 1]]
        weights = graph.data[graph.indptr[u] : graph.indptr[u + 1]]

        for neighbor, weight in zip(neighbors, weights):
            if neighbor not in visited:
                heapq.heappush(pq, (dist + weight, neighbor))

    # Return distances in same order as k_ring, with correction applied
    return np.array([found.get(idx, np.inf) / correction for idx in k_ring])


@njit(parallel=True, cache=True)
def _kring_distances_kernel(
    indptr, indices, data, rings_flat, offsets, correction, n_chunks
):
    """Parallel limited-Dijkstra k-ring distances, in the flat ``rings_flat`` layout.

    For every vertex, computes the corrected graph distance to each of its k-ring targets.
    Parallelizes over chunks: each chunk owns scratch (dist / visited / target / heap)
    allocated once and resets only the touched entries between vertices. A naive parallel
    port of ``_limited_dijkstra_numba`` allocated five O(n_vertices) arrays per call inside
    the prange, which melts the allocator across threads; this avoids that. Output matches
    the serial ``_limited_dijkstra_numba`` (same correction, same heap discipline).
    """
    nv = len(indptr) - 1
    n = offsets.shape[0] - 1
    INF = np.inf

    dist = np.full((n_chunks, nv), INF)
    visited = np.zeros((n_chunks, nv), dtype=np.bool_)
    is_target = np.zeros((n_chunks, nv), dtype=np.bool_)
    touched = np.empty((n_chunks, nv), dtype=np.int64)
    # Lazy-deletion binary heap: a vertex can be pushed multiple times before it is
    # popped/visited, so capacity must exceed nv. Match _limited_dijkstra_numba (3*nv).
    heap_cap = nv * 3
    heap_d = np.empty((n_chunks, heap_cap), dtype=np.float64)
    heap_v = np.empty((n_chunks, heap_cap), dtype=np.int64)
    out = np.empty(offsets[n], dtype=np.float64)

    chunk_size = (n + n_chunks - 1) // n_chunks

    for c in prange(n_chunks):
        d_t = dist[c]
        vis = visited[c]
        tgt = is_target[c]
        tch = touched[c]
        hd = heap_d[c]
        hv = heap_v[c]
        v_start = c * chunk_size
        v_end = min(v_start + chunk_size, n)

        for v in range(v_start, v_end):
            s = offsets[v]
            e = offsets[v + 1]
            m = e - s
            if m == 0:
                continue

            for j in range(m):
                tgt[rings_flat[s + j]] = True

            nt = 0
            d_t[v] = 0.0
            tch[nt] = v
            nt += 1
            hd[0] = 0.0
            hv[0] = v
            hsize = 1
            found = 0

            while hsize > 0 and found < m:
                mi = 0
                md = hd[0]
                for i in range(1, hsize):
                    if hd[i] < md:
                        md = hd[i]
                        mi = i
                du = hd[mi]
                u = hv[mi]
                hsize -= 1
                if mi < hsize:
                    hd[mi] = hd[hsize]
                    hv[mi] = hv[hsize]
                if vis[u]:
                    continue
                vis[u] = True
                if tgt[u]:
                    found += 1
                for p in range(indptr[u], indptr[u + 1]):
                    w = indices[p]
                    if not vis[w]:
                        ndist = du + data[p]
                        if ndist < d_t[w]:
                            if d_t[w] == INF:
                                tch[nt] = w
                                nt += 1
                            d_t[w] = ndist
                            if hsize < heap_cap:
                                hd[hsize] = ndist
                                hv[hsize] = w
                                hsize += 1

            for j in range(m):
                out[s + j] = d_t[rings_flat[s + j]] / correction

            for i in range(nt):
                x = tch[i]
                d_t[x] = INF
                vis[x] = False
            for j in range(m):
                tgt[rings_flat[s + j]] = False

    return out


def compute_kring_geodesic_distances(
    vertices, faces, k, correction=None, use_numba=True, n_threads=None, tqdm_position=0
):
    """Compute geodesic distances from each vertex to its k-ring neighbors.

    Uses graph-based Dijkstra which is fast for local distances and accurate
    for small k since the surface is locally flat.

    With Numba acceleration (default), this is ~50x faster than pure Python.

    Parameters
    ----------
    vertices : ndarray of shape (N, 3)
        Vertex positions
    faces : ndarray of shape (F, 3)
        Face indices
    k : int
        Number of rings to include
    correction : float, optional
        Correction factor for graph distances (uses FreeSurfer default if None)
    use_numba : bool
        If True (default), use Numba-accelerated implementations
    n_threads : int, optional
        Number of threads for parallel k-ring computation
    tqdm_position : int, optional
        Position of tqdm progress bar (for stacking bars in parallel execution)

    Returns
    -------
    k_rings : list of ndarray
        k_rings[i] contains indices of vertices in k-ring of vertex i
    distances : list of ndarray
        distances[i] contains geodesic distances from vertex i
        to each vertex in its k-ring (same order as k_rings[i])
    """
    n_vertices = len(vertices)

    if correction is None:
        correction = GRAPH_DISTANCE_CORRECTION

    # Set thread count for Numba if specified
    if n_threads is not None and use_numba:
        numba.set_num_threads(n_threads)

    # Build mesh graph
    graph = build_mesh_graph(vertices, faces)

    if use_numba:
        # Fully parallel path: build k-rings and distances in flat form with per-chunk
        # scratch, then reconstruct the per-vertex lists. ~Ncore faster than the previous
        # serial per-vertex Dijkstra loop.
        rings_flat, offsets = get_k_ring_fast_flat(faces, n_vertices, k)
        n_chunks = max(1, min(numba.get_num_threads(), max(1, n_vertices // 2000)))
        dist_flat = _kring_distances_kernel(
            graph.indptr,
            graph.indices,
            graph.data,
            rings_flat,
            offsets,
            correction,
            n_chunks,
        )
        k_rings = [rings_flat[offsets[v] : offsets[v + 1]] for v in range(n_vertices)]
        distances = [dist_flat[offsets[v] : offsets[v + 1]] for v in range(n_vertices)]
        return k_rings, distances

    # Pure-Python fallback
    k_rings = get_k_ring(faces, n_vertices, k)
    distances = [
        _limited_dijkstra(v, k_rings[v], graph, correction)
        for v in tqdm(
            range(n_vertices),
            desc="Computing k-ring distances",
            position=tqdm_position,
            leave=True,
        )
    ]
    return k_rings, distances


# =============================================================================
# Numba-accelerated rings by level (~20x faster)
# =============================================================================


@njit(parallel=True, cache=True)
def _get_rings_by_level_numba(adj_flat, adj_offsets, k):
    """Compute k-ring neighbors organized by level in parallel using Numba.

    Parameters
    ----------
    adj_flat : ndarray
        Flattened adjacency list (concatenated neighbor arrays)
    adj_offsets : ndarray
        Offsets into adj_flat for each vertex (length n_vertices + 1)
    k : int
        Number of rings

    Returns
    -------
    rings_flat : ndarray
        Flattened ring results (all levels concatenated)
    level_offsets : ndarray of shape (n_vertices, k+1)
        level_offsets[v, l] is the start offset for vertex v, level l in rings_flat
    """
    n_vertices = len(adj_offsets) - 1

    # First pass: compute sizes for each vertex at each level
    sizes = np.zeros((n_vertices, k), dtype=np.int64)

    for v in prange(n_vertices):
        visited = np.zeros(n_vertices, dtype=np.bool_)
        visited[v] = True

        current_level = np.empty(n_vertices, dtype=np.int64)
        next_level = np.empty(n_vertices, dtype=np.int64)
        current_size = 1
        current_level[0] = v

        for level in range(k):
            next_size = 0
            for i in range(current_size):
                u = current_level[i]
                start = adj_offsets[u]
                end = adj_offsets[u + 1]
                for j in range(start, end):
                    neighbor = adj_flat[j]
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        next_level[next_size] = neighbor
                        next_size += 1
            sizes[v, level] = next_size
            current_level, next_level = next_level, current_level
            current_size = next_size

    # Build offsets: level_offsets[v, l] = start of vertex v, level l
    level_offsets = np.zeros((n_vertices, k + 1), dtype=np.int64)

    # Compute total size and per-vertex offsets
    running_total = 0
    for v in range(n_vertices):
        level_offsets[v, 0] = running_total
        for level in range(k):
            level_offsets[v, level + 1] = level_offsets[v, level] + sizes[v, level]
        running_total = level_offsets[v, k]

    total_size = running_total
    rings_flat = np.empty(total_size, dtype=np.int64)

    # Second pass: fill rings
    for v in prange(n_vertices):
        visited = np.zeros(n_vertices, dtype=np.bool_)
        visited[v] = True

        current_level = np.empty(n_vertices, dtype=np.int64)
        next_level = np.empty(n_vertices, dtype=np.int64)
        current_size = 1
        current_level[0] = v

        for level in range(k):
            next_size = 0
            for i in range(current_size):
                u = current_level[i]
                start = adj_offsets[u]
                end = adj_offsets[u + 1]
                for j in range(start, end):
                    neighbor = adj_flat[j]
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        next_level[next_size] = neighbor
                        next_size += 1

            # Store this level's results
            out_start = level_offsets[v, level]
            for i in range(next_size):
                rings_flat[out_start + i] = next_level[i]

            current_level, next_level = next_level, current_level
            current_size = next_size

    return rings_flat, level_offsets


def get_rings_by_level_fast(faces, n_vertices, k):
    """Get neighbors organized by ring level using Numba acceleration.

    This is ~20x faster than the pure Python version for large meshes.

    Parameters
    ----------
    faces : ndarray of shape (F, 3)
        Face indices
    n_vertices : int
        Number of vertices
    k : int
        Number of rings

    Returns
    -------
    list of list of ndarray
        rings[v][level] contains vertices at exactly level+1 hops from vertex v
        (level 0 = 1-ring, level 1 = 2-ring, etc.)
    """
    # Build adjacency list and flatten for Numba
    adj = igl.adjacency_list(faces.astype(np.int64))

    adj_flat = np.concatenate([np.array(a, dtype=np.int64) for a in adj])
    adj_offsets = np.zeros(n_vertices + 1, dtype=np.int64)
    for i, a in enumerate(adj):
        adj_offsets[i + 1] = adj_offsets[i] + len(a)

    # Compute rings in parallel
    rings_flat, level_offsets = _get_rings_by_level_numba(adj_flat, adj_offsets, k)

    # Convert back to list of list of arrays
    all_rings = []
    for v in range(n_vertices):
        rings = []
        for level in range(k):
            start = level_offsets[v, level]
            end = level_offsets[v, level + 1]
            rings.append(rings_flat[start:end])
        all_rings.append(rings)

    return all_rings


# =============================================================================
# Angular sampling (FreeSurfer-style)
# =============================================================================


def get_rings_by_level(faces, n_vertices, k):
    """Get neighbors organized by ring level (not cumulative).

    Parameters
    ----------
    faces : ndarray of shape (F, 3)
        Face indices
    n_vertices : int
        Number of vertices
    k : int
        Number of rings

    Returns
    -------
    list of list of ndarray
        rings[v][level] contains vertices at exactly level+1 hops from vertex v
        (level 0 = 1-ring, level 1 = 2-ring, etc.)
    """
    adj = igl.adjacency_list(faces.astype(np.int64))

    all_rings = []
    for v in range(n_vertices):
        rings = []
        visited = {v}
        frontier = {v}
        for level in range(k):
            new_frontier = set()
            for u in frontier:
                for neighbor in adj[u]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        new_frontier.add(neighbor)
            rings.append(np.array(sorted(new_frontier), dtype=np.int64))
            frontier = new_frontier
        all_rings.append(rings)

    return all_rings


def compute_vertex_normals(vertices, faces):
    """Compute per-vertex normals via area-weighted face normals.

    Parameters
    ----------
    vertices : ndarray of shape (N, 3)
        Vertex positions
    faces : ndarray of shape (F, 3)
        Face indices

    Returns
    -------
    ndarray of shape (N, 3)
        Unit normals
    """
    # Use igl for robust normal computation
    return igl.per_vertex_normals(vertices, faces.astype(np.int64))


def project_to_tangent_plane(center, normal, neighbors_pos):
    """Project neighbor positions onto the tangent plane at center.

    Parameters
    ----------
    center : ndarray of shape (3,)
        Position of center vertex
    normal : ndarray of shape (3,)
        Unit normal at center
    neighbors_pos : ndarray of shape (M, 3)
        Positions of neighbor vertices

    Returns
    -------
    ndarray of shape (M, 2)
        2D coordinates on tangent plane
    """
    # Build local coordinate frame
    # Choose arbitrary perpendicular vector
    if abs(normal[0]) < 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    else:
        ref = np.array([0.0, 1.0, 0.0])

    u = np.cross(normal, ref)
    u = u / np.linalg.norm(u)
    v = np.cross(normal, u)

    # Project neighbors onto tangent plane
    rel_pos = neighbors_pos - center
    x = np.dot(rel_pos, u)
    y = np.dot(rel_pos, v)

    return np.column_stack([x, y])


def select_angular_samples(angles, n_samples=8):
    """Select n_samples points with best angular spacing.

    Divides the circle into n_samples sectors and picks the point
    closest to each sector center.

    Parameters
    ----------
    angles : ndarray of shape (M,)
        Angles in radians
    n_samples : int
        Number of samples to select

    Returns
    -------
    ndarray
        Indices into original array (length <= n_samples)
    """
    if len(angles) == 0:
        return np.array([], dtype=np.int64)

    if len(angles) <= n_samples:
        return np.arange(len(angles), dtype=np.int64)

    # Normalize angles to [0, 2pi)
    angles = np.mod(angles, 2 * np.pi)

    # Sector centers: 0, 2pi/n, 4pi/n, ...
    sector_width = 2 * np.pi / n_samples
    sector_centers = np.arange(n_samples) * sector_width

    selected = []
    for center in sector_centers:
        # Angular distance to this sector center
        diff = np.abs(angles - center)
        # Handle wrap-around
        diff = np.minimum(diff, 2 * np.pi - diff)

        # Find closest point to sector center
        best_idx = np.argmin(diff)
        # Only add if within half sector width (point is reasonably close)
        if diff[best_idx] < sector_width:
            if best_idx not in selected:
                selected.append(best_idx)

    return np.array(selected, dtype=np.int64)


def set_num_threads(n_threads):
    """Set the number of threads for Numba parallel operations.

    This affects the parallel k-ring computation. Call this before
    running compute_kring_geodesic_distances.

    Parameters
    ----------
    n_threads : int
        Number of threads to use
    """
    numba.set_num_threads(n_threads)


def get_num_threads():
    """Get the current number of threads for Numba parallel operations.

    Returns
    -------
    int
        Current number of threads
    """
    return numba.get_num_threads()


@njit(cache=True)
def _select_angular_samples_njit(angles, n_samples):
    """Numba port of :func:`select_angular_samples` (bit-identical selection).

    Returns indices into ``angles`` (length ``<= n_samples``) chosen one per angular
    sector, closest-to-center, deduplicated, with the same ``< sector_width`` gate and the
    same first-min (``argmin``) tie-breaking as the NumPy version.
    """
    m = angles.shape[0]
    if m == 0:
        return np.empty(0, dtype=np.int64)
    if m <= n_samples:
        out = np.empty(m, dtype=np.int64)
        for i in range(m):
            out[i] = i
        return out

    two_pi = 2.0 * np.pi
    sector_width = two_pi / n_samples
    amod = np.empty(m, dtype=np.float64)
    for i in range(m):
        amod[i] = angles[i] % two_pi

    sel = np.empty(n_samples, dtype=np.int64)
    nsel = 0
    for c in range(n_samples):
        center = c * sector_width
        best_idx = 0
        best_val = np.inf
        for i in range(m):
            d = abs(amod[i] - center)
            d2 = two_pi - d
            if d2 < d:
                d = d2
            if d < best_val:  # strict '<' => first-min, matches np.argmin
                best_val = d
                best_idx = i
        if best_val < sector_width:
            found = False
            for j in range(nsel):
                if sel[j] == best_idx:
                    found = True
                    break
            if not found:
                sel[nsel] = best_idx
                nsel += 1
    return sel[:nsel]


@njit(parallel=True, cache=True)
def _angular_kring_kernel(
    vertices,
    normals,
    rings_flat,
    level_offsets,
    indptr,
    indices,
    data,
    k,
    n_samples,
    correction,
    max_nb,
):
    """Fused, parallel per-vertex angular sampling + limited Dijkstra.

    Reproduces the serial loop of :func:`compute_kring_geodesic_distances_angular`
    bit-for-bit (tangent-plane projection -> per-ring angular sampling -> limited
    Dijkstra) but over a ``prange`` so all CPU cores are used. Each ``prange`` iteration
    writes only its own output row, so there are no races and the result is deterministic.

    Returns dense ``(n_vertices, max_nb)`` neighbor/distance arrays plus a per-vertex count;
    the caller slices each row to ``count`` to rebuild the ragged lists.
    """
    n_vertices = vertices.shape[0]
    out_nb = np.full((n_vertices, max_nb), -1, dtype=np.int64)
    out_dist = np.zeros((n_vertices, max_nb), dtype=np.float64)
    out_count = np.zeros(n_vertices, dtype=np.int64)

    for v in prange(n_vertices):
        cx = vertices[v, 0]
        cy = vertices[v, 1]
        cz = vertices[v, 2]
        nx = normals[v, 0]
        ny = normals[v, 1]
        nz = normals[v, 2]

        # Local tangent frame (matches project_to_tangent_plane exactly).
        if abs(nx) < 0.9:
            rx, ry, rz = 1.0, 0.0, 0.0
        else:
            rx, ry, rz = 0.0, 1.0, 0.0
        ux = ny * rz - nz * ry
        uy = nz * rx - nx * rz
        uz = nx * ry - ny * rx
        un = np.sqrt(ux * ux + uy * uy + uz * uz)
        ux /= un
        uy /= un
        uz /= un
        vx = ny * uz - nz * uy
        vy = nz * ux - nx * uz
        vz = nx * uy - ny * ux

        count = 0
        for level in range(k):
            start = level_offsets[v, level]
            end = level_offsets[v, level + 1]
            m = end - start
            if m == 0:
                continue
            angles = np.empty(m, dtype=np.float64)
            for i in range(m):
                idx = rings_flat[start + i]
                px = vertices[idx, 0] - cx
                py = vertices[idx, 1] - cy
                pz = vertices[idx, 2] - cz
                xx = px * ux + py * uy + pz * uz
                yy = px * vx + py * vy + pz * vz
                angles[i] = np.arctan2(yy, xx)
            sel = _select_angular_samples_njit(angles, n_samples)
            for s in range(sel.shape[0]):
                out_nb[v, count] = rings_flat[start + sel[s]]
                count += 1

        out_count[v] = count
        if count > 0:
            targets = out_nb[v, :count].copy()
            dists = _limited_dijkstra_numba(
                indptr, indices, data, v, targets, correction
            )
            for i in range(count):
                out_dist[v, i] = dists[i]

    return out_nb, out_dist, out_count


def compute_kring_geodesic_distances_angular(
    vertices,
    faces,
    k,
    n_samples_per_ring=8,
    correction=None,
    use_numba=True,
    n_threads=None,
    tqdm_position=0,
):
    """Compute geodesic distances with angular sampling at each ring.

    Implements FreeSurfer-style angular sampling: at each ring level,
    select n_samples_per_ring neighbors with approximately uniform angular
    spacing (2pi/n_samples apart).

    Parameters
    ----------
    vertices : ndarray of shape (N, 3)
        Vertex positions
    faces : ndarray of shape (F, 3)
        Face indices
    k : int
        Number of rings to include
    n_samples_per_ring : int
        Number of samples per ring (default 8, like FreeSurfer)
    correction : float, optional
        Correction factor for graph distances
    use_numba : bool
        If True (default), use Numba-accelerated Dijkstra
    n_threads : int, optional
        Number of threads for Numba
    tqdm_position : int, optional
        Position of tqdm progress bar (for stacking bars in parallel execution)

    Returns
    -------
    k_rings : list of ndarray
        Sampled neighbor indices for each vertex
    distances : list of ndarray
        Geodesic distances to sampled neighbors
    """
    n_vertices = len(vertices)

    if correction is None:
        correction = GRAPH_DISTANCE_CORRECTION

    # Set thread count for Numba if specified
    if n_threads is not None and use_numba:
        numba.set_num_threads(n_threads)

    # Build mesh graph for distance computation
    graph = build_mesh_graph(vertices, faces)

    # Compute vertex normals for tangent plane projection
    print("Computing vertex normals...")
    normals = compute_vertex_normals(vertices.astype(np.float64), faces)

    print(f"Angular sampling ({n_samples_per_ring} per ring)...")
    if use_numba:
        # Fused parallel path: build flat rings, then run the prange kernel over all
        # vertices (tangent projection + angular sampling + limited Dijkstra). Output is
        # bit-identical to the serial loop below but uses all cores.
        print(f"Computing {k}-ring neighbors by level...")
        adj = igl.adjacency_list(faces.astype(np.int64))
        adj_flat = np.concatenate([np.array(a, dtype=np.int64) for a in adj])
        adj_offsets = np.zeros(n_vertices + 1, dtype=np.int64)
        for i, a in enumerate(adj):
            adj_offsets[i + 1] = adj_offsets[i] + len(a)
        rings_flat, level_offsets = _get_rings_by_level_numba(adj_flat, adj_offsets, k)

        verts64 = np.ascontiguousarray(vertices, dtype=np.float64)
        norms64 = np.ascontiguousarray(normals, dtype=np.float64)
        out_nb, out_dist, out_count = _angular_kring_kernel(
            verts64,
            norms64,
            rings_flat,
            level_offsets,
            graph.indptr,
            graph.indices,
            graph.data,
            k,
            n_samples_per_ring,
            correction,
            k * n_samples_per_ring,
        )
        sampled_neighbors = [
            out_nb[v, : out_count[v]].copy() for v in range(n_vertices)
        ]
        sampled_distances = [
            out_dist[v, : out_count[v]].copy() for v in range(n_vertices)
        ]
    else:
        # Serial fallback (kept for parity / debugging).
        print(f"Computing {k}-ring neighbors by level...")
        rings_by_level = get_rings_by_level(faces, n_vertices, k)
        sampled_neighbors = []
        sampled_distances = []

        for v in tqdm(
            range(n_vertices),
            desc="Sampling neighbors",
            position=tqdm_position,
            leave=True,
        ):
            v_neighbors = []
            center = vertices[v]
            normal = normals[v]

            for level in range(k):
                ring = rings_by_level[v][level]
                if len(ring) == 0:
                    continue
                ring_pos = vertices[ring]
                xy = project_to_tangent_plane(center, normal, ring_pos)
                angles = np.arctan2(xy[:, 1], xy[:, 0])
                sample_idx = select_angular_samples(angles, n_samples_per_ring)
                if len(sample_idx) > 0:
                    selected = ring[sample_idx]
                    v_neighbors.extend(selected)

            v_neighbors = np.array(v_neighbors, dtype=np.int64)
            if len(v_neighbors) > 0:
                v_distances = _limited_dijkstra(v, v_neighbors, graph, correction)
            else:
                v_distances = np.array([])

            sampled_neighbors.append(v_neighbors)
            sampled_distances.append(v_distances)

    # Summary stats
    total_neighbors = sum(len(n) for n in sampled_neighbors)
    avg_neighbors = total_neighbors / n_vertices
    print(
        f"Average neighbors per vertex: {avg_neighbors:.1f} "
        f"(max possible: {k * n_samples_per_ring})"
    )

    return sampled_neighbors, sampled_distances
