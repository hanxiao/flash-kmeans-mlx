"""
Batched K-Means implementations for MLX: Euclidean, Cosine, and Dot-product.

API-compatible with flash_kmeans.kmeans_triton_impl:
  batch_kmeans_Euclid(x, n_clusters, max_iters, tol, init_centroids, verbose)
  batch_kmeans_Cosine(...)
  batch_kmeans_Dot(...)
"""

import mlx.core as mx
from flash_kmeans_mlx.ops import (
    euclid_assign,
    cosine_assign,
    dot_assign,
    centroid_update_euclid,
    centroid_update_cosine,
    l2_normalize,
)


# ---------------------------------------------------------------------------
# Single-iteration helpers
# ---------------------------------------------------------------------------

def _euclid_iter(x, x_sq, centroids, x_f16=None):
    cluster_ids = euclid_assign(x, centroids, x_sq, x_f16=x_f16)
    # Use f32 x for centroid update - scatter_add is faster with f32 source
    centroids_new = centroid_update_euclid(x, cluster_ids, centroids)
    diff = (centroids_new.astype(mx.float32) - centroids.astype(mx.float32))
    shift = mx.sqrt((diff * diff).sum(axis=-1)).max()
    return centroids_new, shift, cluster_ids


def _euclid_iter_no_shift(x, x_sq, centroids, x_f16=None):
    """Iteration without shift computation - for tol=0 fast path."""
    cluster_ids = euclid_assign(x, centroids, x_sq, x_f16=x_f16)
    # Use f32 x for centroid update - scatter_add is faster with f32 source
    centroids_new = centroid_update_euclid(x, cluster_ids, centroids)
    return centroids_new, cluster_ids


def _cosine_iter(x_norm, centroids):
    cluster_ids = cosine_assign(x_norm, centroids)
    centroids_new = centroid_update_cosine(x_norm, cluster_ids, centroids)
    diff = (centroids_new.astype(mx.float32) - centroids.astype(mx.float32))
    shift = mx.sqrt((diff * diff).sum(axis=-1)).max()
    return centroids_new, shift, cluster_ids


def _dot_iter(x, centroids):
    cluster_ids = dot_assign(x, centroids)
    # Use cosine update (mean then no normalize) for dot mode - same as original
    centroids_new = centroid_update_cosine(x, cluster_ids, centroids)
    diff = (centroids_new.astype(mx.float32) - centroids.astype(mx.float32))
    shift = mx.sqrt((diff * diff).sum(axis=-1)).max()
    return centroids_new, shift, cluster_ids


# ---------------------------------------------------------------------------
# Compiled variants (shapes must be fixed for mx.compile)
# ---------------------------------------------------------------------------

# We compile the iteration functions lazily based on shapes.
_compiled_cache = {}


def _get_compiled_euclid_iter(B, N, D, K, use_f16=False, no_shift=False):
    key = ("euclid", B, N, D, K, use_f16, no_shift)
    if key not in _compiled_cache:
        if no_shift and use_f16:
            def _iter_ns_f16(x, x_sq, centroids, x_f16):
                return _euclid_iter_no_shift(x, x_sq, centroids, x_f16=x_f16)
            _compiled_cache[key] = mx.compile(_iter_ns_f16)
        elif no_shift:
            _compiled_cache[key] = mx.compile(_euclid_iter_no_shift)
        elif use_f16:
            def _iter_f16(x, x_sq, centroids, x_f16):
                return _euclid_iter(x, x_sq, centroids, x_f16=x_f16)
            _compiled_cache[key] = mx.compile(_iter_f16)
        else:
            _compiled_cache[key] = mx.compile(_euclid_iter)
    return _compiled_cache[key]


def _get_compiled_euclid_multi_iter(B, N, D, K, n_iters, use_f16=True):
    """Get a compiled function that runs n_iters iterations internally."""
    key = ("euclid_multi", B, N, D, K, n_iters, use_f16)
    if key not in _compiled_cache:
        def _multi_iter(x, x_sq, centroids, x_f16):
            for _ in range(n_iters):
                cluster_ids = euclid_assign(x, centroids, x_sq, x_f16=x_f16)
                centroids = centroid_update_euclid(x, cluster_ids, centroids)
            return centroids, cluster_ids
        _compiled_cache[key] = mx.compile(_multi_iter)
    return _compiled_cache[key]


def _get_compiled_cosine_iter(B, N, D, K):
    key = ("cosine", B, N, D, K)
    if key not in _compiled_cache:
        _compiled_cache[key] = mx.compile(_cosine_iter)
    return _compiled_cache[key]


def _get_compiled_dot_iter(B, N, D, K):
    key = ("dot", B, N, D, K)
    if key not in _compiled_cache:
        _compiled_cache[key] = mx.compile(_dot_iter)
    return _compiled_cache[key]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _init_centroids(x, n_clusters):
    """Randomly select n_clusters points from x as initial centroids."""
    B, N, D = x.shape
    indices = mx.random.randint(0, N, shape=(B, n_clusters))  # (B, K)
    # Gather: x[b, indices[b, k], :] for all b, k
    idx_expanded = mx.expand_dims(indices, axis=-1)            # (B, K, 1)
    idx_expanded = mx.broadcast_to(idx_expanded, (B, n_clusters, D))
    centroids = mx.take_along_axis(x, idx_expanded, axis=1)   # (B, K, D)
    return centroids


def _compute_chunk_size_n(B, N, D, K, max_mem_gb):
    """Compute chunk_size_n for the f16 assignment path given a VRAM budget.

    Returns 0 (no chunking) if the full score matrix fits within budget.
    """
    # Fixed allocations (bytes):
    #   x f32: B*N*D*4, x_f16: B*N*D*2, x_sq: B*N*4
    #   centroids f32: B*K*D*4, c_f16+ct: B*K*D*2*2
    #   scatter buffers: B*K*D*4 + B*K*4
    #   cluster_ids: B*N*4
    #   MLX compiled graph overhead: ~2x the tensor allocations
    fixed = (B * N * D * 4  # x
             + B * N * D * 2  # x_f16
             + B * N * 4  # x_sq
             + B * K * D * 4 * 2  # centroids + c_f16/ct
             + B * K * D * 4  # scatter sums (f32)
             + B * N * 4  # scatter x_flat reshape
             + B * K * 4  # scatter counts
             + B * N * 4)  # cluster_ids
    # MLX compiled functions keep input/output buffers + temporaries;
    # empirically ~2.5x the raw tensor sizes for single-iter compiled
    fixed = int(fixed * 2.5)

    budget = int(max_mem_gb * 1e9)
    available = budget - fixed
    if available <= 0:
        # Data alone exceeds budget; use minimal chunks
        return max(1024, K)

    # Score matrix per chunk: chunk_n * K * 2 bytes (f16)
    chunk_n = int(available / (K * 2))
    chunk_n = max(chunk_n, K)  # at least K rows per chunk

    if chunk_n >= N:
        return 0  # no chunking needed
    return chunk_n


def batch_kmeans_Euclid(
    x: mx.array,
    n_clusters: int,
    max_iters: int = 100,
    tol: float = 0.0,
    init_centroids: mx.array = None,
    verbose: bool = False,
    *,
    compiled: bool = True,
    max_mem_gb: float = 0,
) -> tuple:
    """
    Batched K-Means with squared Euclidean distance.

    Args:
        x: (B, N, D) input points.
        n_clusters: number of clusters K.
        max_iters: maximum iterations.
        tol: convergence tolerance on max centroid shift.
        init_centroids: (B, K, D) optional initial centroids.
        verbose: print per-iteration shift.
        compiled: use mx.compile for the iteration loop.
        max_mem_gb: if > 0, limit peak GPU memory to this many GB by
            chunking the assignment step. 0 = no limit (default).

    Returns:
        cluster_ids: (B, N) uint32
        centroids: (B, K, D)
        num_iters: int
    """
    B, N, D = x.shape

    x_sq = (x.astype(mx.float32) * x.astype(mx.float32)).sum(axis=-1)  # (B, N)

    # Pre-compute float16 copy for fast assignment matmul
    x_f16 = x.astype(mx.float16)

    if init_centroids is None:
        centroids = _init_centroids(x, n_clusters)
    else:
        centroids = init_centroids
    centroids = centroids.reshape(B, n_clusters, D)

    # Compute chunk size for memory-constrained mode
    chunk_n = 0
    if max_mem_gb > 0:
        chunk_n = _compute_chunk_size_n(B, N, D, n_clusters, max_mem_gb)
        if verbose and chunk_n > 0:
            print(f"Memory limit {max_mem_gb:.1f} GB: chunking N={N} into chunks of {chunk_n}")

    # When memory-constrained, always use single-iter path. Multi-iter
    # compiled builds the full graph across all iterations, keeping all
    # intermediates alive and using ~max_iters * per-iter memory.
    use_multi_iter = (chunk_n == 0 and max_mem_gb <= 0)

    # Fast path: when tol <= 0 and not verbose, skip shift computation
    needs_shift = tol > 0 or verbose

    if needs_shift:
        if chunk_n > 0:
            # Memory-constrained single-iter path with chunked assignment
            for it in range(max_iters):
                cluster_ids = euclid_assign(
                    x, centroids, x_sq, chunk_size_n=chunk_n, x_f16=x_f16
                )
                centroids_new = centroid_update_euclid(x, cluster_ids, centroids)
                diff = (centroids_new.astype(mx.float32) - centroids.astype(mx.float32))
                shift = mx.sqrt((diff * diff).sum(axis=-1)).max()
                mx.eval(centroids_new, shift, cluster_ids)
                if verbose:
                    print(f"Iter {it}, center shift: {shift.item():.6f}")
                if shift.item() < tol:
                    break
                centroids = centroids_new
        else:
            iter_fn = (_get_compiled_euclid_iter(B, N, D, n_clusters, use_f16=True)
                       if compiled else None)
            for it in range(max_iters):
                if compiled:
                    centroids_new, shift, cluster_ids = iter_fn(
                        x, x_sq, centroids, x_f16
                    )
                else:
                    centroids_new, shift, cluster_ids = _euclid_iter(
                        x, x_sq, centroids, x_f16=x_f16
                    )
                mx.eval(centroids_new, shift, cluster_ids)

                if verbose:
                    print(f"Iter {it}, center shift: {shift.item():.6f}")
                if shift.item() < tol:
                    break
                centroids = centroids_new
    else:
        if compiled and use_multi_iter:
            # Use multi-iteration compiled function for maximum throughput.
            # Runs all iterations inside a single compiled graph, enabling
            # cross-iteration optimization by the MLX compiler.
            multi_fn = _get_compiled_euclid_multi_iter(
                B, N, D, n_clusters, max_iters, use_f16=True)
            centroids, cluster_ids = multi_fn(x, x_sq, centroids, x_f16)
            mx.eval(centroids, cluster_ids)
            it = max_iters - 1
        elif max_mem_gb > 0:
            # Memory-constrained single-iter path (with optional chunking)
            iter_fn_ns = (_get_compiled_euclid_iter(B, N, D, n_clusters, use_f16=True, no_shift=True)
                          if compiled and chunk_n == 0 else None)
            for it in range(max_iters):
                if iter_fn_ns is not None:
                    centroids_new, cluster_ids = iter_fn_ns(
                        x, x_sq, centroids, x_f16
                    )
                else:
                    cluster_ids = euclid_assign(
                        x, centroids, x_sq, chunk_size_n=chunk_n, x_f16=x_f16
                    )
                    centroids_new = centroid_update_euclid(x, cluster_ids, centroids)
                mx.eval(centroids_new, cluster_ids)
                centroids = centroids_new
        else:
            eval_every = 10
            for it in range(max_iters):
                centroids_new, cluster_ids = _euclid_iter_no_shift(
                    x, x_sq, centroids, x_f16=x_f16
                )
                if (it + 1) % eval_every == 0 or it == max_iters - 1:
                    mx.eval(centroids_new, cluster_ids)
                centroids = centroids_new

    return cluster_ids, centroids, it + 1


def batch_kmeans_Cosine(
    x: mx.array,
    n_clusters: int,
    max_iters: int = 100,
    tol: float = 0.0,
    init_centroids: mx.array = None,
    verbose: bool = False,
    *,
    compiled: bool = True,
) -> tuple:
    """
    Batched K-Means with cosine similarity.

    Args:
        x: (B, N, D) input points (will be L2-normalized internally).
        n_clusters: number of clusters K.
        max_iters: maximum iterations.
        tol: convergence tolerance on max centroid shift.
        init_centroids: (B, K, D) optional initial centroids (will be normalized).
        verbose: print per-iteration shift.
        compiled: use mx.compile for the iteration loop.

    Returns:
        cluster_ids: (B, N) uint32
        centroids: (B, K, D) L2-normalized
        num_iters: int
    """
    B, N, D = x.shape
    x_norm = l2_normalize(x, axis=-1)

    if init_centroids is None:
        centroids = _init_centroids(x_norm, n_clusters)
    else:
        centroids = init_centroids
    centroids = centroids.reshape(B, n_clusters, D)
    centroids = l2_normalize(centroids, axis=-1)

    iter_fn = (_get_compiled_cosine_iter(B, N, D, n_clusters)
               if compiled else _cosine_iter)

    for it in range(max_iters):
        centroids_new, shift, cluster_ids = iter_fn(x_norm, centroids)
        mx.eval(centroids_new, shift, cluster_ids)

        if verbose:
            print(f"Iter {it}, center shift: {shift.item():.6f}")
        if shift.item() < tol:
            break
        centroids = centroids_new

    return cluster_ids, centroids, it + 1


def batch_kmeans_Dot(
    x: mx.array,
    n_clusters: int,
    max_iters: int = 100,
    tol: float = 0.0,
    init_centroids: mx.array = None,
    verbose: bool = False,
    *,
    compiled: bool = True,
) -> tuple:
    """
    Batched K-Means with raw dot-product similarity.

    Args:
        x: (B, N, D) input points.
        n_clusters: number of clusters K.
        max_iters: maximum iterations.
        tol: convergence tolerance on max centroid shift.
        init_centroids: (B, K, D) optional initial centroids.
        verbose: print per-iteration shift.
        compiled: use mx.compile for the iteration loop.

    Returns:
        cluster_ids: (B, N) uint32
        centroids: (B, K, D)
        num_iters: int
    """
    B, N, D = x.shape

    if init_centroids is None:
        centroids = _init_centroids(x, n_clusters)
    else:
        centroids = init_centroids
    centroids = centroids.reshape(B, n_clusters, D)

    iter_fn = (_get_compiled_dot_iter(B, N, D, n_clusters)
               if compiled else _dot_iter)

    for it in range(max_iters):
        centroids_new, shift, cluster_ids = iter_fn(x, centroids)
        mx.eval(centroids_new, shift, cluster_ids)

        if verbose:
            print(f"Iter {it} (dot), center shift: {shift.item():.6f}")
        if shift.item() < tol:
            break
        centroids = centroids_new

    return cluster_ids, centroids, it + 1
