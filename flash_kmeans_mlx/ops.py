"""
Core MLX operations for K-Means: distance computation, assignment, centroid update.

All functions operate on (B, N, D) batched tensors using pure mlx.core ops.
No torch dependency.
"""

import mlx.core as mx


# ---------------------------------------------------------------------------
# Custom Metal kernels
# ---------------------------------------------------------------------------

_metal_argmax_cache = {}


def _get_metal_argmax(total_rows, K):
    """Get a cached Metal kernel for fast argmax over axis=-1 on (total_rows, K) f16 data."""
    key = (total_rows, K)
    if key not in _metal_argmax_cache:
        # Use half4 vectorized loads for 4-wide reads; K must be divisible by 4
        assert K % 4 == 0, f"K={K} must be divisible by 4 for vectorized Metal argmax"
        num_vec = K // 4
        source = f'''
            uint n = thread_position_in_grid.x;
            if (n >= {total_rows}u) return;

            float best_score = -1e30f;
            uint best_k = 0;

            const device half4* scores4 = (const device half4*)(scores + n * {K}u);

            for (uint v = 0; v < {num_vec}u; v++) {{
                half4 vals = scores4[v];
                uint base_k = v * 4u;

                float v0 = static_cast<float>(vals[0]);
                float v1 = static_cast<float>(vals[1]);
                float v2 = static_cast<float>(vals[2]);
                float v3 = static_cast<float>(vals[3]);

                if (v0 > best_score) {{ best_score = v0; best_k = base_k; }}
                if (v1 > best_score) {{ best_score = v1; best_k = base_k + 1u; }}
                if (v2 > best_score) {{ best_score = v2; best_k = base_k + 2u; }}
                if (v3 > best_score) {{ best_score = v3; best_k = base_k + 3u; }}
            }}
            out[n] = best_k;
        '''
        _metal_argmax_cache[key] = mx.fast.metal_kernel(
            name=f'fast_argmax_{total_rows}_{K}',
            input_names=['scores'],
            output_names=['out'],
            source=source,
        )
    return _metal_argmax_cache[key]


def _fast_argmax_f16(scores, B, N, K):
    """Fast argmax over axis=-1 for (B, N, K) f16 scores using custom Metal kernel."""
    total_rows = B * N
    kernel = _get_metal_argmax(total_rows, K)
    THREADS = 1024
    grid_size = ((total_rows + THREADS - 1) // THREADS) * THREADS
    flat_scores = scores.reshape(total_rows, K)
    out = kernel(
        inputs=[flat_scores],
        template=[('T', mx.float16)],
        grid=(grid_size, 1, 1),
        threadgroup=(THREADS, 1, 1),
        output_shapes=[(total_rows,)],
        output_dtypes=[mx.uint32],
    )
    return out[0].reshape(B, N)


# ---------------------------------------------------------------------------
# Assignment kernels
# ---------------------------------------------------------------------------

def euclid_assign(
    x: mx.array,
    centroids: mx.array,
    x_sq: mx.array,
    chunk_size_n: int = 0,
    chunk_size_k: int = 0,
    x_f16: mx.array = None,
) -> mx.array:
    """
    Assign each point to the nearest centroid using squared Euclidean distance.

    d^2(x_i, c_k) = ||x_i||^2 - 2 * <x_i, c_k> + ||c_k||^2

    When x_f16 is provided, the matmul and score computation use float16
    for faster throughput and lower memory bandwidth. The argmax result
    is identical in >99.7% of cases.

    Args:
        x: (B, N, D) input points (float32)
        centroids: (B, K, D) cluster centers
        x_sq: (B, N) pre-computed ||x||^2 (unused when x_f16 is provided)
        chunk_size_n: if > 0, process N in chunks to limit memory
        chunk_size_k: if > 0, process K in chunks to limit memory
        x_f16: (B, N, D) float16 copy of x for fast assignment

    Returns:
        cluster_ids: (B, N) uint32 cluster assignment per point
    """
    B, N, D = x.shape
    K = centroids.shape[1]

    # Use float16 fast path when available
    if x_f16 is not None:
        c_f16 = centroids.astype(mx.float16)
        ct = mx.transpose(c_f16, axes=(0, 2, 1))            # (B, D, K)
        c_sq = (c_f16 * c_f16).sum(axis=-1)                 # (B, K)
        c_bias = (-0.5 * c_sq).astype(mx.float16)[:, None, :]  # (B, 1, K)
        # addmm fuses the bias addition into the matmul kernel, avoiding a
        # separate pass over the full NxK score matrix.
        score = mx.addmm(c_bias, x_f16, ct)                 # (B, N, K)
        return _fast_argmax_f16(score, B, N, K)

    c_sq = (centroids * centroids).sum(axis=-1)  # (B, K)

    if chunk_size_n > 0 and chunk_size_n < N:
        # Chunked along N to limit peak memory
        parts = []
        for n_start in range(0, N, chunk_size_n):
            n_end = min(n_start + chunk_size_n, N)
            x_chunk = x[:, n_start:n_end, :]        # (B, n_chunk, D)
            xsq_chunk = x_sq[:, n_start:n_end]      # (B, n_chunk)
            ids_chunk = _assign_chunk(x_chunk, centroids, xsq_chunk, c_sq,
                                      chunk_size_k)
            parts.append(ids_chunk)
        return mx.concatenate(parts, axis=1)

    return _assign_chunk(x, centroids, x_sq, c_sq, chunk_size_k)


def _assign_chunk(
    x: mx.array,
    centroids: mx.array,
    x_sq: mx.array,
    c_sq: mx.array,
    chunk_size_k: int,
) -> mx.array:
    """Assign within a single N-chunk, optionally chunking K."""
    K = centroids.shape[1]

    if chunk_size_k > 0 and chunk_size_k < K:
        # Chunked along K
        best_ids = None
        best_score = None
        for k_start in range(0, K, chunk_size_k):
            k_end = min(k_start + chunk_size_k, K)
            c_chunk = centroids[:, k_start:k_end, :]   # (B, k_chunk, D)
            csq_chunk = c_sq[:, k_start:k_end]          # (B, k_chunk)

            cross = x @ mx.transpose(c_chunk, axes=(0, 2, 1))  # (B, n, k_chunk)
            score = cross - 0.5 * mx.expand_dims(csq_chunk, axis=-2)

            chunk_ids = mx.argmax(score, axis=-1)  # (B, n)
            chunk_max = mx.max(score, axis=-1)     # (B, n)

            if best_ids is None:
                best_ids = chunk_ids + k_start
                best_score = chunk_max
            else:
                mask = chunk_max > best_score
                best_ids = mx.where(mask, chunk_ids + k_start, best_ids)
                best_score = mx.where(mask, chunk_max, best_score)

        return best_ids.astype(mx.uint32)

    # Full matmul: (B, N, D) @ (B, D, K) -> (B, N, K)
    cross = x @ mx.transpose(centroids, axes=(0, 2, 1))
    # Optimization: x_sq is constant across K, so argmin(x_sq + c_sq - 2*cross)
    # = argmin(c_sq - 2*cross) = argmax(cross - 0.5*c_sq).
    # This avoids the x_sq broadcast and the full distance matrix.
    score = cross - 0.5 * mx.expand_dims(c_sq, axis=-2)
    return mx.argmax(score, axis=-1).astype(mx.uint32)


def cosine_assign(x_norm: mx.array, centroids: mx.array) -> mx.array:
    """
    Assign each point to the centroid with highest cosine similarity (dot product
    on L2-normalized vectors).

    Args:
        x_norm: (B, N, D) L2-normalized input points
        centroids: (B, K, D) L2-normalized cluster centers

    Returns:
        cluster_ids: (B, N) uint32
    """
    sim = x_norm @ mx.transpose(centroids, axes=(0, 2, 1))  # (B, N, K)
    return mx.argmax(sim, axis=-1).astype(mx.uint32)


def dot_assign(x: mx.array, centroids: mx.array) -> mx.array:
    """
    Assign each point to the centroid with highest dot-product similarity.

    Args:
        x: (B, N, D)
        centroids: (B, K, D)

    Returns:
        cluster_ids: (B, N) uint32
    """
    sim = x @ mx.transpose(centroids, axes=(0, 2, 1))  # (B, N, K)
    return mx.argmax(sim, axis=-1).astype(mx.uint32)


# ---------------------------------------------------------------------------
# Centroid update
# ---------------------------------------------------------------------------

def centroid_update_euclid(
    x: mx.array,
    cluster_ids: mx.array,
    old_centroids: mx.array,
    x_f16: mx.array = None,
) -> mx.array:
    """
    Recompute centroids as the mean of assigned points (Euclidean mode).

    Uses scatter-add accumulation for memory efficiency.

    Args:
        x: (B, N, D) input points
        cluster_ids: (B, N) uint32 assignments
        old_centroids: (B, K, D) previous centroids (used for empty clusters)
        x_f16: (B, N, D) optional float16 copy for reduced read bandwidth

    Returns:
        centroids_new: (B, K, D)
    """
    return _centroid_update(x, cluster_ids, old_centroids, normalize=False, x_f16=x_f16)


def centroid_update_cosine(
    x: mx.array,
    cluster_ids: mx.array,
    old_centroids: mx.array,
) -> mx.array:
    """Centroid update with L2 normalization (cosine mode)."""
    return _centroid_update(x, cluster_ids, old_centroids, normalize=True)


def _centroid_update(
    x: mx.array,
    cluster_ids: mx.array,
    old_centroids: mx.array,
    normalize: bool,
    x_f16: mx.array = None,
) -> mx.array:
    """
    Core centroid update using scatter-add.

    Instead of building a (B, K, N) indicator matrix (O(B*K*N) memory),
    use scatter_add via .at[].add() to accumulate sums directly into
    a (B, K, D) buffer. Memory: O(B*K*D) vs O(B*K*N).

    When x_f16 is provided, reads from float16 data (halved bandwidth)
    but accumulates into float32 buffer for precision.
    """
    B, N, D = x.shape
    K = old_centroids.shape[1]

    # Use f16 source data for reduced read bandwidth when available
    x_src = x_f16 if x_f16 is not None else x
    ids = cluster_ids.astype(mx.uint32)  # (B, N)

    results_sums = []
    results_counts = []
    for b in range(B):
        cluster_sums = mx.zeros((K, D), dtype=mx.float32)
        cluster_counts = mx.zeros((K,), dtype=mx.float32)
        idx = ids[b]  # (N,)
        cluster_sums = cluster_sums.at[idx].add(x_src[b])
        cluster_counts = cluster_counts.at[idx].add(mx.ones((N,), dtype=mx.float32))
        results_sums.append(cluster_sums)
        results_counts.append(cluster_counts)

    all_sums = mx.stack(results_sums)      # (B, K, D)
    all_counts = mx.stack(results_counts)   # (B, K)

    return _finalize_centroids(
        all_sums, all_counts, old_centroids, x.dtype, normalize
    )



def _finalize_centroids(
    cluster_sums: mx.array,
    cluster_counts: mx.array,
    old_centroids: mx.array,
    out_dtype,
    normalize: bool,
) -> mx.array:
    """Divide sums by counts, handle empty clusters, optionally L2-normalize."""
    counts_safe = mx.maximum(cluster_counts, 1.0)
    centroids_new = cluster_sums / mx.expand_dims(counts_safe, axis=-1)

    # Preserve old centroids for empty clusters
    empty_mask = mx.expand_dims(cluster_counts == 0, axis=-1)  # (B, K, 1)
    centroids_new = mx.where(empty_mask, old_centroids.astype(mx.float32), centroids_new)

    if normalize:
        norms = mx.sqrt((centroids_new * centroids_new).sum(axis=-1, keepdims=True))
        norms = mx.maximum(norms, 1e-12)
        centroids_new = centroids_new / norms

    return centroids_new.astype(out_dtype)


# ---------------------------------------------------------------------------
# L2 normalization helper
# ---------------------------------------------------------------------------

def l2_normalize(x: mx.array, axis: int = -1, eps: float = 1e-12) -> mx.array:
    """L2-normalize along the given axis."""
    norms = mx.sqrt((x * x).sum(axis=axis, keepdims=True))
    norms = mx.maximum(norms, eps)
    return x / norms
