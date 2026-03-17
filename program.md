# flash-kmeans-mlx autoresearch

## Goal
Optimize flash-kmeans-mlx speed on Apple Silicon (M3 Ultra) without quality loss. No non-MLX library dependencies (only mlx, numpy).

## Current state
- Baseline: 1.296s (13.0ms/iter) for N=500K, D=128, K=1000, B=1, 100 iters on M3 Ultra
- Already implemented: mx.addmm fused assign, no-shift fast path, f16 assign matmul
- Gap to H200 Flash-KMeans (Triton): ~5-30x depending on config (batched scenarios worse)
- Gap to H200 fast_pytorch_kmeans: ~2-3x

## Reference code
Study the original Triton implementation in `flash_kmeans_reference/`:
- `assign_euclid_triton.py`: fused distance+assignment kernel
- `centroid_update_triton.py`: sorted-index centroid accumulation (avoids O(B*K*N) indicator matrix)
- `kmeans_triton_impl.py`: iteration loop

## Optimization strategies to try (ONE AT A TIME)
1. **Sorted-index centroid update**: argsort cluster_ids, then segment-sum for centroids. Avoids the O(B*K*N) indicator matmul. Use `mx.argsort` + cumulative segment sums
2. **Chunked/tiled distance computation**: compute distances in N-tiles to reduce peak memory and improve cache usage
3. **Deferred evaluation**: eval every N iterations instead of every iteration (test N=2,3,5,10)
4. **Batch-parallel optimization**: current batched perf is poor. Profile B=8,32 and optimize
5. **Mixed precision centroid update**: f16 accumulation where safe
6. **Memory pre-allocation**: reuse buffers across iterations instead of allocating new tensors
7. **mx.fast operations**: explore mx.fast.* for any applicable fused ops
8. **Reduce Python overhead**: minimize per-iteration Python calls, maximize compiled graph coverage

## Benchmark (MANDATORY for every experiment)
```bash
cd /Users/hanxiao/.openclaw/workspace/flash-kmeans-mlx
source .venv/bin/activate
python -m flash_kmeans_mlx.benchmark --n 500000 --d 128 --k 1000 --max-iters 100 --runs 10 --warmup 2
```

## Correctness gate (MANDATORY before recording)
```python
import numpy as np, mlx.core as mx
from flash_kmeans_mlx import batch_kmeans_Euclid
np.random.seed(42)
X = np.random.randn(20000, 128).astype(np.float32)
init = X[:200].copy()
X_mlx = mx.array(X[None]); init_mlx = mx.array(init[None])
ids, cents, _ = batch_kmeans_Euclid(X_mlx, 200, max_iters=20, tol=0, init_centroids=init_mlx)
mx.eval(ids, cents)
inertia = float(((mx.array(X) - cents[0][ids[0]])**2).sum())
print(f"Inertia: {inertia:.0f}")
# Baseline inertia should be ~2350000 +/- 5000. If >5% off, reject.
```

## Results tracking
Append each experiment to results.tsv with columns:
```
experiment	total_s	per_iter_ms	n_iters	inertia	status	notes
```
status: KEEP or DISCARD

## ABSOLUTE RULES (violation = failure)
1. **EVERY experiment MUST be recorded in results.tsv** - no exceptions, no "micro-benchmarks" that skip recording
2. The cycle is: hypothesize -> implement ONE change -> run FULL benchmark (10 runs) -> run correctness check -> record in results.tsv -> git commit -> next
3. `cat results.tsv` at the START of every iteration to review history
4. NO inline micro-benchmarks as substitutes for the full benchmark
5. NO long analysis chains without recording results
6. If improvement: KEEP, commit code, move to next strategy
7. If no improvement: DISCARD, revert code, record anyway, move to next strategy
8. If 5 consecutive DISCARDs, try a fundamentally different approach
9. Do NOT introduce scipy, torch, or any non-MLX compute dependency
10. Keep all changes in git with descriptive commits
