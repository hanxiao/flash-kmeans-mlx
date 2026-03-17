# flash-kmeans-mlx autoresearch

## Goal
Optimize flash-kmeans-mlx speed on Apple Silicon (M3 Ultra) without quality loss. No non-MLX library dependencies (only mlx, numpy). Study the original Flash-KMeans paper and Triton implementation deeply to port algorithmic insights.

## Background
- Original: https://github.com/svg-project/flash-kmeans (Triton/CUDA kernels)
- Paper: https://arxiv.org/abs/2603.09229 (IO-aware batched K-Means)
- Current MLX port uses indicator-matmul for centroid updates and standard distance matmul for assignment
- The original paper's key insight is IO-awareness: minimizing HBM<->SRAM data movement via tiled/blocked computation
- The Triton kernels use block-sparse patterns, sorted-index accumulation, and fused assign+update
- Current MLX implementation does NOT exploit these optimizations yet

## Reference code to study
Read these files from the original repo (in flash_kmeans/ reference directory if present, or clone fresh):
- `assign_euclid_triton.py`: Triton kernel for distance computation + assignment in one fused pass
- `centroid_update_triton.py`: Triton kernel for sorted-index centroid accumulation (avoids full indicator matrix)
- `kmeans_triton_impl.py`: The iteration loop and heuristic config selection
- The paper itself (fetch via web if needed): tiling strategy, IO complexity analysis

## Key optimization directions
1. **Fused assign+update**: Currently two separate matmuls. Can we restructure to reduce memory traffic?
2. **Sorted-index centroid update**: The original uses argsort by cluster_id then contiguous accumulation, avoiding the O(B*K*N) indicator matrix. MLX has `mx.argsort` - implement this.
3. **mx.compile optimization**: Ensure iteration functions have fixed shapes for compilation. Profile compiled vs uncompiled.
4. **Memory layout**: The indicator matmul approach allocates O(B*K*N) float32. For N=500K K=1000 this is 2GB. The sorted approach is O(B*N) extra.
5. **Half precision**: Test float16 compute with float32 accumulation for centroid sums.
6. **Chunked distance computation**: For very large N*K, compute distances in tiles to stay in GPU cache.
7. **mx.fast operations**: Check if mx.fast.scaled_dot_product_attention or other fused ops help for the distance matmul.

## Benchmark script
```bash
cd /Users/hanxiao/.openclaw/workspace/flash-kmeans-mlx
# Create venv if needed
python3 -m venv .venv && source .venv/bin/activate && pip install -e '.[benchmark]'

# Primary benchmark (use this for results.tsv)
# USE LARGE SCALE + MANY ITERATIONS for stable measurements
python -m flash_kmeans_mlx.benchmark --n 500000 --d 128 --k 1000 --max-iters 100 --runs 5 --warmup 2
```

Record TOTAL time in seconds as the primary metric (not per-iter ms, too noisy).
Per-iter can be secondary. Use N=500K, K=1000, 100 iters for all experiments.

## Correctness gate
After each change, verify correctness:
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
Append each experiment to results.tsv:
```
experiment	per_iter_ms	total_ms	n_iters	inertia	notes
baseline	7.5	149	20	11452316	indicator matmul, mx.compile
```

## Rules
1. Read results.tsv at start of each iteration to review history
2. ONE change per experiment. Benchmark. Record. Decide keep/discard.
3. If 5 consecutive discards, try a fundamentally different approach (escape strategy)
4. Run correctness check before recording results
5. Keep all changes in git with descriptive commits
6. Do NOT introduce scipy, torch, or any non-MLX compute dependency
7. Minimum 10 runs of benchmark for stability before claiming improvement

## CRITICAL: Record EVERY experiment (added 2026-03-17)
You are NOT following the rules. You've done dozens of micro-benchmarks internally but only written 2 entries to results.tsv. This defeats the purpose of autoresearch.

**STRICT PROTOCOL:**
1. EVERY optimization attempt MUST be recorded in results.tsv immediately after benchmarking
2. Use the FULL benchmark script (N=500K, K=1000, 100 iters, 10 runs) - no inline micro-benchmarks as substitutes
3. If an experiment doesn't improve total_s, record it anyway with status DISCARD in notes
4. `cat results.tsv` at the START of every iteration to review history
5. Analysis-only entries (no benchmark run) use `-` for numeric fields
6. STOP doing long analysis chains. The cycle is: hypothesize -> implement -> benchmark -> record -> next
