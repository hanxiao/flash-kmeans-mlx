# Flash-KMeans: Triton vs MLX Performance Gap Analysis

Deep analysis of why the Triton/CUDA implementation is faster and what can (and cannot) be done about it on Apple Silicon.

Reference configuration for this analysis: N=500K, D=128, K=1000, B=1, float32 input, 100 iterations on M3 Ultra.

---

## 1. What Makes the Triton Version Fast

The Flash-KMeans paper identifies two fundamental bottlenecks in standard K-Means on GPUs:

1. **Assignment stage**: IO-bound from materializing the N*K distance matrix in HBM
2. **Centroid update stage**: atomic write contention from scatter-style per-token accumulation

The Triton implementation solves both with kernel-level fusions that are tightly coupled to NVIDIA GPU hardware.

### 1.1 FlashAssign: Fused Distance + Online Argmin

The core innovation. The kernel (`_euclid_assign_kernel` in `assign_euclid_triton.py`) does the following in a single GPU kernel:

```
For each tile of BLOCK_N points (64-128 points):
  1. Load x_tile (BLOCK_N, D) into SRAM once - stays resident for all K iterations
  2. Load x_sq_tile (BLOCK_N,) into SRAM
  3. Initialize best_dist = +inf, best_idx = 0
  4. For k_start in range(0, K, BLOCK_K):        // BLOCK_K = 32-128
     a. Load centroid tile c_tile (D, BLOCK_K) from HBM into SRAM
     b. Load c_sq (BLOCK_K,) from HBM
     c. Compute cross = tl.dot(x_tile, c_tile)    // (BLOCK_N, BLOCK_K) matmul in SRAM
     d. Compute dist = x_sq + c_sq - 2*cross      // elementwise in registers
     e. curr_min = tl.min(dist, axis=1)            // reduce across K dimension
     f. Update best_dist, best_idx with online argmin
  5. Write best_idx (BLOCK_N,) to HBM             // only N int32 values written
```

**Why this is fast on NVIDIA GPUs:**

- **tl.dot**: Triton compiles this to hardware Tensor Core MMA instructions. The (BLOCK_N, D) x (D, BLOCK_K) matmul runs at near-peak FLOPS because the tiles fit in shared memory/registers.
- **Double buffering / async prefetch**: Triton's `num_stages` parameter controls software pipelining. With `num_stages=2-4`, loading the next centroid tile overlaps with computing on the current one. This hides memory latency almost entirely.
- **Online argmin in registers**: The running min/argmin stays in registers across all K/BLOCK_K iterations. The full N*K distance matrix is **never written to HBM**. This eliminates O(N*K) writes + O(N*K) reads.
- **IO reduction**: Standard approach reads/writes 2*N*K*4 bytes for the distance matrix. FlashAssign reads N*D + ceil(K/BLOCK_K)*(BLOCK_K*D + BLOCK_K) bytes and writes only N*4 bytes. For N=500K, K=1000, D=128: standard = 4 GB I/O; FlashAssign = ~0.26 GB reads + 2 MB writes.
- **Auto-tuning / heuristics**: The kernel has GPU-family-specific heuristics for tile sizes, warp counts, and pipeline stages tuned per (N, K, D) configuration.

### 1.2 Sort-Inverse Update: Contention-Free Centroid Accumulation

The centroid update kernel (`_centroid_update_chunk_kernel` in `centroid_update_triton.py`):

```
1. Sort assignment vector a by cluster_id -> sorted_idx, sorted_cluster_ids
2. For each chunk of BLOCK_N consecutive sorted tokens:
   a. Load sorted_cluster_ids and sorted_idx for the chunk
   b. For each unique cluster_id in the chunk:
      - Gather token features from X using sorted_idx (random access)
      - Accumulate local partial sum on-chip
      - Issue ONE atomic_add per segment boundary (not per token)
   c. Issue ONE atomic_add for count
```

**Why this is fast on NVIDIA GPUs:**

- **Atomic count reduction**: Standard approach issues N*D atomic adds (500K*128 = 64M atomics). Sort-inverse issues approximately (K + N/BLOCK_N) * D atomics. For N=500K, K=1000, BLOCK_N=256: ~3000*128 = 384K atomics - a 166x reduction.
- **Sorted access eliminates write contention**: When tokens are sorted by cluster_id, contiguous chunks tend to contain tokens from the same cluster. Multiple tokens accumulate locally before a single atomic write, eliminating the hardware serialization from concurrent writes to the same address.
- **GPU shared memory for local accumulation**: The local partial sums reside in fast shared memory/registers, with only the final merged result going to global memory via atomic.

### 1.3 System-Level: Chunked Stream Overlap

The `kmeans_large.py` shows double-buffered CUDA stream pipelining for out-of-core execution:
- Two work streams alternate processing chunks
- `non_blocking=True` transfers overlap with compute on previous chunks
- Events coordinate dependencies between streams

This is specific to CUDA's explicit stream/event model.

### 1.4 Compilation Model

Triton compiles to PTX/SASS with full access to:
- Tensor Core MMA instructions (hardware matrix multiply accelerators)
- Shared memory (programmable L1 cache, 48-228 KB per SM on H200)
- Software pipelining via `num_stages` (cp.async + double buffering)
- Warp-level primitives for reductions
- L2 cache persistence hints

---

## 2. What Our MLX Version Does and Where We Lose Time

### 2.1 Current MLX Assignment: addmm + Separate Argmax

Our implementation (`ops.py:euclid_assign`, `ops.py:_fast_argmax_f16`):

```python
# 1. Cast centroids to f16, compute transpose and squared norms
c_f16 = centroids.astype(mx.float16)
ct = mx.contiguous(mx.transpose(c_f16, axes=(0, 2, 1)))  # (B, D, K) contiguous
c_sq = (c_f16 * c_f16).sum(axis=-1)                       # (B, K)
c_bias = (-0.5 * c_sq).astype(mx.float16)[:, None, :]     # (B, 1, K)

# 2. Fused addmm: score = c_bias + x_f16 @ ct             # (B, N, K)
score = mx.addmm(c_bias, x_f16, ct)

# 3. Custom Metal argmax kernel
return _fast_argmax_f16(score, B, N, K)
```

**Performance breakdown (measured, N=500K, K=1000, D=128, B=1):**
- addmm: ~5.9 ms (80% of 27 TFLOPS f16 peak on M3 Ultra)
- Metal argmax: ~1.67 ms (95% of achievable bandwidth)
- Centroid update (scatter_add): overlapped with addmm, effectively free
- Total: 7.65 ms/iter

**Where we lose time vs Triton:**

**Gap 1: The N*K score matrix IS materialized.**
Our approach computes the full (N, K) = (500K, 1000) score matrix and writes it to unified memory. That is 500K * 1000 * 2 bytes = 1 GB written, then 1 GB read back by argmax. The Triton FlashAssign kernel never writes this matrix at all.

However: on M3 Ultra with unified memory (no HBM distinction), the "write to memory + read back" penalty is partially mitigated by the large L2 cache (192 MB on M3 Ultra). The N*K matrix at 1 GB exceeds L2, but for smaller K the matrix might partially stay in cache. This gap is proportionally larger at large K.

**Gap 2: addmm uses AMX, not a fused kernel.**
`mx.addmm` dispatches to Apple's AMX (Apple Matrix eXtension) hardware, which is a fixed-function matrix multiply accelerator. It achieves excellent throughput for large matmuls but:
- It cannot be fused with argmin. The AMX unit produces results that must be written back to memory before any custom kernel can read them.
- The bias addition in addmm is the only "fusion" available - it avoids a separate pass to add c_bias.
- There is no way to stream centroid tiles and compute online argmin inside the AMX path.

**Gap 3: Argmax is a separate memory-bound pass.**
The custom Metal argmax kernel reads the full N*K score matrix from memory. Even with vectorized half4 loads, it is bandwidth-limited. At 1 GB of data and ~400 GB/s memory bandwidth on M3 Ultra, the theoretical minimum is ~2.5 ms. The measured 1.67 ms indicates the data partially resides in cache from the addmm write. But it is still an extra pass that FlashAssign avoids entirely.

### 2.2 Current MLX Centroid Update: scatter_add

```python
# Flat scatter_add across all batches
x_flat = x_src.reshape(B * N, D)       # (BN, D)
cluster_sums = mx.zeros((B * K, D), dtype=mx.float32)
cluster_sums = cluster_sums.at[ids_flat].add(x_flat)

cluster_counts = mx.zeros((B * K,), dtype=mx.float32)
cluster_counts = cluster_counts.at[ids_flat].add(ones)
```

**Where we lose time vs Triton:**

**Gap 4: scatter_add is unsorted, causing write contention.**
MLX's `scatter_add` processes tokens in original order. Multiple tokens mapping to the same cluster cause atomic contention on the accumulation buffer. The Triton sort-inverse approach eliminates this.

However: in the compiled multi-iteration path, the centroid update is completely overlapped with the next iteration's addmm. The 100% overlap was confirmed experimentally (3x scatter_add experiment showed identical total time). So the centroid update is currently "free" - it executes concurrently with the dominant addmm. This means optimizing the centroid update alone would yield zero improvement unless the addmm also gets faster.

### 2.3 Compilation and Evaluation Model

MLX uses `mx.compile` to trace and fuse operations into a single computation graph. The multi-iteration compiled path (`_get_compiled_euclid_multi_iter`) runs all 100 iterations inside one compiled graph, enabling:
- Cross-iteration buffer reuse
- Elimination of Python loop overhead
- Potential operator fusion by the MLX compiler

This is architecturally different from Triton's per-kernel compilation but achieves a similar goal of reducing overhead.

---

## 3. Gap-by-Gap Analysis: Can It Be Closed in MLX?

### Gap 1: Distance Matrix Materialization

**Status: LIKELY UNCLOSABLE with current MLX primitives**

The fundamental issue is that MLX's matrix multiplication (`@`, `addmm`, `matmul`) uses Apple's AMX hardware, which is a fixed-function accelerator that produces complete output matrices. There is no API to:
- Stream partial results from AMX to a custom reduction kernel
- Intercept AMX output in SRAM/L2 before it reaches main memory
- Tile the matmul and fuse argmin between tiles

Attempted workarounds and their results:
- **K-tiled matmul with running argmax**: 13-15 ms (2x slower) - multiple small matmuls are less efficient than one large matmul on AMX
- **Custom Metal simdgroup_matrix fused matmul+argmax**: 14.6 ms (2x slower) - GPU compute shaders are 6x slower than AMX for matmul
- **Custom Metal tiled matmul+argmax**: 46 ms (6x slower) - same fundamental issue

The AMX is simply too fast at full matmul for any tiled/fused approach to compete, even though the tiled approach would avoid materializing the full matrix. The break-even point would require K to be very large (probably K > 10000) where the materialized matrix dominates.

**What would be needed**: An MLX primitive like `mx.fast.fused_matmul_argmax(A, B)` that internally uses AMX with streaming output to a reduction unit without full materialization. This would need to be implemented at the MLX C++/Metal framework level. It does not exist today.

**Alternative**: If Apple Silicon had programmable shared memory (like CUDA's shared memory / Triton's SRAM model), a custom Metal kernel could tile the matmul on the GPU shader cores and fuse argmin. But Apple GPU shader cores are 6x slower than AMX for matmul, making this approach nonviable.

### Gap 2: AMX vs Tensor Core Throughput

**Status: HARDWARE LIMIT**

H200 FP16 Tensor Core peak: ~990 TFLOPS (with sparsity: 1979 TFLOPS)
M3 Ultra AMX FP16 peak: ~27 TFLOPS

This is a ~37x raw compute gap. For the assignment matmul (N=500K, D=128, K=1000):
- FLOPs = 2 * 500K * 128 * 1000 = 128 GFLOPS
- H200 theoretical: 128 / 990000 = 0.13 ms
- M3 Ultra theoretical: 128 / 27000 = 4.7 ms
- M3 Ultra measured: 5.9 ms (80% of peak)

The assignment stage will always be ~37x slower on M3 Ultra due to raw compute throughput. This is not a software optimization problem.

### Gap 3: Separate Argmax Pass

**Status: PARTIALLY CLOSABLE**

The custom Metal argmax kernel achieves 95% of the achievable bandwidth. Further optimization:
- vec8/vec16 loads were tried and showed no gain (bandwidth-limited, not latency-limited)
- Cooperative multi-thread per row was tried - no gain
- The kernel is at the hardware limit for a standalone pass

The only way to eliminate this pass entirely would be to fuse it into the matmul (Gap 1), which is blocked by the AMX architecture.

However, at 1.67 ms out of 7.65 ms total (22%), this is the second-largest component. If the addmm score matrix were smaller (smaller K), a larger fraction would stay in L2 cache and the argmax would be faster. But for K=1000 at N=500K, the 1 GB matrix exceeds the 192 MB L2.

### Gap 4: Unsorted Centroid Update

**Status: CLOSABLE BUT IRRELEVANT**

The sorted-index approach was tried (`sorted_index_centroid` experiment) and was 11% slower overall because the argsort overhead exceeded the contention reduction benefit. More importantly, the centroid update is currently 100% overlapped with the next iteration's addmm in the compiled pipeline. It is effectively free.

The Triton version benefits from the sort because on NVIDIA GPUs:
- `torch.sort` on int32 is very fast via CUB radix sort
- The atomic contention on H200 HBM is severe enough to dominate without sorting
- The centroid update is NOT overlapped with FlashAssign (they are separate kernels)

On Apple Silicon:
- `mx.scatter_add` contention is less severe because unified memory has different coherence characteristics
- The centroid update runs concurrently with AMX matmul (different hardware units)
- Sorting would add latency to the critical path without removing any

**Would become relevant if**: addmm gets faster (pushing centroid update onto the critical path) or K becomes very large (more contention).

### Gap 5: Memory Bandwidth (HBM vs Unified Memory)

**Status: HARDWARE LIMIT**

H200 HBM bandwidth: 4.8 TB/s
M3 Ultra unified memory bandwidth: 400-800 GB/s (theoretical peak ~819 GB/s, practical ~400-500 GB/s)

This 6-12x bandwidth gap affects:
- Argmax pass (reading the N*K score matrix)
- Centroid update (reading X for scatter)
- All memory-bound operations

### Gap 6: Kernel Launch and Framework Overhead

**Status: LARGELY CLOSED**

The multi-iteration compiled path eliminates per-iteration Python overhead. All 100 iterations run inside one compiled graph. This is one of the most effective optimizations we have (9% speedup from `multi_iter_compiled`).

The Triton version still has per-iteration Python overhead (calling into Triton kernels from the Python loop), though this is minimal on GPU due to async dispatch.

---

## 4. Optimization Opportunities Ranked by Expected Impact

### Tier 1: Would Help But Blocked by Hardware/Framework

| Opportunity | Expected Impact | Feasibility |
|---|---|---|
| Fused matmul+argmin (FlashAssign equivalent) | 20-30% (eliminate argmax pass + reduce memory) | **Blocked**: requires AMX streaming API or mx.fast primitive that does not exist |
| Lower-precision matmul (FP8, INT8) | 30-50% if ALU-bound | **Blocked**: mx.qqmm NYI on M3 architecture; Apple GPU FP8 support unclear |

### Tier 2: Possible But Marginal (< 5% expected)

| Opportunity | Expected Impact | Feasibility |
|---|---|---|
| Further argmax optimization (warp shuffle, simdgroup reduce) | 1-3% | Tried several approaches, all at bandwidth limit |
| Smarter K-tiling for large K | 5-15% for K > 5000 | Viable only when N*K matrix far exceeds L2; at K=1000 it hurts |
| Precomputed c_sq across iterations (when centroids change slowly) | < 1% | c_sq computation is tiny and overlapped |
| Alternative scatter approaches for centroid update | 0% (overlapped) | Would only help if addmm gets faster |

### Tier 3: Algorithmic / Approximate Methods

| Opportunity | Expected Impact | Feasibility |
|---|---|---|
| Triangle inequality pruning (Elkan-style) | Negative (adds branches, reduces parallelism) | Known to hurt on GPUs; paper explicitly discusses this |
| Random projection dimensionality reduction | 50%+ matmul reduction | **Failed**: 93% assignment mismatch at D=128->64, curse of dimensionality |
| LSH approximate nearest centroid | Huge potential | **Failed**: 1.4% accuracy, completely unusable |
| Hadamard-based dimension reduction | Variable | **Failed**: 13-87% match depending on reduction ratio, unusable |

### Tier 4: Apple Silicon-Specific Opportunities

| Opportunity | Expected Impact | Feasibility |
|---|---|---|
| Exploit Neural Engine (ANE) for matmul | Unknown, potentially 2x | ANE is ~15 TOPS on M3 Ultra; unclear if it helps for this shape and precision |
| Exploit multiple GPU core clusters | 5-20% | M3 Ultra has 80 GPU cores; unclear if MLX already uses all of them |
| Metal feature set exploration (mesh shaders, tile functions) | Unknown | Apple's tile shader model might allow partial fusion, needs investigation |

---

## 5. Theoretical Minimum Latency on M3 Ultra

### Per-Iteration Lower Bounds

**Assignment stage - compute bound:**
- FLOPs = 2 * N * D * K = 2 * 500000 * 128 * 1000 = 128 GFLOPS
- AMX FP16 peak: 27 TFLOPS
- Minimum compute time: 128 / 27000 = 4.74 ms

**Assignment stage - memory bound (reading inputs):**
- Read X: 500K * 128 * 2 bytes = 128 MB (f16)
- Read C: 1000 * 128 * 2 bytes = 0.256 MB (f16)
- Write assignments: 500K * 4 bytes = 2 MB
- Total minimum IO: ~130 MB
- At 400 GB/s practical bandwidth: 0.33 ms
- Assignment is compute-bound, not memory-bound

**Assignment stage - with distance matrix materialization:**
- Write scores: 500K * 1000 * 2 bytes = 1000 MB (f16)
- Read scores for argmax: 1000 MB
- At 400 GB/s: 5.0 ms (memory bound for the argmax pass alone)
- Note: this overlaps with compute if AMX and memory subsystem work concurrently

**Centroid update:**
- Read X: 500K * 128 * 4 bytes = 256 MB (f32)
- Read cluster_ids: 500K * 4 bytes = 2 MB
- Write sums: 1000 * 128 * 4 bytes = 0.5 MB
- At 400 GB/s: 0.65 ms
- Currently overlapped with AMX, so effectively 0 ms on critical path

**Shift computation (when needed):**
- Read two centroids: 2 * 1000 * 128 * 4 bytes = 1 MB
- Negligible

### Theoretical Floor

**With current architecture (addmm + separate argmax):**
- AMX addmm: 4.74 ms (compute bound, 80% utilization achieved = 5.9 ms practical)
- Argmax: cannot overlap with addmm (depends on addmm output)
  - Reading 1 GB score matrix at 400 GB/s = 2.5 ms theoretical
  - Measured: 1.67 ms (cache effects from addmm write)
- Critical path: addmm + argmax = 5.9 + 1.67 = 7.57 ms
- **Current measured: 7.65 ms (within 1% of this floor)**

**With hypothetical fused matmul+argmin (FlashAssign equivalent):**
- Compute: 4.74 ms (same FLOPs)
- No extra memory pass for argmax (fused)
- Memory: read X once (128 MB) + stream centroids ceil(K/BK) times (~0.3 MB each) + write assignments (2 MB)
- But: this is impossible with AMX. A GPU compute shader implementation would need to do the matmul at GPU shader rate, which is ~4.5 TFLOPS on M3 Ultra (vs 27 TFLOPS AMX). So the fused kernel would take: 128 / 4500 = 28.4 ms - much worse.
- **Fused approach only wins if K is large enough that the argmax pass dominates.**

**Absolute theoretical floor (if AMX could stream to fused argmin):**
- Compute: 4.74 ms
- Memory: ~130 MB at 400 GB/s = 0.33 ms
- Total: ~4.74 ms (compute-bound)
- Practical with 80% utilization: ~5.9 ms

### Summary of Limits

| Scenario | Per-iter latency | vs Current |
|---|---|---|
| Current measured | 7.65 ms | baseline |
| Current theoretical floor (addmm+argmax) | 7.57 ms | -1% |
| Hypothetical fused AMX+argmin | ~5.9 ms | -23% |
| AMX compute bound (absolute floor) | 4.74 ms | -38% |
| H200 Flash-KMeans (measured from paper, extrapolated) | ~0.3-1.5 ms | 5-25x faster |

---

## 6. Honest Assessment

### What we have achieved
- The MLX implementation is within 1% of the theoretical floor given the current architectural constraints (materialized score matrix + separate argmax).
- 160-517x faster than sklearn CPU on the same machine.
- All 39 experiments systematically confirmed that the remaining headroom is near zero for the current approach.

### What we cannot achieve
- **Matching H200 Flash-KMeans is impossible.** The 37x compute gap (27 TFLOPS vs 990 TFLOPS) and 6-12x bandwidth gap (400 GB/s vs 4.8 TB/s) are fundamental hardware limits. Even a perfect implementation on M3 Ultra would be 5-25x slower depending on the regime.
- **Fusing matmul+argmin is impossible with current MLX.** The AMX accelerator is a black box - it takes input matrices, produces output matrices. There is no way to intercept intermediate tile results for online reduction. Custom Metal compute shaders that could do fusion are 6x slower at matmul than AMX.
- **Lower precision (FP8/INT4) matmul is blocked.** Apple Silicon M3 does not support `mx.qqmm` quantized matmul.

### The real gap is hardware, not software
The 5-30x gap to H200 Flash-KMeans maps almost exactly to the hardware FLOPS ratio. The Triton implementation's algorithmic innovations (FlashAssign, sort-inverse) provide an additional ~2-20x on top of raw hardware advantage by eliminating memory overhead. But the MLX implementation has already compensated for the equivalent overhead where possible:
- addmm fuses the bias addition (partial FlashAssign equivalent)
- centroid update is fully overlapped with compute
- multi-iteration compilation eliminates Python overhead

The remaining software gap is approximately 20-30% (the argmax pass that could be eliminated with hypothetical fused AMX+argmin). Everything beyond that is hardware.

### Recommendations

1. **For users**: The current implementation at 7.65 ms/iter for N=500K, K=1000, D=128 is at the practical hardware limit of M3 Ultra. Focus on algorithmic improvements (fewer iterations, better initialization) rather than per-iteration speed.

2. **For MLX framework developers**: The single highest-value primitive would be a fused `matmul_argmax` or `matmul_argmin` operation that leverages AMX for the matmul and applies the reduction without materializing the full output. This would save 20-30% on K-Means and likely benefit other workloads (attention, retrieval).

3. **For future Apple Silicon**: If Apple adds programmable shared memory or streaming AMX output support in M4/M5, the FlashAssign approach becomes viable and could close the remaining 20-30% gap.

4. **For large K workloads (K > 5000)**: The materialization overhead grows with K. K-tiled approaches with running argmax would become competitive when the score matrix far exceeds L2 cache. Worth revisiting for K > 10000.
