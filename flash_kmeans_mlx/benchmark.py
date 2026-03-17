"""
Benchmark: flash_kmeans_mlx vs sklearn KMeans on Apple Silicon.

Usage:
    python -m flash_kmeans_mlx.benchmark
    python -m flash_kmeans_mlx.benchmark --n 100000 --d 128 --k 1000 --max-iters 20
"""

import argparse
import time
import numpy as np
import mlx.core as mx


def bench_mlx(X_mlx, K, max_iters, n_warmup=2, n_runs=5):
    """Benchmark mlx_kmeans.batch_kmeans_Euclid."""
    from flash_kmeans_mlx import batch_kmeans_Euclid

    x = X_mlx.reshape(1, X_mlx.shape[0], X_mlx.shape[1])  # (1, N, D)

    # Warmup
    for _ in range(n_warmup):
        ids, cents, ni = batch_kmeans_Euclid(x, K, max_iters=max_iters, compiled=True)
        mx.eval(ids, cents)

    # Timed runs
    times = []
    iters_list = []
    for _ in range(n_runs):
        mx.random.seed(int(time.time() * 1000) % (2**31))
        t0 = time.perf_counter()
        ids, cents, ni = batch_kmeans_Euclid(x, K, max_iters=max_iters, compiled=True)
        mx.eval(ids, cents)
        t1 = time.perf_counter()
        times.append(t1 - t0)
        iters_list.append(ni)

    # Compute inertia from last run
    ids_np = np.array(ids[0])
    cents_np = np.array(cents[0])
    X_np = np.array(X_mlx)
    diffs = X_np - cents_np[ids_np]
    inertia = float((diffs ** 2).sum())

    return {
        "times": times,
        "mean_time": np.mean(times),
        "std_time": np.std(times),
        "mean_iters": np.mean(iters_list),
        "inertia": inertia,
    }


def bench_sklearn(X_np, K, max_iters, n_runs=5):
    """Benchmark sklearn KMeans."""
    from sklearn.cluster import KMeans

    times = []
    iters_list = []
    inertia = None
    for _ in range(n_runs):
        t0 = time.perf_counter()
        km = KMeans(n_clusters=K, max_iter=max_iters, n_init=1,
                    random_state=None, algorithm="lloyd")
        km.fit(X_np)
        t1 = time.perf_counter()
        times.append(t1 - t0)
        iters_list.append(km.n_iter_)
        inertia = km.inertia_

    return {
        "times": times,
        "mean_time": np.mean(times),
        "std_time": np.std(times),
        "mean_iters": np.mean(iters_list),
        "inertia": float(inertia),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark mlx_kmeans vs sklearn")
    parser.add_argument("--n", type=int, default=50000, help="Number of points")
    parser.add_argument("--d", type=int, default=128, help="Dimensionality")
    parser.add_argument("--k", type=int, default=256, help="Number of clusters")
    parser.add_argument("--max-iters", type=int, default=20, help="Max iterations")
    parser.add_argument("--runs", type=int, default=5, help="Number of timed runs")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup runs for MLX")
    parser.add_argument("--skip-sklearn", action="store_true", help="Skip sklearn benchmark")
    args = parser.parse_args()

    N, D, K = args.n, args.d, args.k
    print(f"Config: N={N}, D={D}, K={K}, max_iters={args.max_iters}, runs={args.runs}")
    print(f"Data size: {N * D * 4 / 1e6:.1f} MB (float32)")
    print()

    # Generate data
    np.random.seed(0)
    X_np = np.random.randn(N, D).astype(np.float32)
    X_mlx = mx.array(X_np)
    mx.eval(X_mlx)

    # MLX benchmark
    print("--- mlx_kmeans (MLX GPU) ---")
    mlx_res = bench_mlx(X_mlx, K, args.max_iters, n_warmup=args.warmup, n_runs=args.runs)
    print(f"  Time: {mlx_res['mean_time']:.3f}s +/- {mlx_res['std_time']:.3f}s")
    print(f"  Avg iters: {mlx_res['mean_iters']:.1f}")
    print(f"  Inertia: {mlx_res['inertia']:.1f}")
    print(f"  Per-iter: {mlx_res['mean_time'] / mlx_res['mean_iters'] * 1000:.1f} ms")
    print()

    if not args.skip_sklearn:
        print("--- sklearn KMeans (CPU) ---")
        sk_res = bench_sklearn(X_np, K, args.max_iters, n_runs=args.runs)
        print(f"  Time: {sk_res['mean_time']:.3f}s +/- {sk_res['std_time']:.3f}s")
        print(f"  Avg iters: {sk_res['mean_iters']:.1f}")
        print(f"  Inertia: {sk_res['inertia']:.1f}")
        print(f"  Per-iter: {sk_res['mean_time'] / sk_res['mean_iters'] * 1000:.1f} ms")
        print()

        speedup = sk_res["mean_time"] / mlx_res["mean_time"]
        print(f"=== Speedup: mlx_kmeans is {speedup:.2f}x {'faster' if speedup > 1 else 'slower'} than sklearn ===")
    else:
        print("(sklearn benchmark skipped)")


if __name__ == "__main__":
    main()
