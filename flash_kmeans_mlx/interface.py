"""
FlashKMeans high-level interface for MLX.

API-compatible with flash_kmeans.interface.FlashKMeans.
"""

from __future__ import annotations

from typing import Optional
import mlx.core as mx
from flash_kmeans_mlx.kmeans import batch_kmeans_Euclid, batch_kmeans_Cosine, batch_kmeans_Dot
from flash_kmeans_mlx.ops import euclid_assign


class FlashKMeans:
    """
    Fast batched K-Means clustering on Apple Silicon via MLX.

    Parameters
    ----------
    d : int
        Feature dimensionality.
    k : int
        Number of clusters.
    niter : int, default=25
        Maximum iterations.
    tol : float, default=1e-8
        Convergence tolerance on centroid shift.
    seed : int, default=0
        Random seed for centroid initialization.
    verbose : bool, default=False
        Whether to print per-iteration info.
    dtype : mx.Dtype, optional
        Compute dtype. Defaults to input dtype.
    metric : str, default='euclidean'
        Distance metric: 'euclidean', 'cosine', or 'dot'.
    compiled : bool, default=True
        Use mx.compile for the iteration hot loop.
    """

    def __init__(
        self,
        d: int,
        k: int,
        niter: int = 25,
        tol: float = 1e-8,
        seed: int = 0,
        verbose: bool = False,
        dtype: Optional[mx.Dtype] = None,
        metric: str = "euclidean",
        compiled: bool = True,
        max_mem_gb: float = 0,
    ):
        self.d = int(d)
        self.k = int(k)
        self.niter = int(niter)
        self.tol = float(tol)
        self.seed = int(seed)
        self.verbose = bool(verbose)
        self.dtype = dtype
        self.metric = metric
        self.compiled = compiled
        self.max_mem_gb = float(max_mem_gb)

        self.centroids_b: Optional[mx.array] = None
        self.cluster_ids_b: Optional[mx.array] = None
        self._batch_size: Optional[int] = None

    def train(self, data: mx.array) -> None:
        """
        Fit K-Means on data and store centroids.

        Parameters
        ----------
        data : mx.array
            Shape (N, D) or (B, N, D).
        """
        if data.ndim == 2:
            N, D = data.shape
            B = None
            x_b = mx.expand_dims(data, axis=0)  # (1, N, D)
        elif data.ndim == 3:
            B, N, D = data.shape
            x_b = data
        else:
            raise ValueError("data must be (N, D) or (B, N, D)")

        mx.random.seed(self.seed)

        compute_dtype = self.dtype or x_b.dtype
        x_b = x_b.astype(compute_dtype)

        kmeans_fn = {
            "euclidean": batch_kmeans_Euclid,
            "cosine": batch_kmeans_Cosine,
            "dot": batch_kmeans_Dot,
        }.get(self.metric)

        if kmeans_fn is None:
            raise ValueError(f"Unknown metric: {self.metric!r}. "
                             f"Use 'euclidean', 'cosine', or 'dot'.")

        extra_kwargs = {}
        if self.metric == "euclidean":
            extra_kwargs["max_mem_gb"] = self.max_mem_gb

        cluster_ids_b, centroids_b, _ = kmeans_fn(
            x_b,
            self.k,
            max_iters=self.niter,
            tol=self.tol,
            verbose=self.verbose,
            compiled=self.compiled,
            **extra_kwargs,
        )

        self.centroids_b = centroids_b
        self.cluster_ids_b = cluster_ids_b
        self._batch_size = B

    def fit(self, data: mx.array) -> "FlashKMeans":
        """Fit K-Means; returns self."""
        self.train(data)
        return self

    def predict(self, data: mx.array) -> mx.array:
        """
        Assign each point to the nearest centroid.

        Parameters
        ----------
        data : mx.array
            Shape (N, D) or (B, N, D). Batch size must match training.

        Returns
        -------
        labels : mx.array uint32
            (N,) or (B, N)
        """
        if self.centroids_b is None:
            raise RuntimeError("Model not trained. Call train() or fit() first.")

        if data.ndim == 2:
            B = None
            x_b = mx.expand_dims(data, axis=0)
        elif data.ndim == 3:
            B = data.shape[0]
            x_b = data
        else:
            raise ValueError("data must be (N, D) or (B, N, D)")

        if B != self._batch_size:
            raise ValueError(
                f"Model was trained with batch size B={self._batch_size}, "
                f"but predict received B={B}."
            )

        compute_dtype = self.dtype or x_b.dtype
        x_b = x_b.astype(compute_dtype)

        if self.metric == "euclidean":
            x_sq = (x_b.astype(mx.float32) * x_b.astype(mx.float32)).sum(axis=-1)
            x_f16 = x_b.astype(mx.float16)
            labels_b = euclid_assign(x_b, self.centroids_b, x_sq, x_f16=x_f16)
        elif self.metric in ("cosine", "dot"):
            from flash_kmeans_mlx.ops import l2_normalize, cosine_assign, dot_assign
            if self.metric == "cosine":
                x_n = l2_normalize(x_b, axis=-1)
                labels_b = cosine_assign(x_n, self.centroids_b)
            else:
                labels_b = dot_assign(x_b, self.centroids_b)
        else:
            raise ValueError(f"Unknown metric: {self.metric!r}")

        mx.eval(labels_b)

        if B is None:
            return labels_b.squeeze(0)
        return labels_b

    def fit_predict(self, data: mx.array) -> mx.array:
        """
        Fit and return cluster assignments.

        Returns
        -------
        labels : mx.array uint32
            (N,) or (B, N)
        """
        self.train(data)
        if self._batch_size is None:
            return self.cluster_ids_b.squeeze(0)
        return self.cluster_ids_b
