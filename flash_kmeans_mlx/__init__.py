"""
flash_kmeans_mlx - Fast batched K-Means clustering on Apple Silicon via MLX.

Port of flash-kmeans (Triton/PyTorch) to pure MLX.
No torch dependency.
"""

from flash_kmeans_mlx.kmeans import (
    batch_kmeans_Euclid,
    batch_kmeans_Cosine,
    batch_kmeans_Dot,
)
from flash_kmeans_mlx.interface import FlashKMeans
from flash_kmeans_mlx.ops import (
    euclid_assign,
    cosine_assign,
    dot_assign,
    centroid_update_euclid,
    centroid_update_cosine,
    l2_normalize,
)

__version__ = "0.1.0"

__all__ = [
    "batch_kmeans_Euclid",
    "batch_kmeans_Cosine",
    "batch_kmeans_Dot",
    "FlashKMeans",
    "euclid_assign",
    "cosine_assign",
    "dot_assign",
    "centroid_update_euclid",
    "centroid_update_cosine",
    "l2_normalize",
]
