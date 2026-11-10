import numpy as np
from pyannote.core import SlidingWindow, SlidingWindowFeature

from pyannote.audio.pipelines.clustering import AgglomerativeClustering, VBxClustering


def test_agglomerative_clustering_num_cluster():
    """
    Make sure AgglomerativeClustering doesn't "over-merge" clusters when initial
    clustering already matches target num_clusters, cf
    https://github.com/pyannote/pyannote-audio/issues/1525
    """

    # 2 embeddings different enough
    embeddings = np.array([[1.0, 1.0, 1.0, 1.0], [1.0, 2.0, 1.0, 2.0]])

    # clustering with params that should yield 1 cluster per embedding
    clustering = AgglomerativeClustering().instantiate(
        {
            "method": "centroid",
            "min_cluster_size": 0,
            "threshold": 0.0,
        }
    )

    # request 2 clusters
    clusters = clustering.cluster(
        embeddings=embeddings, min_clusters=2, max_clusters=2, num_clusters=2
    )
    assert np.array_equal(clusters, np.array([0, 1]))


class IdentityPLDA:
    """Minimal stand-in for PLDA (identity transform into PLDA space)"""

    def __init__(self, dimension: int):
        self.phi = np.ones(dimension)

    def __call__(self, embeddings: np.ndarray) -> np.ndarray:
        return embeddings


def test_vbx_clustering_more_clusters_than_embeddings():
    """
    Make sure VBxClustering does not crash when more clusters are requested
    than there are embeddings to cluster (e.g. `num_speakers` larger than
    what a short file can provide) and caps the number of clusters instead.
    """

    num_chunks, num_frames, num_speakers, dimension = 3, 10, 1, 4

    # 3 (orthogonal) embeddings
    embeddings = np.eye(dimension)[:num_chunks, np.newaxis, :]
    segmentations = SlidingWindowFeature(
        np.ones((num_chunks, num_frames, num_speakers)),
        SlidingWindow(start=0.0, duration=10.0, step=1.0),
    )

    clustering = VBxClustering(plda=IdentityPLDA(dimension)).instantiate(
        {"threshold": 0.6, "Fa": 0.07, "Fb": 0.8}
    )

    # request 5 clusters
    hard_clusters, soft_clusters, centroids = clustering(
        embeddings,
        segmentations=segmentations,
        num_clusters=5,
        min_clusters=5,
        max_clusters=5,
    )

    assert centroids.shape == (num_chunks, dimension)
    assert soft_clusters.shape == (num_chunks, num_speakers, num_chunks)
    assert np.array_equal(np.sort(hard_clusters.ravel()), np.arange(num_chunks))
