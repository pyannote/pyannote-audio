import numpy as np

from pyannote.audio.pipelines.clustering import AgglomerativeClustering


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
