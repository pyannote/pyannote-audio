import numpy as np

from pyannote.audio.pipelines.clustering import AgglomerativeClustering, KMeansGPU


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
    print(clusters)
    assert np.array_equal(clusters, np.array([0, 1]))


def test_kmeans_clustering_num_cluster_gpu_too_small():
    clustering = KMeansGPU().instantiate({})

    embeddings = np.array([[1.0, 1.0, 1.0, 1.0], [1.0, 2.0, 1.0, 2.0]])

    # request 2 clusters
    clusters = clustering.cluster(
        embeddings=embeddings, min_clusters=2, max_clusters=2, num_clusters=2
    )

    assert np.array_equal(clusters, np.array([0, 1]))

    # generate a 256-dimensional random vector
    v = np.random.rand(256)

    # define the range and standard deviation of the generated cluster center
    cluster_center_std_dev = 2.0

    # generate 8 cluster centers randomly
    num_clusters = 8
    cluster_centers = np.random.normal(
        np.mean(v), cluster_center_std_dev, size=(num_clusters, 256)
    )

    a, b, c = clustering.set_num_clusters(num_clusters, 10, 1, 10)
    assert a == 8

    a, b, c = clustering.set_num_clusters(num_clusters, None, 1, 10)
    assert a is None and b == 1 and c == 8

    a, b, c = clustering.set_num_clusters(num_clusters, None, 8, 10)
    assert a == 8

    a, b, c = clustering.set_num_clusters(num_clusters, None, 7, 10)
    assert a is None and b == 7 and c == 8

    clustering.cluster(
        embeddings=cluster_centers, num_clusters=a, min_clusters=b, max_clusters=c
    )


def test_kmeans_clustering_num_cluster_gpu_large():
    clustering = KMeansGPU().instantiate({})

    # generate a 256-dimensional random vector
    v = np.random.rand(256)

    # define the range and standard deviation of the generated cluster center
    cluster_center_std_dev = 2.0
    vector_std_dev = 1

    # generate 5 cluster centers randomly
    num_clusters = 5
    cluster_centers = np.random.normal(
        np.mean(v), cluster_center_std_dev, size=(num_clusters, 256)
    )

    # generate 2000 * 32 vectors
    num_vectors_per_cluster = int(2000 * 32 / num_clusters)
    all_vectors = []

    for center in cluster_centers:
        vectors = np.random.normal(
            center, vector_std_dev, size=(num_vectors_per_cluster, 256)
        )
        all_vectors.append(vectors)

    # stack all vectors
    all_vectors = np.vstack(all_vectors)

    np.random.shuffle(all_vectors)

    clusters = clustering.cluster(
        embeddings=all_vectors, min_clusters=2, max_clusters=10
    )
    assert np.unique(clusters).shape[0] == num_clusters
