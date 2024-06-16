import numpy as np

from pyannote.audio.pipelines.clustering import (
    AgglomerativeClustering,
    AgglomerativeClusteringGPU,
)


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


def test_agglomerative_clustering_num_cluster_gpu():
    clustering = AgglomerativeClusteringGPU().instantiate(
        {"method": "single", "min_cluster_size": 0}
    )

    embeddings = np.array([[1.0, 1.0, 1.0, 1.0], [1.0, 2.0, 1.0, 2.0]])

    # request 2 clusters
    clusters = clustering.cluster(
        embeddings=embeddings, min_clusters=2, max_clusters=2, num_clusters=2
    )
    print(clusters)
    assert np.array_equal(clusters, np.array([0, 1]))

    # 生成一个256维的随机向量
    v = np.random.rand(256)

    # 定义簇中心的生成范围和标准差
    cluster_center_std_dev = 2.0  # 可以根据需要调整
    vector_std_dev = 1  # 可以根据需要调整

    # 随机生成5个簇中心
    num_clusters = 5
    cluster_centers = np.random.normal(
        np.mean(v), cluster_center_std_dev, size=(num_clusters, 256)
    )

    # 在每个簇中心附近生成20个向量
    num_vectors_per_cluster = 100
    all_vectors = []

    for center in cluster_centers:
        vectors = np.random.normal(
            center, vector_std_dev, size=(num_vectors_per_cluster, 256)
        )
        all_vectors.append(vectors)

    # 将所有向量合并到一个数组中
    all_vectors = np.vstack(all_vectors)

    np.random.shuffle(all_vectors)

    clusters = clustering.cluster(
        embeddings=all_vectors, min_clusters=2, max_clusters=10
    )
    assert np.unique(clusters).shape[0] == num_clusters
