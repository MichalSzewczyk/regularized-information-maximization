from sklearn.cluster import KMeans


def perform_kmeans_clustering(X, k):
    kmeans = KMeans(k)
    kmeans_result = kmeans.fit(X)
    return kmeans_result.labels_, kmeans_result.n_clusters
