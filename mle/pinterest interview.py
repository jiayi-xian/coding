# pinterest interview
import random
from scipy.spatial.distance import cdist

import numpy as np


y = np.array([1,0,1,2,3])
label = np.array([1,0,1,2,3])
accu = np.sum(y==label)


def calculate_distances(data, centroids):
    """
    Step 1: Calculate distance between each data point and the k centroids
    """
    return cdist(data, centroids, "cosine").tolist()


def make_clusters(distances):
    """
    Step 2: Assign each data point to it's nearest centroid
    """
    # implement this
    import numpy as np
    dists = np.array(distances)
    # collect centroids
    N, n_cluster = dists.shape
    clusters = dists.argmin(axis=1)
    
    return clusters.tolist()


def update_clusters(clusters, data, k, iterations):
    """
    Step 3: Average the data points in each cluster to update
    the centroids' locations and repeat for set number of iterations
    """
    # implement this
    import numpy as np
    iter = 0
    X = np.array(data)
    n_clusters = len(set(clusters))
    clusters = np.array(clusters)
    centroids = [np.mean(X[clusters==i], axis = 0) for i in range(n_clusters)]
    while iter < iterations:
        prev_centroids = centroids
        Dists = calculate_distances(X, centroids)
        Dists = np.array(Dists)
        clusters = np.argmin(Dists, axis=1)
        centroids = [np.mean(X[clusters==i], axis = 0) for i in range(n_clusters)]

        prev_centroids = centroids
        iter += 1
        # self.centroids = [np.mean(cluster, axis = 0) for cluster in clusters]

    
    return clusters.tolist()

# pull everything together
def solution(data, k, centroids, iterations):
    distances = calculate_distances(data, centroids)
    clusters = make_clusters(distances)
    clusters = update_clusters(clusters, data, k, iterations)
    return clusters