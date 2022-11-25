import numpy as np


def distance(X, data):

    return np.linalg.norm(data-X, ord=2, axis = 1)

class Kmean:
    def __init__(self, n_clusters, max_iter):

        self.max_iter = max_iter
        self.n_clusters = n_clusters
        self.centroids = []

    def initialize_simple(self, X):

        self.centroids = np.random.uniform(0, len(X)-1, self.n_clusters)
        
    def initialize_plus(self, X):
        N, M = X.shape
        self.centroids.append(X[np.random.choice(N)])

        while len(self.centroids) <= self.n_clusters:
            dists = np.apply_along_axis(distance, 1, X, *[self.centroids])
            aver_dists = np.mean(dists, axis=1)
            #centroid = X[np.argmin(aver_dists)]
            centroid_idx = np.random.choice(len(X),p = aver_dists/np.sum(aver_dists)) # size = 1
            self.centroids.append(X[centroid_idx])


    def fit(self, X):

        # iterate with max_iter:
        # compute dists for X to any centorids (N, K)
        # compute mean for each cluster which becomes new centroids

        # update centroid

        
        self.initialize_plus(X)
        prev_centroids = self.centroids
        iter = 0
        while iter <= self.max_iter:
            dists = np.apply_along_axis(distance, 1, X, *[self.centroids])
            centroids_idx = np.argmin(dists, axis=1)
            self.centroids = [np.mean(X[centroids_idx == i], axis = 0) for i in range(self.n_clusters)]

            for i in range(self.n_clusters):
                if self.centroids[i] is np.NaN:
                    self.centroids[i] = prev_centroids[i]

            prev_centroids = self.centroids
            iter += 1
        
    def evaluate(self, X):

        clusters = [[] for _ in range(self.n_clusters)]
        dists = np.apply_along_axis(distance, 1, X, *[self.centroids])
        labels = np.argmin(dists, axis = 1)

        for i in range(self.n_clusters):
            clusters[i].extend(X[labels == i])
        
        return [self.centroids[i] for i in labels], labels

    def accuracy(self, labels, true_labels):

        return np.sum(labels==true_labels)/len(true_labels)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from numpy.random import uniform
from sklearn.datasets import make_blobs
import seaborn as sns
import random
from metrics import kmean_euclidean
def show_result(X_train, model, true_labels):
        class_centers, classification = model.evaluate(X_train)
        sns.scatterplot(x=[X[0] for X in X_train],
                        y=[X[1] for X in X_train],
                        hue=true_labels,
                        style=classification,
                        palette="deep",
                        legend=None
                        )
        plt.plot([x for x, _ in model.centroids],
                [y for _, y in model.centroids],
                '+',
                markersize=10,
                )
        plt.show()
        print("Done")

if __name__ == "__main__":

    centers = 5
    X_train, true_labels = make_blobs(n_samples = 100, centers = centers, random_state=42)
    X_train = StandardScaler().fit_transform(X_train)

    # Fit centroids to dataset
    kmeans = Kmean(centers, 100)
    kmeans.fit(X_train)

    _, labels = kmeans.evaluate(X_train)
    accu = kmeans.accuracy(labels, true_labels)
    show_result(X_train, kmeans, true_labels)