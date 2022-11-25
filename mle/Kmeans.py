import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from numpy.random import uniform
from sklearn.datasets import make_blobs
import seaborn as sns
import random
from metrics import kmean_euclidean



class KMeans:
    def __init__(self, n_clusters = 6, max_iter = 100) -> None:
        self.centroids = []
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.c2l = None


    def fit(self, X, y):
        """
        Train KMean model until centroids converge or iteration reaches the maximal limit.

        Parameters:
        -----------
        X: (N, M)
          N: number of samples
        Returns:
        --------
        None
        """
        self.initialize_plus(X)
        iter = 0
        clusters = [[] for _ in range(self.n_clusters)]
        prev_centroids = None
        # np.not_equal(self.centroids, prev_centroids).any()
        while np.not_equal(self.centroids, prev_centroids).any() and iter < self.max_iter:
            prev_centroids = self.centroids
            for x in X:
                dists = kmean_euclidean(x, self.centroids)
                centroid_idx = np.argmin(dists)
                clusters[centroid_idx].append(x)

            Dists = np.apply_along_axis(kmean_euclidean, 1, X, *[self.centroids]) # X[i] and self.centroids are sent into kmean_euclidean
            X_centroid_idx = np.argmin(Dists, axis=1)
            centroids2 = [np.mean(X[X_centroid_idx==i], axis = 0) for i in range(self.n_clusters)]
            self.centroids = [np.mean(cluster, axis = 0) for cluster in clusters]

            for i, centroid in enumerate(self.centroids):
                if np.isnan(centroid).any():
                    self.centroids[i] = prev_centroids[i]
            iter += 1
        
        self.centroid2label(X_centroid_idx, y)

    def initialize(self, X):
        _min, _max = np.min(X, axis = 0), np.max(X, axis = 0)
        self.centroids = np.random.uniform(low=_min, high = _max, size = self.n_clusters)

    def initialize_plus(self, X):
        """
        Pick a random point from train data for first centroid, iteratively choose the next centroid w/ probabilities proportional to the mean dist from current centroids, until the number of centroids reach the maximum limit.

        Parameters:
        -----------
        X: (N, M)
        
        Returns:
        --------
        None
        """
        # pick a random datapoint as the first, the rest are initialized w/ probabilities proportional to their distances to the first

        # Pick a random point from train data for first centroid, iteratively choose the next centroid w/ probabilities proportional to the mean dist from current centroids, until the number of centroids reach the maximum limit.
        first = X[np.random.choice(len(X))]
        self.centroids.append(first)
        for i in range(self.n_clusters - 1):
        
            # calculate dists from points to the centroids
            dists = np.mean([kmean_euclidean(centroid, X) for centroid in self.centroids], axis = 0)
            # Normalize the distances
            dists = dists / np.sum(dists)
            # choose remaining points based on their distances
            new_centroids_idx = np.random.choice(int(X.shape[0]), p = dists)
            self.centroids.append(X[new_centroids_idx])
    
    def accuracy(self, labels, true_labels):

        return np.sum(labels==true_labels)/len(true_labels)
    
    def evaluate(self, X):
        """
        
        Parameters:
        -----------
        X: (N, M)

        Returns:
        --------
        centroids: (N, )
            the centroid for each input sample
        centroid_idx: (N, )
            the clusters' idx samples assigned to.
        """

        centroids, centroid_idxes = [], []
        for x in X:
            dists = kmean_euclidean( x, self.centroids)
            centroid_idx = np.argmin(dists)
            centroids.append(self.centroids[centroid_idx])
            centroid_idxes.append(centroid_idx)

        X_dists = np.apply_along_axis(kmean_euclidean, 1, X, *[self.centroids]) #(N, K)
        X_centroid_idx = np.argmin(X_dists, axis = 1) #(N, )
        self.centroid = [np.mean(X[X_centroid_idx==i], axis = 0) for i in range(self.n_clusters)]
        labels = self.c2l[X_centroid_idx]
        return centroids, centroid_idxes, labels

    def centroid2label(self, centroid_idxs, true_labels):
        
        self.c2l = [0 for _ in range(self.n_clusters)]
        for i in range(self.n_clusters):
            majority = np.bincount(true_labels[centroid_idxs == i])
            self.c2l[i] = np.argmax(majority)
        self.c2l = np.array(self.c2l)

        #majorities = np.bincount(true_labels[centroid_idxs == i] for i in range(self.n_clusters))
        #self.c2l = np.argmax(majorities, axis = 1)




def show_result(X_train, model, true_labels):
        class_centers, centroids, classification = model.evaluate(X_train)
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

def show_data(X_train, labels):
    sns.scatterplot( x = [x[0] for x in X_train],
                    y = [x[1] for x in X_train],
                    hue = labels,
                    palette="deep",
                    legend=None)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

if __name__ == "__main__":

    centers = 5
    X_train, true_labels = make_blobs(n_samples = 100, centers = centers, random_state=42)
    X_train = StandardScaler().fit_transform(X_train)

    # Fit centroids to dataset
    kmeans = KMeans(n_clusters=centers)
    kmeans.fit(X_train, true_labels)

    _, centroids, labels = kmeans.evaluate(X_train)
    #accu = kmeans.accuracy(labels, true_labels)
    show_result(X_train, kmeans, true_labels)
    accu = kmeans.accuracy(labels, true_labels)
    print("")


    

    