import numpy as np

from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import cdist

from FSDL.plate_recognizer.utils.logger import get_logger

logger = get_logger(__name__)


class Cluster():
    def __init__(self, dimensions=2):
        self.dimensions = dimensions

    def get_pca_reduced(self, X_train):
        X_train_flatten = X_train.reshape(X_train.shape[0], -1)
        pca = PCA(self.dimensions)

        X_train_pca_reduced = pca.fit_transform(X_train_flatten)

        return X_train_pca_reduced, pca
    
    def get_clusters(self, X_train_pca_reduced, K):
        kmeans = KMeans(n_clusters=K, random_state=0)
        X_train_pca_clusters = kmeans.fit(X_train_pca_reduced)

        return X_train_pca_clusters, kmeans

    def get_feature_map_clusters(self, X, K):
        """
        param X: input data
        param K: number of clusters
        returns: X_clusters - clustered input data

        (side effect): plots the frequency histogram of clusters
        """
        X_fm, _ = self.get_feature_maps(X)
        # use cosine distance to find similarities
        X_fm_normalized = preprocessing.normalize(X_fm.reshape(len(X_fm), -1))

        return self.get_clusters(X_fm_normalized)

    @staticmethod
    def to_cluster_idx(cluster_labels, bins):
        """
        param bins: range of K
        param labels: cluster labels
        returns: dictionary of cluster IDs
        """
        cluster_dict = dict()
        for cluster_id in bins:
            cluster_dict[cluster_id] = np.where(cluster_labels == cluster_id)[0]
        return cluster_dict

    @staticmethod
    def to_clusters_dict(X, y, X_clusters, K):
        """
        given X_clusters, put X & y into the correct clusters
        and return the dictionary
        """
        cluster_idx = to_cluster_idx(X_clusters.labels_, range(K))

        X_dict = {}
        y_dict = {}
        for id in range(K):
            ids = cluster_idx[id]
            X_dict[id] = X[ids]
            y_dict[id] = y[ids]

        return X_dict, y_dict

    @staticmethod
    def get_merged_data(clusters_d, id=-1):
        if id != -1:
            return clusters_d[id]
        else:
            merged = []
            for cluster_id, cluster in clusters_d.items():
                if cluster_id == 0:
                    merged = cluster
                else:
                    merged = np.hstack((merged, cluster))

            return merged

    @staticmethod
    def find_duplicates(X_train_pca):
        # Calculate distances of all points
        distances = cdist(X_train_pca, X_train_pca)

        # Find duplicates (very similar images)
        # dupes = np.array([np.where(distances[id] < 1) for id in range(distances.shape[0])]).reshape(-1)
        dupes = [np.array(np.where(distances[id] < 1)).reshape(-1).tolist() \
                for id in range(distances.shape[0])]

        to_remove = set()
        for d in dupes:
            if len(d) > 1:
                for id in range(1, len(d)):
                    to_remove.add(d[id])
        logger.info("Found {} duplicates".format(len(to_remove)))
        return to_remove
