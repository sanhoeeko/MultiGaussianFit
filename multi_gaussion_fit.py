# Import necessary libraries
import numpy as np
from scipy.optimize import curve_fit
import warnings


# auxiliary function
def weighted_mean(points: list[list]):
    arr = np.array(points)
    x = arr[:, 0]
    w = arr[:, 1]
    return np.dot(w, x) / np.sum(w)


# KMeans class: Written by Cursor AI
class KMeans:
    def __init__(self, n_clusters, max_iter=300):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids = []

    def fit(self, X, weight):
        # Initialize centroids randomly
        self.centroids = X[np.random.choice(range(len(X)), self.n_clusters, replace=False)]

        # Iterate until convergence or max iterations reached
        for i in range(self.max_iter):
            # Assign each point to the closest centroid
            clusters = [[] for _ in range(self.n_clusters)]
            for x, y in zip(X, weight):
                distances = [np.linalg.norm(x - c) for c in self.centroids]
                closest_cluster = np.argmin(distances)
                clusters[closest_cluster].append([x, y])

            # Update centroids to be the mean of their cluster
            prev_centroids = self.centroids
            self.centroids = []
            for cluster in clusters:
                if cluster:
                    self.centroids.append(weighted_mean(cluster))
                else:
                    # If a centroid has no points, randomly reinitialize it
                    self.centroids.append(X[np.random.choice(range(len(X)), 1)][0])

            # If centroids have not moved, convergence has been reached
            if np.allclose(prev_centroids, self.centroids):
                break

    def predict(self, X):
        # Assign each point to the closest centroid
        clusters = [[] for _ in range(self.n_clusters)]
        for x in X:
            distances = [np.linalg.norm(x - c) for c in self.centroids]
            closest_cluster = np.argmin(distances)
            clusters[closest_cluster].append(x)

        # Return the cluster assignments
        return [i for i, cluster in enumerate(clusters) for _ in cluster]

    def get_center(self):
        return self.centroids


def gauss(x, A, mu, sigma):
    return A * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))


def linear_transform(x):
    b = np.max(x)
    a = np.min(x)
    u = 1 / (b - a)
    v = -a / (b - a)
    y = u * x + v
    return y, (u, v)


def inv_linear_transform(u, v, y):
    return (y - v) / u


class FitResult:
    def __init__(self, res, cov):
        self.coefs = res
        self.uncertain = np.max(cov)  # it is not a valid value, but it reflects the uncertainty
        if self.uncertain > 1:
            warnings.warn("multi_gaussian_fit: Uncertainty is large.")

    def func(self):
        def target(x):
            return sum([gauss(x, *list(i)) for i in self.coefs])

        return target


def multi_gaussian_fit(x, y, peaks):
    x, (a, b) = linear_transform(x)  # normalize
    maxy = max(y)  # normalize, if not, it doesn't converge
    y /= maxy
    km = KMeans(n_clusters=peaks)
    km.fit(x, y)
    centers = km.get_center()
    print(centers)
    p0 = np.ones((peaks, 3))
    p0[:, 0] = np.array([1, 1, 1])
    p0[:, 2] = np.array([0.1, 0.1, 0.1])
    p0[:, 1] = np.array(centers)
    p0 = p0.reshape(-1)

    def target(x, *param):
        a = np.array(param).reshape((-1, 3))
        return sum([gauss(x, *list(i)) for i in a])

    res, cov = curve_fit(target, x, y, p0=p0, method='dogbox')
    res = res.reshape((-1, 3))
    res[:, 0] *= maxy
    res[:, 1:] = inv_linear_transform(a, b, res[:, 1:])
    return FitResult(res, cov)


# test
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    x = np.arange(0, 10, 0.01)
    eps = np.random.normal(0, 1, x.shape)
    y = gauss(x, 10, 1, 0.3) + gauss(x, 4, 4, 0.8) + gauss(x, 3, 8, 2) + eps
    plt.plot(x, y)

    res = multi_gaussian_fit(x, y, 3)
    print(res.coefs)
    y_pred = res.func()(x)
    plt.plot(x, y_pred)
    plt.show()
