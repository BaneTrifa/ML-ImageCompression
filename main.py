import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from PIL import Image
import os


def find_closest_centroids(X, centroids):
    """
        Determines the closest cluster to each pixel

        Args:
            X (ndarray): (H*W, color_ch) input image
            centroids (ndarray): (K, color_ch) centroids

        Returns:
            idx (array_like): (H*W,) closest centroids

        """

    # Calculate the closest centroid for each pixel
    distances = np.linalg.norm(X[:, np.newaxis, :] - centroids, axis=2)

    indexes = np.argmin(distances, axis=1)

    return indexes


def compute_centroids(X, idx, K):
    """
        Returns the new centroids by computing the means of the
        data points assigned to each centroid.

        Args:
            X (ndarray):   (H*W, color_ch) input image
            idx (ndarray): (H*W,) Array containing index of closest centroid for each
                           pixel in X. Concretely, idx[i] contains the index of
                           the centroid closest to example i
            K (int):       number of centroids

        Returns:
            centroids (ndarray): (K, n) New centroids computed
        """

    m, n = X.shape

    centroids = np.zeros((K, n))

    counter = [0] * K
    accumulator = np.zeros((K, n))

    for j in range(m):
        accumulator[idx[j]] += X[j]
        counter[idx[j]] += 1

    for i in range(K):
        if counter[i] != 0:
            centroids[i] = accumulator[i] / counter[i]

    return centroids


def k_means_init_centroids(X, K):
    """
        This function initializes K centroids that are to be used in K-Means on the image X

        Args:
            X (ndarray):   (H*W, color_ch) input image which we want to compress
            K (int):       number of clusters

        Returns:
                init_centroids (ndarray): (K, color_ch) list of initial centroids
    """

    shuffled_X = deepcopy(X)
    np.random.shuffle(shuffled_X)

    init_centroids = shuffled_X[:K, :]

    return init_centroids


def run_k_means(X, initial_centroids, max_iterations=10):
    """
    Runs the K-Means algorithm on data matrix X.

    Args:
        X (ndarray):   (H*W, color_ch) input image which we want to compress
        initial_centroids (ndarray(K, color_ch)):   array with initial centroids
        max_iterations (int):                       number of maximum iterations

    Returns:
         centroids (ndarray): (K, n) final list of centroids
    """

    K = initial_centroids.shape[0]
    centroids = initial_centroids

    # Main loop
    for i in range(max_iterations):
        # Output progress
        print("K-Means iteration %d/%d" % (i, max_iters - 1))

        indexes = find_closest_centroids(X, centroids)
        centroids = compute_centroids(X, indexes, K)

    return centroids


if __name__ == '__main__':

    dir_list = os.listdir(".//in_img")

    for i, image in enumerate(dir_list):

        original_img = plt.imread('.//in_img//' + image)

        print("Shape of original_img is:", original_img.shape)
        X_img = np.reshape(original_img, (original_img.shape[0] * original_img.shape[1], 3))

        K = 16
        max_iters = 10

        # Using the function you have implemented above.
        initial_centroids = k_means_init_centroids(X_img, K)

        # Run K-Means - returns position of every cluster centorid
        centroids = run_k_means(X_img, initial_centroids, max_iters)

        # Find the closest centroid of each pixel
        idx = find_closest_centroids(X_img, centroids)

        # Replace each pixel with the color of the closest centroid
        X_recovered = centroids[idx, :]

        # Reshape image into proper dimensions
        X_recovered = np.reshape(X_recovered, original_img.shape)

        # Save compressed image
        im = Image.fromarray((X_recovered * 255).astype(np.uint8))
        im.save(fr'./out_img/comppressed_{image}')