import numpy as np

def get_random_centroids(X, k):
    '''
    Each centroid is a point in RGB space (color) in the image. 
    This function should uniformly pick `k` centroids from the dataset.A
    Input: a single image of shape `(num_pixels, 3)` and `k`, the number of centroids. 
    Notice we are flattening the image to a two dimentional array.
    Output: Randomly chosen centroids of shape `(k,3)` as a numpy array. 
    '''
    centroids = []
    # Randomly pick `k` unique indices from the dataset
    random_indices = np.random.choice(X.shape[0], k, replace=False)
    # Select the centroids using the random indices
    centroids = X[random_indices]
    # make sure you return a numpy array
    return np.asarray(centroids).astype(np.float64) 

def lp_distance(X, centroids, p=2):
    '''
    Inputs: 
    A single image of shape (num_pixels, 3)
    The centroids (k, 3)
    The distance parameter p

    output: numpy array of shape `(k, num_pixels)` thats holds the distances of 
    all points in RGB space from all centroids
    '''
    distances = []
    k = len(centroids)
    # Calculate the distance between each pixel and each centroid
    centroids_dif = np.abs(X[np.newaxis] - centroids[:, np.newaxis])
    # Calculate the Lp distance
    distances = np.sum(centroids_dif**p, axis=2)**(1/p)
    return distances


def kmeans(X, k, p ,max_iter=100):
    """
    Inputs:
    - X: a single image of shape (num_pixels, 3).
    - k: number of centroids.
    - p: the parameter governing the distance measure.
    - max_iter: the maximum number of iterations to perform.

    Outputs:
    - The calculated centroids as a numpy array.
    - The final assignment of all RGB points to the closest centroids as a numpy array.
    """
    classes = []
    centroids = get_random_centroids(X, k)
    for _ in range(max_iter):
        # Calculate the distance between each pixel and each centroid
        distance = lp_distance(X, centroids, p)
        # Assign each pixel to the closest centroid
        classes = np.argmin(distance, axis=0)
        previous_centroids = centroids
        # Update the centroids
        centroids = np.array([np.mean(X[classes == i, :], axis=0) for i in range(k)])
        # If the centroids didn't change, break
        if np.all(previous_centroids ==centroids):
            break
    return centroids, classes

def kmeans_pp(X, k, p ,max_iter=100):
    """
    Your implenentation of the kmeans++ algorithm.
    Inputs:
    - X: a single image of shape (num_pixels, 3).
    - k: number of centroids.
    - p: the parameter governing the distance measure.
    - max_iter: the maximum number of iterations to perform.

    Outputs:
    - The calculated centroids as a numpy array.
    - The final assignment of all RGB points to the closest centroids as a numpy array.
    """
    classes = []
    # Step 1: Choose a centroid uniformly at random among the data points
    centroids = np.array([X[np.random.choice(X.shape[0])]])

    # Step 2 and 3: Choose k-1 new centroids using the kmeans++ algorithm
    for _ in range(k-1):
        # Compute the distance between each data point and the nearest centroid
        min_distances = np.min(lp_distance(X, centroids, p), axis=0)
        # Compute the probability distribution for choosing the next centroid
        probability = min_distances**2 / np.sum(min_distances**2)
        # Choose a new centroid based on the probability distribution
        new_centroid = X[np.random.choice(X.shape[0], p=probability)]
        # Add the new centroid to the list of centroids
        centroids = np.vstack((centroids, new_centroid))
    # Step 5: Perform standard k-means clustering using the founded centroids
    for _ in range(max_iter):
        distance = lp_distance(X, centroids, p)
        classes = np.argmin(distance, axis=0)
        new_centroids = np.array([np.mean(X[classes == i, :], axis=0) for i in range(k)])
        if np.all(new_centroids == centroids):
            break
        centroids = new_centroids
    return centroids, classes