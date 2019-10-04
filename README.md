# Equal Groups K-Means clustering
"""Equal Groups K-Means clustering utlizing the scikit-learn api and related utilities."""

From:
https://github.com/ndanielsen/Same-Size-K-Means/blob/master/clustering/equal_groups.py

And made to work with python3

"""Equal Groups K-Means clustering
90 percent of this is the Kmeans implmentations with the equal groups logic
located in `_labels_inertia_precompute_dense()` which follows the steps laid
out in the Elki Same-size k-Means Variation tutorial.
https://elki-project.github.io/tutorial/same-size_k_means
Please note that this implementation only works in scikit-learn 17.X as later
versions having breaking changes to this implementation.
Parameters
----------
n_groups : int, optional, default: 8
    The number of clusters to form as well as the number of
    centroids to generate.
max_iter : int, default: 300
    Maximum number of iterations of the k-means algorithm for a
    single run.
n_init : int, default: 10
    Number of time the k-means algorithm will be run with different
    centroid seeds. The final results will be the best output of
    n_init consecutive runs in terms of inertia.
init : {'k-means++', 'random' or an ndarray}
    Method for initialization, defaults to 'k-means++':
    'k-means++' : selects initial cluster centers for k-mean
    clustering in a smart way to speed up convergence. See section
    Notes in k_init for more details.
    'random': choose k observations (rows) at random from data for
    the initial centroids.
    If an ndarray is passed, it should be of shape (n_groups, n_features)
    and gives the initial centers.
precompute_distances : {'auto', True, False}
    Precompute distances (faster but takes more memory).
    'auto' : do not precompute distances if n_samples * n_groups > 12
    million. This corresponds to about 100MB overhead per job using
    double precision.
    True : always precompute distances
    False : never precompute distances
tol : float, default: 1e-4
    Relative tolerance with regards to inertia to declare convergence
random_state : integer or numpy.RandomState, optional
    The generator used to initialize the centers. If an integer is
    given, it fixes the seed. Defaults to the global numpy random
    number generator.
verbose : int, default 0
    Verbosity mode.
copy_x : boolean, default True
    When pre-computing distances it is more numerically accurate to center
    the data first.  If copy_x is True, then the original data is not
    modified.  If False, the original data is modified, and put back before
    the function returns, but small numerical differences may be introduced
    by subtracting and then adding the data mean.

Attributes
----------
cluster_centers_ : array, [n_groups, n_features]
    Coordinates of cluster centers
labels_ :
    Labels of each point
inertia_ : float
    Sum of distances of samples to their closest cluster center.

Notes
------
The k-means problem is solved using Lloyd's algorithm.
The average complexity is given by O(k n T), were n is the number of
samples and T is the number of iteration.
The worst case complexity is given by O(n^(k+2/p)) with
n = n_samples, p = n_features. (D. Arthur and S. Vassilvitskii,
'How slow is the k-means method?' SoCG2006)
In practice, the k-means algorithm is very fast (one of the fastest
clustering algorithms available), but it falls in local minima. That's why
it can be useful to restart it several times.
See also
--------
MiniBatchKMeans:
    Alternative online implementation that does incremental updates
    of the centers positions using mini-batches.
    For large scale learning (say n_samples > 10k) MiniBatchKMeans is
    probably much faster to than the default batch implementation.
"""