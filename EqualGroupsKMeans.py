import numpy as np


def equal_group_kmeans(X, n_groups, max_iter=10):
    """Calculate equal sized groups k-means clustering."""

    # Init cluster centers by randomly sampling datapoints
    centers = X[np.random.choice(len(X), n_groups, replace=False)]

    # Precalculate n_samples
    n_samples = X.shape[0]

    # Init distance storage
    distances = np.zeros(shape=(n_samples,))

    # Init class label array and minimum distance array
    labels = -1 * np.ones(n_samples, dtype=np.int32)
    mindist = np.infty * np.ones(n_samples)

    # Initialize placeholder return values
    best_labels, best_inertia, best_centers = None, None, None

    # Iterate
    for i in range(max_iter):
        # Store the old cluster centroids
        centers_old = centers.copy()

        # Calculate Euclidean distances between point/centers pairs
        all_distances = ((centers * centers).sum(axis=1)[:, np.newaxis] -
                        2 * np.dot(centers, X.T) +
                        (X * X).sum(axis=1)[np.newaxis, :])

        for point_id in np.arange(n_samples):
            # Sorts the points by distance
            sorted_points = sorted([(a, b) for a, b in enumerate(all_distances[:, point_id])],
                                key=lambda x: x[1])

            # Initial assignment of labels and mindist
            for cluster_id, point_dist in sorted_points:
                if not (len(np.where(labels==cluster_id)[0]) >= n_samples//n_groups):
                    labels[point_id], mindist[point_id] = cluster_id, point_dist
                    break

        # Refine Clustering
        transfer_list = []
        best_mindist = mindist.copy()
        best_labels = labels.copy()

        # Iterate through points by distance (highest to lowest)
        for point_id in np.argsort(mindist)[::-1]:
            point_cluster = labels[point_id]

            # see if there is an opening on the best cluster for this point
            cluster_id, point_dist = sorted([(a, b) for a, b in enumerate(all_distances[:, point_id])],
                                            key=lambda x: x[1])[0]

            if not ((len(np.where(labels==cluster_id)[0]) >= n_samples//n_groups) and
                (point_cluster != cluster_id)):
                labels[point_id], mindist[point_id] = cluster_id, point_dist
                best_labels, best_mindist = labels.copy(), mindist.copy()
                continue

            # iterate through candidates in the transfer list
            for swap_candidate_id in transfer_list:
                if point_cluster != labels[swap_candidate_id]:
                    # get the current distance of swap candidate
                    cand_distance = mindist[swap_candidate_id]

                    # get the potential distance of point
                    point_distance = all_distances[labels[swap_candidate_id], point_id]

                    # proceeed if transfer will improve distance
                    if point_distance < cand_distance:
                        labels[point_id] = labels[swap_candidate_id]
                        mindist[point_id] = all_distances[labels[swap_candidate_id], point_id]
                        labels[swap_candidate_id] = point_cluster
                        mindist[swap_candidate_id] = all_distances[point_cluster, swap_candidate_id]

                        if np.abs(mindist).sum() <  np.abs(best_mindist).sum():
                            # update the labels since the transfer was a success
                            best_labels, best_mindist = labels.copy(), mindist.copy()
                            break
                        else:
                            # reset since the transfer was not a success
                            labels, mindist = best_labels.copy(), best_mindist.copy()

            # Add point to transfer list
            transfer_list.append(point_id)

        # Calculate final intertia
        inertia = best_mindist.sum()
        labels = best_labels

        # Recalculate centers
        centers = []
        for group_id in range(n_groups):
            centers.append(np.mean(X[np.where(best_labels == group_id)[0]], 0))
        centers = np.array(centers)

        # UPDATE RETURN VALUES
        if best_inertia is None or inertia < best_inertia:
            best_labels = labels.copy()
            best_centers = centers.copy()
            best_inertia = inertia

    # Test group sizes and raise AssertionError on non-uniform group sizes
    best_group_sizes = [len(np.where(best_labels == group_id)[0]) for
                        group_id in range(n_groups)]
    assert best_group_sizes == [len(X) // n_groups] * n_groups
    return best_centers, best_labels, best_inertia, best_group_sizes, i + 1