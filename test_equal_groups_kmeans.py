import numpy as np
from collections import Counter
from sklearn.datasets.samples_generator import make_blobs
from EqualGroupsKMeans import EqualGroupsKMeans


def compute_equal_group_kmeans(data, n_groups):
  """Wrapper for computing equal size group kmeans."""

  clf = EqualGroupsKMeans(n_clusters=n_groups)
  clf.fit(data)
  labels = clf.labels_

  # Test group sizes and throw AssertionError on failure
  C = Counter(labels)
  group_sizes = list(C.values())
  try:
    n_members = len(data) // n_groups
    assert group_sizes == [n_members] * n_groups
  except AssertionError as e:
    print('Unequal group sizes: {}'.format(group_sizes))
    labels = []

  # Return the equal group size label list
  return labels, group_sizes


for random_state in [1017]:
    for test_n in [32]:
      for test_groups in [4]:
        n_samples = test_n
        n_groups = test_groups
        data, y_true = make_blobs(
            n_samples=n_samples,
            centers=n_groups,
            cluster_std=0.50,
            random_state=random_state)

        labels, group_sizes = compute_equal_group_kmeans(data, n_groups)
        # print(y_true)
        print('Group sizes: {}'.format(group_sizes))


nx, ny = 4, 8
xs = np.linspace(0, 1, nx)
ys = np.linspace(0, 1, ny)
x, y = np.meshgrid(xs, ys) + np.random.normal(scale=0.01, size=(ny, nx))
X = np.zeros(shape=(len(x.flatten()), 2))
X[:, 0] = x.flatten()
X[:, 1] = y.flatten()
labels, group_sizes = compute_equal_group_kmeans(data, n_groups)
print('Group sizes: {}'.format(group_sizes))