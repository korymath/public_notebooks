import numpy as np

from sklearn.datasets.samples_generator import make_blobs
from EqualGroupsKMeans import equal_group_kmeans


for random_state in [1017]:
    for test_n in [8]:
      for test_groups in [2, 4]:
        for cluster_std in [0.5]:
          n_samples = test_n
          n_groups = test_groups
          data, y_true = make_blobs(
              n_samples=n_samples,
              centers=n_groups,
              cluster_std=cluster_std,
              random_state=random_state)

          (centers, labels,
           interia, group_sizes, n_iter) = equal_group_kmeans(data, n_groups)
          # print(y_true)
          print('n: {}, n_g: {}, group_sizes: {}'.format(
            n_samples, n_groups, group_sizes))


# Full sweep testing.
if True:
  print('Test Blobs')
  for random_state in [27, 420, 1017]:
      for test_n in [8, 16, 32, 64, 128, 256]:
        for test_groups in [2, 4, 8]:
          for cluster_std in [0.5, 0.8]:
            n_samples = test_n
            n_groups = test_groups
            data, y_true = make_blobs(
                n_samples=n_samples,
                centers=n_groups,
                cluster_std=cluster_std,
                random_state=random_state)

            (centers, labels,
             interia, group_sizes, n_iter) = equal_group_kmeans(data, n_groups)
            # print(y_true)
            print('n: {}, n_g: {}, group_sizes: {}'.format(
              n_samples, n_groups, group_sizes))


  print('Test Box')
  nx, ny = 4, 8
  xs = np.linspace(0, 1, nx)
  ys = np.linspace(0, 1, ny)

  for test_groups in [2, 4, 8, 16]:
    n_groups = test_groups
    for noise_level in [0.01, 0.1, 0.5]:
      x, y = (np.meshgrid(xs, ys) +
              np.random.normal(scale=noise_level, size=(ny, nx)))
      data = np.zeros(shape=(len(x.flatten()), 2))
      n_samples = len(data)
      data[:, 0] = x.flatten()
      data[:, 1] = y.flatten()
      (centers, labels,
       interia, group_sizes, n_iter) = equal_group_kmeans(data, n_groups)
      print('n: {}, n_g: {}, group_sizes: {}'.format(
        n_samples, n_groups, group_sizes))