#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/korymath/public_notebooks/blob/master/Building_Equal_Size_Clusters_Kyle_Mathewson_Sept_2019.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Imports

# In[1]:


# get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans
from collections import Counter
import numpy as np
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

random_state = 1017


# # Generate and Visualize Data

# In[2]:


n_samples = 6
n_groups = 3
n_members = 2

# ensure that the calculus works out
assert n_groups * n_members == n_samples

X, y_true = make_blobs(n_samples=n_samples, centers=n_groups,
                       cluster_std=0.50, random_state=random_state)
plt.scatter(X[:, 0], X[:, 1], s=50);


# In[3]:


for x in X:
  print('{};'.format(x))


# # K-Means Clustering

# In[4]:


kmeans = KMeans(n_clusters=n_groups, n_init=100, max_iter=1000)
kmeans.fit(X)
labels = kmeans.predict(X)

plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);


# In[5]:


# test the group size, AssertionError on failure
C = Counter(labels)
print('Group sizes: {}'.format(C))

try:
  assert list(C.values()) == [n_members] * n_groups
except AssertionError as e:
  print('Unequal group sizes')


# # (optional) Explicit Algorithm Details

# In[6]:


from sklearn.metrics import pairwise_distances_argmin

def find_clusters(X, n_groups, rseed=random_state):
    # 1. Randomly choose clusters
    rng = np.random.RandomState(rseed)
    i = rng.permutation(X.shape[0])[:n_groups]
    centers = X[i]

    while True:
        # 2a. Assign labels based on closest center
        labels = pairwise_distances_argmin(X, centers)

        # 2b. Find new centers from means of points
        new_centers = np.array([X[labels == i].mean(0)
                                for i in range(n_groups)])

        # 2c. Check for convergence
        if np.all(centers == new_centers):
            break
        centers = new_centers

    return centers, labels

centers, labels = find_clusters(X=X, n_groups=n_groups)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis');


# # Limitations of K-Means
#
# 1. Global optimum not guaranteed
# 2. n_groups must be selected beforehand
# 3. limited to linear cluster boundaries
# 4. slow for large n_samples
# 5. group sizes unequal

# In[7]:


# To address limitation 1, we can increase n_init for different random
# starting points on centroids. We can also increase the number of iterations
# particularly if there is a small n_samples

# To address limitation 3, we can use spectral clustering

# use a kernel transformation to project the data into a higher dimension where
# a linear separation is possible.
# Allow k-means to discover non-linear boundaries.

from sklearn.cluster import SpectralClustering
model = SpectralClustering(n_clusters=n_groups, affinity='nearest_neighbors',
                           assign_labels='kmeans', n_neighbors=n_members,
                           n_init=100, random_state=random_state)
labels = model.fit_predict(X)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis');


# In[8]:


# test the group size, AssertionError on failure
C = Counter(labels)
print('Group sizes: {}'.format(C))

try:
  assert list(C.values()) == [n_members] * n_groups
except AssertionError as e:
  print('Unequal group sizes')


# # Contrained Group Size k-means Clustering

# In[9]:


def average_data_distance_error(n_groups, memberships, distances):
    '''Calculate average distance between data in clusters.'''
    error = 0
    for k in range(n_groups):
        # indices of datapoints belonging to class k
        i = np.where(memberships == k)[0]
        error += np.mean(distances[tuple(np.meshgrid(i, i))])
    return error / n_groups

def cluster_equal_groups(data, n_groups=None, n_members=None, verbose=False):
    # equal-size clustering based on data exchanges between pairs of clusters

    # given two of three num_points, num_clusters, group_size
    # the third is trivial to calculate
    n_samples, _ = data.shape
    if n_members is None and n_groups is not None:
        n_members = n_samples // n_groups
    elif n_groups is None and n_members is not None:
        n_groups = n_samples // n_members
    else:
        raise Exception('must specify either n_members or n_groups')

    # distance matrix
    distances = squareform(pdist(data))
    # print(distances)

    # Random initial membership
    # np.random.seed(random_state)
    # memberships = np.random.permutation(n_samples) % n_groups

    # Initial membership
    kmeans = KMeans(n_clusters=n_groups, n_init=100, max_iter=1000)
    kmeans.fit(data)
    memberships = kmeans.predict(data)

    current_err = average_data_distance_error(n_groups, memberships, distances)
    # print(n_groups, memberships)
    t = 1
    while True:
        past_err = current_err
        for a in range(n_samples):
            for b in range(a):
                # exchange membership
                memberships[a], memberships[b] = memberships[b], memberships[a]
                # calculate new error
                test_err = average_data_distance_error(n_groups, memberships, distances)
                if verbose:
                  print("{}: {}<->{} E={}".format(t, a, b, current_err))
                if test_err < current_err:
                    current_err = test_err
                else:
                    # put them back
                    memberships[a], memberships[b] = memberships[b], memberships[a]
        if past_err == current_err:
            break
        t += 1
    return memberships


# In[10]:


import time

n_samples = 32
n_groups = 8
n_members = n_samples // n_groups

# ensure that the calculus works out
assert n_groups * n_members == n_samples

X, y_true = make_blobs(n_samples=n_samples,
                       centers=n_groups,
                       cluster_std=0.50,
                       random_state=random_state)
plt.scatter(X[:, 0], X[:, 1], s=50);

t0 = time.time()
labels = cluster_equal_groups(X, n_groups=n_groups, verbose=False)
t1 = time.time()

# test the group size, AssertionError on failure
C = Counter(labels)
print('Group sizes: {}'.format(C))

try:
  assert list(C.values()) == [n_members] * n_groups
  print('Success, group sizes are equal!')
except AssertionError as e:
  print('Unequal group sizes')

print('Equal group memberships found in {} s'.format(round(t1-t0, 2)))

# Plot the memberships
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis');


# In[14]:


nx, ny = 4, 8
xs = np.linspace(0, 1, nx)
ys = np.linspace(0, 1, ny)
x, y = np.meshgrid(xs, ys) + np.random.normal(scale=0.01, size=(ny, nx))
print(x.shape, y.shape)


# In[15]:


X = np.zeros(shape=(len(x.flatten()), 2))
X[:, 0] = x.flatten()
X[:, 1] = y.flatten()
plt.scatter(X[:, 0], X[:, 1], s=50);


# In[16]:


labels = cluster_equal_groups(X, n_groups=n_groups, verbose=False)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='jet');

# test the group size, AssertionError on failure
C = Counter(labels)
print('Group sizes: {}'.format(C))

try:
  assert list(C.values()) == [n_members] * n_groups
except AssertionError as e:
  print('Unequal group sizes')


# In[17]:


# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

X, y_true = make_blobs(n_samples=n_samples,
                       centers=n_groups,
                       n_features=3,
                       cluster_std=0.50,
                       random_state=random_state)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], s=50)


# In[18]:


labels = cluster_equal_groups(X, n_groups=n_groups)


# In[19]:


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels, s=50, cmap='viridis');


# In[20]:


np.random.permutation(n_samples) % 4


# In[21]:


distances = squareform(pdist(X))
distances


# In[22]:


distances[np.meshgrid((0,1), (0,1))]


# In[23]:


np.mean(distances[tuple(np.meshgrid([0,1], [0,1]))])


# In[24]:


distances[tuple(np.meshgrid([2,3], [2,3]))]


# In[25]:


np.meshgrid([1,2,3],[1,2,3])
# distances[tuple(np.meshgrid([1,2,3],[1,2,3]))]


# In[26]:


memberships = np.random.permutation(n_samples) % n_groups
np.where(memberships == 3)[0]


# In[27]:


tuple(np.meshgrid([1,2,3], [1,2,3])[0])


# In[28]:


distances[np.meshgrid([1,2,3], [1,2,3])]





# In[30]:


X


# In[31]:


n_samples = 128
n_groups = 16
n_members = 8

# ensure that the calculus works out
assert n_groups * n_members == n_samples

X, y_true = make_blobs(n_samples=n_samples, centers=n_groups,
                       cluster_std=0.50, random_state=random_state)
plt.scatter(X[:, 0], X[:, 1], s=50);
