import pandas as pd
import numpy as np
import pprint
from sklearn.preprocessing import StandardScaler
from matplotlib import *
import matplotlib.pyplot as plt
from matplotlib.cm import register_cmap
from scipy import stats
from sklearn.decomposition import PCA as sklearnPCA
import seaborn

#Load data
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')
ratings.drop(['timestamp'], axis=1, inplace=True)

def replace_name(x):
    return movies[movies['movieId']==x].title.values[0]

ratings.movieId = ratings.movieId.map(replace_name)

# Change those read csv into pivot table
M = ratings.pivot_table(index=['userId'], columns=['movieId'], values='rating')
m = M.shape

# Replace NaN value into 0 (because PCA can only be applied to numerical values)
df1 = M.replace(np.nan, 0, regex=True)

# Standardized the value so that each values are contributed equally to the analysis.
X_std = StandardScaler().fit_transform(df1)

# Get the covariance matrix

# This one is used for getting the mean of values
mean_vec = np.mean(X_std, axis=0)

# Let's calculate the covariance matrix!
# cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)

# OR

cov_mat = np.cov(X_std.T)

# pprint.pprint('Covariance matrix \n%s' %cov_mat)

# Now perform eigendecomposition so that we could get its eigenvectors and eigenvalues
# NOTES:
# - Each eigenvector has a corresponding eigenvalue
# - The sum of the eigenvalues represents all of the variance within the entire dataset

# Perform eigendecomposition on covariance matrix
eig_vals, eig_vecs = np.linalg.eig(cov_mat)

# pprint.pprint('Eigenvectors \n%s' %eig_vecs)
# pprint.pprint('\nEigenvalues \n%s' %eig_vals)

# Visually confirm that the list is correctly sorted by decreasing eigenvalues
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]

print('Eigenvalues in descending order:')
for i in eig_pairs:
    print(i[0])

# Explained variance ration of the first two principal components

pca = sklearnPCA(n_components=2)
pca.fit_transform(df1)
print (pca.explained_variance_ratio_)

# Scree plot of explained variance
pca = sklearnPCA().fit(X_std)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()

