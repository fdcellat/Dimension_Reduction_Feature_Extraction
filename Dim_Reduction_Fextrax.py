# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 11:40:36 2022

@author: T006940
"""

#PCA 
# Load libraries
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import datasets
# Load the data
digits = datasets.load_digits()
# Standardize the feature matrix
features = StandardScaler().fit_transform(digits.data)
# Create a PCA that will retain 99% of variance
pca = PCA(n_components=0.99, whiten=True)
# Conduct PCA
features_pca = pca.fit_transform(features)
# Show results
print("Original number of features:", features.shape[1])
print("Reduced number of features:", features_pca.shape[1])

#Kernel PCA inseperable datada kullanılır 
# Load libraries
from sklearn.decomposition import PCA, KernelPCA
from sklearn.datasets import make_circles
# Create linearly inseparable data
features, _ = make_circles(n_samples=1000, random_state=1, noise=0.1, factor=0.1)
# Apply kernal PCA with radius basis function (RBF) kernel
kpca = KernelPCA(kernel="rbf", gamma=15, n_components=1)
features_kpca = kpca.fit_transform(features)
print("Original number of features:", features.shape[1])
print("Reduced number of features:", features_kpca.shape[1])

#Hedef değişkeni belli olan datalarda kullanılır (Normal dağılıma uygunluk ve featurelar araası korelasyon olmaması )
# Load libraries
from sklearn import datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# Load Iris flower dataset:
iris = datasets.load_iris()
features = iris.data
target = iris.target
# Create and run an LDA, then use it to transform the features
lda = LinearDiscriminantAnalysis(n_components=1)
features_lda = lda.fit(features, target).transform(features)
# Print the number of features
print("Original number of features:", features.shape[1])
print("Reduced number of features:", features_lda.shape[1])

#Non negative matrislerde kullanılır
# Load libraries
from sklearn.decomposition import NMF
from sklearn import datasets
# Load the data
digits = datasets.load_digits()
# Load feature matrix
features = digits.data
# Create, fit, and apply NMF
nmf = NMF(n_components=10, random_state=1)
features_nmf = nmf.fit_transform(features)
# Show results
print("Original number of features:", features.shape[1])
print("Reduced number of features:", features_nmf.shape[1])

#SparseMatrislerde 2/3ü 0 olan matrislerde kullanılır
# Load libraries
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
from sklearn import datasets
import numpy as np
# Load the data
digits = datasets.load_digits()
# Standardize feature matrix
features = StandardScaler().fit_transform(digits.data)
# Make sparse matrix
features_sparse = csr_matrix(features)
# Create a TSVD
tsvd = TruncatedSVD(n_components=10)

# Sum of first three components' explained variance ratios
tsvd.explained_variance_ratio_[0:3].sum()

# Create and run an TSVD with one less than number of features
tsvd = TruncatedSVD(n_components=features_sparse.shape[1]-1)
features_tsvd = tsvd.fit(features)
# List of explained variances
tsvd_var_ratios = tsvd.explained_variance_ratio_
# Create a function
def select_n_components(var_ratio, goal_var):
    # Set initial variance explained so far
 total_variance = 0.0
 # Set initial number of features
 n_components = 0
 # For the explained variance of each feature:
 for explained_variance in var_ratio:
 # Add the explained variance to the total
     total_variance += explained_variance
 # Add one to the number of components
     n_components += 1
 # If we reach our goal level of explained variance
     if total_variance >= goal_var:
 # End the loop
         break
 # Return the number of components
     return n_components
# Run function
select_n_components(tsvd_var_ratios, 0.95)