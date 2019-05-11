##############################################################
# author: Samuel
# date:   20181215
# topic:  perform principal component analysis (PCA) 
#         with graphical display for explained variance ratio
##############################################################
# Some general import statements for modules go here
import numpy as np
import pandas as pd
import numpy.linalg as la
import matplotlib.pyplot as plt

##############################################################
# Some common self-defined functions go here
def print_vector(V): 
    for i in range (len(V)):
            print('%0.4f' % V[i], end=' ')
    print("\n") 
    return       
def print_matrix(M):
    M_numrows = M.shape[0] # shape[0]=number of rows
    M_numcols = M.shape[1] # shape[1]=number of cols
    for i in range (M_numrows):
        for j in range (M_numcols):
            print('%0.4f' % M[i][j], end=' ')
        print("\n")
    return
#############################################################
# Import a CSV file as dataframe and name it df_admission
df_admission=pd.read_csv('./0312_PCA_data.csv')
X=df_admission[['IELTS','GMAT','GPA','U_Rank','Interview','WKExp']]

#############################################################
# normalize the dataset at its centroid
X=X.values
X=X-X.mean(0) # normalize the dataset at its centroid
print_matrix(X)

#############################################################
# construct the covariance matrix and 
# find the eigen values & eigen vectors using linalg
cov = np.cov(X.T)
eigen_values, eigen_vectors = la.eig(cov)
print('\nEigenvalues:')
print_vector(eigen_values)

print('Eigenvectors:')
print_matrix(eigen_vectors)

#############################################################
# construct a variance explained vector
total= sum(eigen_values)
var_exp = [(i/total) for i in sorted(eigen_values, reverse=True)]

print('Variance explained in descending order:')
print_vector(var_exp)

################################################################
# a vector of cumulative sum is computed for the variance exp
cum_var_exp = np.cumsum(var_exp)
print('Cumulative sum of variance explained:')
print_vector(cum_var_exp)

################################################################
# Use matplotlib to show the variance explained  
# both their values and cumulative values
plt.ylabel('Explained Variance Ratio')
plt.xlabel('Principal Components')

plt.bar(range(len(var_exp)), var_exp, alpha=0.5, \
		align='center', label='here')
plt.step(range(len(var_exp)), cum_var_exp, \
		where='mid',label='cumulative explained variance')

plt.show()
#############################################################
# Create a PCA that will retain 85% of the variance
from sklearn.decomposition import PCA

pca = PCA(n_components=0.85)

#############################################################
# Find the new coordinates of the transformed data points 
X_pca = pca.fit_transform(X)

#############################################################
# Display for checking
print('\nOriginal number of dimensions:', X.shape[1])
print('\nReduced number of dimensions:', X_pca.shape[1])
print('\nNew coordinates of the transformed data points:')
print_matrix(X_pca)



