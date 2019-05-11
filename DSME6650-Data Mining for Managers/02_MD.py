##############################################################
# author: Samuel
# date:   20181208
# topic:  calcalate mahalanobis distances (MD) from scratch
#         normalized at its centroid  
##############################################################
# some general import statements for modules go here
import numpy as np
import pandas as pd
import numpy.linalg as la

##############################################################
# some common self-defined functions go here
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
            print('%0.3f' % M[i][j], end=' ')
        print("\n")
    return
#############################################################
# import a CSV file as dataframe and name it df
df=pd.read_csv('./0299-MD-Data.csv')
X=df[['x1','x2','x3','x4']]

#############################################################
# convert the df into an array
X=X.values
X=X-X.mean(0) # normalize the dataset at its centroid
print('Normalized input matrix:\n')
print_matrix(X)

#############################################################
# covariance in numpy and inverse matrix in linalg
cov=np.cov(X.T)
inv_cov = la.inv(cov)
print('Inverse covariance matrix:\n')
print_matrix(inv_cov)

#############################################################
# matrix multiplication using A.dot(B.T) in numpy
# Y= X.dot(inv_cov) 
# MD = Y.dot(X.T)
# or you can put them into one single line
MD = X.dot(inv_cov.dot(X.T))

print('Final MD matrix:\n')
print_matrix(MD)

#############################################################
# display for checking
print('MD vector:\n')
print_vector(np.sqrt(MD.diagonal()))
