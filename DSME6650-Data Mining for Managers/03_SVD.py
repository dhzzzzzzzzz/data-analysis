##############################################################
# author: Samuel
# date:   20181215
# topic:  perform singular value decomposition (SVD) 
#         and reconstruct the compressed matrix
##############################################################
# Some general import statements for modules go here
import pandas as pd
from numpy import diag
from numpy import zeros
from scipy.linalg import svd

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
# Import a CSV file as dataframe and name it df
df=pd.read_csv('./0335_SVD.csv')
# df=pd.read_csv('./0335T_SVD.csv')
X=df[['x1','x2','x3','x4','x5']]

#############################################################
# Singular-value decomposition
U, D, VT = svd(X,full_matrices=False) # "full_matrices = False" for thin SVD 

print('Diagonal vector D:')
print_vector(D)

print('Matrix U:')
print_matrix(U)

print('Matrix VT:')
print_matrix(VT)

#############################################################
# Trim the diagonal matrix D based on a threshold 
threshold = 0.8 

# calculate the total of singular values
total = 0
for i in range(len(D)):
    total= total + D[i]

# retain the significant singular values and wipe off the minor ones 
sum = 0
for i in range(len(D)):
    if sum/total < threshold:
        sum = sum + D[i] 
    else: 
        D[i]=0

# display the final diagonal vector
end_of_D = 0
for i in range(len(D)):
    if D[i]==0:
        end_of_D=i
        break

print("\n# most sig. singular values retained: %s if threshold = %0.2f" % (end_of_D, threshold))        
print("\nDiagonal vector D is trimmed down to:")
print_vector(D)

######################################################################################
# Reconstruct the matrix with the most significant singular values
# declare an empty matrix with length D 
tr_matrix = zeros((len(D), len(D)))
print('Empty square matrix with length D:')
print_matrix(tr_matrix)

# assign the diagonal vector into the empty matrix  
tr_matrix= diag(D)
print('Diagonal matrix:')
print_matrix(tr_matrix)

# reconstruct the final compressed matrix using the most sigificant singular values  
T = U.dot(tr_matrix.dot(VT))
print('Final compressed matrix' )
print_matrix(T)

