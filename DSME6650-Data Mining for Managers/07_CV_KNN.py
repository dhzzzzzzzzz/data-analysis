##############################################################
# author: Samuel
# date:   20190131
# topic:  Cross Validation (CV) metric: F-measure (F1 micro) 
#         using K-NN example in Lecture 5
#               
##############################################################
# Some general import statements go here
import numpy as np

##############################################################
# self defined functions go here, if any
def print_vector(V): 
    print('It is a vector with length = %s' %len(V))
    for i in range (len(V)):
            print('%0.4f' % V[i], end=' ')
    print("\n") 
    return 
#############################################################
# Read data from a CSV 
import pandas as pd
df=pd.read_csv('./0527_Cereals.csv')

#############################################################
# define the training and target attributes  
X=df[['calories','protein','fat','sodium','fiber','carbo','sugars','potass','vitamins','shelf','weight','cups']]
Y=df['One_zero']

#################################################################
# Apply K-NN with odd values of K and cross validation 

from sklearn.neighbors import KNeighborsClassifier     
from sklearn.model_selection import cross_val_score

# creating odd list of K for KNN
neighbors = list(range(1,20,2))

# create an empty list that holds all cv_scores 
cv_scores = []

# perform 10-fold cross validation (CV)
for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X, Y, cv=10, scoring='f1_micro')
    cv_scores.append(scores.mean())

print('\nCV scores: ')
print_vector(cv_scores)

#----------------------------------------------------------
# create a list of errors 
error = [1 - x for x in cv_scores]

#----------------------------------------------------------
# find the best value of k
optimal_k = neighbors[error.index(min(error))]
print ('\nThe optimal number of neighbors is %d' % optimal_k)

#############################################################
# plot error vs k and highlight the min.
import matplotlib.pyplot as plt

plt.xlabel('Number of Neighbors K')
plt.ylabel('Classification error under CV10 using F1-micro')
plt.plot(neighbors, error)
plt.axvline(optimal_k, color="red", linestyle="--")
plt.show()

##############################################################