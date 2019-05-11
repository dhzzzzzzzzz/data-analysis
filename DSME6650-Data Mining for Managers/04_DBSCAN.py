##############################################################
# author: Samuel
# date:   20190102
# topic:  compare K-means++ and DBSCAN in eye-brow dataset 
#         and display the results using matplotlib
##############################################################
# Some general import statements go here
import pandas as pd

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
            print('%0.1f' % M[i][j], end=' ')
        print("\n")
    return
#############################################################
# Read data from a CSV 
df=pd.read_csv('./04_eye_brows2.csv')
X=df[['x1','x2']]

#############################################################
# turn a dataframe into an array
X=X.values

# print and check
print_matrix(X)

#############################################################
# plot to visualize the data patterns
import matplotlib.pyplot as plt
plt.scatter(X[:,0],X[:,1], color='black', marker='*', label='initial')
plt.legend()
plt.grid()
plt.show()

#############################################################
# apply KMeans++ in sklearn
from sklearn.cluster import KMeans

km = KMeans(n_clusters=2, init='k-means++',\
            n_init=10, max_iter=300, tol=1e-04, \
			random_state=0)
y_km = km.fit_predict(X)

#############################################################
# plot result as shown in K-means example

plt.scatter(X[y_km==0,0],X[y_km ==0,1],color='blue',marker='s',label='cluster 1')
plt.scatter(X[y_km ==1,0],X[y_km ==1,1],color='red',marker='o',label='cluster 2')
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],\
            marker='*',color='black',label='centroids')

plt.legend()
plt.grid()
plt.show()

#############################################################
# apply DBSCAN from sklearn
from sklearn.cluster import DBSCAN

db = DBSCAN(eps=0.2, min_samples=3, metric='euclidean')
y_db = db.fit_predict(X)

#############################################################
# plot result. Unlike K-means, there are no cluster centers 
# defined in DBSCAN

plt.scatter(X[y_db==0,0], X[y_db==0,1], color='darkgreen', \
            marker='s', label='cluster 1')
plt.scatter(X[y_db==1,0], X[y_db==1,1], color='darkorange',\
            marker='o', label='cluster 2')

plt.legend()
plt.grid()
plt.show()
