##############################################################
# author: Samuel
# date:   20190102
# topic:  (1) K-means and K-means++  
#         (2) display the results using matplotlib
#         (3) calculation of Silhouette coefficient (SC)           
##############################################################
# Some general import statements for modules go here
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
##############################################################
# Some common self-defined functions go here
def print_vector(V): 
    for i in range (len(V)):
            print('%0.3f' % V[i], end=' ')
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
# Import a CSV file as dataframe and name it df
df=pd.read_csv('./0440_2D_data.csv')
X=df[['SA','SB']]

#------------------------------------------------------------
# turn a dataframe to an arrary
X=X.values
print_matrix(X) # dump all data points for checking

#############################################################
# scatter plot the initial unclustered data points
plt.scatter(X[:,0],X[:,1], color='black', marker='*', label='initial')
plt.xlabel('Sales in A')
plt.ylabel('Sales in B')
plt.legend()
plt.grid()
plt.show()

#############################################################
# apply KMeans or KMeans++ clustering technique
# for standard KMeans, init='random'
# km = KMeans(n_clusters=3,init='random',n_init=10,max_iter=300,tol=1e-04,random_state=0)

# apply KMeans++ clustering technique
km = KMeans(n_clusters=3, init='k-means++',
            n_init=10, max_iter=300, tol=1e-04, random_state=0) #random_state can lead to different result.

# y_km is the index of the cluster each sample belongs to after clustering
y_km = km.fit_predict(X)

#############################################################
# scatter plot all the finalized clusters

plt.scatter(X[y_km==0,0],X[y_km ==0,1],color='green',marker='s',label='cluster 1')
plt.scatter(X[y_km ==1,0],X[y_km ==1,1],color='red',marker='o',label='cluster 2')
plt.scatter(X[y_km ==2,0],X[y_km ==2,1],color='blue',marker='v',label='cluster 3')
plt.scatter(km.cluster_centers_[:,0],
            km.cluster_centers_[:,1],marker='*',color='black',label='centroids')
plt.xlabel('Sales in A')
plt.ylabel('Sales in B')
plt.legend()
plt.grid()
plt.show()

#####################################################################
# calculate the silhouette coeff.(SC)
from sklearn.metrics import silhouette_samples

# silhouette_vals is a vector holding all SCs 
silhouette_vals = silhouette_samples(X, y_km, metric='euclidean')
print('All Silhouette coeff. are:')
print_vector(silhouette_vals)

#####################################################################
# calculate the average SC for each cluster
for i in range (3):
    sc = np.mean(silhouette_vals[y_km == i])
    print('SC for cluster-%s = %0.3f' %(i, sc))

# calculate the overall SC for all data points
silhouette_avg = np.mean(silhouette_vals)
print('\nOverall Silhouette coefficient=%0.3f'%silhouette_avg)

#####################################################################
