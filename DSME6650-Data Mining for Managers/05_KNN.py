##############################################################
# author: Samuel
# date:   20190111
# topic:  Classification using K-NN and plot the result
#               
##############################################################
# Some general import statements go here

##############################################################
# self defined functions go here, if any

#############################################################
# Read data from a CSV 
import pandas as pd
df=pd.read_csv('./0527_Cereals.csv')

#############################################################
# define the training and target attributes  
X=df[['calories','protein','fat','sodium','fiber','carbo','sugars','potass','vitamins','shelf','weight','cups']]
Y=df['One_zero']

#############################################################
# define the train-test split
from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test=\
train_test_split(X,Y,test_size=0.2,random_state=0)

#############################################################
# Apply K-NN with matplotlib 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import mean_squared_error,mean_absolute_error,accuracy_score

#----------------------------------------------------------
# define a list of k neighbors and print them out for checking
neighbors = list(range(1,20,2)) # start from 1 to 20 with step=2

# "*" is used to print the list in a single line with space
print(*neighbors)

# create an empty list that holds all accuracy_scores 
accuracy_scores = []
#----------------------------------------------------------
# create an object knn and trigger its methods
for k in neighbors: 
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, Y_train)
    Y_predict=knn.predict(X_test)
    print('K-NN: (K = %s)' %k)
    print('Mean Squared Error of K-NN: %0.5f'\
          % mean_squared_error(Y_test,Y_predict))
    print('Mean Absolute Error of K-NN: %0.5f' \
          % mean_absolute_error(Y_test,Y_predict))
    acc_score = accuracy_score(Y_test,Y_predict)
    print('Accuracy: %.2f\n' % acc_score)
    accuracy_scores.append(acc_score)

#----------------------------------------------------------
# create a list of errors and dump it out for checking
error = [1 - x for x in accuracy_scores]
print('The error list for K=', *neighbors,'is')
print(*error)

#----------------------------------------------------------
# determining best k
# index() finds the given element in a list and returns its position.

optimal_k = neighbors[error.index(min(error))]
print ('The optimal number of neighbors is %d' % optimal_k)
#############################################################
# plot error vs k and highlight the min.
import matplotlib.pyplot as plt

plt.xlabel('Number of Neighbors K')
plt.ylabel('Error')
plt.plot(neighbors, error)
plt.axvline(optimal_k, color="red", linestyle="--")
plt.show()	
