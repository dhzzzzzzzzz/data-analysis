##############################################################
# author: Samuel
# date:   20190111
# topic:  Classification using Logistic Regression (LR) 
#
##############################################################
# Some general import statements go here
import pandas as pd

##############################################################
# Some common self-defined functions go here
def print_vector(V): 
    print('It is a vector with length = %s' %len(V))
    for i in range (len(V)):
            print('%0.4f' % V[i], end=' ')
    print("\n") 
    return       
def print_matrix(M):
    M_numrows = M.shape[0] # shape[0]=number of rows
    M_numcols = M.shape[1] # shape[1]=number of cols
    print('It is an %s X %s matrix' %(M_numrows,M_numcols))
    for i in range (M_numrows):
        for j in range (M_numcols):
            print('%0.2f' % M[i][j], end=' ')
        print("\n")
    return    
#############################################################
# Read data from a CSV 
df=pd.read_csv('./0527_Cereals.csv')

#############################################################
# define the training and target attributes  
X=df[['calories','protein','fat','sodium','fiber','carbo','sugars','potass','vitamins','shelf','weight','cups']]

Y=df['One_zero']

#############################################################
# define the training-test split
from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test\
=train_test_split(X,Y,test_size=0.1,random_state=0)

#############################################################
#Apply Logistic Regression
from sklearn.linear_model import LogisticRegression

# initialize a logistic regression object
lr = LogisticRegression(solver='liblinear', C = 1e9)

# build a LR model using training data 
model =lr.fit(X_train, Y_train)

# display the coefficients of the attributes and intercept 
print('\nCoefficients of the attributes are:')
print_matrix(lr.coef_)
print('Intercept is: ')
print_vector(lr.intercept_)

# making prediction using the trained model
Y_predict=lr.predict(X_test)

#############################################################
#Apply some metrics to evaluate results 
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error,accuracy_score,confusion_matrix

print('Logistic Regression')
print('R-squared of Logistic Regression: %.5f' % r2_score(Y_test,Y_predict))
print('Mean Squared Error of Logistic Regression: %0.4f' % mean_squared_error(Y_test,Y_predict))
print('Mean Absolute Error of Logistic Regression: %0.4f' % mean_absolute_error(Y_test,Y_predict))
print('Accuracy: %.3f\n' % accuracy_score(Y_test,Y_predict))
print('Confusion matrix:')
print_matrix(confusion_matrix(Y_test,Y_predict))
#############################################################
# Remarks:
# The result may be different from GRETL or Weka because
# (a) different parameters in initializing the LR object
# (b) differnt random seed
# (c) train-test split= 90:10 split vs 100% training 
