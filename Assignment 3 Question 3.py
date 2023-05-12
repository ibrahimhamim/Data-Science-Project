#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
import pandas as pd
from sklearn import datasets
#from sklearn.linear_model import Lasso
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier



users = pd.read_csv('u(1).user.csv', header = None)
movies =pd.read_csv("u.item.csv", header = None, encoding = 'latin-1')
ratings = pd.read_csv("u(1).data.csv",sep=',' , header =None, names = ['userid', 'movieid', 'rating', 'time'])


#print(users)
#print(ratings.shape)


# In[ ]:




#X = [[0., 0.], [1., 1.]]
#y = [0, 1]
#clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
#clf.fit(X, y)
#print(clf.coefs_)



#print(users)
print(movies.iloc[:,[5,7]])

N = len(users.index) # Number of users
M = len(movies.index) #Number of movies
userItem = np.zeros((N,M)) #Initialize the userItem matrix

for ind in ratings.index:
    row = ratings.iloc[ind,0]
    column = ratings.iloc[ind,1]
    userItem[row-1,column-1] = ratings.iloc[ind,2] #Assign the #row-user's rating on the #col-movie
    #if ratings.iloc[ind,2] >= 4:
    #    userItem[row - 1, column - 1] = 1   #cast the lablel to 1 or 0, representing "like" or "dislike" when we use logistic models
    #else:
    #    userItem[row - 1, column - 1] = 0



#selectRow = [1,3]
print(userItem.shape)


#The index of user 1 is ind=0
targetUser = 1
selectRowInd = []
predList = []
for col in range(M):
    if userItem[targetUser-1][col] == 0: #Select the movies that need recoomendation for the target user
        predList.append(col)


#lr = linear_model.LinearRegression()
mlp = MLPClassifier(solver='lbfgs', activation='relu',learning_rate='constant',learning_rate_init=0.001,alpha=1e-5,hidden_layer_sizes=(2,8),max_iter=100000, random_state=1)
accuracy =[]
for mid in predList: # need predict for all movies in predList for the target user
    selectRowInd = [] # there exists a specific user group for each target movie (the movie to be predicted).
    for row in range(N):
        if row != targetUser-1 and userItem[row][mid] > 0: # Eleminate the target user and the users haven't rated on this movie mid
            selectRowInd.append(row)
    #prepare the work data by eliminateing the some rows and the target column
    data = userItem[selectRowInd,:]
    X = np.delete(data,mid,axis=1)
    Y = data[:,mid]
    Y_label = np.zeros((len(selectRowInd),5), dtype=int)

    for ind in range(len(selectRowInd)):
        row = selectRowInd[ind] #the row index in the corresponding userItem index.
        rating = int(userItem[row][mid])
        Y_label[ind][rating-1] = 1

    #print(Y_label)

    if len(Y_label) >= 10:
            X_train, X_test, y_train, y_test = train_test_split(X, Y_label, test_size=0.3, random_state=42)

        #Some datasets have only one type of labels due to the small numbers of instances. We can only use the training datasets having both labels
     #if (np.count_nonzero(y_train > 0) >= 1) and (np.count_nonzero(y_train < 0) >= 1):

            mlp.fit(X_train, y_train)
            #lr.fit(X_train, y_train)
            #prediction = lr.predict(X_test)
            prediction = mlp.predict(X_test)
            diff= abs(prediction - y_test)
            #prediction = clf.predict(X_test)
            diff= abs(prediction - y_test)
            #print(diff)
            correct = (diff<= 0.5).sum()
            ratio = correct/len(y_test)
            #print(y_test) 
            accuracy.append(ratio)
            #print(ratio)
            #print("results",prediction)
       

        #print('accuracy')
#print(accuracy)
print("the average accuracy of neural network(classifier) model on this user is ", sum(accuracy)/len(accuracy))






# In[11]:


targetUser = 50
selectRowInd = []
predList = []
for col in range(M):
    if userItem[targetUser-1][col] == 0: #Select the movies that need recoomendation for the target user
        predList.append(col)


#lr = linear_model.LinearRegression()
mlp = MLPClassifier(solver='lbfgs', activation='relu',learning_rate='constant',learning_rate_init=0.001,alpha=1e-5,hidden_layer_sizes=(2,8), max_iter=100000,random_state=1)
accuracy =[]
for mid in predList: # need predict for all movies in predList for the target user
    selectRowInd = [] # there exists a specific user group for each target movie (the movie to be predicted).
    for row in range(N):
        if row != targetUser-1 and userItem[row][mid] > 0: # Eleminate the target user and the users haven't rated on this movie mid
            selectRowInd.append(row)
    #prepare the work data by eliminateing the some rows and the target column
    data = userItem[selectRowInd,:]
    X = np.delete(data,mid,axis=1)
    Y = data[:,mid]
    Y_label = np.zeros((len(selectRowInd),5), dtype=int)

    for ind in range(len(selectRowInd)):
        row = selectRowInd[ind] #the row index in the corresponding userItem index.
        rating = int(userItem[row][mid])
        Y_label[ind][rating-1] = 1

    #print(Y_label)

    if len(Y_label) >= 10:
            X_train, X_test, y_train, y_test = train_test_split(X, Y_label, test_size=0.3, random_state=42)

        #Some datasets have only one type of labels due to the small numbers of instances. We can only use the training datasets having both labels
     #if (np.count_nonzero(y_train > 0) >= 1) and (np.count_nonzero(y_train < 0) >= 1):

            mlp.fit(X_train, y_train)
            #lr.fit(X_train, y_train)
            #prediction = lr.predict(X_test)
            prediction = mlp.predict(X_test)
            diff= abs(prediction - y_test)
            #prediction = clf.predict(X_test)
            diff= abs(prediction - y_test)
            #print(diff)
            correct = (diff<= 0.5).sum()
            ratio = correct/len(y_test)
            #print(y_test) 
            accuracy.append(ratio)
            #print(ratio)
            #print("results",prediction)
       

        #print('accuracy')
#print(accuracy)
print("the average accuracy of neural network(classifier) model on this user is ", sum(accuracy)/len(accuracy))


# In[12]:


targetUser = 100
selectRowInd = []
predList = []
for col in range(M):
    if userItem[targetUser-1][col] == 0: #Select the movies that need recoomendation for the target user
        predList.append(col)


#lr = linear_model.LinearRegression()
mlp = MLPClassifier(solver='lbfgs', activation='relu',learning_rate='constant',learning_rate_init=0.001,max_iter=100000,alpha=1e-5,hidden_layer_sizes=(2,8), random_state=1)
accuracy =[]
for mid in predList: # need predict for all movies in predList for the target user
    selectRowInd = [] # there exists a specific user group for each target movie (the movie to be predicted).
    for row in range(N):
        if row != targetUser-1 and userItem[row][mid] > 0: # Eleminate the target user and the users haven't rated on this movie mid
            selectRowInd.append(row)
    #prepare the work data by eliminateing the some rows and the target column
    data = userItem[selectRowInd,:]
    X = np.delete(data,mid,axis=1)
    Y = data[:,mid]
    Y_label = np.zeros((len(selectRowInd),5), dtype=int)

    for ind in range(len(selectRowInd)):
        row = selectRowInd[ind] #the row index in the corresponding userItem index.
        rating = int(userItem[row][mid])
        Y_label[ind][rating-1] = 1

    #print(Y_label)

    if len(Y_label) >= 10:
            X_train, X_test, y_train, y_test = train_test_split(X, Y_label, test_size=0.3, random_state=42)

        #Some datasets have only one type of labels due to the small numbers of instances. We can only use the training datasets having both labels
     #if (np.count_nonzero(y_train > 0) >= 1) and (np.count_nonzero(y_train < 0) >= 1):

            mlp.fit(X_train, y_train)
            #lr.fit(X_train, y_train)
            #prediction = lr.predict(X_test)
            prediction = mlp.predict(X_test)
            diff= abs(prediction - y_test)
            #prediction = clf.predict(X_test)
            diff= abs(prediction - y_test)
            #print(diff)
            correct = (diff<= 0.5).sum()
            ratio = correct/len(y_test)
            #print(y_test) 
            accuracy.append(ratio)
            #print(ratio)
            #print("results",prediction)
       

        #print('accuracy')
#print(accuracy)
print("the average accuracy of neural network(classifier) model on this user is ", sum(accuracy)/len(accuracy))


# In[ ]:




