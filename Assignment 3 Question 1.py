#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn import datasets
#from sklearn.linear_model import Lasso
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

users = pd.read_csv('u(1).user.csv', header = None)
movies =pd.read_csv("u.item.csv", header = None, encoding = 'latin-1')
ratings = pd.read_csv("u(1).data.csv",sep=',' , header =None, names = ['userid', 'movieid', 'rating', 'time'])


#print(users)
#print(ratings.shape)


# In[2]:


N = len(users.index) # Number of users
M = len(movies.index) #Number of movies
userItem = np.zeros((N,M)) #Initialize the userItem matrix

for ind in ratings.index:
    row = ratings.iloc[ind,0]
    column = ratings.iloc[ind,1]
    userItem[row-1,column-1] = ratings.iloc[ind,2] #Assign the #row-user's rating on the #col-movie

#selectRow = [1,3]
print(userItem.shape)


#The index of user 1 is ind=0
targetUser = 1
selectRowInd = []
predList = []
for col in range(M):
    if userItem[targetUser-1][col] == 0: #Select the movies that need recoomendation for the target user
        predList.append(col)

print(len(predList))

#print(predList)

#lr = linear_model.LinearRegression()
regr=MLPRegressor(random_state=1, max_iter=500)
accuracy =[]
for mid in predList: # need predict for all movies in predList for the target user
    selectRowInd = []
    for row in range(N):
        if row != targetUser and userItem[row][mid] != 0: # Eleminate the target user and the users haven't rated on this movie mid
            selectRowInd.append(row)
    #prepare the work data by eliminateing the some rows and the target column
    data = userItem[selectRowInd,:]
    X = np.delete(data,mid,axis=1)
    Y = data[:,mid]
    #print('Y---> ',Y)

    if len(Y) >= 10:
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
        #print("target label",y_test)
        regr.fit(X_train, y_train)
        prediction = regr.predict(X_test)
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
print("the average accuracy of neural regression model on this user is ", sum(accuracy)/len(accuracy))






# In[3]:



targetUser = 50
selectRowInd = []
predList = []
for col in range(M):
    if userItem[targetUser-1][col] == 0: #Select the movies that need recoomendation for the target user
        predList.append(col)

print(len(predList))

#print(predList)

regr=MLPRegressor(random_state=1, max_iter=500)
accuracy =[]
for mid in predList: # need predict for all movies in predList for the target user
    selectRowInd = []
    for row in range(N):
        if row != targetUser and userItem[row][mid] != 0: # Eleminate the target user and the users haven't rated on this movie mid
            selectRowInd.append(row)
    #prepare the work data by eliminateing the some rows and the target column
    data = userItem[selectRowInd,:]
    X = np.delete(data,mid,axis=1)
    Y = data[:,mid]
    #print('Y---> ',Y)

    if len(Y) >= 10:
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
        #print("target label",y_test)
        regr.fit(X_train, y_train)
        prediction = regr.predict(X_test)
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
print("the average accuracy of neural regression model on this user is ", sum(accuracy)/len(accuracy))






# In[5]:



targetUser = 100
selectRowInd = []
predList = []

for col in range(M):
    if userItem[targetUser-1][col] == 0: #Select the movies that need recoomendation for the target user
        predList.append(col)

print(len(predList))

#print(predList)

regr=MLPRegressor(random_state=1, max_iter=500)
accuracy =[]
for mid in predList: # need predict for all movies in predList for the target user
    selectRowInd = []
    for row in range(N):
        if row != targetUser and userItem[row][mid] != 0: # Eleminate the target user and the users haven't rated on this movie mid
            selectRowInd.append(row)
    #prepare the work data by eliminateing the some rows and the target column
    data = userItem[selectRowInd,:]
    X = np.delete(data,mid,axis=1)
    Y = data[:,mid]
    #print('Y---> ',Y)

    if len(Y) >= 10:
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
        #print("target label",y_test)
        regr.fit(X_train, y_train)
        prediction = regr.predict(X_test)
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
print("the average accuracy of neural regression model on this user is ", sum(accuracy)/len(accuracy))


# In[ ]:





# In[ ]:




