#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[24]:


import numpy as np
import pandas as pd
from scipy import stats
import math
from sklearn import datasets
#from sklearn.linear_model import Lasso
from sklearn import linear_model
from sklearn.model_selection import train_test_split
#!pip install pingouin
from statsmodels.stats.weightstats import ttest_ind
from sklearn.linear_model import Lasso
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# In[25]:



ibr_user=pd.read_csv("u(1).user.csv")
ibr_user.columns = ['userid','age','gender','occpuation','zip']

ibr_data=pd.read_csv("u(1).data.csv",sep=',',header =None)
#ibr_item=pd.read_csv("u.item.csv", encoding='latin-1')

ibr_data.columns = ['userid','itemid','rating', 'timestamp']

#ibr_new_1 = pd.merge(ibr_data,ibr_item,on='itemid')
#ibr_new_2 = pd.merge(ibr_new_1,ibr_user,on='userid')
ibr_item=pd.read_csv("u.item.csv", encoding='latin1',header = None)
ibr_item.columns = ['itemid','title and year','date','gap','imdb','unknown','action','adventure','animation','children','comedy','crime','documentary','drama','fantasy','film-noir','Horror','Musical','Mystery','Romance','Sci-fi','Thriller','War','Western']
ibr_new_1 = ibr_data.merge(ibr_item, on ='itemid', how ='left')
ibr_new_2 = ibr_new_1.merge(ibr_user, on ='userid', how ='left')
ibr_new_2


# In[32]:


N = len(ibr_user.index) # Number of users
M = len(ibr_item.index) #Number of movies
print('N: ',N,' M: ',M)
userItem = np.zeros((N,M)) #Initialize the userItem matrix
userItem


# In[42]:


for ind in ibr_data.index:
    row = ibr_data.iloc[ind,0]
    column = ibr_data.iloc[ind,1]
   # print('row: ',row,'column: ',column)
    userItem[row-1,column-1] = ibr_data.iloc[ind,2] 

#prediction for user 1
print(userItem.shape)
targetUser = 1
selectRowInd = []
predList = []
selectRowInd = []
predList = []
for col in range(M):
    if userItem[targetUser-1][col] == 0: 
        predList.append(col)
lr = linear_model.LinearRegression()
accuracy =[]
for mid in predList: 
    selectRowInd = []
    for row in range(N):
        if row != targetUser-1 and userItem[row][mid] != 0: 
            selectRowInd.append(row)
    data = userItem[selectRowInd,:]
    X = np.delete(data,mid,axis=1)
    Y = data[:,mid]

    if len(Y) >= 10: 
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
        lr.fit(X_train, y_train)
        prediction = lr.predict(X_test)
        diff= abs(prediction - y_test)
        
        correct = (diff<= 0.5).sum() 
        ratio = correct/len(y_test)
        accuracy.append(ratio)
user_1=sum(accuracy)/len(accuracy)
print("Average rating accuracy of linear regression model on user 1 is ", user_1)


# In[41]:




#prediction for user 10
targetUser = 10
selectRowInd = []
predList = []
for col in range(M):
    if userItem[targetUser-1][col] == 0: 
        predList.append(col)
lr = linear_model.LinearRegression()
accuracy =[]
for mid in predList: 
    selectRowInd = []
    for row in range(N):
        if row != targetUser-1 and userItem[row][mid] != 0:
            selectRowInd.append(row)
    data = userItem[selectRowInd,:]
    X = np.delete(data,mid,axis=1)
    Y = data[:,mid]

    if len(Y) >= 10:
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
        lr.fit(X_train, y_train)
        prediction = lr.predict(X_test)
        diff= abs(prediction - y_test)
        correct = (diff<= 0.5).sum()
        ratio = correct/len(y_test)
        accuracy.append(ratio)
user_10=sum(accuracy)/len(accuracy)
print("Average rating accuracy of linear regression model on user 10 is ", user_10) 

#prediction for user 20
targetUser = 20
selectRowInd = []
predList = []
for col in range(M):
    if userItem[targetUser-1][col] == 0:
        predList.append(col)
lr = linear_model.LinearRegression()
accuracy =[]
for mid in predList: 
    selectRowInd = []
    for row in range(N):
        if row != targetUser-1 and userItem[row][mid] != 0: 
            selectRowInd.append(row)
    data = userItem[selectRowInd,:]
    X = np.delete(data,mid,axis=1)
    Y = data[:,mid]

    if len(Y) >= 10: 
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42) 
        lr.fit(X_train, y_train)
        prediction = lr.predict(X_test)
        diff= abs(prediction - y_test)
        #print(diff)
        correct = (diff<= 0.5).sum()
        ratio = correct/len(y_test)
        accuracy.append(ratio)
user_20=sum(accuracy)/len(accuracy)
print("Average rating accuracy of linear regression model on user 20 is ", user_20) 

#prediction for user 50
targetUser = 50
selectRowInd = []
predList = []
for col in range(M):
    if userItem[targetUser-1][col] == 0: 
        predList.append(col)
lr = linear_model.LinearRegression()
accuracy =[]
for mid in predList: 
    selectRowInd = []
    for row in range(N):
        if row != targetUser-1 and userItem[row][mid] != 0: 
            selectRowInd.append(row)
    data = userItem[selectRowInd,:]
    X = np.delete(data,mid,axis=1)
    Y = data[:,mid]

    if len(Y) >= 10: 
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42) 
        lr.fit(X_train, y_train)
        prediction = lr.predict(X_test)
        diff= abs(prediction - y_test)
        #print(diff)
        correct = (diff<= 0.5).sum() 
        ratio = correct/len(y_test)
        accuracy.append(ratio)
user_50=sum(accuracy)/len(accuracy)
print("Average rating accuracy of linear regression model on user 50 is ", user_50) 

linear_regression = (user_1+user_10+user_20+user_50)/4
print("the average accuracy of the linear regression model is",linear_regression)




# In[38]:


#logistic_model
userItemlogistic = np.zeros((N,M)) 
for ind in ibr_data.index:
    row = ibr_data.iloc[ind,0]
    column = ibr_data.iloc[ind,1]
    
    if ibr_data.iloc[ind,2] >= 3:
        userItem[row - 1, column - 1] = 1
    else:
        userItem[row - 1, column - 1] = -1

#prediction for user 1
targetUser = 1
selectRowInd = []
predList = []
for col in range(M):
    if userItem[targetUser-1][col] == 0: 
        predList.append(col)
log = linear_model.LogisticRegression()
accuracy =[]
for mid in predList: 
    selectRowInd = []
    for row in range(N):
        if row != targetUser-1 and userItem[row][mid] != 0: 
            selectRowInd.append(row)
    data = userItem[selectRowInd,:]
    X = np.delete(data,mid,axis=1)
    Y = data[:,mid]
    if len(Y) >= 10: 
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42) 
        if (np.count_nonzero(y_train > 0) >= 1) and (np.count_nonzero(y_train < 0) >= 1):
            log.fit(X_train, y_train)
            prediction =log.predict(X_test)
            diff= abs(prediction - y_test)
            correct = (diff<= 0.5).sum() 
            ratio = correct/len(y_test)
            accuracy.append(ratio)
user_1 =sum(accuracy)/len(accuracy)
print("the average rating accuracy of logistic regression model on user 1 is ", user_1) 

#prediction for user 10
targetUser = 10
selectRowInd = []
predList = []
for col in range(M):
    if userItem[targetUser-1][col] == 0:
        predList.append(col)
log = linear_model.LogisticRegression()
accuracy =[]
for mid in predList: 
    selectRowInd = []
    for row in range(N):
        if row != targetUser-1 and userItem[row][mid] != 0: 
            selectRowInd.append(row)
   
    data = userItem[selectRowInd,:]
    X = np.delete(data,mid,axis=1)
    Y = data[:,mid]

    if len(Y) >= 10: 
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
        if (np.count_nonzero(y_train > 0) >= 1) and (np.count_nonzero(y_train < 0) >= 1):
            log.fit(X_train, y_train)
            prediction = log.predict(X_test)
            diff= abs(prediction - y_test)
            correct = (diff<= 0.5).sum()
            ratio = correct/len(y_test)
            accuracy.append(ratio)
user_10=sum(accuracy)/len(accuracy)
        
print("the average rating accuracy of logistic regression model on user 10 is ", user_10) 

#prediction for user 20
targetUser = 20
selectRowInd = []
predList = []
for col in range(M):
    if userItem[targetUser-1][col] == 0: 
        predList.append(col)
log = linear_model.LogisticRegression()
accuracy =[]
for mid in predList: 
    selectRowInd = []
    for row in range(N):
        if row != targetUser-1 and userItem[row][mid] != 0: 
            selectRowInd.append(row)
    data = userItem[selectRowInd,:]
    X = np.delete(data,mid,axis=1)
    Y = data[:,mid]

    if len(Y) >= 10: 
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
        if (np.count_nonzero(y_train > 0) >=1) and   (np.count_nonzero(y_train < 0) >=1):
            log.fit(X_train, y_train)
            prediction = log.predict(X_test)
            diff= abs(prediction - y_test)
            #print(diff)
            correct = (diff<= 0.5).sum() 
            ratio = correct/len(y_test)
            accuracy.append(ratio)
user_20=sum(accuracy)/len(accuracy)
        #print("results",prediction)
print("Average accuracy rating of logistic regression model on user 20 is ", user_20) 

#prediction for user 50
targetUser = 50
selectRowInd = []
predList = []
for col in range(M):
    if userItem[targetUser-1][col] == 0: 
        predList.append(col)
log = linear_model.LogisticRegression()
accuracy =[]
for mid in predList: 
    selectRowInd = []
    for row in range(N):
        if row != targetUser-1 and userItem[row][mid] != 0:
            selectRowInd.append(row)
    data = userItem[selectRowInd,:]
    X = np.delete(data,mid,axis=1)
    Y = data[:,mid]
    if len(Y) >= 10: 
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)    
        if (np.count_nonzero(y_train > 0) >=1) and   (np.count_nonzero(y_train < 0) >=1):
            log.fit(X_train, y_train)
            prediction = log.predict(X_test)
            diff= abs(prediction - y_test)
            correct = (diff<= 0.5).sum() 
            ratio = correct/len(y_test)
            accuracy.append(ratio)
user_50=sum(accuracy)/len(accuracy)
print("Average rating accuracy of logistic regression model on user 50 is ", user_50) 
logistic_regression = (user_1+user_10+user_20+user_50)/4

#average accuracy for users 1,10, 20,and 50
print("the average accuracy of the logistic regression model is",logistic_regression)



# In[39]:


if (logistic_regression>linear_regression): 
    print("We prefer logistic regression")
else:
    print("We prefer Linear Regression")


# In[ ]:




