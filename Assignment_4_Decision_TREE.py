#!/usr/bin/env python
# coding: utf-8

# In[129]:


import numpy as np
import pandas as pd
from sklearn import datasets
#from sklearn.linear_model import Lasso
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

#users = pd.read_csv('u(1).user.csv', header = None)
#movies =pd.read_csv("u.item.csv", header = None, encoding = 'latin-1')
#ratings = pd.read_csv("u(1).data.csv",sep=',' , header =None, names = ['userid', 'movieid', 'rating', 'time'])

air_pollution=pd.read_csv('airQualityData.csv')
air_pollution


# In[130]:


air_pollution["SO2"]=air_pollution["SO2"].fillna(air_pollution["SO2"].mean())
air_pollution["PM10"]=air_pollution["PM10"].fillna(air_pollution["PM10"].mode())
air_pollution["CO"]=air_pollution["CO"].fillna(air_pollution["CO"].mode())
air_pollution["O3_8"]=air_pollution["O3_8"].fillna(air_pollution["O3_8"].mode())


# In[131]:


air_pollution_DT=air_pollution.drop(['PM25','station','airPressure','sunHours','highTemperature','lowHumidity','year','month','season','longitude','day','latitude','date','latitude','NO2','cityname'], axis=1)


# In[132]:


air_pollution_DT=air_pollution_DT.drop(air_pollution_DT.columns[[0]],axis=1)


# In[133]:


air_pollution_DT


# In[134]:


air_pollution_DT


# In[135]:


#PM25_ap=air_pollution['PM25']
air_pollution['PM25C'] = air_pollution['PM25'].apply(lambda x: 1 if x > 50 else -1)
air_pollution_c1=air_pollution['PM25C']
air_pollution['PM25C2'] = air_pollution['PM25'].apply(lambda x: 1 if x > 100 else -1)
air_pollution_c2=air_pollution['PM25C2']
air_pollution['PM25C3'] = air_pollution['PM25'].apply(lambda x: 1 if x > 150 else -1)
air_pollution_c3=air_pollution['PM25C3']
air_pollution['PM25C4'] = air_pollution['PM25'].apply(lambda x: 1 if x > 200 else -1)
air_pollution_c4=air_pollution['PM25C4']
air_pollution['PM25C5'] = air_pollution['PM25'].apply(lambda x: 1 if x > 250 else -1)
air_pollution_c5=air_pollution['PM25C5']


# In[136]:


from sklearn.model_selection import train_test_split
X1_train, X1_test, y1_train, y1_test = train_test_split(air_pollution_DT, air_pollution_c1, test_size=0.20, random_state=42)



# In[137]:


from sklearn import tree
#from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import plot_roc_curve
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
#RF=RandomForestRegressor(random_state=42, n_estimators=100, max_depth=20) #I need to know what n estimators mean
DT1=DecisionTreeClassifier(random_state=42, max_depth=10)
DT1.fit(X1_train,y1_train)


# In[138]:


DT1.score(X1_test,y1_test)


# In[139]:


#RF_pred = rf1.predict(X_train)
DT1_pred_test = DT1.predict(X1_test)
#r3 = mean_absolute_error(y_train, y_pred)
MAE_DT1= mean_absolute_error(y1_test, DT1_pred_test)
#print(“MAE of Random Forest Regressor on training set: {:.3f}”.format(r3))
print("MAE of Gradient Boosting Tree : {:.4f}".format(MAE_DT1))


# In[140]:


import math
meanSquaredError_DT1=mean_squared_error(y1_test, DT1_pred_test)
print("MSE:", meanSquaredError_DT1)
rootMeanSquaredError_DT1 = math.sqrt(meanSquaredError_DT1)
print("RMSE:", rootMeanSquaredError_DT1)


# In[141]:


plot_roc_curve(DT1,X1_test, y1_test)


# In[142]:


from sklearn.model_selection import train_test_split
X2_train, X2_test, y2_train, y2_test = train_test_split(air_pollution_DT, air_pollution_c2, test_size=0.20, random_state=42)



# In[143]:


DT2=DecisionTreeClassifier(random_state=42, max_depth=10)
DT2.fit(X2_train,y2_train)


# In[144]:


DT2.score(X2_test,y2_test)


# In[145]:


#RF_pred = rf1.predict(X_train)
DT2_pred_test = DT2.predict(X2_test)
#r3 = mean_absolute_error(y_train, y_pred)
MAE_DT2= mean_absolute_error(y2_test, DT2_pred_test)
#print(“MAE of Random Forest Regressor on training set: {:.3f}”.format(r3))
print("MAE of Gradient Boosting Tree : {:.4f}".format(MAE_DT2))


# In[146]:


import math
meanSquaredError_DT2=mean_squared_error(y2_test, DT2_pred_test)
print("MSE:", meanSquaredError_DT2)
rootMeanSquaredError_DT2 = math.sqrt(meanSquaredError_DT2)
print("RMSE:", rootMeanSquaredError_DT2)


# In[147]:


plot_roc_curve(DT2,X2_test, y2_test)


# In[148]:


from sklearn.model_selection import train_test_split
X3_train, X3_test, y3_train, y3_test = train_test_split(air_pollution_DT, air_pollution_c3, test_size=0.20, random_state=42)



# In[149]:


DT3=DecisionTreeClassifier(random_state=42, max_depth=10)
DT3.fit(X3_train,y3_train)


# In[150]:


DT3.score(X3_test,y3_test)


# In[151]:


plot_roc_curve(DT3,X3_test, y3_test)


# In[152]:


from sklearn.model_selection import train_test_split
X4_train, X4_test, y4_train, y4_test = train_test_split(air_pollution_DT, air_pollution_c4, test_size=0.20, random_state=42)



# In[153]:


DT4=DecisionTreeClassifier(random_state=42, max_depth=10)
DT4.fit(X4_train,y4_train)


# In[154]:


DT4.score(X4_test,y4_test)


# In[155]:


plot_roc_curve(DT4,X4_test, y4_test)


# In[156]:


from sklearn.model_selection import train_test_split
X5_train, X5_test, y5_train, y5_test = train_test_split(air_pollution_DT, air_pollution_c5, test_size=0.20, random_state=42)



# In[157]:


DT5=DecisionTreeClassifier(random_state=42, max_depth=10)
DT5.fit(X5_train,y5_train)


# In[158]:


DT4.score(X5_test,y5_test)


# In[159]:


plot_roc_curve(DT4,X4_test, y4_test)


# In[160]:


disp=plot_roc_curve(DT1,X1_test, y1_test)
plot_roc_curve(DT2,X2_test, y2_test,ax=disp.ax_);
plot_roc_curve(DT3,X3_test, y3_test,ax=disp.ax_);
plot_roc_curve(DT4,X4_test, y4_test,ax=disp.ax_);
plot_roc_curve(DT5,X5_test, y5_test,ax=disp.ax_); 


# In[ ]:




