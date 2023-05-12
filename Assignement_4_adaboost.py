#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
from sklearn import datasets
#from sklearn.linear_model import Lasso
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

#users = pd.read_csv('u(1).user.csv', header = None)
#movies =pd.read_csv("u.item.csv", header = None, encoding = 'latin-1')
#ratings = pd.read_csv("u(1).data.csv",sep=',' , header =None, names = ['userid', 'movieid', 'rating', 'time'])

air_pollution=pd.read_csv('airQualityData.csv')


# In[3]:


air_pollution["SO2"]=air_pollution["SO2"].fillna(air_pollution["SO2"].mean())
air_pollution["PM10"]=air_pollution["PM10"].fillna(air_pollution["PM10"].mode())
air_pollution["CO"]=air_pollution["CO"].fillna(air_pollution["CO"].mode())
air_pollution["O3_8"]=air_pollution["O3_8"].fillna(air_pollution["O3_8"].mode())


# In[4]:


air_pollution_new=air_pollution.drop(['PM25','station','airPressure','sunHours','highTemperature','lowHumidity','year','month','season','longitude','day','latitude','date','latitude','NO2','cityname'], axis=1)


# In[5]:


air_pollution_new1=air_pollution_new.drop(air_pollution_new.columns[[0]],axis=1)


# In[6]:


air_pollution_new1


# In[7]:


air_pollution['PM25C1'] = air_pollution['PM25'].apply(lambda x: 1 if x > 50 else -1)
air_pollution_c1=air_pollution['PM25C1']
air_pollution['PM25C2'] = air_pollution['PM25'].apply(lambda x: 1 if x > 100 else -1)
air_pollution_c2=air_pollution['PM25C2']
air_pollution['PM25C3'] = air_pollution['PM25'].apply(lambda x: 1 if x > 150 else -1)
air_pollution_c3=air_pollution['PM25C3']
air_pollution['PM25C4'] = air_pollution['PM25'].apply(lambda x: 1 if x > 200 else -1)
air_pollution_c4=air_pollution['PM25C4']
air_pollution['PM25C5'] = air_pollution['PM25'].apply(lambda x: 1 if x > 250 else -1)
air_pollution_c5=air_pollution['PM25C5']


# In[8]:


from sklearn.model_selection import train_test_split
X1_train, X1_test, y1_train, y1_test = train_test_split(air_pollution_new1, air_pollution['PM25C1'], test_size=0.20, random_state=42)



# In[9]:



from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import plot_roc_curve
from sklearn.ensemble import BaggingClassifier
from sklearn import metrics
#RF=RandomForestRegressor(random_state=42, n_estimators=100, max_depth=20) #I need to know what n estimators mean
ada1=AdaBoostClassifier(n_estimators=100, random_state=0)
ada1.fit(X1_train,y1_train)


# In[10]:


ada1.score(X1_test,y1_test)


# In[11]:



#RF_pred = rf1.predict(X_train)
ada1_pred_test = ada1.predict(X1_test)
#r3 = mean_absolute_error(y_train, y_pred)
MAE_ada1= mean_absolute_error(y1_test, ada1_pred_test)
#print(“MAE of Random Forest Regressor on training set: {:.3f}”.format(r3))
print("MAE of Gradient Boosting Tree : {:.4f}".format(MAE_ada1))


# In[12]:


import math
meanSquaredError_ada1=mean_squared_error(y1_test, ada1_pred_test)
print("MSE:", meanSquaredError_ada1)
rootMeanSquaredError_ada1 = math.sqrt(meanSquaredError_ada1)
print("RMSE:", rootMeanSquaredError_ada1)


# In[13]:


p1=plot_roc_curve(ada1,X1_test, y1_test)


# In[14]:


from sklearn.model_selection import train_test_split
X2_train, X2_test, y2_train, y2_test = train_test_split(air_pollution_new1, air_pollution['PM25C2'], test_size=0.20, random_state=42)



# In[15]:


ada2=AdaBoostClassifier(n_estimators=100, random_state=0)
ada2.fit(X1_train,y1_train)


# In[16]:


ada2.score(X2_test,y2_test)


# In[17]:


p2=plot_roc_curve(ada2,X2_test, y2_test)


# In[24]:


from sklearn.model_selection import train_test_split
X3_train, X3_test, y3_train, y3_test = train_test_split(air_pollution_new1, air_pollution['PM25C3'], test_size=0.20, random_state=42)



# In[25]:


ada3=AdaBoostClassifier(n_estimators=100, random_state=0)
ada3.fit(X3_train,y3_train)


# In[26]:


ada3.score(X3_test,y3_test)


# In[30]:


p3=plot_roc_curve(ada3,X3_test, y3_test)


# In[28]:


from sklearn.model_selection import train_test_split
X4_train, X4_test, y4_train, y4_test = train_test_split(air_pollution_new1, air_pollution['PM25C4'], test_size=0.20, random_state=42)



# In[29]:


ada4=AdaBoostClassifier(n_estimators=100, random_state=0)
ada4.fit(X4_train,y4_train)


# In[34]:


ada4.score(X4_test,y4_test)


# In[31]:


p4=plot_roc_curve(ada4,X4_test, y4_test)


# In[32]:


from sklearn.model_selection import train_test_split
X5_train, X5_test, y5_train, y5_test = train_test_split(air_pollution_new1, air_pollution['PM25C4'], test_size=0.20, random_state=42)



# In[33]:


ada5=AdaBoostClassifier(n_estimators=100, random_state=0)
ada5.fit(X5_train,y5_train)


# In[35]:


ada5.score(X5_test,y5_test)


# In[36]:


p5=plot_roc_curve(ada5,X5_test, y5_test)


# In[37]:


disp=plot_roc_curve(ada1,X1_test, y1_test)
plot_roc_curve(ada2,X2_test, y2_test,ax=disp.ax_);
plot_roc_curve(ada3,X3_test, y3_test,ax=disp.ax_);
plot_roc_curve(ada4,X4_test, y4_test,ax=disp.ax_);
plot_roc_curve(ada5,X5_test, y5_test,ax=disp.ax_); 


# In[ ]:




