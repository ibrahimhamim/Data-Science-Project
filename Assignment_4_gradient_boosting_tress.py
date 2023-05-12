#!/usr/bin/env python
# coding: utf-8

# In[83]:


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


# In[84]:


air_pollution["SO2"]=air_pollution["SO2"].fillna(air_pollution["SO2"].mean())
air_pollution["PM10"]=air_pollution["PM10"].fillna(air_pollution["PM10"].mode())
air_pollution["CO"]=air_pollution["CO"].fillna(air_pollution["CO"].mode())
air_pollution["O3_8"]=air_pollution["O3_8"].fillna(air_pollution["O3_8"].mode())


# In[85]:


air_pollution_new1=air_pollution.drop(['PM25','station','airPressure','sunHours','highTemperature','lowHumidity','year','month','season','longitude','day','latitude','date','latitude','NO2','cityname'], axis=1)


# In[86]:


air_pollution_new1=air_pollution_new1.drop(air_pollution_new1.columns[[0]],axis=1)


# In[87]:


air_pollution_new1


# In[88]:


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


# In[89]:


from sklearn.model_selection import train_test_split
X1_train, X1_test, y1_train, y1_test = train_test_split(air_pollution_new1, air_pollution['PM25C1'], test_size=0.20, random_state=42)
#fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)


# In[90]:



from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import plot_roc_curve
from sklearn.ensemble import BaggingClassifier
from sklearn import metrics
from sklearn import datasets, metrics, model_selection, svm
#RF=RandomForestRegressor(random_state=42, n_estimators=100, max_depth=20) #I need to know what n estimators mean
grad1=GradientBoostingClassifier(random_state=0)
grad1.fit(X1_train,y1_train)


# In[91]:


grad1.score(X1_test,y1_test)


# In[ ]:





# In[92]:


#RF_pred = rf1.predict(X_train)
grad1_pred_test = grad1.predict(X1_test)
#r3 = mean_absolute_error(y_train, y_pred)
MAE_grad1= mean_absolute_error(y1_test, grad1_pred_test)
#print(“MAE of Random Forest Regressor on training set: {:.3f}”.format(r3))
print("MAE of Gradient Boosting Tree : {:.4f}".format(MAE_grad1))


# In[93]:


import math
meanSquaredError_grad1=mean_squared_error(y1_test, grad1_pred_test)
print("MSE:", meanSquaredError_grad1)
rootMeanSquaredError_grad1 = math.sqrt(meanSquaredError_grad1)
print("RMSE:", rootMeanSquaredError_grad1)


# In[94]:


p1=plot_roc_curve(grad1,X1_test, y1_test)


# In[113]:


from sklearn.model_selection import train_test_split
X2_train, X2_test, y2_train, y2_test = train_test_split(air_pollution_new1, air_pollution['PM25C2'], test_size=0.20, random_state=42)



# In[114]:


grad2=GradientBoostingClassifier(random_state=0)
grad2.fit(X2_train,y2_train)


# In[115]:


grad2.score(X2_test,y2_test)


# In[116]:


p2=plot_roc_curve(grad2,X2_test, y2_test)


# In[117]:


from sklearn.model_selection import train_test_split
X3_train, X3_test, y3_train, y3_test = train_test_split(air_pollution_new1, air_pollution['PM25C3'], test_size=0.20, random_state=42)



# In[118]:


grad3=GradientBoostingClassifier(random_state=0)
grad3.fit(X3_train,y3_train)


# In[119]:


grad3.score(X3_test,y3_test)


# In[120]:


p3=plot_roc_curve(grad3,X3_test, y3_test)


# In[103]:


from sklearn.model_selection import train_test_split
X4_train, X4_test, y4_train, y4_test = train_test_split(air_pollution_new1, air_pollution['PM25C4'], test_size=0.20, random_state=42)



# In[104]:


grad4=GradientBoostingClassifier(random_state=0)
grad4.fit(X4_train,y4_train)


# In[105]:


grad4.score(X4_test,y4_test)


# In[106]:


p4=plot_roc_curve(grad4,X4_test, y4_test)


# In[107]:


X4_test


# In[108]:


from sklearn.model_selection import train_test_split
X5_train, X5_test, y5_train, y5_test = train_test_split(air_pollution_new1, air_pollution['PM25C5'], test_size=0.20, random_state=42)



# In[109]:


grad5=GradientBoostingClassifier(random_state=0)
grad5.fit(X5_train,y5_train)


# In[110]:


grad5.score(X5_test,y5_test)


# In[111]:


p5=plot_roc_curve(grad5,X5_test, y5_test)


# In[112]:


disp=plot_roc_curve(grad1,X1_test, y1_test)
plot_roc_curve(grad2,X2_test, y2_test,ax=disp.ax_);
plot_roc_curve(grad3,X3_test, y3_test,ax=disp.ax_);
plot_roc_curve(grad4,X4_test, y4_test,ax=disp.ax_);
plot_roc_curve(grad5,X5_test, y5_test,ax=disp.ax_); 

