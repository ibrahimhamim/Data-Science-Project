#!/usr/bin/env python
# coding: utf-8

# In[34]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
from sklearn.metrics import plot_roc_curve
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import mean_squared_error
from sklearn.inspection import permutation_importance

#import shap

#shap.initjs()



# read csv file
data = pd.read_csv('airQualityData.csv')

# create list of columns to extract
extract = ['cityname', 'date', 'SO2', 'NO2', 'PM10', 'CO', 'O3_8', 'PM25', 'year', 'month','day']

# extract columns into dataframe airQuality
airQuality = data[extract].dropna(axis = 'index').head(1000)
y = airQuality['PM25'].to_numpy()
airQuality['PM25C'] = airQuality['PM25'].apply(lambda x: 1 if x > 50 else -1)
yc = airQuality['PM25C'].to_numpy()
X= airQuality[['SO2', 'NO2', 'PM10', 'CO', 'O3_8']].to_numpy()

#print(X)
#print(y)
x_train, x_test, y_train, y_test  = train_test_split(X,yc)



# In[2]:


RF=RandomForestClassifier(random_state=42, n_estimators=100, max_depth=20)
RF.fit(x_train,y_train)
RF.score(x_test,y_test)


# In[3]:



#RF_pred = rf1.predict(X_train)
RF_pred_test = RF.predict(x_test)
#r3 = mean_absolute_error(y_train, y_pred)
MAE= mean_absolute_error(y_test, RF_pred_test)
#print(“MAE of Random Forest Regressor on training set: {:.3f}”.format(r3))
print("MAE of Gradient Boosting Tree : {:.4f}".format(MAE))


# In[4]:


import math
meanSquaredError=mean_squared_error(y_test, RF_pred_test)
print("MSE:", meanSquaredError)
rootMeanSquaredError = math.sqrt(meanSquaredError)
print("RMSE:", rootMeanSquaredError)


# In[5]:


plot_roc_curve(RF, x_test, y_test)
plt.show()


# #Gradient_Boosting

# In[6]:


GB=GradientBoostingClassifier(random_state=42, n_estimators=100, max_depth=20)
GB.fit(x_train,y_train)
GB.score(x_test,y_test)


# In[7]:


plot_roc_curve(GB, x_test, y_test)
plt.show()


# In[10]:



#RF_pred = rf1.predict(X_train)
GB_pred_test = GB.predict(x_test)
#r3 = mean_absolute_error(y_train, y_pred)
MAE= mean_absolute_error(y_test, GB_pred_test)
#print(“MAE of Random Forest Regressor on training set: {:.3f}”.format(r3))
print("MAE of Gradient Boosting Tree : {:.4f}".format(MAE))


# In[11]:


import math
meanSquaredError=mean_squared_error(y_test, GB_pred_test)
print("MSE:", meanSquaredError)
rootMeanSquaredError = math.sqrt(meanSquaredError)
print("RMSE:", rootMeanSquaredError)


# #Neural_Network

# In[23]:


NN=MLPClassifier(alpha=1e-5,hidden_layer_sizes=(4,4), random_state=42)
NN.fit(x_train,y_train)


# In[25]:


NN.score(x_test,y_test)


# In[26]:


plot_roc_curve(NN, x_test, y_test)
plt.show()


# In[27]:


disp=plot_roc_curve(RF,x_test, y_test)
plot_roc_curve(GB,x_test, y_test,ax=disp.ax_);
plot_roc_curve(NN,x_test, y_test,ax=disp.ax_);


# In[39]:


RF.feature_importances_
feature_names=['SO2', 'NO2', 'PM10', 'CO', 'O3_8']
no_of_features=len(feature_names)
importance=RF.feature_importances_
airpollution_FI=pd.DataFrame({"features":feature_names,"importances":importance})
airpollution_FI

#X.columns[(sel.get_support())]


# In[41]:


import seaborn as sns
plt.figure(figsize=(20,10))
sns.barplot(data=airpollution_FI, x='features',y='importances')


# In[ ]:





# In[42]:


len(feature_names)


# In[ ]:




