#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from sklearn.inspection import PartialDependenceDisplay
from sklearn.datasets import make_hastie_10_2
from sklearn.inspection import plot_partial_dependence


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


RF=RandomForestRegressor(random_state=42, n_estimators=100, max_depth=20)
RF.fit(x_train,y_train)
RF.score(x_test,y_test)


# In[3]:


features = ['SO2','NO2','PM10','CO','O3_8']
#fig, ax = plt.subplots(1,5)
#pdp = plot_partial_dependence(GB, x_train, features, feature_names=['SO2','NO2','PM10','CO','O3_8'], n_jobs=3, ax=ax, grid_resolution=100)
#pdp.axes_[0][0].set_ylabel("Failure Probability")

#fig.suptitle("PDP")
#plt.show()


# In[4]:


get_ipython().system('pip install shap')
import shap

shap.initjs()

explainer = shap.Explainer(RF,feature_names=features)
shap_test = explainer(x_test)
print(f"Sample shap value:\n{shap_test[0]}")
shap.plots.bar(shap_test)

shap.summary_plot(shap_test)


# In[ ]:




