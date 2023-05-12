#!/usr/bin/env python
# coding: utf-8

# In[101]:


#import pakages
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

#import pingouin as pg


# In[102]:





ibr_user=pd.read_csv("u(1).user.csv")
ibr_user.columns = ['userid','age','gender','occupation','zip']

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

                                             


# In[103]:


avg_rating = ibr_new_2['rating'].mean()

print('avg of all movies: ',avg_rating)
#calculate population standard deviation
std_data = ibr_new_2['rating'].std()
print('standard_deviation of all movies: ',std_data)

avg_mean_action=ibr_new_2[ibr_new_2['action'] == 1]['rating'].mean()
print('average of all action movies: ',avg_mean_action)


# In[104]:


number_of_action_movies=452
number_of_movies=1682


# In[105]:


number_of_action_movies=ibr_new_2[ibr_new_2['action'] == 1].count()
print('number of action movies: ',number_of_action_movies)
action_detector_new=ibr_new_2.loc[action_detector, ['rating']]


# In[ ]:





# In[ ]:





# In[ ]:





# In[110]:



#std_action = action_detector_new.std()
print('standard_deviation of all action movies: ',std_action)
#action_std
#sample_size=len(action_detector)
#print('action movie size: ',sample_size)
#action_mv_rating=samples.mean()
#print('average rating of all action movies: ',samples.mean())

#sample_size=len(ibr_new_2[ibr_new_2['action']==1]['rating'])
#caluclate action movie average rating
#action_mv_rating = ibr_new_2[ibr_new_2['action']==1]['rating'].mean()

#hypothesis testing: is action movie's rating than the average rating at .1 level
#t_test = (action_mv_rating-avg_rating)/(std_action/(math.sqrt(sample_size)))
#t_test
#print('t score: ',t_test)
#tcrit = 1.282

#result = ttest_ind(ibr_new_2['rating'],ibr_new_2['action'])#pg.ttest(ibr_new_2['rating'],ibr_new_2['action'],correction=True)

#print('t test result: ',result)
#df=9999
#if (tobs > tcrit): 
  #  print("according to the t-test action movie average rating is higher than average rating at .1 level ")
#else: 
 #   print("according to the t-test action movie average rating is not higher than average rating at .1 level ")

    
t_val=(3.48-3.53)/math.sqrt(std_data*(1/number_of_action_movies)+(1/number_of_movies))
print('t_val: ',t_val)

#for alpha=0.05

t_critical = 1.645

if(abs(t_val)>t_critical):
    print('We reject null hypothesis and accept alternative hypothesis')
else:
    print('We accept null hypothesis')
    
    
#Confidence_Interval

upper_limit = (3.48-3.53) + (t_critical * std_data* math.sqrt((1/number_of_action_movies)+(1/number_of_movies)))
                                           
lower_limit = (3.48-3.53) - (t_critical * std_data*math.sqrt((1/number_of_action_movies)+(1/number_of_movies)))
print(lower_limit)
                                           
print('confidence_interval: (',upper_limit,' ',lower_limit,')')
                                           


# In[112]:


t_val=(3.48-3.53)/math.sqrt(std_data*(1/number_of_action_movies)+(1/number_of_movies))
print('t_val: ',t_val)

#for alpha=0.1

t_critical = 1.645

if(abs(t_val)>t_critical):
    print('We reject null hypothesis and accept alternative hypothesis')
else:
    print('We accept null hypothesis')
    
    
#Confidence_Interval

upper_limit = (3.48-3.53) + (t_critical * std_data* (math.sqrt((1/number_of_action_movies)+(1/number_of_movies))))
                                           
lower_limit = (3.48-3.53) - (t_critical * std_data*math.sqrt((1/number_of_action_movies)+(1/number_of_movies)))
print(lower_limit)
                                           
print('confidence_interval: (',upper_limit,' ',lower_limit,')')
                                           


# In[109]:





# In[ ]:


#for alpha=0.01

t_val=(3.48-3.53)/math.sqrt(std_data*(1/number_of_action_movies)+(1/number_of_movies))
print('t_val: ',t_val)



t_critical = 2.326

if(abs(t_val)>t_critical):
    print('We reject null hypothesis and accept alternative hypothesis')
else:
    print('We accept null hypothesis')
    
    
#Confidence_Interval

upper_limit = (3.48-3.53) + (t_critical * std_data* math.sqrt((1/number_of_action_movies)+(1/number_of_movies)))
                                           
lower_limit = (3.48-3.53) - (t_critical * std_data*math.sqrt((1/number_of_action_movies)+(1/number_of_movies)))
print(lower_limit)
                                           
print('confidence_interval: (',upper_limit,' ',lower_limit,')')
                                           


# In[ ]:


print('action movies rating is lower than the average rating as we are accepting null hypothesis')


# # (b)

# In[107]:


#avg_rating_new = ibr_new_2['rating'].mean()

#print('avg of all movies: ',avg_rating)
#calculate population standard deviation
#std_data = ibr_new_2['rating'].std()
print('standard_deviation of all movies: ',std_data)

avg_mean_student=ibr_new_2[ibr_new_2['occupation'] == 'student']['rating'].mean()
print('average of all students: ',avg_mean_student)


# In[117]:


number_of_students=21957


# In[118]:


t_val=(3.48-3.53)/math.sqrt(std_data*(1/number_of_students)+(1/number_of_movies))
print('t_val: ',t_val)

#for alpha=0.05

t_critical = 1.645

if(abs(t_val)>t_critical):
    print('We reject null hypothesis and accept alternative hypothesis')
else:
    print('We accept null hypothesis')
    
    
#Confidence_Interval

upper_limit = (3.48-3.53) + (t_critical * std_data* math.sqrt((1/number_of_students)+(1/number_of_movies)))
                                           
lower_limit = (3.48-3.53) - (t_critical * std_data*math.sqrt((1/number_of_students)+(1/number_of_movies)))
print(lower_limit)
                                           
print('confidence_interval: (',upper_limit,' ',lower_limit,')')


# In[119]:


t_val=(3.48-3.53)/math.sqrt(std_data*(1/number_of_students)+(1/number_of_movies))
print('t_val: ',t_val)

#for alpha=0.01

t_critical = 2.326

if(abs(t_val)>t_critical):
    print('We reject null hypothesis and accept alternative hypothesis')
else:
    print('We accept null hypothesis')
    
    
#Confidence_Interval

upper_limit = (3.48-3.53) + (t_critical * std_data* math.sqrt((1/number_of_students)+(1/number_of_movies)))
                                           
lower_limit = (3.48-3.53) - (t_critical * std_data*math.sqrt((1/number_of_students)+(1/number_of_movies)))
print(lower_limit)
                                           
print('confidence_interval: (',upper_limit,' ',lower_limit,')')


# In[120]:


t_val=(3.48-3.53)/math.sqrt(std_data*(1/number_of_students)+(1/number_of_movies))
print('t_val: ',t_val)

#for alpha=0.1

t_critical = 1.645

if(abs(t_val)>t_critical):
    print('We reject null hypothesis and accept alternative hypothesis')
else:
    print('We accept null hypothesis')
    
    
#Confidence_Interval

upper_limit = (3.48-3.53) + (t_critical * std_data* math.sqrt((1/number_of_students)+(1/number_of_movies)))
                                           
lower_limit = (3.48-3.53) - (t_critical * std_data*math.sqrt((1/number_of_students)+(1/number_of_movies)))
print(lower_limit)
                                           
print('confidence_interval: (',upper_limit,' ',lower_limit,')')


# In[121]:


print('students rating is higher than the average rating as we are rejecting null hypothesis')


# In[ ]:




