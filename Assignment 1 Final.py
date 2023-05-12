#!/usr/bin/env python
# coding: utf-8

# # 1

# In[2]:


import pandas as pd
import numpy as np

ibr_user=pd.read_csv("u(1).user.csv")
ibr_user.columns = ['userid','age','gender','occpuation','zip']

ibr_data=pd.read_csv("u(1).data.csv",sep=',' , header =None)
ibr_item=pd.read_csv("u.item.csv", encoding='latin-1')

ibr_data.columns = ['userid', 'itemid', 'rating', 'timestamp']

numberOfUsers = len(ibr_user)+1  #ibr_user.count()+1
numberOfItems = len(ibr_item)+1 #ibr_item.count()+1
print('Number of users: ', numberOfUsers, 'Number of Items: ', numberOfItems)


# In[3]:


ibr_user.head()


# In[4]:


print(ibr_data)


# # 2

# In[6]:


mean_rating = ibr_data['rating'].mean()
mean_rating


# # 3

# In[8]:


variance = ibr_data['rating'].var()
variance


# # 4

# In[10]:


std = ibr_data['rating'].std()
std


# # 5

# In[12]:


meanSTD = ibr_data['rating'].mad()
meanSTD


# # 6

# In[14]:


checkUser = (ibr_data['userid'] == 10)
avRatingUser10 = ibr_data.loc[checkUser, ['rating']]
avRatingUser10.mean()


# #7

# In[15]:


checkRating = (ibr_data['itemid'] == 10)
avRating10 = ibr_data.loc[checkRating, ['rating']]
avRating10.mean()


# In[16]:


ibr_item=pd.read_csv("u.item.csv", encoding='latin1',header = None)
ibr_item.columns = ['itemid','tit/yr','date','Nan','web','unknow','Acti','Adv','Anim','Child','Comd','Crime','Docm','Dra','Fant','F-Noir','Horror','Musc','Myst','Rom','Sci-fi','Thril','War','Western']
#print (ibr_item)
ibr_item.head()


# In[17]:


ibr_data=pd.read_csv("u(1).data.csv",sep=',' , header =None)
ibr_data.columns = ['userid', 'itemid', 'rating', 'timestamp']
ibr_data.head()


# In[ ]:





# In[18]:


merge = ibr_data.merge(ibr_item, on ="itemid", how = "left")
merge.head()


# ## 8.

# In[19]:


a1=merge[merge['Comd']==1]['rating'].mean()


# # 9
# 

# In[20]:


a2=merge[merge['Acti']==1]['rating'].mean()
    
    


# In[21]:


a3=merge[merge['Adv']==1]['rating'].mean()


# In[22]:


a4=merge[merge['Anim']==1]['rating'].mean()


# In[23]:


a5=merge[merge['Child']==1]['rating'].mean()


# In[24]:


a6=merge[merge['Crime']==1]['rating'].mean()


# In[25]:


a7=merge[merge['Docm']==1]['rating'].mean()


# In[26]:


a8=merge[merge['Dra']==1]['rating'].mean()


# In[27]:


a9=merge[merge['Fant']==1]['rating'].mean()


# In[28]:


a10=merge[merge['F-Noir']==1]['rating'].mean()


# In[29]:


a11=merge[merge['Horror']==1]['rating'].mean()


# In[30]:


a12=merge[merge['Musc']==1]['rating'].mean()


# In[31]:


a13=merge[merge['Myst']==1]['rating'].mean()


# In[32]:


a14=merge[merge['Rom']==1]['rating'].mean()


# In[33]:


a15=merge[merge['Sci-fi']==1]['rating'].mean()


# In[34]:


a16=merge[merge['Thril']==1]['rating'].mean()


# In[35]:


a17=merge[merge['War']==1]['rating'].mean()


# In[36]:


a18=merge[merge['Western']==1]['rating'].mean()


# In[37]:


a19=merge[merge['unknow']==1]['rating'].mean()


# In[38]:


array_movie_rank=np.array([a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19])


# In[39]:


array_movie_rank.sort()
print(array_movie_rank)


# # 10

# In[40]:


combine_occupation = ibr_user.merge(ibr_data, on ='userid', how = "left")
print(combine_occupation)


# In[46]:


ibra_rank = combine_occupation.groupby("occpuation")["rating"].mean().rank(ascending=False)
print(ibra_rank)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




