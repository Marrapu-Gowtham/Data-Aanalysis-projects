#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


import os


# In[3]:


os.listdir(r'/Users/gowthammarrapu/Documents/untitled folder 2/Datasets')


# In[4]:


uber_15 = pd.read_csv(r'/Users/gowthammarrapu/Documents/untitled folder 2/Datasets/uber-raw-data-janjune-15_sample.csv')


# In[5]:


uber_15.shape


# In[6]:


uber_15.isnull().sum()


# In[7]:


uber_15.duplicated().sum()


# In[8]:


uber_15.drop_duplicates(inplace=True)


# In[9]:


uber_15.duplicated().sum()


# In[10]:


uber_15.shape


# In[11]:


uber_15.dtypes


# In[12]:


uber_15.head(2)


# In[13]:


uber_15['Pickup_date'][0]


# In[14]:



type(uber_15['Pickup_date'][0])


# In[15]:


uber_15['Pickup_date'] = pd.to_datetime(uber_15['Pickup_date'])


# In[16]:


uber_15['Pickup_date'].dtypes


# In[17]:


type(uber_15['Pickup_date'][0])


# In[18]:


uber_15.dtypes


# In[19]:


uber_15['month'] = uber_15['Pickup_date'].dt.month_name()


# In[20]:


uber_15['month'].value_counts().plot(kind = 'bar')


# In[21]:


uber_15['weekday'] = uber_15['Pickup_date'].dt.day_name()
uber_15['day'] = uber_15['Pickup_date'].dt.day
uber_15['hour'] = uber_15['Pickup_date'].dt.hour
uber_15['minute'] = uber_15['Pickup_date'].dt.minute


# In[22]:


uber_15.head(3)


# In[23]:


pivort = pd.crosstab(index= uber_15['month'], columns = uber_15['weekday'])


# In[24]:


pivort


# In[25]:


pivort.plot(kind='bar',figsize=(16,8))


# In[26]:


summary = uber_15.groupby(['weekday', 'hour'], as_index=False).size()


# In[27]:


plt.figure(figsize=(8,6))
sns.pointplot(x= 'hour', y='size', hue='weekday' , data= summary)


# In[28]:


get_ipython().system('pip install plotly')
get_ipython().system('pip install chart_studio')


# In[60]:


import chart_studio.plotly as py
import plotly .graph_objs as go
import plotly.express as px

from plotly.offline import download_plotlyjs, init_notebook_mode ,plot , iplot


# In[61]:


init_notebook_mode(connected=True)


# In[62]:


os.listdir(r'/Users/gowthammarrapu/Documents/untitled folder 2/Datasets')


# In[63]:


uber_foil = pd.read_csv(r'/Users/gowthammarrapu/Documents/untitled folder 2/Datasets/Uber-Jan-Feb-FOIL.csv')


# In[64]:


uber_foil.columns


# In[65]:


px.box(x = 'dispatching_base_number', y = 'active_vehicles', data_frame = uber_foil)


# In[66]:


files = os.listdir(r'/Users/gowthammarrapu/Documents/untitled folder 2/Datasets')[-13:]


# In[67]:


files 


# In[68]:


files.remove('Uber-Jan-Feb-FOIL.csv')


# In[69]:


files.remove('other-Highclass_B01717.csv')


# In[70]:


files.remove('other-Federal_02216.csv')


# In[71]:


files.remove('other-Carmel_B00256.csv')
             


# In[72]:



files.remove('other-Diplo_B01196.csv')


# In[73]:


files.remove('other-Dial7_B00887.csv')


# In[74]:


files.remove('other-Prestige_B01338.csv')


# In[75]:


files


# In[76]:


final = pd.DataFrame()
path = r'/Users/gowthammarrapu/Documents/untitled folder 2/Datasets'

for file in files:
    current_df = pd.read_csv(path+'/'+file)
    final = pd.concat([current_df,final])


# In[77]:


final


# In[78]:


final.shape


# In[79]:


final.duplicated().sum()


# In[80]:


final.drop_duplicates(inplace=True)


# In[81]:


final.duplicated().sum()


# In[82]:


final.shape


# In[83]:



rush_uber = final.groupby(['Lat', 'Lon'], as_index=False).size()


# In[84]:


rush_uber


# In[85]:


get_ipython().system('pip install folium')


# In[55]:


import folium


# In[86]:


basemap = folium.Map()


# In[87]:


basemap


# In[88]:


from folium.plugins import HeatMap


# In[89]:


HeatMap(rush_uber).add_to(basemap)


# In[90]:


basemap


# In[91]:


final.head(3)


# In[92]:


final.dtypes


# In[93]:


final['Date/Time'][0]


# In[94]:


final['Date/Time'] = pd.to_datetime(final['Date/Time'], format="%m/%d/%Y %H:%M:%S")


# In[95]:


final['Date/Time'].dtypes


# In[116]:


final['day'] = final['Date/Time'].dt.day
final['hour'] = final['Date/Time'].dt.hour


# In[117]:


pivortt = final.groupby(['day', 'hour']).size().unstack()


# In[118]:




# Assuming you've already converted 'Date/Time' to datetime
#final['Date/Time'] = pd.to_datetime(final['Date/Time'], format="%m/%d/%Y %H:%M:%S")

# Extracting minutes
#final['Minutes'] = final['Date/Time'].dt.minute


# In[119]:


pivortt


# In[120]:


pivortt.style.background_gradient()


# # automation

# In[100]:


def gen_pivort_table(df,col1,col2):
    pivortt = final.groupby(['day', 'hour']).size().unstack()
    return pivortt.style.background_gradient()  


# In[101]:


gen_pivort_table(final,'day','hour')


# In[200]:


final.columns


# In[102]:


pivortt


# In[103]:


final


# In[ ]:




