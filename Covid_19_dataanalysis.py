#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import plotly.graph_objects as go
import plotly.express as px

import warnings
warnings.filterwarnings('ignore')


# In[2]:


import os


# In[3]:


files=os.listdir(r'/Users/gowthammarrapu/Documents/untitled folder 2/Covid-19')
files


# In[4]:


def read_data(path,filename):
    return pd.read_csv(path+'/'+filename)


# In[6]:


path='/Users/gowthammarrapu/Documents/untitled folder 2/Covid-19'
world_data=read_data(path,'worldometer_data.csv')


# In[7]:


day_wise=read_data(path,files[2])


# In[8]:


group_data=read_data(path,files[3])


# In[9]:


usa_data=read_data(path,files[4])


# In[10]:


usa_data=read_data(path,files[4])


# In[11]:


#### Which Country has maximum Total cases, Deaths, Recovered & active cases 
#### lets create TreeMap Representation of our data


# In[12]:


world_data.columns


# In[13]:


columns=['TotalCases','TotalDeaths','TotalRecovered','ActiveCases']
for i in columns:
    fig=px.treemap(world_data[0:20],values=i,path=['Country/Region'],template="plotly_dark",title="<b>TreeMap representation of different Countries w.r.t. their {}</b>".format(i))
    fig.show()


# In[14]:


### what is the trend of Confirmed Deaths Recovered Active cases


# In[15]:


fig=px.line(day_wise,x="Date",y=["Confirmed","Deaths","Recovered","Active"],title="covid cases w.r.t. date",template="plotly_dark")
fig.show()


# In[16]:


### finding 20 most effected countries


# In[17]:


pop_test_ratio=world_data.iloc[0:20]['Population']/world_data.iloc[0:20]['TotalTests']


# In[18]:


pop_test_ratio


# In[19]:


fig=px.bar(world_data.iloc[0:20],color='Country/Region',y=pop_test_ratio,x='Country/Region',template="plotly_dark",title="<b>population to tests done ratio</b>")
fig.show()


# In[20]:


### 20 countries that are badly affected by corona 


# In[21]:


fig=px.bar(world_data.iloc[0:20],x='Country/Region',y=['Serious,Critical','TotalDeaths','TotalRecovered','ActiveCases','TotalCases'],template="plotly_dark")


# In[22]:


fig.update_layout({'title':"Coronavirus cases w.r.t. time"})
fig.show()


# In[23]:


world_data.head()


# In[24]:


world_data['Country/Region'].nunique()


# In[25]:


### Top 20 countries of Total Confirmed cases


# In[26]:


fig=px.bar(world_data.iloc[0:20],y='Country/Region',x='TotalCases',color='TotalCases',text="TotalCases")
fig.update_layout(template="plotly_dark",title_text="<b>Top 20 countries of Total confirmed cases</b>")
fig.show()


# In[27]:


### Top 20 countries of Total deaths


# In[28]:


fig=px.bar(world_data.sort_values(by='TotalDeaths',ascending=False)[0:20],y='Country/Region',x='TotalDeaths',color='TotalDeaths',text="TotalDeaths")
fig.update_layout(template="plotly_dark",title_text="<b>Top 20 countries of Total deaths</b>")
fig.show()


# In[29]:


### Top 20 countries of Total active cases


# In[30]:


fig=px.bar(world_data.sort_values(by='ActiveCases',ascending=False)[0:20], y='Country/Region',x='ActiveCases',color='ActiveCases',text='ActiveCases')
fig.update_layout(template="plotly_dark",title_text="<b>Top 20 countries of Total Active cases")
fig.show()


# In[31]:


### Top 20 countries of Total Recoveries


# In[32]:


fig=px.bar(world_data.sort_values(by='TotalRecovered',ascending=False)[:20],y='Country/Region',x='TotalRecovered',color='TotalRecovered',text='TotalRecovered')
fig.update_layout(template="plotly_dark",title_text="<b>Top 20 countries of Total Recovered")
fig.show()


# In[33]:


world_data.columns


# In[34]:


world_data[0:15]['Country/Region'].values


# In[35]:


### Pie Chart Representation of stats of worst affected countries


# In[36]:


labels=world_data[0:15]['Country/Region'].values
cases=['TotalCases','TotalDeaths','TotalRecovered','ActiveCases']
for i in cases:
    fig=px.pie(world_data[0:15],values=i,names=labels,template="plotly_dark",hole=0.3,title=" {} Recordeded w.r.t. to WHO Region of 15 worst effected countries ".format(i))
    fig.show()


# In[37]:


### Deaths to Confirmed ratio


# In[38]:


deaths_to_confirmed=((world_data['TotalDeaths']/world_data['TotalCases']))
fig = px.bar(world_data,x='Country/Region',y=deaths_to_confirmed)
fig.update_layout(title={'text':"Death to confirmed ratio of some  worst effected countries",'xanchor':'left'},template="plotly_dark")
fig.show()


# In[39]:


### Deaths to recovered ratio


# In[40]:


deaths_to_recovered=((world_data['TotalDeaths']/world_data['TotalRecovered']))
fig = px.bar(world_data,x='Country/Region',y=deaths_to_recovered)
fig.update_layout(title={'text':"Death to recovered ratio of some  worst effected countries",'xanchor':'left'},template="plotly_dark")
fig.show()


# In[41]:


### Tests to Confirmed Ratio


# In[42]:


tests_to_confirmed=((world_data['TotalTests']/world_data['TotalCases']))
fig = px.bar(world_data,x='Country/Region',y=tests_to_confirmed)
fig.update_layout(title={'text':"Tests to confirmed ratio of some  worst effected countries",'xanchor':'left'},template="plotly_dark")
fig.show()


# In[43]:


### Serious to Deaths Ratio


# In[44]:


serious_to_death=((world_data['Serious,Critical']/world_data['TotalDeaths']))
fig = px.bar(world_data,x='Country/Region',y=serious_to_death)
fig.update_layout(title={'text':"serious to Death ratio of some  worst effected countries",'xanchor':'left'},template="plotly_dark")
fig.show()


# In[45]:


#### Visualize Confirmed,  Active,  Recovered , Deaths Cases(entire statistics ) of a particular country


# In[46]:


group_data.head()


# In[47]:


from plotly.subplots import make_subplots
import plotly.graph_objects as go


# In[48]:



def country_visualization(group_data,country):
    
    data=group_data[group_data['Country/Region']==country]
    df=data.loc[:,['Date','Confirmed','Deaths','Recovered','Active']]
    fig = make_subplots(rows=1, cols=4,subplot_titles=("Confirmed", "Active", "Recovered",'Deaths'))
    fig.add_trace(
        go.Scatter(name="Confirmed",x=df['Date'],y=df['Confirmed']),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(name="Active",x=df['Date'],y=df['Active']),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(name="Recovered",x=df['Date'],y=df['Recovered']),
        row=1, col=3
    )

    fig.add_trace(
        go.Scatter(name="Deaths",x=df['Date'],y=df['Deaths']),
        row=1, col=4
    )

    fig.update_layout(height=600, width=1000, title_text="Date Vs Recorded Cases of {}".format(country),template="plotly_dark")
    fig.show()


# In[52]:


country_visualization(group_data,'India')


# In[53]:


country_visualization(group_data,'US')

