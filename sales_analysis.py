#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[6]:


all_data = pd.read_feather(r'/Users/gowthammarrapu/Documents/untitled folder 2/Sales_data.ftr')


# In[7]:


all_data


# In[8]:


all_data.isnull().sum()


# In[9]:


all_data = all_data.dropna(how="all")


# In[11]:


all_data.isnull().sum()


# In[13]:


all_data.duplicated()


# In[14]:


all_data[all_data.duplicated()]


# In[15]:


all_data = all_data.drop_duplicates()


# In[16]:


all_data.shape


# In[17]:


all_data[all_data.duplicated()]


# In[19]:


all_data.head(2)


# In[21]:


all_data['Order Date'][0]


# In[24]:


'04/19/19 08:46'.split(' ')[0]


# In[26]:


'04/19/19 08:46'.split(' ')[0].split('/')[0]


# In[28]:


all_data['Order Date'][0].split('/')[0]


# In[31]:


def return_month(x):
    return x.split('/')[0]


# In[37]:


all_data['month'] = all_data['Order Date'].apply(return_month)


# In[42]:


all_data.dtypes


# In[43]:


all_data['month'].unique()


# In[44]:


filter1 = all_data['month']=='Order Date'


# In[45]:


filter1


# In[46]:


all_data[filter1]


# In[48]:


all_data = all_data[~filter1]


# In[49]:


all_data


# In[59]:


import warnings
from warnings import filterwarnings
filterwarnings('ignore')


# In[60]:


all_data['month'] = all_data['month'].astype(int)


# In[52]:


all_data['month'].unique()


# In[63]:


all_data.dtypes


# In[64]:


all_data['Quantity Ordered'] = all_data['Quantity Ordered'].astype(int)
all_data['Price Each'] = all_data['Price Each'].astype(float)


# In[65]:


all_data.dtypes


# In[66]:


all_data['sales'] = all_data['Quantity Ordered'] * all_data['Price Each']


# In[67]:


all_data['sales']


# In[69]:


all_data.groupby(['month'])['sales'].sum()


# In[71]:


all_data.groupby(['month'])['sales'].sum().plot(kind= 'bar')


# In[ ]:





# In[73]:


all_data.head(2)


# In[76]:


all_data['Purchase Address'][0]


# In[77]:


all_data['Purchase Address'][0].split(',')[1]


# In[78]:


def return_city(x):
    return x.split(',')[1]


# In[83]:


all_data['city'] = all_data['Purchase Address'].apply(return_city)


# In[84]:


all_data_city


# In[85]:


# all_data['Purchase Address'].str.split(',').str.get(1)


# In[89]:


pd.value_counts(all_data['city'])


# In[92]:


pd.value_counts(all_data['city']).plot(kind='pie', autopct = '%1.0f%%')


# In[93]:


all_data.head(3)


# In[96]:


#all_data.groupby(['Product'])['Quantity Ordered'].sum()


# In[98]:


count_df = all_data.groupby(['Product']).agg({'Quantity Ordered':'sum','Price Each':'mean'})


# In[99]:


count_df


# In[100]:


count_df =count_df.reset_index()


# In[101]:


count_df


# In[104]:


products = count_df['Product'].values


# In[105]:


products


# In[112]:


fig, ax1 = plt.subplots()
ax2 = ax1.twinx()

ax1.bar(count_df['Product'],count_df['Quantity Ordered'], color = 'g')
ax2.plot(count_df['Product'],count_df['Price Each'])
ax1.set_xticklabels(products , rotation='vertical', fontsize = 8)

ax1.set_ylabel('Order Count')
ax1.set_xlabel('Avg Price of Product')


# In[113]:


all_data.head(2)


# In[118]:


most_sold_product = all_data['Product'].value_counts()[0:5].index


# In[119]:


all_data['Product'].isin(most_sold_product)


# In[123]:


most_sold_product_df = all_data[all_data['Product'].isin(most_sold_product)]


# In[125]:


most_sold_product_df.head(4)


# In[130]:


pivot = most_sold_product_df.groupby(['month', 'Product']).size().unstack()


# In[131]:


pivot.plot(figsize=(8,6))


# In[ ]:





# In[ ]:





# In[133]:


df_duplicate = all_data[all_data['Order ID'].duplicated(keep=False)]


# In[134]:


df_duplicate


# In[146]:


df_duplicate.groupby(['Order ID'])['Product'].apply(lambda x : ','.join(x))


# In[147]:


all_data.head(6)


# In[148]:


all_data = all_data.drop('Grouped_orders',axis=1)


# In[149]:


all_data


# In[153]:


dup_products = df_duplicate.groupby(['Order ID'])['Product'].apply(lambda x : ','.join(x)).reset_index().rename(columns={'Product':'Grouped_products'})


# In[155]:


dup_products


# In[157]:


dup_products_df = df_duplicate.merge(dup_products, how='left', on='Order ID')


# In[158]:


dup_products_df


# In[160]:


no_dup_df = dup_products.drop_duplicates(subset=['Order ID'])


# In[161]:


no_dup_df


# In[168]:


no_dup_df['Grouped_products'].value_counts()[0:5].plot.pie()


# In[ ]:




