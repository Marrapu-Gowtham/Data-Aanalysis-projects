#!/usr/bin/env python
# coding: utf-8

# In[34]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[35]:


import glob


# In[36]:


glob.glob(r'/Users/gowthammarrapu/Documents/untitled folder 2/individual_stocks_5yr/*csv')


# In[37]:


len(glob.glob(r'/Users/gowthammarrapu/Documents/untitled folder 2/individual_stocks_5yr/*csv')
   )


# In[ ]:





# In[38]:


company_list = [r'/Users/gowthammarrapu/Documents/untitled folder 2/individual_stocks_5yr/AAPL_data.csv',
               r'/Users/gowthammarrapu/Documents/untitled folder 2/individual_stocks_5yr/AMZN_data.csv',
               r'/Users/gowthammarrapu/Documents/untitled folder 2/individual_stocks_5yr/GOOG_data.csv',
               r'/Users/gowthammarrapu/Documents/untitled folder 2/individual_stocks_5yr/MSFT_data.csv']
                


# In[39]:


import warnings
from warnings import filterwarnings
filterwarnings('ignore')


# In[40]:


all_data = pd.DataFrame()

for file in company_list:
    
    current_df = pd.read_csv(file)
    
    all_data = current_df.append(all_data, ignore_index = True)
    ##full_df = pd.concat([full_df, current_df], ignore_index = True)


# In[41]:


all_data.head(6)


# In[42]:


all_data['Name'].unique()


# In[ ]:





# In[43]:


all_data.isnull().sum()


# In[44]:


all_data.dtypes


# In[45]:


all_data['date'] = pd.to_datetime(all_data['date'])


# In[46]:


all_data['date']


# In[47]:


all_data['date'].dtypes


# In[48]:


tech_list = all_data['Name'].unique()


# In[49]:


tech_list


# In[ ]:





# In[52]:


plt.figure(figsize= (20,12))

for index, company in enumerate(tech_list,1):
    plt.subplot(2,2, index)
    filter1 = all_data['Name']==company
    df = all_data[filter1]
    plt.plot(df['date'], df['close'])
    plt.title(company)
    


# In[ ]:





# In[ ]:





# In[53]:


all_data.head(15)


# In[56]:


all_data['close'].rolling(window=10).mean().head(14)


# In[57]:


new_data = all_data.copy()


# In[75]:


ma_day = [10,20,50]

for ma in ma_day:
    new_data['close_'+str(ma)] = new_data['close'].rolling(ma).mean()


# In[76]:


new_data.head(6)


# In[116]:


new_data.set_index('date', inplace=True)


# In[78]:


new_data


# In[79]:


new_data.columns


# In[80]:


plt.figure(figsize= (20,12))

for index, company in enumerate(tech_list,1):
    plt.subplot(2,2, index)
    filter1 = new_data['Name']==company
    df = new_data[filter1]
    df[[ 'close_20','close_50']].plot(ax=plt.gca())
    plt.title(company)


# In[82]:


apple = pd.read_csv(r'/Users/gowthammarrapu/Documents/untitled folder 2/individual_stocks_5yr/AAPL_data.csv')


# In[83]:


apple


# In[85]:


apple['Daily return(in %) '] = apple['close'].pct_change()*100


# In[86]:


apple.head(5)


# In[87]:


import plotly.express as px


# In[91]:


px.line(apple, x='date', y ='Daily return(in %) ')


# In[92]:


apple['date'] = pd.to_datetime(apple['date'])


# In[93]:


apple['date']


# In[94]:


apple.set_index('date', inplace=True)


# In[95]:


apple.head(4)


# In[96]:


apple['close'].resample('M').mean().plot()


# In[97]:


apple['close'].resample('Y').mean().plot()


# In[99]:


apple['close'].resample('Q').mean().plot()


# In[ ]:





# In[100]:


company_list 


# In[101]:


company_list[0]


# In[103]:


app = pd.read_csv(r'/Users/gowthammarrapu/Documents/untitled folder 2/individual_stocks_5yr/AAPL_data.csv')
amzn = pd.read_csv(r'/Users/gowthammarrapu/Documents/untitled folder 2/individual_stocks_5yr/AMZN_data.csv')
google = pd.read_csv(r'/Users/gowthammarrapu/Documents/untitled folder 2/individual_stocks_5yr/GOOG_data.csv')
msft = pd.read_csv(r'/Users/gowthammarrapu/Documents/untitled folder 2/individual_stocks_5yr/MSFT_data.csv')


# In[104]:


closing_price = pd.DataFrame()


# In[110]:


closing_price['apple_close'] = app['close']
closing_price['amzn_close'] = amzn['close']
closing_price['google_close'] = google['close']
closing_price['msft_close'] = msft['close']


# In[111]:


closing_price


# In[112]:


sns.pairplot(closing_price)


# In[113]:


closing_price.corr()


# In[115]:


sns.heatmap(closing_price.corr(), annot=True)


# In[119]:


(closing_price['apple_close'] - closing_price['apple_close'].shift(1))/closing_price['apple_close'].shift(1) *100


# In[120]:


closing_price.columns


# In[121]:


for col in closing_price.columns:
    closing_price[col + '_pct_chane'] = (closing_price[col] - closing_price[col].shift(1))/closing_price[col].shift(1) *100


# In[123]:


closing_price.columns


# In[126]:


closing_p = closing_price[['apple_close_pct_chane', 'amzn_close_pct_chane',
       'google_close_pct_chane', 'msft_close_pct_chane']]


# In[127]:


closing_p


# In[131]:


g = sns.PairGrid(data = closing_p)
g.map_diag(sns.histplot)
g.map_lower(sns.scatterplot)
g.map_upper(sns.kdeplot)


# In[132]:


closing_p.corr()


# In[133]:


sns.heatmap(closing_p.corr(), annot=True)


# In[ ]:




