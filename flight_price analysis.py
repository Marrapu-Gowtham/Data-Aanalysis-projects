#!/usr/bin/env python
# coding: utf-8

# In[158]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[8]:


train_data = pd.read_excel(r'/Users/gowthammarrapu/Documents/untitled folder 2/Data_Train.xlsx')


# In[9]:


train_data.head(5)


# In[10]:


train_data.isnull().sum()


# In[12]:


train_data.duplicated().sum()


# In[13]:


train_data.info()


# In[15]:


train_data[train_data['Total_Stops'].isnull()]


# In[20]:


train_data.dropna(inplace=True)


# In[21]:


train_data.isnull().sum()


# In[22]:


train_data.dtypes


# In[23]:


train_data.info(memory_usage= 'deep')


# In[24]:


data = train_data.copy()


# In[26]:


data .columns


# In[27]:


data.head(2)


# In[28]:


import warnings
from warnings import filterwarnings
filterwarnings ('ignore')


# In[30]:


def change_into_Datetime(col):
    data[col] = pd.to_datetime(data[col])


# In[31]:


data.columns


# In[34]:


for feature in ['Dep_Time', 'Arrival_Time','Date_of_Journey']:
    change_into_Datetime(feature)


# In[36]:


data.dtypes


# In[44]:


data['Journey_day'] = data['Date_of_Journey'].dt.day


# In[45]:


data['Journey_month'] = data['Date_of_Journey'].dt.month


# In[46]:


data['Journey_year'] = data['Date_of_Journey'].dt.year


# In[47]:


data.head(4)


# In[48]:


data.columns


# In[49]:


def extract_hour_min(df,col):
    df[col + '_hour'] = df[col].dt.hour
    df[col + '_minute'] = df[col].dt.minute
    return df.head(3)
    


# In[50]:


extract_hour_min(data,'Dep_Time')


# In[51]:


extract_hour_min(data,'Arrival_Time')


# In[52]:


col_drop = ['Arrival_Time','Dep_Time']


# In[53]:


data.drop(col_drop,axis=1,inplace=True)


# In[54]:


data.head(3)


# In[55]:


def flight_dep_time (x):
    if (x>4) and (x<=8):
        return 'Early Morning'
    elif (x>8) and (x<=12):
        return 'Morning'
    elif (x>12) and (x<=16):
        return 'Noon'
    elif (x>16) and (x<=20):
        return 'Evening'
    elif (x>20) and (x<=24):
        return 'Night'
    else:
        return 'Late Night'
    
    


# In[63]:


data['Dep_Time_hour'].apply(flight_dep_time).value_counts().plot(kind = 'bar', color = 'g')


# In[64]:


get_ipython().system('pip install plotly')
get_ipython().system('pip install chart_studio')


# In[68]:


pip install cufflinks


# In[72]:


import plotly
import cufflinks as cf
from cufflinks.offline import go_offline
from plotly.offline import plot, iplot, init_notebook_mode,download_plotlyjs
init_notebook_mode(connected=True)
cf.go_offline()


# In[73]:


data['Dep_Time_hour'].apply(flight_dep_time).value_counts().iplot(kind = "bar" )


# In[74]:


data['Duration']


# In[122]:


def preprocess_duration(x):
    if 'h' not in x:
        x = '0h' + ' ' + x
    elif 'm' not in x:
        x =  x + ' ' + '0m'
    return x


# In[123]:


data['Duration'].apply(preprocess_duration)


# In[ ]:





# In[124]:


data['Duration'][0]


# In[125]:


'2h 50m'.split(' ')[0][0:-1]


# In[126]:


type('2h 50m'.split(' ')[0][0:-1])


# In[127]:


int('2h 50m'.split(' ')[0][0:-1])


# In[128]:


int('2h 50m'.split(' ')[1][0:-1])


# In[129]:


data['Duration'].apply(lambda x : int(x.split(' ')[0][0:-1]) )


# In[132]:


def extract_hours_minutes(duration):
    try:
        parts = duration.split(' ')
        hours = int(parts[0][:-1]) if 'h' in parts[0] else 0
        minutes = int(parts[1][:-1]) if len(parts) > 1 and 'm' in parts[1] else 0
        return hours, minutes
    except (IndexError, ValueError):
        return None, None  # Handle the error gracefully

# Apply the function to the 'Duration' column
data['Duration_hours'], data['Duration_minutes'] = zip(*data['Duration'].apply(extract_hours_minutes))

# Check the result
print(data)


# In[135]:


data.head(3)


# In[136]:


# data['Duration'].apply(lambda x : int(x.split(' ')[1][0:-1]) )


# In[120]:


data.dtypes


# In[121]:


data.head(3)


# In[137]:


data.isnull().sum()


# In[144]:


data.drop('Duration_int', axis=1,inplace=True)


# In[145]:


data.head(5)


# In[146]:


2*60


# In[147]:


'2*60'


# In[148]:


eval('2*60')


# In[155]:


data['Duration_total_mins'] = data['Duration'].str.replace('h',"*60").str.replace(' ','+').str.replace('m','*1').apply(eval)


# In[156]:


data.head(4)


# In[160]:


sns.scatterplot(x = 'Duration_total_mins',y='Price', hue = 'Total_Stops',data = data)


# In[161]:


sns.lmplot(x = 'Duration_total_mins',y='Price',data = data)


# In[166]:


data['Airline']== 'Jet Airways'


# In[167]:


data[data['Airline']== 'Jet Airways']


# In[171]:


data[data['Airline']== 'Jet Airways'].groupby('Route').size().sort_values(ascending=False)


# In[180]:


plt.figure(figsize=(20,12))

sns.boxplot(y = "Price", x = 'Airline', data=data.sort_values('Price', ascending=False))
plt.xticks(rotation='vertical')
plt.show()


# In[181]:


cat_col = [col for col in data.columns if data[col].dtypes == 'object']


# In[183]:


cat_col


# In[182]:


num_col = [col for col in data.columns if data[col].dtypes != 'object']


# In[184]:


num_col


# In[185]:


data['Source'].unique()


# In[187]:


for sub_category in data['Source'].unique():
     data['Source_'+sub_category]= data['Source'].apply(lambda x : 1 if x == sub_category else 0)
    


# In[186]:


data['Source'].apply(lambda x : 1 if x == 'Banglore' else 0)


# In[188]:


data.columns


# In[189]:


data.head(4)


# In[194]:


data.groupby(['Airline'])['Price'].mean().sort_values()


# In[198]:


airlines =  data.groupby(['Airline'])['Price'].mean().sort_values().index


# In[199]:


airlines 


# In[206]:


dict_airlines = {key: index for index, key in enumerate(airlines,0)}
    


# In[207]:


dict_airlines


# In[221]:


data['Airline'] = data['Airline'].map(dict_airlines)


# In[222]:


data['Airline']


# In[223]:


data['Destination'].unique()


# In[224]:


data['Destination'].replace('New Delhi', 'Delhi',inplace=True)


# In[225]:


data['Destination'].unique()


# In[226]:


dest =  data.groupby(['Destination'])['Price'].mean().sort_values().index


# In[227]:


dest


# In[228]:


dict_dest = {key: index for index, key in enumerate(dest,0)}


# In[229]:


dict_dest


# In[230]:


data['Destination'] = data['Destination'].map(dict_dest)


# In[231]:


data['Destination']


# In[232]:


data.head(4)


# In[332]:


data['Total_Stops'].unique()


# In[333]:


stop = {'non-stop':0,'2 stops':2,'1 stop':1,'4 stops':4}


# In[334]:


stop


# In[335]:


data['Total_Stops'] = data['Total_Stops'].map(stop)


# In[336]:


data['Total_Stops']


# In[240]:


data.head(4)


# In[243]:


data['Additional_Info'].value_counts()/len(data)*100


# In[244]:


data.columns


# In[247]:


data.drop(columns=['Additional_Info','Date_of_Journey','Duration_total_mins','Source','Journey_year'],axis=1,inplace=True)


# In[248]:


data.columns


# In[249]:


data.drop(columns=['Route'],axis=1,inplace=True)


# In[250]:


data.columns


# In[251]:


data.head(5)


# In[260]:


def plot(df,col):
    fig, (ax1,ax2,ax3) = plt.subplots(3,1)
    
    sns.distplot(df[col], ax = ax1)
    sns.boxplot(df[col], ax = ax2)
    sns.distplot(df[col], ax = ax3, kde=False)


# In[261]:


plot(data,'Price')


# In[264]:


q1 = data['Price'].quantile(0.25)
q3 = data['Price'].quantile(0.75)

iqr = q3 - q1

maximum = q3 + 1.5*iqr
minumum = q1 - 1.5*iqr


# In[265]:


maximum


# In[266]:


minumum 


# In[268]:


print([price for price in data['Price'] if price>maximum or price<minumum])


# In[269]:


len([price for price in data['Price'] if price>maximum or price<minumum])


# In[271]:


data['Price'] = np.where(data['Price']>35000, data['Price'].median(),data['Price'])


# In[285]:


plot(data,'Price')


# In[344]:


#data[data['Total_Stops'].isnull()]


# In[347]:


#data.drop(columns=('Total_Stops'),axis = 1,inplace = True)


# In[348]:


X = data.drop(['Price'],axis = 1)


# In[349]:


y = data['Price']


# In[351]:


from sklearn.feature_selection import mutual_info_regression

imp = mutual_info_regression(X,y)


# In[355]:


imp_df = pd.DataFrame(imp, index=X.columns)


# In[356]:


imp_df.columns = ['importance']


# In[357]:


imp_df


# In[358]:


imp_df.sort_values(by='importance',ascending=False)


# In[ ]:




