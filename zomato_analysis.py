#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


import sqlite3


# In[3]:


con = sqlite3.connect(r'/Users/gowthammarrapu/Documents/untitled folder 2/zomato_rawdata.sqlite')


# In[4]:


df = pd.read_sql_query("SELECT * FROM Users", con)


# In[5]:


df.head(4)


# In[6]:


df.isnull().sum()


# In[7]:


df.isnull().sum()/len(df)*100


# In[8]:


df['rate'].unique()


# In[9]:


df['rate'].replace(('NEW','-'), np.nan, inplace=True)


# In[10]:


df['rate'].unique()


# In[11]:


'4.1/5'


# In[12]:


'4.1/5'.split('/')[0]


# In[13]:


float('4.1/5'.split('/')[0])


# In[14]:


df['rate'] = df['rate'].apply(lambda x : float(x.split('/')[0]) if type(x)==str else x)


# In[15]:


df['rate']


# In[ ]:





# In[16]:


df


# In[17]:


df.columns


# In[18]:


x = df[['rate','online_order']]


# In[19]:


x


# In[20]:












x = pd.crosstab(df['rate'],df['online_order'])


# In[21]:


x


# In[22]:


x.plot(kind='bar', stacked=True)


# In[23]:


x.sum(axis =1).astype(float)


# In[24]:


normalise_df = x.div(x.sum(axis =1).astype(float),axis=0)


# In[25]:


(normalise_df*100).plot(kind='bar', stacked=True)


# In[26]:


df['rest_type'].isnull().sum()


# In[27]:


data = df.dropna(subset=['rest_type'])


# In[28]:


data['rest_type'].isnull().sum()


# In[29]:


data['rest_type'].unique()


# In[30]:


quick_Bites_df = data[data['rest_type'].str.contains('Quick Bites')]


# In[31]:


quick_Bites_df.shape


# In[32]:


quick_Bites_df.columns


# In[33]:


quick_Bites_df['reviews_list']


# In[34]:


quick_Bites_df['reviews_list']=quick_Bites_df['reviews_list'].apply(lambda x:x.lower())


# In[35]:


from nltk import RegexpTokenizer 


# In[36]:


tokenizer = RegexpTokenizer("[a-zA-Z]+")


# In[37]:


tokenizer


# In[38]:


tokenizer.tokenize(quick_Bites_df['reviews_list'][3])


# In[39]:


sample = data[0:1000]


# In[40]:


reviews_token = sample['reviews_list'].apply(tokenizer.tokenize)


# In[41]:


reviews_token


# In[42]:


from nltk.corpus import stopwords


# In[43]:


stop = stopwords.words('english')


# In[44]:


print(stop)


# In[45]:


stop.extend(['rated','n','nan','x','RATED','Rated'])


# In[46]:


print(stop)


# In[47]:


rev3 = reviews_token[3]
print(rev3)


# In[ ]:





# In[48]:


print([token for token in rev3 if token not in stop])


# In[ ]:





# In[49]:


reviews_token_clean=reviews_token.apply(lambda each_review: [token for token in rev3 if token not in stop])


# In[50]:




print(reviews_token_clean)


# In[51]:


type(reviews_token_clean)


# In[52]:


total_reviews_2D = list(reviews_token_clean)


# In[53]:


total_reviews_1D = []

for review in total_reviews_2D:
    for word in review:
        total_reviews_1D.append(word)
    


# In[54]:


total_reviews_1D


# In[55]:


from nltk import FreqDist


# In[56]:


fd = FreqDist()


# In[57]:


for word in total_reviews_1D:
    fd[word] =     fd[word] + 1


# In[58]:


fd.most_common(20)


# In[59]:


plt.figure(figsize=(8,6))
fd.plot()


# In[60]:


from nltk import FreqDist, bigrams,trigrams


# In[61]:


bi_grams = bigrams(total_reviews_1D)


# In[62]:


bi_grams


# In[63]:


fd_bigrams = FreqDist()

for bigram in bi_grams:
    fd_bigrams[bigram]=fd_bigrams[bigram]+1


# In[64]:


fd_bigrams.most_common(20)


# In[65]:


fd_bigrams.plot(20)


# In[ ]:





# In[66]:


tri_gram = trigrams(total_reviews_1D)


# In[67]:


fd_trigrams = FreqDist()

for trigram in tri_gram:
    fd_trigrams[trigram]=fd_trigrams[trigram]+1


# In[68]:


fd_trigrams.plot(20)


# In[69]:


fd_trigrams


# In[70]:


fd_trigrams.most_common(20)


# In[71]:


get_ipython().system('pip install geocoder')
get_ipython().system('pip install geopy')


# In[72]:


len(df['location'].unique())


# In[73]:


df['location'] = df['location'] + " , Banglore, Karnataka, India"


# In[74]:


df['location'].unique()


# In[75]:


df_copy = df.copy()


# In[76]:


df_copy['location'].isnull().sum()


# In[77]:


df_copy = df_copy.dropna(subset=['location'])


# In[78]:


df_copy['location'].isnull().sum()


# In[79]:


locations = pd.DataFrame(df_copy['location'].unique())


# In[80]:


locations.columns = ['name']


# In[81]:


locations


# In[82]:


get_ipython().system('pip install geopy')


# In[83]:


from geopy.geocoders import Nominatim


# In[84]:


geolocator = Nominatim(user_agent="app" , timeout=None)


# In[85]:


geolocator


# In[86]:


mylocations = pd.DataFrame({
    'name': [
        'Banashankari, Bangalore, Karnataka, India',
        'Basavanagudi, Bangalore, Karnataka, India',
        'Mysore Road, Bangalore, Karnataka, India',
        'Jayanagar, Bangalore, Karnataka, India',
        'Kumaraswamy Layout, Bangalore, Karnataka, India',
        'West Bangalore, Bangalore, Karnataka, India',
        'Magadi Road, Bangalore, Karnataka, India',
        'Yelahanka, Bangalore, Karnataka, India',
        'Sahakara Nagar, Bangalore, Karnataka, India',
        'Peenya, Bangalore, Karnataka, India'
    ]
})
mylocations


# In[87]:


mylocations = pd.DataFrame({
    'name': [
        'Cubbon Park, Bangalore, Karnataka, India',
        'Brigade Road, Bangalore, Karnataka, India',
        'MG Road, Bangalore, Karnataka, India',
        'Koramangala, Bangalore, Karnataka, India',
        'Indiranagar, Bangalore, Karnataka, India',
        'Whitefield, Bangalore, Karnataka, India',
        'Bangalore Palace, Bangalore, Karnataka, India',
        'Vidhana Soudha, Bangalore, Karnataka, India',
        'Lalbagh Botanical Garden, Bangalore, Karnataka, India',
        'Bannerghatta National Park, Bangalore, Karnataka, India'
    ]
})

# Initialize the geolocator
geolocator = Nominatim(user_agent="app" , timeout=None)

# Loop through each row in the DataFrame
for index, row in mylocations.iterrows():
    location_name = row['name']  # Access the value in the 'name' column
    geoLocation = geolocator.geocode(location_name)
    
    if geoLocation is None:
        print(f"{index}: {location_name}: Not found")
    else:
        print(f"{index}: {location_name}: Latitude = {geoLocation.latitude}, Longitude = {geoLocation.longitude}")


# In[88]:


geolocator = Nominatim(user_agent="app" , timeout=None)
lat = []
lon = []

for location in locations['name']:
    geoLocation = geolocator.geocode(location)
    if geoLocation is None:
        lat.append(np.nan)
        lon.append(np.nan)
    else:
        lat.append(geoLocation.latitude)
        lon.append(geoLocation.longitude)

# Assigning the lists to new columns in the DataFrame
locations['latitude'] = lat
locations['longitude'] = lon
locations


# In[89]:


lat=[]
lon=[]

for location in locations['name']:
    geoLocation = geolocator.geocode(location)
    if geoLocation is None:
        lat.append(np.nan)
        lon.append(np.nan)
    else:
        lat.append(location.latitude)
        lon.append(locaton.longitude)
        


# In[90]:


locations['latitude'] = lat
locations['longitude'] = lon


# In[91]:


locations


# In[92]:


locations.isnull().sum()


# In[93]:


df['cuisines'].isnull().sum()


# In[94]:


df = df.dropna(subset='cuisines')


# In[95]:


north_india = df[df['cuisines'].str.contains('North Indian')]


# In[96]:


north_india.shape


# In[97]:


north_india.head(2)


# In[98]:


north_india_rest_count=north_india['location'].value_counts().reset_index().rename(columns={'index':'name', "location":"count"})


# In[99]:


north_india_rest_count


# In[100]:


heatmap_df = north_india_rest_count.merge(locations, on= 'name', how = 'left')


# In[101]:


heatmap_df


# In[102]:


import folium


# In[103]:


basemap = folium.Map()


# In[104]:


basemap


# In[105]:


heatmap_df.columns


# In[106]:


from folium.plugins import HeatMap


# In[107]:


HeatMap(heatmap_df[['latitude', 'longitude', 'count']]).add_to(basemap)


# In[ ]:





# In[ ]:




