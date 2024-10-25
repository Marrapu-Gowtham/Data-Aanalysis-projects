#!/usr/bin/env python
# coding: utf-8

# In[74]:


import numpy as np
import pandas as pd


# In[75]:


import warnings
from warnings import filterwarnings
filterwarnings('ignore')


# In[76]:


comments = pd.read_csv(r"/Users/gowthammarrapu/Documents/untitled folder 2/UScomments.csv" , error_bad_lines=False)


# In[77]:


comments.head()


# In[78]:


comments.isnull().sum()


# In[79]:


comments.dropna(inplace=True)


# In[80]:


from textblob import TextBlob


# In[81]:


comments.head(6)


# In[82]:


TextBlob("Logan Paul it's yo big day ‼️‼️‼️").sentiment.polarity


# In[83]:


sample_df = comments[0:1000]


# In[84]:


sample_df


# In[85]:


sample_df['comment_text']


# In[86]:


polarity = []

for comment in sample_df['comment_text']:
    try:
        polarity.append(TextBlob(comment).sentiment.polarity)
    except:
        polarity.append(0)


# In[87]:


len(polarity)


# In[88]:


sample_df['polarity'] = polarity


# In[89]:


comments.head(5)


# In[90]:


sample_df.shape


# In[93]:


comments_positive = sample_df[sample_df['polarity'] == 1.0]

top_comments = filtered_comments.head(5)

top_comments


# In[67]:



from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt


# In[61]:


from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

comments_text = ' '.join(comments_positive['comment_text'])

stopwords = set(STOPWORDS)

wordcloud = WordCloud(stopwords=stopwords, background_color='white', width=800, height=400).generate(comments_text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')  
plt.show()


# In[62]:


comments_positive = ' '.join(comments_positive['comment_text'])


# In[63]:


wordcloud(stopwords=set(STOPWORDS)).generate()


# In[52]:


import emoji


# In[53]:


comments['comment_text'].head(6)


# In[54]:


comment = 'trending 😉'


# In[55]:


[char for char in comment if char in emoji.EMOJI_DATA]


# In[56]:


all_emoji_list = []
for comment in comments['comment_text'].dropna():
    for char in comment:
        if char in emoji.EMOJI_DATA:
            all_emoji_list.append(char)


# In[57]:


all_emoji_list


# In[58]:


from collections import Counter


# In[59]:


Counter(all_emoji_list).most_common(10)[3][0]


# In[60]:


emojis = [Counter(all_emoji_list).most_common(10)[i][0] for i in range(10)]


# In[61]:


frequency = [Counter(all_emoji_list).most_common(10)[i][1] for i in range(10)]


# In[62]:


emojis


# In[63]:


frequency


# In[64]:


import plotly.graph_objs as go
from plotly.offline import iplot


# In[65]:


trace = go.Bar(x= emojis, y=frequency)


# In[66]:


trace


# In[67]:


iplot([trace])


# In[68]:


import os


# In[69]:


files = os.listdir(r'/Users/gowthammarrapu/Documents/untitled folder 2/additional_data')


# In[70]:


files_csv = [file for file in files if '.csv' in file]


# In[71]:


import warnings
from warnings import filterwarnings
filterwarnings('ignore')


# In[72]:


full_df = pd.DataFrame()
path = r'/Users/gowthammarrapu/Documents/untitled folder 2/additional_data'

for files in files_csv:
    current_df = pd.read_csv(path+'/'+files ,encoding='iso-8859-1', error_bad_lines=False)
    full_df = pd.concat([full_df, current_df],ignore_index=True)
    


# In[73]:


full_df.shape


# In[74]:


full_df[full_df.duplicated()].shape


# In[75]:


full_df = full_df.drop_duplicates()


# In[76]:


full_df.shape


# In[77]:


full_df[0:1000].to_csv(r'/Users/gowthammarrapu/Documents/untitled folder 2/youtube_sample.csv', index=False)


# In[78]:


full_df[0:1000].to_json(r'/Users/gowthammarrapu/Documents/untitled folder 2/youtube_sample.json')


# In[ ]:





# In[79]:


from sqlalchemy import create_engine


# In[80]:


engine = create_engine(r'sqlite:////Users/gowthammarrapu/Documents/untitled folder 2/youtube_sample.sqlite')


# In[81]:


engine


# In[82]:


full_df[0:1000].to_sql('Users', con=engine, if_exists='append')


# In[83]:


full_df.head(5)


# In[84]:


full_df['category_id'].unique()


# In[85]:


json_df = pd.read_json(r'/Users/gowthammarrapu/Documents/untitled folder 2/additional_data/US_category_id.json')


# In[86]:


json_df


# In[87]:


json_df['items'][0]


# In[88]:


json_df['items'][1]


# In[89]:


cat_dict = {}

for item in json_df['items'].values:
    cat_dict[int(item['id'])]=item['snippet']['title']


# In[91]:


cat_dict


# In[92]:


full_df['category_name'] = full_df['category_id'].map(cat_dict)


# In[93]:


full_df.head(7)


# In[94]:


import seaborn as sns
import matplotlib.pyplot as plt 


# In[95]:


plt.figure(figsize=(12,8))
sns.boxplot(x= 'category_name', y = 'likes', data= full_df)
plt.xticks(rotation='vertical')


# In[ ]:





# In[185]:


full_df['like_rate'] = (full_df['likes']/full_df['views'])*100
full_df['dislikes_rate'] = (full_df['dislikes']/full_df['views'])*100
full_df['comment_count_rate'] = (full_df['comment_count']/full_df['views'])*100


# In[186]:


full_df.columns


# In[194]:


plt.figure(figsize=(16,8))
sns.regplot(x='views', y = 'likes', data = full_df)


# In[196]:


full_df[['views', 'likes', 'dislikes']].corr()


# In[98]:


sns.heatmap(full_df[['views', 'likes', 'dislikes']].corr(),annot=True)


# In[200]:


full_df['channel_title'].value_counts()


# In[101]:


cdf = full_df.groupby(['channel_title']).size().sort_values(ascending=False).reset_index()


# In[102]:


cdf= cdf.rename(columns={0:'total_videos'})


# In[ ]:





# In[103]:


import plotly.express as px


# In[104]:


px.bar(data_frame=cdf[0:20], x = 'channel_title', y='total_videos')


# In[216]:


full_df['title'][0]


# In[217]:


import string


# In[218]:


string.punctuation


# In[222]:


len([char for char in full_df['title'][0] if char in string.punctuation])


# In[225]:


def punc_count(text):
    return len([char for char in text if char in string.punctuation])


# In[226]:


sample = full_df[0:1000]


# In[230]:


sample['count_punc'] = sample['title'].apply(punc_count)


# In[231]:


sample['count_punc']


# In[234]:


plt.figure(figsize=(8,6))
sns.boxplot(x = 'count_punc', y = 'views',data= sample)
plt.show()


# In[235]:


plt.figure(figsize=(8,6))
sns.boxplot(x = 'count_punc', y = 'likes',data= sample)
plt.show()

