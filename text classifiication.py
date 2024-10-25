#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.feature_extraction.text import CountVectorizer


# In[2]:


import pandas as pd
import numpy as np


# In[3]:


x = ['which book is this', 'this is book and this is math']


# In[4]:


cv = CountVectorizer()
count = cv.fit_transform(x)


# In[5]:


count.toarray()


# In[6]:


cv.get_feature_names()


# In[7]:


bow = pd.DataFrame(count.toarray(),columns = cv.get_feature_names())


# In[8]:


bow


# In[11]:


tf = bow.copy()
for index,row in enumerate(tf.iterrows()):
    for col in row[1].index:
        tf.loc[index,col] = tf.loc[index,col]/sum(row[1].values)


# In[12]:


tf


# In[15]:


bb = bow.astype('bool')
bb


# In[16]:


bb['is'].sum()


# In[17]:


cols = bb.columns
cols


# In[20]:


nz = []
for col in cols:
    nz.append(bb[col].sum())
nz


# In[27]:


N = 2
idf = []
for index,col in enumerate(cols):
    idf.append(np.log((N+1)/(nz[index]+1))+1)
idf


# In[28]:


x


# In[30]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[32]:


tfid = TfidfVectorizer()
X = tfid.fit_transform(x)


# In[33]:


X


# In[35]:


print(X.toarray())


# In[38]:


print(tfid.idf_)


# In[39]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[44]:


df = pd.read_csv(r'/Users/gowthammarrapu/Documents/untitled folder 2/nlp-for-beginners-udemy/14-Spam Text Classification/spam.tsv',sep = '\t')
df.head()


# In[47]:


ham = df[df['label']=='ham']
ham.shape


# In[49]:


spam = df[df['label']=='spam']
spam.shape


# In[50]:


ham = ham.sample(spam.shape[0])


# In[52]:


ham.shape,spam.shape


# In[53]:


data = ham.append(spam, ignore_index=True)


# In[54]:


data


# In[55]:


data.shape


# In[57]:


data['label'].value_counts()


# In[62]:


plt.hist(ham['length'],bins = 100,alpha =0.7,label = 'Ham')
plt.hist(spam['length'],bins = 100,alpha =0.7,label = 'spam')
plt.show()


# In[63]:


plt.hist(ham['punct'],bins = 100,alpha =0.7,label = 'Ham')
plt.hist(spam['punct'],bins = 100,alpha =0.7,label = 'spam')
plt.show()


# In[64]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline


# In[65]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[66]:


data.head()


# In[67]:


tfidf = TfidfVectorizer()
X = tfidf.fit_transform(data['message'])


# In[68]:


X = X.toarray()


# In[69]:


X_train, X_test, y_train, y_test = train_test_split(X, data['label'], test_size = 0.2, random_state = 0, stratify = data['label'])


# In[70]:


X_train.shape, X_test.shape


# In[71]:


clf = RandomForestClassifier(n_estimators=100, n_jobs= -1)


# In[72]:


clf.fit(X_train, y_train)


# In[73]:


y_pred = clf.predict(X_test)


# In[74]:


confusion_matrix(y_test, y_pred)


# In[75]:


print(classification_report(y_test, y_pred))


# In[81]:


clf


# In[85]:


def predict(x):
    x = tfidf.transform([x])
    x = x.toarray()
    pred = clf.predict(x)
    return pred


# In[86]:


predict('hey, whassup')


# In[87]:


predict('you have got free tickets to the usa this summer')


# In[ ]:




