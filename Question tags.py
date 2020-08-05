#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import datetime
import numpy as np
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer


# In[2]:


que_t = pd.read_csv('C:/Users/Preetham/Desktop/question tags/Tags.csv')

que = pd.read_csv('C:/Users/Preetham/Desktop/question tags/questions_s.csv', nrows= 500, encoding='latin-1')
que.head(5)


# In[4]:


que_c = que.drop(['CreationDate', 'ClosedDate','Score','Body','OwnerUserId'], axis=1)


# In[5]:


que_c['Title'] = que_c['Title'].map(lambda Title: re.sub(r'\W+', ' ', Title))


# In[6]:


que_c['Title']= que_c['Title'].str.lower()


# In[7]:


data = pd.merge(que_t, que_c,on='Id')


# In[8]:


data_c = data.drop(['Id'], axis=1)


# In[9]:


data_c = data_c[~data_c.Tag.isna()]


# In[10]:


X = data_c['Title']
y = data_c['Tag']


# In[11]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[12]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

vect = CountVectorizer(stop_words='english',max_df=.3)


# In[13]:


X_train_dtm = vect.fit_transform(X_train)

X_test_dtm = vect.transform(X_test)


# In[14]:


from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()

#fitting the model into train data 
nb.fit(X_train_dtm, y_train)


# In[15]:


#predicting the model on train and test data
y_pred_class_test = nb.predict(X_test_dtm)
y_pred_class_train = nb.predict(X_train_dtm)


# In[16]:


nb.predict(X_test_dtm)


# In[17]:


from sklearn import metrics
print(metrics.accuracy_score(y_test, y_pred_class_test))
print(metrics.accuracy_score(y_train, y_pred_class_train))


# In[ ]:





# In[ ]:




