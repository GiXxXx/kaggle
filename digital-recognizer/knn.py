
# coding: utf-8

# In[58]:


import pandas as pd


# In[59]:


train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')


# In[60]:


train_data.tail(5)


# In[61]:


test_data.tail(5)


# In[62]:


from sklearn.neighbors import KNeighborsClassifier
knnClf = KNeighborsClassifier(weights='distance', n_jobs=-1)


# In[63]:


knnClf.fit(train_data.values[:,1:], train_data.values[:,0])


# In[64]:


testLabel = knnClf.predict(test_data.values)


# In[65]:


result = pd.DataFrame(data=test_data.index, index=None, columns=['ImageId'])


# In[66]:


result.head(5)


# In[67]:


result['Label'] = testLabel


# In[70]:


result.head(5)
testLabel[0:10]
test_data.values[0:10]
result.describe()
result.ImageId = result.ImageId + 1


# In[71]:


result.to_csv('simple-knn.csv',index=False)

