#!/usr/bin/env python
# coding: utf-8

# In[9]:


# Importing Libraries required
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[10]:


from sklearn.datasets import load_iris


# In[11]:


# Loading the dataset
dataset = load_iris()


# In[14]:


# Understanding dataset, knowing which keys are there
for keys in dataset.keys() :
    print(keys)



# In[15]:


# Splitting dataset into train and test
X = dataset.data
y = dataset.target


# In[16]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=0, train_size = .75)


# In[17]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# In[18]:


# Creating an instance for Random Forest model
model = RandomForestClassifier(n_estimators =100, criterion= 'entropy',random_state = 0)


# In[19]:


# Training the dataset
model.fit(X_train,y_train)


# In[20]:


y_pred = model.predict(X_test)


# In[29]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix (y_test,y_pred,)


# In[22]:


cm


# In[30]:


from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)


# In[31]:


# Calculating the accuracy of the model
model.score(X_test,y_test)


# ## https://builtin.com/data-science/random-forest-python-deep-dive

# In[ ]:




