#!/usr/bin/env python
# coding: utf-8

# In[6]:


# Importing
import sklearn
import pandas as pd
import numpy as  np
import seaborn as sns
from sklearn.datasets import load_boston


# In[7]:


# Load the dataset
df = pd.read_csv('Iris.csv')
 
# IQR
Q1 = np.percentile(df['SepalWidthCm'], 25,
                interpolation = 'midpoint')
 
Q3 = np.percentile(df['SepalWidthCm'], 75,
                interpolation = 'midpoint')
IQR = Q3 - Q1

sns.boxplot(x='SepalWidthCm', data=df)
print("Old Shape: ", df.shape)
 
# Upper bound
upper = np.where(df['SepalWidthCm'] >= (Q3+1.5*IQR))
 
# Lower bound
lower = np.where(df['SepalWidthCm'] <= (Q1-1.5*IQR))
 
# Removing the Outliers
df.drop(upper[0], inplace = True)
df.drop(lower[0], inplace = True)


# In[8]:


print("New Shape: ", df.shape)
 
sns.boxplot(x='SepalWidthCm', data=df)


# In[ ]:




