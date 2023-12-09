#!/usr/bin/env python
# coding: utf-8

# In[26]:


# PCA requires data with a mean = 0 and a variance = 1 to function properly 


# In[22]:


from sklearn.decomposition import PCA
import pandas as pd 
import io
import os 
import pyarrow.parquet as pq #parquet specific analysis 
import numpy as np
import stats
import fastparquet # make sure that this is installed in your anaconda environment
from scipy import stats
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
from sklearn.datasets import make_spd_matrix
from mpl_toolkits.mplot3d import Axes3D


# In[2]:


# import sample 
data = pd.read_csv("dropped_correlation.csv")
# convert dictionary into dataframe
data = pd.DataFrame(data)


# In[3]:


# drop the first column because it is strange
data.drop('Unnamed: 0',inplace = True, axis = 1)


# In[4]:


#we're going to break the rows down into different made-up drug treatments. rows 1-100 = drug 1, 101-200 = drug 2, 200-384 = drug 3
treatment1 = pd.DataFrame(['Drug 1']*100)
treatment2 = pd.DataFrame(['Drug 2']*100)
treatment3 = pd.DataFrame(['Drug 3']*184)
treatments = pd.concat([treatment1, treatment2,treatment3], axis = 0)
treatments.columns = ['Treatments']


# In[5]:


# sometimes you need to reset the index if they have already been indexed so pandas can use the clean versions
data = data.reset_index(drop=True)
treatments = treatments.reset_index(drop=True) 
final = pd.concat([treatments, data], axis=1)


# In[68]:


# perform PCA 
#set the number of components
pca = PCA(n_components=3)
# fit the data 
principalComponents = pca.fit_transform(data)
# transform components into a dataframe
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['PC1', 'PC2','PC3'])
# create a final data frame with the treatment 'metadata'
finalDf = pd.concat([treatments, principalDf], axis=1)


# In[69]:


# plot the principal components graph
fig = plt.figure(figsize = (5,5))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)

targets = ['Drug 1', 'Drug 2', 'Drug 3']
colors = ['r', 'g', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['Treatments'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'PC1']
               , finalDf.loc[indicesToKeep, 'PC2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()


# In[71]:


# now extract the optimal number of components needed to explain 80% of variance with PCA

pca = PCA(n_components=None)
pca.fit(data)

# extract the smallest number of components which explain at least p% (e.g. 80%) of the variance
p = 0.80
# arg max takes index value of the largest value 
n_components = 1 + np.argmax(np.cumsum(pca.explained_variance_ratio_) >= p)
print(n_components)

# i will set this at 3 again for visualization later on, but the printed value is the optimal one
n_components = 3

# check how much variance is explained by each principal component
variance = pca.explained_variance_ratio_


# In[74]:


# import relevant libraries for 3D graph
from mpl_toolkits.mplot3d import Axes3D
#Z = Z.reset_index(drop=True)
data = data.reset_index(drop=True)
finalDf = finalDf.reset_index(drop=True)
fig = plt.figure(figsize=(10,10))
# choose projection 3d for creating a 3d graph
axis = fig.add_subplot(111, projection='3d')

# Define colors for each treatment type
colors = {'Drug 1': 'red', 'Drug 2': 'blue', 'Drug 3': 'red'}
treatment = final2['Treatments']
axis.scatter(final2['PC1'],final2['PC2'],final2['PC3'])
axis.set_title('3D PCA', fontsize = 20)
axis.set_xlabel("PC1", fontsize=10)
axis.set_ylabel("PC2", fontsize=10)
axis.set_zlabel("PC3", fontsize=10)


# In[73]:


#open the figure
fig = plt.figure(figsize=(8,8))
#set the axis, with 3D
ax = fig.add_subplot(111, projection='3d')
#define what the drug treatments groups are 
treat_groups = ['Drug 1', 'Drug 2', 'Drug 3']
#define the colours we want to use
colours = ['red', 'green', 'blue']
for treat_group, colour in zip(treat_groups,colours):
    indicesToKeep = finalDf['Treatments'] == treat_group
    ax.scatter(finalDf.loc[indicesToKeep, 'PC1']
               , finalDf.loc[indicesToKeep, 'PC2']
               , finalDf.loc[indicesToKeep, 'PC3']
               , c = colour
               , s = 25
               , marker = 'o')
ax.legend(treat_groups)
ax.set_title('3D PCA')
ax.set_xlabel('PC1', fontsize=10)
ax.set_ylabel('PC2', fontsize=10)
ax.set_zlabel('PC3', fontsize=10)
ax.grid()


# In[ ]:




