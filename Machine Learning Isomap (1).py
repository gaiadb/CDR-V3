#!/usr/bin/env python
# coding: utf-8

# In[126]:


from sklearn.decomposition import PCA
import pandas as pd 
import io
import os 
import math
import scipy.io
import pyarrow.parquet as pq #parquet specific analysis 
import numpy as np
import stats
import fastparquet #make sure that this is installed in your anaconda environment
from scipy import stats
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
from sklearn.datasets import make_spd_matrix
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import Isomap
from sklearn.datasets import load_digits
from sklearn.datasets import make_swiss_roll
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import silhouette_score
from sklearn import manifold 
from sklearn.datasets import load_iris


# In[127]:


# import data 
data = pd.read_csv("dropped_correlation.csv")
# convert dictionary into dataframe
data = pd.DataFrame(data)


# In[128]:


# drop the first column because it is strange
data.drop('Unnamed: 0',inplace = True, axis = 1)


# In[129]:


#we're going to break the rows down into different made-up drug treatments. rows 1-100 = drug 1, 101-200 = drug 2, 200-384 = drug 3
treatment1 = pd.DataFrame(['Drug 1']*100)
treatment2 = pd.DataFrame(['Drug 2']*100)
treatment3 = pd.DataFrame(['Drug 3']*184)
treatments = pd.concat([treatment1, treatment2,treatment3], axis = 0)
treatments.columns = ['Treatments']


# In[130]:


# sometimes you need to reset the index if they have already been indexed so pandas can use the clean versions
data = data.reset_index(drop=True)
treatments = treatments.reset_index(drop=True) 


# In[131]:


# perform isomap
n_comps = 3
embedding = Isomap(n_components=n_comps)
isomap = embedding.fit_transform(data) #do the data without the metadata
isomap.shape
# convert array to a dataframe 
isomap_df = pd.DataFrame(isomap, columns = ['Isomap 1','Isomap 2','Isomap 3'])
final = pd.concat([treatments, isomap_df], axis=1)


# In[132]:


# plot isomap

plt.scatter(isomap[:,0],isomap[:,1])
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.title('Isomap Projection')


# In[133]:


fig = plt.figure(figsize = (5,5))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Isomap Component 1', fontsize = 15)
ax.set_ylabel('Isomap Component 2', fontsize = 15)
ax.set_title('Isomap Projection')

targets = ['Drug 1', 'Drug 2', 'Drug 3']
colors = ['r', 'g', 'b']

for target, color in zip(targets,colors):
    indicesToKeep = final['Treatments'] == target
    ax.scatter(final.loc[indicesToKeep, 'Isomap 1'],
               final.loc[indicesToKeep, 'Isomap 2'],
               c = color,
               s = 50)
ax.legend(targets)
ax.grid()


# In[134]:


#open the figure
fig = plt.figure(figsize=(8,8))
#set the axis, with 3D
ax = fig.add_subplot(111, projection='3d')
#define what the drug treatments groups are 
treat_groups = ['Drug 1', 'Drug 2', 'Drug 3']
#define the colours we want to use
colours = ['red', 'green', 'blue']
for treat_group, colour in zip(treat_groups,colours):
    indicesToKeep = final['Treatments'] == treat_group
    ax.scatter(final.loc[indicesToKeep, 'Isomap 1']
               , final.loc[indicesToKeep, 'Isomap 2']
               , final.loc[indicesToKeep, 'Isomap 3']
               , c = colour
               , s = 25
               , marker = 'o')
ax.legend(treat_groups)
ax.set_title('3D Isomap')
ax.set_xlabel('Isomap 1', fontsize=10)
ax.set_ylabel('Isomap 2', fontsize=10)
ax.set_zlabel('Isomap 3', fontsize=10)
ax.grid()


# In[135]:


# determine the embedding parameters
params = embedding.get_params()
# obtain the feature names
features = embedding.get_feature_names_out(input_features=None)
# get the reconstruction error 
error = embedding.reconstruction_error()


# In[136]:


params, features, error


# In[137]:


# perform the elbow method to determine the optimal neighbours based on reconstruction error.
# the elbow point is where the graph begins to level off

n_neighbors_values = range(5,30) # set a neighbours range to go through
# The reconstruction error measures how well the lower-dimensional representation 
# preserves the original data's pairwise distances.
reconstruction_errors = []

for n_neighbors in n_neighbors_values:
    isomap = Isomap(n_neighbors=n_neighbors, n_components=2)
    data_reduced = isomap.fit_transform(data)
    reconstruction_error = isomap.reconstruction_error()
    reconstruction_errors.append(reconstruction_error)

# Plot the reconstruction errors
plt.plot(n_neighbors_values, reconstruction_errors, marker='o')
plt.xlabel('Number of Neighbors')
plt.ylabel('Reconstruction Error')
plt.title('Elbow Method for Optimal n_neighbors')
plt.show()


# In[138]:


# do above again for optimal components

n_neighbors = 15 # set at level off point above
n_components_values = range(1,10)
reconstruction_errors = []

for n_components in n_components_values:
    isomap = Isomap(n_neighbors=n_neighbors, n_components=n_components)
    data_reduced = isomap.fit_transform(data)
    reconstruction_error = isomap.reconstruction_error()
    reconstruction_errors.append(reconstruction_error)

# Plot the reconstruction errors
plt.plot(n_components_values, reconstruction_errors, marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Reconstruction Error')
plt.title('Elbow Method for Optimal n_neighbors')
plt.show()


# In[ ]:




