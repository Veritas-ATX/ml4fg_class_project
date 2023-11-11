#!/usr/bin/env python
# coding: utf-8

# In[25]:


# Import libraries
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns


# In[26]:


# Specify the path to the CSV files
data_csv = 'data.csv'
labels_csv = 'labels.csv'

# Read the CSV files into a pandas DataFrame
df_data = pd.read_csv(data_csv)
df_labels = pd.read_csv(labels_csv)

# Merge the DataFrames as a table and display it
df_merged = pd.merge(df_data, df_labels, left_on='Unnamed: 0', right_on='Unnamed: 0')
df_merged


# In[31]:


# Get summary statistics

# Descriptive statistics
descriptive_stats = df.describe()
print(descriptive_stats)

# Correlation matrix - unneccesary?
# correlation_matrix = df.corr()
# print(correlation_matrix)


# In[28]:


# Perform dimensionality reduction using PCA

# Assuming X contains your gene expression data
X = df_merged.iloc[:, 1:20531].values

# Apply PCA for dimensionality reduction
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)


# In[29]:


# Perform clustering

# Assuming y contains your cancer classification labels
y = df_merged['Class'].values

# Apply K-Means clustering
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(X_pca)

# Add the cluster information to the merged DataFrame
df_merged['Cluster'] = clusters

# Scatter plot using PCA
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=clusters, palette='viridis')
plt.title('PCA Clustering')
plt.show()


# In[ ]:




