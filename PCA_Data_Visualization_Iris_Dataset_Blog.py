
# coding: utf-8

# <h1 align="center"> Principle Component Analysis (PCA) for Data Visualization </h1>

# In[27]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
get_ipython().magic(u'matplotlib inline')


# ## Load Iris Dataset

# In[28]:


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"


# In[29]:


# loading dataset into Pandas DataFrame
df = pd.read_csv(url
                 , names=['sepal length','sepal width','petal length','petal width','target'])


# In[30]:


df.head()


# ## Standardize the Data

# Since PCA yields a feature subspace that maximizes the variance along the axes, it makes sense to standardize the data, especially, if it was measured on different scales. Although, all features in the Iris dataset were measured in centimeters, let us continue with the transformation of the data onto unit scale (mean=0 and variance=1), which is a requirement for the optimal performance of many machine learning algorithms.

# In[31]:


features = ['sepal length', 'sepal width', 'petal length', 'petal width']
x = df.loc[:, features].values


# In[32]:


y = df.loc[:,['target']].values


# In[33]:


x = StandardScaler().fit_transform(x)


# In[34]:


pd.DataFrame(data = x, columns = features).head()


# ## PCA Projection to 2D

# In[35]:


#pca = PCA(n_components=2)
pca = PCA(n_components=3)

# In[36]:


principalComponents = pca.fit_transform(x)


# In[37]:


#principalDf = pd.DataFrame(data = principalComponents
             #, columns = ['principal component 1', 'principal component 2'])
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2','principal component 3'])

# In[38]:


principalDf.head(5)


# In[39]:


df[['target']].head()


# In[40]:


finalDf = pd.concat([principalDf, df[['target']]], axis = 1)
finalDf.head(5)


# ## Visualize 2D Projection

# Use a PCA projection to 2d to visualize the entire data set. You should plot different classes using different colors or shapes. Do the classes seem well-separated from each other?

# In[41]:


fig = plt.figure(figsize = (8,8))
#ax = fig.gca(projection='3d')
ax = fig.add_subplot(1,1,1, projection='3d')
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_zlabel('Principal Component 3', fontsize = 15)
#ax.set_title('2 Component PCA', fontsize = 20)
ax.set_title('3 Component PCA', fontsize = 20)

targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
colors = ['r', 'g', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['target'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , finalDf.loc[indicesToKeep, 'principal component 3']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()


# The three classes appear to be well separated!
#
# iris-virginica and iris-versicolor could be better separated, but still good!

# ## Explained Variance

# The explained variance tells us how much information (variance) can be attributed to each of the principal components.

# In[42]:


pca.explained_variance_ratio_


# Together, the first two principal components contain 95.80% of the information. The first principal component contains 72.77% of the variance and the second principal component contains 23.03% of the variance. The third and fourth principal component contained the rest of the variance of the dataset.

# ## What are other applications of PCA (other than visualizing data)?

# If your learning algorithm is too slow because the input dimension is too high, then using PCA to speed it up is a reasonable choice. (<b>most common application in my opinion</b>). We will see this in the MNIST dataset.

# If memory or disk space is limited, PCA allows you to save space in exchange for losing a little of the data's information. This can be a reasonable tradeoff.

# ## What are the limitations of PCA?

# - PCA is not scale invariant. check: we need to scale our data first.
# - The directions with largest variance are assumed to be of the most interest
# - Only considers orthogonal transformations (rotations) of the original variables
# - PCA is only based on the mean vector and covariance matrix. Some distributions (multivariate normal) are characterized by this, but some are not.
# - If the variables are correlated, PCA can achieve dimension reduction. If not, PCA just orders them according to their variances.
