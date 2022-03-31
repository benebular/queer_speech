## classification for queer speech
# author: Ben lang
# e: blang@ucsd.edu

# modules
import sklearn
import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import make_scorer, get_scorer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import datasets
# np.set_printoptions(threshold=sys.maxsize)

# set up directory and read in csv
dir = '/Users/bcl/Documents/GitHub/queer_speech'
os.chdir(dir)
ratings_features_fname = os.path.join(dir, 'feature_extraction', 'ratings_features_all.csv')
data = pd.read_csv(ratings_features_fname)
# with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
#     print(finalDf)

# data = data.dropna()
# fill creak nans with 0
data['percent_creak'] = data['percent_creak'].fillna(0)

gender_id = data[data['Condition']=='gender_id']
sexual_orientation = data[data['Condition']=='sexual_orientation']
voice_id = data[data['Condition']=='voice_id']


## PCA
features = ['F0_mean','F0_range','F0_std','percent_creak','vowel_avg_dur','dispersion']

# Separating out the features
x = data.loc[:, features].values
# Separating out the target
y = data.loc[:,['Condition']].values
# Standardizing the features
x = StandardScaler().fit_transform(x)

pca = PCA(n_components=6, random_state=42)
principalComponents = pca.fit_transform(x)
# principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2',
                                                                    'principal component 3', 'principal component 4',
                                                                    'principal component 5', 'principal component 6'])

finalDf = pd.concat([principalDf, data['Condition']], axis=1)

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = ['gender_id','sexual_orientation','voice_id']
colors = ['r', 'g', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['Condition'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()
plt.show()

### K-means

x = data[['Participant','Rating','Condition','WAV']]
x = x.pivot_table(index = ['Participant','WAV'], columns = 'Condition', values='Rating')
y = x.iloc[:, [0,1,2]].values

# with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
#     print(x)


# Collecting the distortions into list
distortions = []
K = range(1,10)
for k in K:
     kmeanModel = KMeans(n_clusters=k)
     kmeanModel.fit(y)
     distortions.append(kmeanModel.inertia_)

 # Plotting the distortions
plt.figure(figsize=(16,8))
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal clusters')
plt.show()

# Define the model
kmeans_model = KMeans(n_clusters=4, random_state=42)
# Fit into our dataset fit
kmeans_predict = kmeans_model.fit_predict(y)

x['Cluster'] = kmeans_predict

# Visualising the clusters
plt.scatter(y[kmeans_predict == 0, 0], y[kmeans_predict == 0, 1], s = 50, c = 'red', label = 'Identity 1')
plt.scatter(y[kmeans_predict == 1, 0], y[kmeans_predict == 1, 1], s = 50, c = 'blue', label = 'Identity 2')
plt.scatter(y[kmeans_predict == 2, 0], y[kmeans_predict == 2, 1], s = 50, c = 'green', label = 'Identity 3')
plt.scatter(y[kmeans_predict == 3, 0], y[kmeans_predict == 3, 1], s = 50, c = 'black', label = 'Identity 4')
# plt.scatter(y[kmeans_predict == 4, 0], y[kmeans_predict == 4, 1], s = 50, c = 'pink', label = 'Identity 5')


# Plotting the centroids of the clusters
plt.scatter(kmeans_model.cluster_centers_[:, 0], kmeans_model.cluster_centers_[:,1], s = 100, c = 'yellow', label = 'Centroids')
plt.legend()
plt.show()
