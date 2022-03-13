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
# np.set_printoptions(threshold=sys.maxsize)


# set up directory and read in csv
dir = '/Users/bcl/Documents/GitHub/queer_speech'
os.chdir(dir)
ratings_features_fname = os.path.join(dir, 'ratings_features_all.csv')
data = pd.read_csv(ratings_features_fname)
with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(data.head(5))

## PCA
features = ['participant_prox_social','participant_prox_affiliation','participant_prox_media']

# Separating out the features
x = data.loc[:, features].values
# Separating out the target
y = data.loc[:,['Condition']].values
# Standardizing the features
x = StandardScaler().fit_transform(x)

pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])

finalDf = pd.concat([principalDf, data[['Condition']]], axis = 1)

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = ['gender_id', 'sexual_orientation', 'voice_id']
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
