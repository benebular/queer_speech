## classification for queer speech
# author: Ben lang
# e: blang@ucsd.edu

# modules
import sklearn
import numpy as np
import pandas as pd
import os
import sys
import time
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import make_scorer, get_scorer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn import datasets
# np.set_printoptions(threshold=sys.maxsize)

# set up directory and read in csv
dir = '/Users/bcl/Documents/GitHub/queer_speech'
os.chdir(dir)
ratings_features_fname = os.path.join(dir, 'feature_extraction', 'ratings_features_all.csv')
data = pd.read_csv(ratings_features_fname)
data = data.drop('Unnamed: 0', axis=1) # might need to change column name when it imports in, there's a random addition at the beginning

# with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
#     print(finalDf)

# data = data.dropna()
# fill creak nans with 0
data['percent_creak'] = data['percent_creak'].fillna(0)

gender_id = data[data['Condition']=='gender_id']
sexual_orientation = data[data['Condition']=='sexual_orientation']
voice_id = data[data['Condition']=='voice_id']

### K-means

x = data[['Participant','Rating_z_score','Condition','WAV']]
x = x.pivot_table(index = ['WAV'], columns = 'Condition', values='Rating_z_score', aggfunc=np.mean)
y = x.iloc[:, [0,1,2]].values


# # Collecting the distortions into list
distortions = []
K = range(1,10)
for k in K:
     kmeanModel = KMeans(n_clusters=k)
     kmeanModel.fit(y)
     distortions.append(kmeanModel.inertia_)

 # Plotting the distortions
# plt.figure(figsize=(16,8))
# plt.plot(K, distortions, 'bx-')
# plt.xlabel('k')
# plt.ylabel('Distortion')
# plt.title('The Elbow Method showing the optimal clusters')
# plt.show()

# Define the model for 4 clusters
kmeans_model = KMeans(n_clusters=4, random_state=42)
# Fit into our dataset fit
kmeans_predict = kmeans_model.fit_predict(y)
x['kmeans_4_cluster'] = kmeans_predict
abc = x
abc = abc.drop(columns=['gender_id','sexual_orientation','voice_id'])
data_merged = pd.merge(data, abc, on='WAV')

### 5 clusters
kmeans_model = KMeans(n_clusters=5, random_state=42)
# Fit into our dataset fit
kmeans_predict = kmeans_model.fit_predict(y)
x['kmeans_5_cluster'] = kmeans_predict
abc = x
abc = abc.drop(columns=['gender_id','sexual_orientation','voice_id','kmeans_4_cluster'])
df = pd.merge(data_merged, abc, on='WAV')

# # Visualising the clusters
# plt.scatter(y[kmeans_predict == 0, 0], y[kmeans_predict == 0, 1], s = 100, c = 'red', label = 'Identity 1')
# plt.scatter(y[kmeans_predict == 1, 0], y[kmeans_predict == 1, 1], s = 100, c = 'blue', label = 'Identity 2')
# plt.scatter(y[kmeans_predict == 2, 0], y[kmeans_predict == 2, 1], s = 100, c = 'green', label = 'Identity 3')
# plt.scatter(y[kmeans_predict == 3, 0], y[kmeans_predict == 3, 1], s = 100, c = 'black', label = 'Identity 4')
# # plt.scatter(y[kmeans_predict == 4, 0], y[kmeans_predict == 4, 1], s = 100, c = 'pink', label = 'Identity 5')
#
#
# # Plotting the centroids of the clusters
# plt.scatter(kmeans_model.cluster_centers_[:, 0], kmeans_model.cluster_centers_[:,1], s = 100, c = 'yellow', label = 'Centroids')
# plt.title("4 Clusters", fontsize=20, fontweight='bold')
# plt.legend()
# plt.show()


# xyz_melt = xyz.melt(id_vars=['Cluster'])
# abc = pd.melt(abc.reset_index(), id_vars=['Participant','WAV'],value_vars=['gender_id','sexual_orientation','voice_id','4_Cluster'])
# abc_cluster1 = abc[abc['Condition'] == 'Cluster']
# abc_cluster2 = abc[abc['Condition'] == 'Cluster']
# abc_cluster3 = abc[abc['Condition'] == 'Cluster']
# abc_cluster = pd.concat([abc_cluster1,abc_cluster2,abc_cluster3], axis = 0)
# abc_conditions = abc[abc['Condition'] != 'Cluster']
#
# abc_cluster = abc_cluster.drop(columns=['Condition']).rename(columns={"value": 'Cluster'})

## add new column for group descriptors

# first summarize high or low conditions

# assign label based on conditions


### Random Forest

# for each group from the K-means clustering, slice out the relevant z-scored ratings, with all of the other acoustics features
# what is the acoustic information that is important to the ratings in a particular separate clusters

df = df.drop(['Rating','Rating_z_score','kmeans_4_cluster','participant_other_langs','participant_race_free_response','participant_gender_pso_free_response',
                'survey_experience','survey_feedback','Condition','WAV'], axis=1)
df_dummies_all = pd.get_dummies(df)
df_dummies_clusters = pd.get_dummies(df['kmeans_5_cluster'], prefix="cluster")
df_dummies_all = pd.concat([df_dummies_all,df_dummies_clusters], axis=1)
df_dummies_all = df_dummies_all.drop(['kmeans_5_cluster'], axis=1)
df_dummies_all.iloc[:,5:].head(5)

## random number column for sanity checks
# df_dummies_all['rando_baseline'] = np.round(np.random.uniform(1.0,8.0,len(df_dummies_all)), 1)
df_dummies_all['rando_baseline_z_score'] = stats.zscore(np.round(np.random.uniform(1.0,8.0,len(df_dummies_all)), 1), axis=0)

df_group_1 = df_dummies_all.drop(['cluster_1','cluster_2','cluster_3','cluster_4'], axis=1)
df_group_2 = df_dummies_all[df_dummies_all['cluster_1'] == 1]
df_group_3 = df_dummies_all[df_dummies_all['cluster_2'] == 1]
df_group_4 = df_dummies_all[df_dummies_all['cluster_3'] == 1]
df_group_5 = df_dummies_all[df_dummies_all['cluster_4'] == 1]

# df_group_1 = pd.get_dummies(df[df['kmeans_5_cluster'] == 0])
# df_group_1.iloc[:,5:].head(5)

###### BASELINE #######
# Labels are the values we want to predict
labels = np.array(df_group_1['cluster_0'])
# Remove the labels from the features
# axis 1 refers to the columns
df_group_1 = df_group_1.drop('cluster_0', axis = 1)
# Saving feature names for later use
feature_list = list(df_group_1.columns)
# Convert to numpy array
df_group_1 = np.array(df_group_1)

# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(df_group_1, labels, test_size = 0.25, random_state = 42)

# The baseline predictions are the historical averages
baseline_preds = test_features[:, feature_list.index('rando_baseline_z_score')]
# Baseline errors, and display average baseline error
baseline_errors = abs(baseline_preds - test_labels)
print('Average baseline error: ', round(np.mean(baseline_errors), 2))



# Labels are the values we want to predict
labels = np.array(df_group_1['cluster_0'])
# Remove the labels from the features
# axis 1 refers to the columns
df_group_1 = df_group_1.drop('cluster_0', axis = 1)
# Saving feature names for later use
feature_list = list(df_group_1.columns)
# Convert to numpy array
df_group_1 = np.array(df_group_1)

# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(df_group_1, labels, test_size = 0.25, random_state = 42)

# The baseline predictions are the historical averages
baseline_preds = test_features[:, feature_list.index('rando_baseline_z_score')]
# Baseline errors, and display average baseline error
baseline_errors = abs(baseline_preds - test_labels)
print('Average baseline error: ', round(np.mean(baseline_errors), 2))

# Create our imputer to replace missing values with the mean e.g.,
# Imputing values shouldn't matter at this stage because the ratings have already been clustered so
# the effect of a missing values of a fricatives within a given group doesn't change that WAVs membership to a group *in this data*
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp = imp.fit(train_features)
train_features_imp = imp.transform(train_features)
imp = imp.fit(test_features)
test_features_imp = imp.transform(test_features)


# Import the model we are using
from sklearn.ensemble import RandomForestRegressor
# Instantiate model with 1000 decision trees
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
# Train the model on training data
print ("Training Random Forest...")
t = time.time()
rf.fit(train_features_imp, train_labels);
elapsed_rf = time.time() - t
print("Random Forest elapsed time (in sec): %s" %elapsed_rf)

# Use the forest's predict method on the test data
predictions = rf.predict(test_features_imp)
# Calculate the absolute errors
errors = abs(predictions - test_labels)
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')

# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / test_labels)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')

# Get numerical feature importances
importances = list(rf.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];

# Import matplotlib for plotting and use magic command for Jupyter Notebooks
# Set the style
# plt.style.use('fivethirtyeight')
fig = plt.figure(figsize = (8,8))

# list of x locations for plotting
x_values = list(range(len(importances)))
# Make a bar chart
plt.bar(x_values, importances, orientation = 'vertical')
# Tick labels for x axis
plt.xticks(x_values, feature_list, rotation='vertical')
# Axis labels and title
plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances');

## save
print ("Saving data as queer_data.csv")
df.to_csv(os.path.join(dir,'feature_extraction','queer_data.csv'), index=True, encoding='utf-8')
df_dummies_all.to_csv(os.path.join(dir,'feature_extraction','dummies_data.csv'), index=True, encoding='utf-8')


with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(df_dummies_all)

# ## PCA
# features = ['F0_mean','F0_range','F0_std','percent_creak','vowel_avg_dur','dispersion']
#
# # Separating out the features
# x = data.loc[:, features].values
# # Separating out the target
# y = data.loc[:,['Condition']].values
# # Standardizing the features
# x = StandardScaler().fit_transform(x)
#
# pca = PCA(n_components=6, random_state=42)
# principalComponents = pca.fit_transform(x)
# # principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
# principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2',
#                                                                     'principal component 3', 'principal component 4',
#                                                                     'principal component 5', 'principal component 6'])
#
# finalDf = pd.concat([principalDf, data['Condition']], axis=1)
#
# fig = plt.figure(figsize = (8,8))
# ax = fig.add_subplot(1,1,1)
# ax.set_xlabel('Principal Component 1', fontsize = 15)
# ax.set_ylabel('Principal Component 2', fontsize = 15)
# ax.set_title('2 component PCA', fontsize = 20)
# targets = ['gender_id','sexual_orientation','voice_id']
# colors = ['r', 'g', 'b']
# for target, color in zip(targets,colors):
#     indicesToKeep = finalDf['Condition'] == target
#     ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
#                , finalDf.loc[indicesToKeep, 'principal component 2']
#                , c = color
#                , s = 50)
# ax.legend(targets)
# ax.grid()
# plt.show()
