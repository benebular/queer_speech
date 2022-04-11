## classification for queer speech
# author: Ben lang
# e: blang@ucsd.edu

# modules
import numpy as np
import pandas as pd
import os
import sys
import time
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.impute import SimpleImputer

# np.set_printoptions(threshold=sys.maxsize)

# set up directory and read in csv
dir = '/Users/bcl/Documents/GitHub/queer_speech'
os.chdir(dir)
ratings_features_fname = os.path.join(dir, 'feature_extraction', 'queer_data.csv')
data = pd.read_csv(ratings_features_fname)
data = data.drop('Unnamed: 0', axis=1) # might need to change column name when it imports in, there's a random addition at the beginning

# with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
#     print(finalDf)

# data = data.dropna()
# fill creak nans with 0
data['percent_creak'] = data['percent_creak'].fillna(0)

# gender_id = data[data['Condition']=='gender_id']
# sexual_orientation = data[data['Condition']=='sexual_orientation']
# voice_id = data[data['Condition']=='voice_id']

df = data
# df['rando_baseline_z_score'] = stats.zscore(np.round(np.random.uniform(1.0,8.0,len(df)), 1), axis=0)
df = df.drop(['Rating','Rating_z_score','kmeans_4_cluster','participant_gender_id','participant_sexual_orientation','participant_voice_id','participant_cis_trans',
                'participant_prox_social','participant_prox_affiliation', 'participant_prox_media', 'participant_race','participant_race_hispanic','eng_primary_early','eng_primary_current',
                'participant_other_langs','participant_race_free_response','participant_gender_pso_free_response', 'participant_age', 'deaf_hoh','rando',
                'survey_experience','survey_feedback','Condition','WAV','color','spectral_S_start','spectral_Z_start','spectral_F_start','spectral_V_start','spectral_JH_start','spectral_SH_start'], axis=1)

## impute cluster means for each cluster in the loop
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp = imp.fit(df)
df_imp = imp.transform(df)
df_imp = pd.DataFrame(df_imp, columns = df.columns)

# make cluster dfs with imputed grand means
df_cluster_dummies = pd.get_dummies(df_imp['kmeans_5_cluster'], prefix='cluster')
df_grand = pd.concat([df_imp, df_cluster_dummies], axis=1)
df_sans = df_grand.drop('kmeans_5_cluster', axis=1)
df_sans = df_sans.rename(columns={'cluster_0.0':'cluster_0', 'cluster_1.0':'cluster_1', 'cluster_2.0':'cluster_2', 'cluster_3.0':'cluster_3', 'cluster_4.0':'cluster_4'})
df_group_0 = df_sans.drop(['cluster_1','cluster_2','cluster_3','cluster_4'], axis=1)
df_group_1 = df_sans.drop(['cluster_0','cluster_2','cluster_3','cluster_4'], axis=1)
df_group_2 = df_sans.drop(['cluster_0','cluster_1','cluster_3','cluster_4'], axis=1)
df_group_3 = df_sans.drop(['cluster_0','cluster_1','cluster_2','cluster_4'], axis=1)
df_group_4 = df_sans.drop(['cluster_0','cluster_1','cluster_2','cluster_3'], axis=1)

###### BASELINE #######
# Labels are the values we want to predict
labels = np.array(df_group_0['cluster_0'])
# Remove the labels from the features
# axis 1 refers to the columns
df_group_0 = df_group_0.drop('cluster_0', axis = 1)
# Saving feature names for later use
feature_list = list(df_group_0.columns)
# Convert to numpy array
df_group_0 = np.array(df_group_0)

# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
Xtrain, Xtest, ytrain, ytest = train_test_split(df_group_0, labels, test_size = 0.25, random_state = 42)

## logistic regressions
# defining the dependent and independent variables
# Xtrain = df
# ytrain = df[['kmeans_5_cluster']]

# building the model and fitting the data
# loading the training dataset
# df = pd.read_csv('logit_train1.csv', index_col = 0)
log_reg = sm.Logit(ytrain, Xtrain).fit()

print(log_reg.summary())

# loading the testing dataset
# df = pd.read_csv('logit_test1.csv', index_col = 0)

# defining the dependent and independent variables
# Xtest = df
# ytest = df['kmeans_5_cluster']

# performing predictions on the test datdaset
yhat = log_reg.predict(Xtest)
prediction = list(map(round, yhat))

# comparing original and predicted values of y
print('Actual values', list(ytest.values))
print('Predictions :', prediction)

from sklearn.metrics import (confusion_matrix, accuracy_score)
# confusion matrix
cm = confusion_matrix(ytest, prediction)
print ("Confusion Matrix : \n", cm)

# accuracy score of the model
print('Test accuracy = ', accuracy_score(ytest, prediction))
