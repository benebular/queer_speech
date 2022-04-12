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
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import make_scorer, get_scorer, accuracy_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.inspection import permutation_importance

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

# df['rando_baseline_z_score'] = stats.zscore(np.round(np.random.uniform(1.0,8.0,len(df)), 1), axis=0)

# import random
# get_colors = lambda n: list(map(lambda i: "#" + "%06x" % random.randint(0, 0xFFFFFF),range(n)))
# colors = pd.DataFrame(get_colors(5), columns={'color'}) # sample return:  ['#8af5da', '#fbc08c', '#b741d0', '#e599f1', '#bbcb59', '#a2a6c0']
# colors['cluster'] = [0,1,2,3,4]

colors = pd.DataFrame({"color": ['#648FFF','#785EF0','#DC267F','#FE6100','#FFB000']})
colors['kmeans_5_cluster'] = [0,1,2,3,4]
# 0: straight men, 1: straight women, 2: queer NB, men, and women, 3: queer women, 4: queer men
df = pd.merge(df, colors, on = 'kmeans_5_cluster', how = "outer")
df_orig = df


print ("Saving data as queer_data.csv")
df.to_csv(os.path.join(dir,'feature_extraction','queer_data.csv'), index=True, encoding='utf-8')

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

########################### RANDOM FOREST ##########################

# for each group from the K-means clustering, slice out the relevant z-scored ratings, with all of the other acoustic features
# what is the acoustic information that is important to the ratings in a particular cluster

df = df.drop(['Rating','Rating_z_score','kmeans_4_cluster','participant_gender_id','participant_sexual_orientation','participant_voice_id','participant_cis_trans',
                'participant_prox_social','participant_prox_affiliation', 'participant_prox_media', 'participant_race','participant_race_hispanic','eng_primary_early','eng_primary_current',
                'participant_other_langs','participant_race_free_response','participant_gender_pso_free_response', 'participant_age', 'deaf_hoh', 'Participant',
                'survey_experience','survey_feedback','Condition','WAV','color','spectral_S_start','spectral_Z_start','spectral_F_start','spectral_V_start','spectral_JH_start','spectral_SH_start'], axis=1)

number_of_features = len(df.columns)

## impute for the cross validation, random forest needs values in each, below we are creating one large df with imputed column means from teh entire set, and then 5 cluster groups with the gran imputed means
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp = imp.fit(df)
df_imp = imp.transform(df) # this is the big df
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

###### BASELINE ####### Big ol' df with imputed means on everyone
# Labels are the values we want to predict
labels = np.array(df_imp['kmeans_5_cluster'])
# Remove the labels from the features
# axis 1 refers to the columns
df_imp = df_imp.drop('kmeans_5_cluster', axis = 1)
# Saving feature names for later use
feature_list = list(df_imp.columns)
# Convert to numpy array
df_imp = np.array(df_imp)

# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(df_imp, labels, test_size = 0.75, random_state = 42)

# # # The baseline predictions are the historical averages
# baseline_preds = test_features[:, feature_list.index('rando')]
# # # # Baseline errors, and display average baseline error
# baseline_errors = abs(baseline_preds - test_labels)
# print('Average baseline error: ', round(np.mean(baseline_errors), 3))

# # Create our imputer to replace missing values with the mean e.g.,
# # Imputing values shouldn't matter at this stage because the ratings have already been clustered so
# # the effect of a missing values of a fricatives within a given group doesn't change that WAVs membership to a group *in this data*
# # impute for final model
# imp = SimpleImputer(missing_values=np.nan, strategy='mean')
# imp = imp.fit(train_features)
# train_features_imp = imp.transform(train_features)
# imp = imp.fit(test_features)
# test_features_imp = imp.transform(test_features)

# Import the model we are using
# Instantiate model with 1000 decision trees
rf = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators = 5000, random_state = 42, n_jobs = -1, max_features = 10, max_depth = 5))
# Train the model on training data
print ("Training Random Forest on %s features for all clusters..."%number_of_features)
t = time.time()
rf.fit(train_features, train_labels);
elapsed_rf = time.time() - t
print("Random Forest elapsed time (in sec): %s" %elapsed_rf)
print("Accuracy on train data: {:.2f}".format(rf.score(train_features, train_labels)))

# cross validation
print ("Cross-validation...")
# rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
t = time.time()
scores = cross_val_score(rf, train_features, train_labels, cv=10)
elapsed_rf = time.time() - t
print("CV elapsed time (in sec): %s" %elapsed_rf)
scores
print('Mean CV Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))

# # accuracy score
# predictions=rf.predict(test_features)
# print("Accuracy:", accuracy_score(test_features, predictions))

# # Get and reshape confusion matrix data
# matrix = confusion_matrix(test_labels, predictions)
# matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
#
# # Build the plot
# plt.figure(figsize=(16,7))
# sns.set(font_scale=1.4)
# sns.heatmap(matrix, annot=True, annot_kws={'size':10},
#             cmap=plt.cm.Greens, linewidths=0.2)
#
# # Add labels to the plot
# class_names = feature_list
# tick_marks = np.arange(len(class_names))
# tick_marks2 = tick_marks + 0.5
# plt.xticks(tick_marks, class_names, rotation=25)
# plt.yticks(tick_marks2, class_names, rotation=0)
# plt.xlabel('Predicted label')
# plt.ylabel('True label')
# plt.title('Confusion Matrix for Random Forest Model')
# plt.show()

# # Use the forest's predict method on the test data
predictions = rf.predict(test_features)
# # Calculate the absolute errors
# errors = abs(predictions - test_labels)
# # # Print out the mean absolute error (mae)
# print('Mean Absolute Error:', round(np.mean(errors), 3), 'degrees.')
# print("Accuracy on test data: {:.2f}".format(rf.score(test_features, test_labels)))

# View the classification report for test data and predictions
print(confusion_matrix(test_labels,predictions))
print(accuracy_score(test_labels, predictions))
print(classification_report(test_labels, predictions))

# # Calculate mean absolute percentage error (MAPE)
# mape = 100 * (errors / test_labels)
# # Calculate and display accuracy
# accuracy = 100 - np.mean(mape)
# print('Accuracy:', round(accuracy, 2), '%.')

# Get numerical feature importances
importances = list(rf.steps[1][1].feature_importances_)
# importances = list(result.importances_mean)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 5)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
grand_features = pd.DataFrame(feature_importances, columns = {'feature': '0','importance': '1'})
grand_features_50 = grand_features['importance'].quantile(0.5)
grand_features = grand_features[grand_features['importance'] > grand_features_50]
# Print out the feature and importances
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];
print(grand_features)

## make variables to save importances

# Import matplotlib for plotting and use magic command for Jupyter Notebooks
# Set the style
# plt.style.use('fivethirtyeight')
fig = plt.figure(figsize = (100,8))

# grand_features_list = grand_features['feature'].to_list()

# list of x locations for plotting
x_values = list(range(len(importances)))
# Make a bar chart
plt.bar(x_values, importances, orientation = 'vertical')
# Tick labels for x axis
plt.xticks(x_values, feature_list, rotation='vertical', fontsize = 8)
# Axis labels and title
plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances');

plt.savefig(os.path.join(dir,'figs', 'randomforest_grandmean.png'), bbox_inches='tight', dpi=300)
plt.close()


# start_time = time.time()
# result = permutation_importance(
#     rf, test_features, test_labels, n_repeats=10, random_state=42, n_jobs=5,
#     scoring = 'f1'
# )
# elapsed_time = time.time() - start_time
# print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")
#
# forest_importances = pd.Series(result.importances_mean, index=feature_list)
#
# fig, ax = plt.subplots()
# forest_importances.plt.bar(yerr=result.importances_std, ax=ax)
# ax.set_title("Feature importances using permutation on full model")
# ax.set_ylabel("Mean accuracy decrease")
# fig.tight_layout()
# plt.show()



######### CLUSTERS ########
### clusters using grand mean, no additional imputing needed

cluster_dict  = {'df_group_0':df_group_0,'df_group_1':df_group_1,'df_group_2':df_group_2,'df_group_3':df_group_3,'df_group_4':df_group_4}
for cluster, value in cluster_dict.items():
    cluster_number = 'cluster_' + cluster[-1:]
    ###### BASELINE #######
    # Labels are the values we want to predict
    labels = np.array(value[cluster_number])
    # Remove the labels from the features
    # axis 1 refers to the columns
    value = value.drop('%s'%cluster_number, axis = 1)
    # Saving feature names for later use
    feature_list = list(value.columns)
    # Convert to numpy array
    value = np.array(value)

    # Using Skicit-learn to split data into training and testing sets
    from sklearn.model_selection import train_test_split
    # Split the data into training and testing sets
    train_features, test_features, train_labels, test_labels = train_test_split(value, labels, test_size = 0.25, random_state = 42)

    # # # The baseline predictions are the historical averages
    # baseline_preds = test_features[:, feature_list.index('rando')]
    # # # # Baseline errors, and display average baseline error
    # baseline_errors = abs(baseline_preds - test_labels)
    # print('Average baseline error: ', round(np.mean(baseline_errors), 3))

    # Instantiate model with 1000 decision trees
    # rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
    rf = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators = 5000, random_state = 42, n_jobs = -1, max_features = 10, max_depth = 5))
    # Train the model on training data
    print ("Training Random Forest on %s features for %s..."%(number_of_features, cluster_number))
    t = time.time()
    rf.fit(train_features, train_labels);
    elapsed_rf = time.time() - t
    print("Random Forest elapsed time (in sec): %s" %elapsed_rf)

    # cross validation
    print ("Cross-validation...")
    t = time.time()
    scores = cross_val_score(rf, train_features, train_labels, cv=10)
    elapsed_rf = time.time() - t
    print("CV elapsed time (in sec): %s" %elapsed_rf)
    scores
    print('Mean CV Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))

    # # accuracy score
    # predictions=rf.predict(test_features)
    # print("Accuracy:", accuracy_score(test_features, predictions))

    # # Calculate mean absolute percentage error (MAPE)
    # mape = 100 * (errors / test_labels)
    # # Calculate and display accuracy
    # accuracy = 100 - np.mean(mape)
    # print('Accuracy:', round(accuracy, 2), '%.')


    # # Use the forest's predict method on the test data
    predictions = rf.predict(test_features)
    # # # Calculate the absolute errors
    # errors = abs(predictions - test_labels)
    # # # Print out the mean absolute error (mae)
    # print('Mean Absolute Error:', round(np.mean(errors), 3), 'degrees.')
    # print("Accuracy on test data: {:.2f}".format(rf.score(test_features, test_labels)))

    # View the classification report for test data and predictions
    print(confusion_matrix(test_labels,predictions))
    print(accuracy_score(test_labels, predictions))
    print(classification_report(test_labels, predictions))

    # Get numerical feature importances
    importances = list(rf.steps[1][1].feature_importances_)
    # List of tuples with variable and importance
    feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
    # Sort the feature importances by most important first
    feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
    cluster_features = pd.DataFrame(feature_importances, columns = {'feature': '0','importance': '1'})
    cluster_features_50 = cluster_features['importance'].quantile(0.5)
    cluster_features = cluster_features[cluster_features['importance'] > cluster_features_50]

    # Print out the feature and importances
    [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];
    print(cluster_features)

    # Import matplotlib for plotting and use magic command for Jupyter Notebooks
    # Set the style
    # plt.style.use('fivethirtyeight')
    fig = plt.figure(figsize = (100,8))

    # list of x locations for plotting
    x_values = list(range(len(importances)))
    # Make a bar chart
    plt.bar(x_values, importances, orientation = 'vertical')
    # Tick labels for x axis
    plt.xticks(x_values, feature_list, rotation='vertical', fontsize = 8)
    # Axis labels and title
    plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances %s'%cluster_number);

    print("Saving Variable Importances for %s as figure..."%cluster_number)
    plt.savefig(os.path.join(dir,'figs', 'randomforest_grandmean_%s.png'%cluster_number), bbox_inches='tight', dpi=300)
    plt.close()


# ## loop for clusters using cluster means, instead of grand means
# df = df_orig
# # df['rando_baseline_z_score'] = stats.zscore(np.round(np.random.uniform(1.0,8.0,len(df)), 1), axis=0)
# df = df.drop(['Rating','Rating_z_score','kmeans_4_cluster','participant_gender_id','participant_sexual_orientation','participant_voice_id','participant_cis_trans',
#                 'participant_prox_social','participant_prox_affiliation', 'participant_prox_media', 'participant_race','participant_race_hispanic','eng_primary_early','eng_primary_current',
#                 'participant_other_langs','participant_race_free_response','participant_gender_pso_free_response', 'participant_age', 'deaf_hoh', 'Participant',
#                 'survey_experience','survey_feedback','Condition','WAV','color','spectral_S_start','spectral_Z_start','spectral_F_start','spectral_V_start','spectral_JH_start','spectral_SH_start'], axis=1)
# number_of_features = len(df.columns)
# df_cluster_dummies = pd.get_dummies(df['kmeans_5_cluster'], prefix='cluster')
# df = pd.concat([df, df_cluster_dummies], axis=1)
# df = df.drop('kmeans_5_cluster', axis=1)
# df_group_0 = df.drop(['cluster_1','cluster_2','cluster_3','cluster_4'], axis=1)
# df_group_1 = df.drop(['cluster_0','cluster_2','cluster_3','cluster_4'], axis=1)
# df_group_2 = df.drop(['cluster_0','cluster_1','cluster_3','cluster_4'], axis=1)
# df_group_3 = df.drop(['cluster_0','cluster_1','cluster_2','cluster_4'], axis=1)
# df_group_4 = df.drop(['cluster_0','cluster_1','cluster_2','cluster_3'], axis=1)
#
# cluster_dict  = {'df_group_0':df_group_0,'df_group_1':df_group_1,'df_group_2':df_group_2,'df_group_3':df_group_3,'df_group_4':df_group_4}
# for cluster, value in cluster_dict.items():
#     cluster_number = 'cluster_' + cluster[-1:]
#
#     ## impute cluster means for each cluster in the loop
#     imp = SimpleImputer(missing_values=np.nan, strategy='mean')
#     imp = imp.fit(value)
#     value_imp = imp.transform(value)
#     value_imp = pd.DataFrame(value_imp, columns = value.columns)
#
#     ###### BASELINE #######
#     # Labels are the values we want to predict
#     labels = np.array(value_imp[cluster_number])
#     # Remove the labels from the features
#     # axis 1 refers to the columns
#     value_imp = value_imp.drop('%s'%cluster_number, axis = 1)
#     # Saving feature names for later use
#     feature_list = list(value_imp.columns)
#     # Convert to numpy array
#     value_imp = np.array(value_imp)
#
#     # Using Skicit-learn to split data into training and testing sets
#     from sklearn.model_selection import train_test_split
#     # Split the data into training and testing sets
#     train_features, test_features, train_labels, test_labels = train_test_split(value_imp, labels, test_size = 0.25, random_state = 42)
#
#     # # # The baseline predictions are the historical averages
#     # baseline_preds = test_features[:, feature_list.index('rando')]
#     # # # # Baseline errors, and display average baseline error
#     # baseline_errors = abs(baseline_preds - test_labels)
#     # print('Average baseline error: ', round(np.mean(baseline_errors), 3))
#
#     # Instantiate model with 1000 decision trees
#     # rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
#     rf = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators = 5000, random_state = 42, n_jobs = -1, max_features = 184, max_depth = 5))
#     # Train the model on training data
#     print ("Training Random Forest on %s features for %s..."%(number_of_features, cluster_number))
#     t = time.time()
#     rf.fit(train_features, train_labels);
#     elapsed_rf = time.time() - t
#     print("Random Forest elapsed time (in sec): %s" %elapsed_rf)
#
#     # cross validation
#     print ("Cross-validation...")
#     t = time.time()
#     scores = cross_val_score(rf, train_features, train_labels, cv=10)
#     elapsed_rf = time.time() - t
#     print("CV elapsed time (in sec): %s" %elapsed_rf)
#     scores
#     print('Mean CV Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))
#
#     # # accuracy score
#     # predictions=rf.predict(test_features)
#     # print("Accuracy:", accuracy_score(test_features, predictions))
#
#     # # Calculate mean absolute percentage error (MAPE)
#     # mape = 100 * (errors / test_labels)
#     # # Calculate and display accuracy
#     # accuracy = 100 - np.mean(mape)
#     # print('Accuracy:', round(accuracy, 2), '%.')
#
#     # # Use the forest's predict method on the test data
#     predictions = rf.predict(test_features)
#     # # # Calculate the absolute errors
#     # errors = abs(predictions - test_labels)
#     # # # Print out the mean absolute error (mae)
#     # print('Mean Absolute Error:', round(np.mean(errors), 3), 'degrees.')
#     # print("Accuracy on test data: {:.2f}".format(rf.score(test_features, test_labels)))
#
#     # View the classification report for test data and predictions
#     print(confusion_matrix(test_labels,predictions))
#     print(accuracy_score(test_labels, predictions))
#     print(classification_report(test_labels, predictions))
#
#     # Get numerical feature importances
#     importances = list(rf.steps[1][1].feature_importances_)
#     # List of tuples with variable and importance
#     feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
#     # Sort the feature importances by most important first
#     feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
#     cluster_features = pd.DataFrame(feature_importances, columns = {'feature': '0','importance': '1'})
#     cluster_features_50 = cluster_features['importance'].quantile(0.5)
#     cluster_features = cluster_features[cluster_features['importance'] > cluster_features_50]
#
#     # Print out the feature and importances
#     [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];
#     print(cluster_features)
#
#     # Import matplotlib for plotting and use magic command for Jupyter Notebooks
#     # Set the style
#     # plt.style.use('fivethirtyeight')
#     fig = plt.figure(figsize = (100,8))
#
#     # list of x locations for plotting
#     x_values = list(range(len(importances)))
#     # Make a bar chart
#     plt.bar(x_values, importances, orientation = 'vertical')
#     # Tick labels for x axis
#     plt.xticks(x_values, feature_list, rotation='vertical', fontsize = 8)
#     # Axis labels and title
#     plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances %s'%cluster_number);
#
#     print("Saving Variable Importances for %s as figure..."%cluster_number)
#     plt.savefig(os.path.join(dir,'figs', 'randomforest_clustermean_%s.png'%cluster_number), bbox_inches='tight', dpi=300)
#     plt.close()

## save
# df_dummies_all.to_csv(os.path.join(dir,'feature_extraction','dummies_data.csv'), index=True, encoding='utf-8')


# # impute for the cross validation
# imp = SimpleImputer(missing_values=np.nan, strategy='mean')
# imp = imp.fit(df)
# df_imp = imp.transform(df)
#
#
# df_dummies_all = pd.get_dummies(df)
# df_dummies_clusters = pd.get_dummies(df['kmeans_5_cluster'], prefix="cluster")
# df_dummies_all = pd.concat([df_dummies_all,df_dummies_clusters], axis=1)
# df_dummies_all = df_dummies_all.drop(['kmeans_5_cluster'], axis=1)
# df_dummies_all.iloc[:,5:].head(5)
#
# # random number column for sanity checks
# df_dummies_all['rando_baseline'] = np.round(np.random.uniform(1.0,8.0,len(df_dummies_all)), 1)
#
# df_group_1 = df_dummies_all.drop(['cluster_1','cluster_2','cluster_3','cluster_4'], axis=1)
# df_group_2 = df_dummies_all[df_dummies_all['cluster_1'] == 1]
# df_group_3 = df_dummies_all[df_dummies_all['cluster_2'] == 1]
# df_group_4 = df_dummies_all[df_dummies_all['cluster_3'] == 1]
# df_group_5 = df_dummies_all[df_dummies_all['cluster_4'] == 1]
#
# df_group_1 = pd.get_dummies(df[df['kmeans_5_cluster'] == 0])
# df_group_1.iloc[:,5:].head(5)

#
# with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
#     print(df_dummies_all)

## PCA
features = ['F0_mean','F0_range','F0_std','percent_creak','vowel_avg_dur']

# Separating out the features
x = data.loc[:, features].values
# Separating out the target
y = data.loc[:,['kmeans_5_cluster']].values
# Standardizing the features
x = StandardScaler().fit_transform(x)

pca = PCA(n_components=6, random_state=42)
principalComponents = pca.fit_transform(x)
# principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2',
                                                                    'principal component 3', 'principal component 4',
                                                                    'principal component 5', 'principal component 6'])

finalDf = pd.concat([principalDf, data['kmeans_5_cluster']], axis=1)

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
