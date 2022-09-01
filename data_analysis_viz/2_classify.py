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
dir = '/Users/bcl/GitHub/queer_speech'
fig_dir = '/Users/bcl/Library/CloudStorage/GoogleDrive-blang@ucsd.edu/My Drive/Comps/figs/'
os.chdir(dir)
ratings_features_fname = os.path.join(dir, 'feature_extraction', 'ratings_features_all.csv')
data = pd.read_csv(ratings_features_fname)
data = data.drop('Unnamed: 0', axis=1) # might need to change column name when it imports in, there's a random addition at the beginning
data_orig = data

# set random state for K-Means, RF, train_test_split, and PCA
random_state = 42

# with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
#     print(gender_id.columns)

# fill creak nans with 0
data['percent_creak'] = data['percent_creak'].fillna(0)

gender_id = data[data['Condition']=='gender_id']
sexual_orientation = data[data['Condition']=='sexual_orientation']
voice_id = data[data['Condition']=='voice_id']

# start a timer for the entire analysis loop
total_start_time = time.time()

### K-means
print("K-Means clustering...")
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
plt.figure(figsize=(16,8))
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal clusters')
plt.savefig(os.path.join(fig_dir, 'kmeans_cluster_elbow.png'), bbox_inches='tight', dpi=300)
plt.close()

# Define the model for 3 clusters
kmeans_model = KMeans(n_clusters=3, random_state=42)
# Fit into our dataset fit
kmeans_predict = kmeans_model.fit_predict(y)
x['kmeans_3_cluster'] = kmeans_predict

# Define the model for 4 clusters
kmeans_model = KMeans(n_clusters=4, random_state=42)
# Fit into our dataset fit
kmeans_predict = kmeans_model.fit_predict(y)
x['kmeans_4_cluster'] = kmeans_predict

### 5 clusters
kmeans_model = KMeans(n_clusters=5, random_state=42)
# Fit into our dataset fit
kmeans_predict = kmeans_model.fit_predict(y)
x['kmeans_5_cluster'] = kmeans_predict

abc = x
abc = abc.drop(columns=['gender_id','sexual_orientation','voice_id'])
data_merged = pd.merge(data, abc, on='WAV')

# df['rando_baseline_z_score'] = stats.zscore(np.round(np.random.uniform(1.0,8.0,len(df)), 1), axis=0)

# import random
# get_colors = lambda n: list(map(lambda i: "#" + "%06x" % random.randint(0, 0xFFFFFF),range(n)))
# colors = pd.DataFrame(get_colors(5), columns={'color'}) # sample return:  ['#8af5da', '#fbc08c', '#b741d0', '#e599f1', '#bbcb59', '#a2a6c0']
# colors['cluster'] = [0,1,2,3,4]

# 3 clusters
colors = pd.DataFrame({"color_3_cluster": ['#785EF0','#DC267F','#648FFF']})
colors['kmeans_3_cluster'] = [0,1,2]
# 0: straight women, 1: queer NB, men, and women, 2: straight men
df = pd.merge(data_merged, colors, on = 'kmeans_3_cluster', how = "outer")

# 4 clusters
colors = pd.DataFrame({"color_4_cluster": ['#DC267F','#785EF0','#648FFF','#FFB000']})
colors['kmeans_4_cluster'] = [0,1,2,3]
# 0: queer NB, men, and women, 1: straight women, 2: straight men, 3: queer men
df = pd.merge(df, colors, on = 'kmeans_4_cluster', how = "outer")

# 5 clusters
colors = pd.DataFrame({"color_5_cluster": ['#FFB000','#785EF0','#648FFF','#FE6100','#DC267F']})
colors['kmeans_5_cluster'] = [0,1,2,3,4]
# 0: queer men, 1: straight women, 2: straight men, 3: queer women, 4: queer NB, men, and women
df = pd.merge(df, colors, on = 'kmeans_5_cluster', how = "outer")

### double check group identities in clusters
print(df.pivot_table(index='kmeans_3_cluster', columns = 'Condition', values = 'Rating_z_score'))
print(df.pivot_table(index='kmeans_4_cluster', columns = 'Condition', values = 'Rating_z_score'))
print(df.pivot_table(index='kmeans_5_cluster', columns = 'Condition', values = 'Rating_z_score'))

df['3_rando_classes'] = np.random.randint(0, 3, df.shape[0])
df['4_rando_classes'] = np.random.randint(0, 4, df.shape[0])
df['5_rando_classes'] = np.random.randint(0, 5, df.shape[0])


print ("Saving data as queer_data.csv")
df.to_csv(os.path.join(dir,'data_analysis_viz','queer_data.csv'), index=True, encoding='utf-8')

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

########################### RANDOM FOREST ##########################

# for each group from the K-means clustering, slice out the relevant z-scored ratings, with all of the other acoustic features
# what is the acoustic information that is important to the ratings in a particular cluster
type_list = ['baseline','raw_ablation','raw_by_feature','PCA_red_corr']
# cluster_quantity = ['kmeans_3_cluster','kmeans_4_clusters','kmeans_5_clusters']
# min_max = ['min','max']

## set RF params
n_estimators = 1000
random_state = 42
n_jobs = -1

## train, test, split params
test_size = 0.75

## cross vall params
cv = 5

accuracy_list = []
n_features_list = []
df_kind_list = []
removed_features_list = []
one_feature_names_list = []

cluster_accuracy_list = []
cluster_n_features_list = []
cluster_df_kind_list = []
cluster_removed_features_list = []
cluster_one_feature_names_list = []


PCA_accuracy_list = []
PCA_n_features_list = []
PCA_df_kind_list = []
PCA_removed_features_list = []

PCA_cluster_accuracy_list = []
PCA_cluster_n_features_list = []
PCA_cluster_df_kind_list = []
PCA_cluster_removed_features_list = []

df_temp = df[['Rating','Rating_z_score','kmeans_4_cluster','kmeans_3_cluster','3_rando_classes','4_rando_classes','participant_gender_id','participant_sexual_orientation','participant_voice_id','participant_cis_trans',
                'participant_prox_social','participant_prox_affiliation', 'participant_prox_media', 'participant_race','participant_race_hispanic','eng_primary_early','eng_primary_current',
                'participant_other_langs','participant_race_free_response','participant_gender_pso_free_response', 'participant_age', 'deaf_hoh', 'Participant',
                'survey_experience','survey_feedback','Condition','WAV','color_3_cluster','color_4_cluster','color_5_cluster','spectral_S_start','spectral_Z_start','spectral_F_start','spectral_V_start','spectral_JH_start','spectral_SH_start', 'spectral_TH_start', 'spectral_DH_start',]]


df = df.drop(['Rating','Rating_z_score','kmeans_4_cluster','kmeans_3_cluster','3_rando_classes','4_rando_classes','participant_gender_id','participant_sexual_orientation','participant_voice_id','participant_cis_trans',
                'participant_prox_social','participant_prox_affiliation', 'participant_prox_media', 'participant_race','participant_race_hispanic','eng_primary_early','eng_primary_current',
                'participant_other_langs','participant_race_free_response','participant_gender_pso_free_response', 'participant_age', 'deaf_hoh', 'Participant',
                'survey_experience','survey_feedback','Condition','WAV','color_3_cluster','color_4_cluster','color_5_cluster','spectral_S_start','spectral_Z_start','spectral_F_start','spectral_V_start','spectral_JH_start','spectral_SH_start', 'spectral_TH_start', 'spectral_DH_start',
                'S_avg_dur','Z_avg_dur','F_avg_dur','V_avg_dur','JH_avg_dur','SH_avg_dur','TH_avg_dur','DH_avg_dur'], axis=1)

# feature_importances_all = pd.DataFrame({'feature':list(df.columns)})

## impute for the cross validation, random forest needs values in each, below we are creating one large df with imputed column means from teh entire set, and then 5 cluster groups with the gran imputed means
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp = imp.fit(df)
df_imp = imp.transform(df) # this is the big df
df_imp = pd.DataFrame(df_imp, columns = df.columns)

df_orig = df_imp

## Establish chance, grand and clusters should be around 20% as a baseline with randomly assigned classes
for type in type_list:
    if type == 'baseline':
        df_imp = df_orig

        df_imp = df_imp.drop('kmeans_5_cluster', axis=1)

        # make cluster dfs with imputed grand means
        df_cluster_dummies = pd.get_dummies(df_imp['5_rando_classes'], prefix='cluster')
        df_grand = pd.concat([df_imp, df_cluster_dummies], axis=1)
        df_sans = df_grand.drop('5_rando_classes', axis=1)
        df_sans = df_sans.rename(columns={'cluster_0.0':'cluster_0', 'cluster_1.0':'cluster_1', 'cluster_2.0':'cluster_2', 'cluster_3.0':'cluster_3', 'cluster_4.0':'cluster_4'})
        df_group_0 = df_sans.drop(['cluster_1','cluster_2','cluster_3','cluster_4'], axis=1)
        df_group_1 = df_sans.drop(['cluster_0','cluster_2','cluster_3','cluster_4'], axis=1)
        df_group_2 = df_sans.drop(['cluster_0','cluster_1','cluster_3','cluster_4'], axis=1)
        df_group_3 = df_sans.drop(['cluster_0','cluster_1','cluster_2','cluster_4'], axis=1)
        df_group_4 = df_sans.drop(['cluster_0','cluster_1','cluster_2','cluster_3'], axis=1)

        ###### GRAND ####### Big ol' df with imputed means on everyone
        # Labels are the values we want to predict
        labels = np.array(df_imp['5_rando_classes'])
        # Remove the labels from the features
        # axis 1 refers to the columns
        df_imp = df_imp.drop('5_rando_classes', axis = 1)
        number_of_features = len(df_imp.columns)
        # Saving feature names for later use
        feature_list = list(df_imp.columns)
        # Convert to numpy array
        df_imp = np.array(df_imp)

        # Using Skicit-learn to split data into training and testing sets
        from sklearn.model_selection import train_test_split
        # Split the data into training and testing sets
        train_features, test_features, train_labels, test_labels = train_test_split(df_imp, labels, test_size = test_size, random_state = random_state)

        # Import the model we are using
        # Instantiate model with 1000 decision trees
        rf = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators = n_estimators, random_state = random_state, n_jobs = n_jobs))
        # Train the model on training data
        print ("Training Random Forest on %s features for all random classes..."%number_of_features)
        t = time.time()
        rf.fit(train_features, train_labels);
        elapsed_rf = time.time() - t
        print("Random Forest elapsed time (in sec): %s" %elapsed_rf)

        # cross validation
        print ("Cross-validation...")
        # rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
        t = time.time()
        scores = cross_val_score(rf, train_features, train_labels, cv=cv)
        elapsed_rf = time.time() - t
        print("CV elapsed time (in sec): %s" %elapsed_rf)
        scores
        print('Mean CV Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))

        # # Use the forest's predict method on the test data
        predictions = rf.predict(test_features)

        # View the classification report for test data and predictions
        print("Accuracy on test data..")
        print(accuracy_score(test_labels, predictions))
        print("Confusion matrix on test data..")
        print(confusion_matrix(test_labels,predictions))
        print("Classification report on test data...")
        print(classification_report(test_labels, predictions))

        # Get numerical feature importances
        importances = list(rf.steps[1][1].feature_importances_)
        # importances = list(result.importances_mean)
        # List of tuples with variable and importance
        feature_importances = [(feature, round(importance, 5)) for feature, importance in zip(feature_list, importances)]
        # Sort the feature importances by most important first
        feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
        grand_features = pd.DataFrame(feature_importances, columns = {'feature': '0','grand_importance': '1'})
        grand_features_50 = grand_features['grand_importance'].quantile(0.5)
        grand_features_50 = grand_features[grand_features['grand_importance'] > grand_features_50]
        # Print out the feature and importances
        # [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];
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

        plt.savefig(os.path.join(fig_dir, 'randomforest_grandmean_rando.png'), bbox_inches='tight', dpi=300)
        plt.close()

        ######### CLUSTERS ########
        ### clusters using grand mean, no additional imputing needed

        cluster_dict  = {'df_group_0':df_group_0,'df_group_1':df_group_1,'df_group_2':df_group_2,'df_group_3':df_group_3,'df_group_4':df_group_4}
        for cluster, value in cluster_dict.items():
            cluster_number = 'cluster_' + cluster[-1:]
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
            train_features, test_features, train_labels, test_labels = train_test_split(value, labels, test_size = test_size, random_state = random_state)

            # Instantiate model with 1000 decision trees
            # rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
            rf = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators = n_estimators, random_state = random_state, n_jobs = n_jobs))
            # Train the model on training data
            print ("Training Random Forest on %s features for random class %s..."%(number_of_features, cluster_number))
            t = time.time()
            rf.fit(train_features, train_labels);
            elapsed_rf = time.time() - t
            print("Random Forest elapsed time (in sec): %s" %elapsed_rf)

            # cross validation
            print ("Cross-validation...")
            t = time.time()
            scores = cross_val_score(rf, train_features, train_labels, cv=cv)
            elapsed_rf = time.time() - t
            print("CV elapsed time (in sec): %s" %elapsed_rf)
            scores
            print('Mean CV Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))

            # # Use the forest's predict method on the test data
            predictions = rf.predict(test_features)

            # View the classification report for test data and predictions
            print("Accuracy on test data..")
            print(accuracy_score(test_labels, predictions))
            print("Confusion matrix on test data..")
            print(confusion_matrix(test_labels,predictions))
            print("Classification report on test data...")
            print(classification_report(test_labels, predictions))

            # Get numerical feature importances
            importances = list(rf.steps[1][1].feature_importances_)
            # List of tuples with variable and importance
            feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
            # Sort the feature importances by most important first
            feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
            cluster_features = pd.DataFrame(feature_importances, columns = {'feature': '0','importance_%s'%cluster_number: '1'})
            # cluster_features_50 = cluster_features['importance_%s'%cluster_number].quantile(0.5)
            # cluster_features_50 = cluster_features[cluster_features['importance_%s'%cluster_number] > cluster_features_50]

            # Print out the feature and importances
            # [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];
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
            plt.savefig(os.path.join(fig_dir, 'randomforest_grandmean_%s_rando.png'%cluster_number), bbox_inches='tight', dpi=300)
            plt.close()

    if type == 'raw_ablation':
        df_imp = df_orig

        df_imp = df_imp.drop('5_rando_classes', axis=1)

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


        for i in range(len(df_orig.columns)-2):
            ###### GRAND ####### Big ol' df with imputed means on everyone
            # Labels are the values we want to predict
            labels = np.array(df_imp['kmeans_5_cluster'])
            # Remove the labels from the features
            # axis 1 refers to the columns
            df_array = df_imp.drop('kmeans_5_cluster', axis = 1)
            number_of_features = len(df_array.columns)
            # Saving feature names for later use
            feature_list = list(df_array.columns)
            # Convert to numpy array
            df_array = np.array(df_array)

            # Using Skicit-learn to split data into training and testing sets
            from sklearn.model_selection import train_test_split
            # Split the data into training and testing sets
            train_features, test_features, train_labels, test_labels = train_test_split(df_array, labels, test_size = test_size, random_state = random_state)

            # Import the model we are using
            # Instantiate model with 1000 decision trees
            rf = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators = n_estimators, random_state = random_state, n_jobs = n_jobs))
            # Train the model on training data
            print ("Training Random Forest on %s features for all clusters..."%number_of_features)
            t = time.time()
            rf.fit(train_features, train_labels);
            elapsed_rf = time.time() - t
            print("Random Forest elapsed time (in sec): %s" %elapsed_rf)

            # cross validation
            print ("Cross-validation...")
            # rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
            t = time.time()
            scores = cross_val_score(rf, train_features, train_labels, cv=cv)
            elapsed_rf = time.time() - t
            print("CV elapsed time (in sec): %s" %elapsed_rf)
            scores
            print('Mean CV Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))

            # # Use the forest's predict method on the test data
            predictions = rf.predict(test_features)

            # View the classification report for test data and predictions
            print("Accuracy on test data..")
            print(accuracy_score(test_labels, predictions))
            print("Confusion matrix on test data..")
            print(confusion_matrix(test_labels,predictions))
            print("Classification report on test data...")
            print(classification_report(test_labels, predictions))

            # Get numerical feature importances
            importances = list(rf.steps[1][1].feature_importances_)
            # importances = list(result.importances_mean)
            # List of tuples with variable and importance
            feature_importances = [(feature, round(importance, 20)) for feature, importance in zip(feature_list, importances)]
            # Sort the feature importances by most important first
            feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
            grand_features = pd.DataFrame(feature_importances, columns = {'feature': '0','grand_importance': '1'})
            # grand_features_50 = grand_features['grand_importance'].quantile(0.5)
            # grand_features_50 = grand_features[grand_features['grand_importance'] > grand_features_50]
            # Print out the feature and importances
            # [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];
            # print(grand_features)

            ## make variables to save importances

            # Import matplotlib for plotting and use magic command for Jupyter Notebooks
            # Set the style
            # plt.style.use('fivethirtyeight')
            if i == 0:
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

                plt.savefig(os.path.join(fig_dir, 'ablation','randomforest_grandmean_raw_all_features.png'), bbox_inches='tight', dpi=300)
                plt.close()

                print ("Saving data as grand_feature_importances.csv")
                grand_features.to_csv(os.path.join(dir,'data_analysis_viz','grand_feature_importances.csv'), index=True, encoding='utf-8')

            # feature_importances_all = pd.merge(feature_importances_all, grand_features, on = 'feature', how = 'outer')

            ## feature ablation ##
            # save accuracy and number of features on original RF
            print ('Appending grand values...')
            accuracy_list.append(accuracy_score(test_labels, predictions))
            n_features_list.append(len(grand_features['feature']))
            df_kind_list.append('grand_raw')
            # print(accuracy_list)
            # print(n_features_list)
            # print(df_kind_list)
            # find the min of the features
            idx_max_feature = grand_features[['grand_importance']].idxmax()[0]
            max_feature = grand_features.iloc[idx_max_feature][0]
            removed_features_list.append(max_feature)
            # remove min
            print('Dropping maximum feature: %s...'%max_feature)
            df_imp = df_imp.drop(max_feature, axis=1)
            # do the RF again
            # save accuracy and number of features
            # repeat til 1 feature left
            # plot x number of features and y accuracy

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
                value_array = value.drop('%s'%cluster_number, axis = 1)
                number_of_cluster_features = len(value_array.columns)
                # Saving feature names for later use
                feature_list = list(value_array.columns)
                # Convert to numpy array
                value_array = np.array(value_array)

                # Using Skicit-learn to split data into training and testing sets
                from sklearn.model_selection import train_test_split
                # Split the data into training and testing sets
                train_features, test_features, train_labels, test_labels = train_test_split(value_array, labels, test_size = test_size, random_state = random_state)

                # Instantiate model with 1000 decision trees
                # rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
                rf = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators = n_estimators, random_state = random_state, n_jobs = n_jobs))
                # Train the model on training data
                print ("Training Random Forest on %s features for %s..."%(number_of_cluster_features, cluster_number))
                t = time.time()
                rf.fit(train_features, train_labels);
                elapsed_rf = time.time() - t
                print("Random Forest elapsed time (in sec): %s" %elapsed_rf)

                # cross validation
                print ("Cross-validation...")
                t = time.time()
                scores = cross_val_score(rf, train_features, train_labels, cv=cv)
                elapsed_rf = time.time() - t
                print("CV elapsed time (in sec): %s" %elapsed_rf)
                scores
                print('Mean CV Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))

                # Use the forest's predict method on the test data
                predictions = rf.predict(test_features)

                # View the classification report for test data and predictions
                print("Accuracy on test data..")
                print(accuracy_score(test_labels, predictions))
                print("Confusion matrix on test data..")
                print(confusion_matrix(test_labels,predictions))
                print("Classification report on test data...")
                print(classification_report(test_labels, predictions))

                # Get numerical feature importances
                importances = list(rf.steps[1][1].feature_importances_)
                # List of tuples with variable and importance
                feature_importances = [(feature, round(importance, 20)) for feature, importance in zip(feature_list, importances)]
                # Sort the feature importances by most important first
                feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
                cluster_features = pd.DataFrame(feature_importances, columns = {'feature': '0','importance_%s'%cluster_number: '1'})
                # cluster_features_50 = cluster_features['importance_%s'%cluster_number].quantile(0.5)
                # cluster_features_50 = cluster_features[cluster_features['importance_%s'%cluster_number] > cluster_features_50]

                # Print out the feature and importances
                # [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];
                # print(cluster_features)

                # Import matplotlib for plotting and use magic command for Jupyter Notebooks
                # Set the style
                # plt.style.use('fivethirtyeight')
                if i == 0:
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
                    plt.savefig(os.path.join(fig_dir, 'ablation', 'randomforest_%s_raw_all_features_ablated.png'%cluster_number), bbox_inches='tight', dpi=300)
                    plt.close()

                    print ("Saving data as cluster_feature_importances.csv")
                    cluster_features.to_csv(os.path.join(dir,'data_analysis_viz','cluster_feature_importances_%s.csv'%cluster_number), index=True, encoding='utf-8')

                print ('Appending cluster values...')
                cluster_accuracy_list.append(accuracy_score(test_labels, predictions))
                cluster_n_features_list.append(len(cluster_features['feature']))
                cluster_df_kind_list.append(cluster_number)
                # print(cluster_accuracy_list)
                # print(cluster_n_features_list)
                # print(cluster_df_kind_list)
                # find the min of the features
                idx_max_feature = cluster_features[['importance_%s'%cluster_number]].idxmax()[0]
                max_feature = cluster_features.iloc[idx_max_feature][0]
                cluster_removed_features_list.append(max_feature)
                # remove min
                print('Dropping maximum feature: %s...'%max_feature)
                if cluster == 'df_group_0':
                    df_group_0 = df_group_0.drop(max_feature, axis=1)
                if cluster == 'df_group_1':
                    df_group_1 = df_group_1.drop(max_feature, axis=1)
                if cluster == 'df_group_2':
                    df_group_2 = df_group_2.drop(max_feature, axis=1)
                if cluster == 'df_group_3':
                    df_group_3 = df_group_3.drop(max_feature, axis=1)
                if cluster == 'df_group_4':
                    df_group_4 = df_group_4.drop(max_feature, axis=1)
                if i == 371:
                    ablated_df = pd.DataFrame({'accuracy':cluster_accuracy_list, 'n_features': cluster_n_features_list, 'kind': cluster_df_kind_list, 'removed_feature': cluster_removed_features_list})

                    fig = plt.figure(figsize = (100,8))
                    plt.title('Model Accuracy by Features Ablated (Elbow Plot)')
                    # plt.plot(ablated_df['n_features'].sort_values(), ablated_df['accuracy'])
                    # plt.plot([0, 20], 'k-', lw=2)
                    plt.bar(ablated_df['n_features'].sort_values(), ablated_df['accuracy'])
                    plt.xlabel('Number of Features')
                    plt.ylabel('Model Accuracy')
                    # plt.show()

                    print("Saving feature ablation for Variable Importances for %s as figure..."%cluster_number)
                    plt.savefig(os.path.join(fig_dir, 'ablation', 'randomforest_feature_ablation_%s.png'%cluster_number), bbox_inches='tight', dpi=300)
                    plt.close()

                    print ("Saving data as cluster_ablated_df.csv")
                    ablated_df.to_csv(os.path.join(dir,'data_analysis_viz','cluster_ablated_df.csv'), index=True, encoding='utf-8')

                # feature_importances_all = pd.merge(feature_importances_all, cluster_features, on = 'feature', how = 'outer', suffixes=('_%s'%cluster_number, 'y'))
                # print ("Saving raw feature importances as feature_importances_all.csv")
                # df.to_csv(os.path.join(dir,'data_analysis_viz','feature_importances_all.csv'), index=True, encoding='utf-8')

            if i == 371:
                grand_ablated_df = pd.DataFrame({'accuracy':accuracy_list, 'n_features': n_features_list, 'kind': df_kind_list, 'removed_feature': removed_features_list})

                fig = plt.figure(figsize = (100,8))
                plt.title('Model Accuracy by Features Ablated (Elbow Plot)')
                # plt.plot(grand_ablated_df['n_features'].sort_values(), grand_ablated_df['accuracy'])
                # plt.plot([0, 20], 'k-', lw=2)
                plt.bar(grand_ablated_df['n_features'].sort_values(), grand_ablated_df['accuracy'])
                plt.xlabel('Number of Features')
                plt.ylabel('Model Accuracy')
                # plt.show()

                print("Saving feature ablation for Variable Importances for all clusters as figure...")
                plt.savefig(os.path.join(fig_dir, 'ablation', 'randomforest_feature_ablation_all.png'), bbox_inches='tight', dpi=300)
                plt.close()

                print ("Saving data as grand_ablated_df.csv")
                grand_ablated_df.to_csv(os.path.join(dir,'data_analysis_viz','grand_ablated_df.csv'), index=True, encoding='utf-8')

    if type == 'raw_by_feature':
        df_imp = df_orig

        df_imp = df_imp.drop('5_rando_classes', axis=1)

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

        accuracy_list = []
        n_features_list = []
        df_kind_list = []
        removed_features_list = []
        one_feature_names_list = []

        cluster_accuracy_list = []
        cluster_n_features_list = []
        cluster_df_kind_list = []
        cluster_removed_features_list = []
        cluster_one_feature_names_list = []

        ###### GRAND ####### Big ol' df with imputed means on everyone
        # Labels are the values we want to predict
        labels = np.array(df_imp['kmeans_5_cluster'])
        # Remove the labels from the features
        # axis 1 refers to the columns
        df_array = df_imp.drop('kmeans_5_cluster', axis = 1)
        # Saving feature names for later use
        feature_list_all = list(df_array.columns)

        loop_number = 0
        for column in feature_list_all:
            df_one_feature = df_array[column]
            # number_of_features = len(df_one_feature.columns)
            # Saving feature names for later use
            feature_list = list(df_one_feature.name)
            feature_name = df_one_feature.name
            # Convert to numpy array
            df_one_feature = np.array(df_one_feature)

            # Using Skicit-learn to split data into training and testing sets
            from sklearn.model_selection import train_test_split
            # Split the data into training and testing sets
            train_features, test_features, train_labels, test_labels = train_test_split(df_one_feature, labels, test_size = test_size, random_state = random_state)

            train_features = train_features.reshape(-1, 1)
            # train_labels = train_labels.reshape(-1, 1)
            test_features = test_features.reshape(-1, 1)
            # test_labels = test_labels.reshape(-1, 1)

            # Import the model we are using
            # Instantiate model with 1000 decision trees
            rf = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators = n_estimators, random_state = random_state, n_jobs = n_jobs))
            # Train the model on training data
            print ("Training Random Forest on %s feature for all clusters..."%feature_name)
            t = time.time()
            rf.fit(train_features, train_labels);
            elapsed_rf = time.time() - t
            print("Random Forest elapsed time (in sec): %s" %elapsed_rf)

            # cross validation
            print ("Cross-validation...")
            # rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
            t = time.time()
            scores = cross_val_score(rf, train_features, train_labels, cv=cv)
            elapsed_rf = time.time() - t
            print("CV elapsed time (in sec): %s" %elapsed_rf)
            scores
            print('Mean CV Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))

            # # Use the forest's predict method on the test data
            predictions = rf.predict(test_features)

            # View the classification report for test data and predictions
            print("Accuracy on test data..")
            print(accuracy_score(test_labels, predictions))
            print("Confusion matrix on test data..")
            print(confusion_matrix(test_labels,predictions))
            print("Classification report on test data...")
            print(classification_report(test_labels, predictions))

            # Get numerical feature importances
            importances = list(rf.steps[1][1].feature_importances_)
            # importances = list(result.importances_mean)
            # List of tuples with variable and importance
            feature_importances = [(feature, round(importance, 20)) for feature, importance in zip(feature_list, importances)]
            # Sort the feature importances by most important first
            feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
            grand_features = pd.DataFrame(feature_importances, columns = {'feature': '0','grand_importance': '1'})
            # grand_features_50 = grand_features['grand_importance'].quantile(0.5)
            # grand_features_50 = grand_features[grand_features['grand_importance'] > grand_features_50]
            # Print out the feature and importances
            # [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];
            # print(grand_features)

            ## make variables to save importances

            #Import matplotlib for plotting and use magic command for Jupyter Notebooks
            #Set the style
            #plt.style.use('fivethirtyeight')
            # fig = plt.figure(figsize = (100,8))
            #
            # # grand_features_list = grand_features['feature'].to_list()
            #
            # # list of x locations for plotting
            # x_values = list(range(len(importances)))
            # # Make a bar chart
            # plt.bar(x_values, importances, orientation = 'vertical')
            # # Tick labels for x axis
            # plt.xticks(x_values, feature_list, rotation='vertical', fontsize = 8)
            # # Axis labels and title
            # plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances');
            #
            # plt.savefig(os.path.join(dir,'figs', 'ablation','randomforest_grandmean_raw_%s_features_ablated.png'%i), bbox_inches='tight', dpi=300)
            # plt.close()

            # feature_importances_all = pd.merge(feature_importances_all, grand_features, on = 'feature', how = 'outer')

            ## feature ablation ##
            # save accuracy and number of features on original RF
            print ('Appending grand values...')
            accuracy_list.append(accuracy_score(test_labels, predictions))
            n_features_list.append(len(grand_features['feature']))
            df_kind_list.append('grand_raw')
            one_feature_names_list.append(column)
            # print(accuracy_list)
            # print(n_features_list)
            # print(df_kind_list)
            # find the min of the features
            # idx_max_feature = grand_features[['grand_importance']].idxmax()[0]
            # max_feature = grand_features.iloc[idx_max_feature][0]
            # removed_features_list.append(max_feature)
            # remove min
            # print('Dropping maximum feature: %s...'%max_feature)
            # df_imp = df_imp.drop(max_feature, axis=1)
            # do the RF again
            # save accuracy and number of features
            # repeat til 1 feature left
            # plot x number of features and y accuracy

            ######### CLUSTERS ########
            ### clusters using grand mean, no additional imputing needed
            cluster_dict  = {'df_group_0':df_group_0,'df_group_1':df_group_1,'df_group_2':df_group_2,'df_group_3':df_group_3,'df_group_4':df_group_4}
            loop_number = loop_number + 1
            for cluster, value in cluster_dict.items():
                cluster_number = 'cluster_' + cluster[-1:]

                ###### BASELINE #######
                # Labels are the values we want to predict
                labels = np.array(value[cluster_number])
                # Remove the labels from the features
                # axis 1 refers to the columns
                value_array = value.drop('%s'%cluster_number, axis = 1)
                # number_of_cluster_features = len(value_array.columns)
                # Saving feature names for later use
                feature_list = list(value_array.columns)
                value_array_one_feature = value_array[column]
                # Convert to numpy array
                value_array_one_feature = np.array(value_array_one_feature)

                # Using Skicit-learn to split data into training and testing sets
                from sklearn.model_selection import train_test_split
                # Split the data into training and testing sets
                train_features, test_features, train_labels, test_labels = train_test_split(value_array_one_feature, labels, test_size = test_size, random_state = random_state)

                train_features = train_features.reshape(-1, 1)
                # train_labels = train_labels.reshape(-1, 1)
                test_features = test_features.reshape(-1, 1)
                # test_labels = test_labels.reshape(-1, 1)

                # Instantiate model with 1000 decision trees
                # rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
                rf = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators = n_estimators, random_state = random_state, n_jobs = n_jobs))
                # Train the model on training data
                print ("Training Random Forest on %s feature for %s..."%(feature_name, cluster_number))
                t = time.time()
                rf.fit(train_features, train_labels);
                elapsed_rf = time.time() - t
                print("Random Forest elapsed time (in sec): %s" %elapsed_rf)

                # cross validation
                print ("Cross-validation...")
                t = time.time()
                scores = cross_val_score(rf, train_features, train_labels, cv=cv)
                elapsed_rf = time.time() - t
                print("CV elapsed time (in sec): %s" %elapsed_rf)
                scores
                print('Mean CV Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))

                # Use the forest's predict method on the test data
                predictions = rf.predict(test_features)

                # View the classification report for test data and predictions
                print("Accuracy on test data..")
                print(accuracy_score(test_labels, predictions))
                print("Confusion matrix on test data..")
                print(confusion_matrix(test_labels,predictions))
                print("Classification report on test data...")
                print(classification_report(test_labels, predictions))

                # Get numerical feature importances
                importances = list(rf.steps[1][1].feature_importances_)
                # List of tuples with variable and importance
                feature_importances = [(feature, round(importance, 20)) for feature, importance in zip(feature_list, importances)]
                # Sort the feature importances by most important first
                feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
                cluster_features = pd.DataFrame(feature_importances, columns = {'feature': '0','importance_%s'%cluster_number: '1'})
                # cluster_features_50 = cluster_features['importance_%s'%cluster_number].quantile(0.5)
                # cluster_features_50 = cluster_features[cluster_features['importance_%s'%cluster_number] > cluster_features_50]

                # Print out the feature and importances
                # [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];
                # print(cluster_features)

                # Import matplotlib for plotting and use magic command for Jupyter Notebooks
                # Set the style
                # plt.style.use('fivethirtyeight')
                # fig = plt.figure(figsize = (100,8))
                #
                # # list of x locations for plotting
                # x_values = list(range(len(importances)))
                # # Make a bar chart
                # plt.bar(x_values, importances, orientation = 'vertical')
                # # Tick labels for x axis
                # plt.xticks(x_values, feature_list, rotation='vertical', fontsize = 8)
                # # Axis labels and title
                # plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances %s'%cluster_number);
                #
                # print("Saving Variable Importances for %s as figure..."%cluster_number)
                # plt.savefig(os.path.join(dir,'figs', 'ablation', 'randomforest_grandmean_%s_raw_%s_features_ablated.png'%(cluster_number,i)), bbox_inches='tight', dpi=300)
                # plt.close()

                print ('Appending cluster values...')
                cluster_accuracy_list.append(accuracy_score(test_labels, predictions))
                cluster_n_features_list.append(len(cluster_features['feature']))
                cluster_df_kind_list.append(cluster_number)
                cluster_one_feature_names_list.append(column)
                # print(cluster_accuracy_list)
                # print(cluster_n_features_list)
                # print(cluster_df_kind_list)
                # find the min of the features
                # idx_max_feature = cluster_features[['importance_%s'%cluster_number]].idxmax()[0]
                # max_feature = cluster_features.iloc[idx_max_feature][0]
                # cluster_removed_features_list.append(max_feature)
                # remove min
                # print('Dropping maximum feature: %s...'%max_feature)
                # if cluster == 'df_group_0':
                #     df_group_0 = df_group_0.drop(max_feature, axis=1)
                # if cluster == 'df_group_1':
                #     df_group_1 = df_group_1.drop(max_feature, axis=1)
                # if cluster == 'df_group_2':
                #     df_group_2 = df_group_2.drop(max_feature, axis=1)
                # if cluster == 'df_group_3':
                #     df_group_3 = df_group_3.drop(max_feature, axis=1)
                # if cluster == 'df_group_4':
                #     df_group_4 = df_group_4.drop(max_feature, axis=1)
                if loop_number == 371:
                    ablated_df = pd.DataFrame({'accuracy':cluster_accuracy_list, 'n_features': cluster_n_features_list, 'kind': cluster_df_kind_list, 'feature_name': cluster_one_feature_names_list})

                    fig = plt.figure(figsize = (100,8))
                    plt.title('Model Accuracy by Feature')
                    # plt.plot(ablated_df['n_features'].sort_values(), ablated_df['accuracy'])
                    # plt.plot([0, 20], 'k-', lw=2)
                    plt.bar(ablated_df['feature_name'], ablated_df['accuracy'])
                    plt.xticks(ablated_df['feature_name'], rotation='vertical', fontsize = 8)
                    plt.xlabel('Number of Features')
                    plt.ylabel('Model Accuracy')
                    # plt.show()

                    print("Saving single features for Variable Importances for %s as figure..."%cluster_number)
                    plt.savefig(os.path.join(fig_dir, 'randomforest_byfeature_%s.png'%cluster_number), bbox_inches='tight', dpi=300)
                    plt.close()

                    print ("Saving data as cluster_byfeature_df.csv")
                    ablated_df.to_csv(os.path.join(dir,'data_analysis_viz','cluster_byfeature_df.csv'), index=True, encoding='utf-8')

                # feature_importances_all = pd.merge(feature_importances_all, cluster_features, on = 'feature', how = 'outer', suffixes=('_%s'%cluster_number, 'y'))
                # print ("Saving raw feature importances as feature_importances_all.csv")
                # df.to_csv(os.path.join(dir,'data_analysis_viz','feature_importances_all.csv'), index=True, encoding='utf-8')

            if loop_number == 371:
                grand_ablated_df = pd.DataFrame({'accuracy':accuracy_list, 'n_features': n_features_list, 'kind': df_kind_list, 'feature_name': one_feature_names_list})

                fig = plt.figure(figsize = (100,8))
                plt.title('Model Accuracy by Feature')
                # plt.plot(grand_ablated_df['n_features'].sort_values(), grand_ablated_df['accuracy'])
                # plt.plot([0, 20], 'k-', lw=2)
                plt.bar(grand_ablated_df['feature_name'], grand_ablated_df['accuracy'])
                plt.xticks(grand_ablated_df['feature_name'], rotation='vertical', fontsize = 8)
                plt.xlabel('Number of Features')
                plt.ylabel('Model Accuracy')
                # plt.show()

                print("Saving single features for Variable Importances for all clusters as figure...")
                plt.savefig(os.path.join(fig_dir, 'randomforest_byfeature_all.png'), bbox_inches='tight', dpi=300)
                plt.close()

                print ("Saving data as grand_byfeature_df.csv")
                grand_ablated_df.to_csv(os.path.join(dir,'data_analysis_viz','grand_byfeature_df.csv'), index=True, encoding='utf-8')

    if type == 'PCA_red_corr':
        df_imp = df_orig

        df_imp = df_imp.drop('5_rando_classes', axis=1)

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

        ## PCA
        data = df_imp

        features = list(data.columns)
        feature_number = len(features)

        # Separating out the features
        x = data.loc[:, features].values
        # Separating out the target
        y = data.loc[:,['kmeans_5_cluster']].values
        # Standardizing the features
        x = StandardScaler().fit_transform(x)

        pca = PCA(n_components=10, random_state=random_state)
        print("Fitting PCA on grand df...")
        principalComponents = pca.fit_transform(x)
        pc_columns = []
        for j in range(len(principalComponents[1])):
            pc_string = 'Principal Component' + " " + str(j + 1)
            pc_columns.append(pc_string)
        # principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
        principalDf = pd.DataFrame(data = principalComponents, columns = pc_columns)

        finalDf = pd.concat([principalDf, data['kmeans_5_cluster']], axis=1)

        print ("Saving data as principal_components.csv")
        finalDf.to_csv(os.path.join(dir,'data_analysis_viz','principal_components.csv'), index=True, encoding='utf-8')

        exp_var_all = pd.DataFrame({"explained_variance": pca.explained_variance_ratio_})
        print ("Saving data as pca_explained_variance_all.csv")
        exp_var_all.to_csv(os.path.join(dir,'data_analysis_viz','pca_explained_variance_all.csv'), index=True, encoding='utf-8')

        for i in range(0,10):

            ###### BASELINE ####### Big ol' df with imputed means on everyone
            # Labels are the values we want to predict
            labels = np.array(finalDf['kmeans_5_cluster'])
            # Remove the labels from the features
            # axis 1 refers to the columns
            finalDf_array = finalDf.drop('kmeans_5_cluster', axis = 1)
            number_of_features = len(finalDf_array.columns)
            # Saving feature names for later use
            feature_list = list(finalDf_array.columns)
            # Convert to numpy array
            finalDf_array = np.array(finalDf_array)

            # Using Skicit-learn to split data into training and testing sets
            from sklearn.model_selection import train_test_split
            # Split the data into training and testing sets
            train_features, test_features, train_labels, test_labels = train_test_split(finalDf_array, labels, test_size = test_size, random_state = random_state)

            # Import the model we are using
            # Instantiate model with 1000 decision trees
            rf = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators = n_estimators, random_state = random_state, n_jobs = n_jobs))
            # Train the model on training data
            print ("Training Random Forest on %s PCs for all clusters..."%(10-i))
            t = time.time()
            rf.fit(train_features, train_labels);
            elapsed_rf = time.time() - t
            print("Random Forest elapsed time (in sec): %s" %elapsed_rf)

            # cross validation
            print ("Cross-validation...")
            # rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
            t = time.time()
            scores = cross_val_score(rf, train_features, train_labels, cv=cv)
            elapsed_rf = time.time() - t
            print("CV elapsed time (in sec): %s" %elapsed_rf)
            scores
            print('Mean CV Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))

            # # Use the forest's predict method on the test data
            predictions = rf.predict(test_features)

            # View the classification report for test data and predictions
            print("Accuracy on test data..")
            print(accuracy_score(test_labels, predictions))
            print("Confusion matrix on test data..")
            print(confusion_matrix(test_labels,predictions))
            print("Classification report on test data...")
            print(classification_report(test_labels, predictions))

            # Get numerical feature importances
            importances = list(rf.steps[1][1].feature_importances_)
            # importances = list(result.importances_mean)
            # List of tuples with variable and importance
            feature_importances = [(feature, round(importance, 5)) for feature, importance in zip(feature_list, importances)]
            # Sort the feature importances by most important first
            feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
            grand_PCA = pd.DataFrame(feature_importances, columns = {'component': '0','grand_importance': '1'})
            # grand_PCA_50 = grand_PCA['grand_importance'].quantile(0.5)
            # grand_PCA_50 = grand_PCA[grand_PCA['grand_importance'] > grand_PCA_50]
            # Print out the feature and importances
            # [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];
            # print(grand_PCA)

            ## make variables to save importances

            # Import matplotlib for plotting and use magic command for Jupyter Notebooks
            # Set the style
            # plt.style.use('fivethirtyeight')
            # fig = plt.figure(figsize = (100,8))
            #
            # # grand_features_list = grand_features['feature'].to_list()
            #
            # # list of x locations for plotting
            # x_values = list(range(len(importances)))
            # # Make a bar chart
            # plt.bar(x_values, importances, orientation = 'vertical')
            # # Tick labels for x axis
            # plt.xticks(x_values, feature_list, rotation='vertical', fontsize = 8)
            # # Axis labels and title
            # plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances');
            #
            # plt.savefig(os.path.join(dir,'figs', 'ablation', 'randomforest_grandmean_PCA_%s_components.png'%i), bbox_inches='tight', dpi=300)
            # plt.close()

            if i == 0:
                ### correlation ###
                components_list = principalDf.columns
                df_pc = finalDf
                features_corr = features[:372]
                pc_corr_r_all = pd.DataFrame(components_list, columns={'PC'})
                pc_corr_p_all = pd.DataFrame(components_list, columns={'PC'})
                for feature in features_corr:
                    corr_list_r = []
                    corr_list_p = []
                    for pc in components_list:
                        pc_correlation_r = stats.pearsonr(data[feature], df_pc[pc])[0]
                        pc_correlation_p = stats.pearsonr(data[feature], df_pc[pc])[1]
                        corr_list_r.append(pc_correlation_r)
                        corr_list_p.append(pc_correlation_p)
                    single_feature_corr_r = pd.DataFrame(corr_list_r, columns={feature})
                    single_feature_corr_p = pd.DataFrame(corr_list_p, columns={feature})
                    pc_corr_r_all = pd.concat([pc_corr_r_all, single_feature_corr_r], axis=1)
                    pc_corr_p_all = pd.concat([pc_corr_p_all, single_feature_corr_p], axis=1)

                print ("Saving data as grand_corr_r.csv and others...")
                pc_corr_r_all.to_csv(os.path.join(dir,'data_analysis_viz','grand_corr_r.csv'), index=True, encoding='utf-8')
                print ("Saving data as grand_corr_p.csv and others...")
                pc_corr_p_all.to_csv(os.path.join(dir,'data_analysis_viz','grand_corr_p.csv'), index=True, encoding='utf-8')

                ## subsets ##
                data_corr_subsets = pd.concat([data, df_temp], axis=1)
                data_corr_subsets = pd.concat([data_corr_subsets, df_pc], axis=1)
                gender_id = data_corr_subsets[data_corr_subsets['Condition']=='gender_id']
                sexual_orientation = data_corr_subsets[data_corr_subsets['Condition']=='sexual_orientation']
                voice_id = data_corr_subsets[data_corr_subsets['Condition']=='voice_id']

                pc_corr_r_all = pd.DataFrame(components_list, columns={'PC'})
                pc_corr_p_all = pd.DataFrame(components_list, columns={'PC'})
                condition_dict = {'gender_id': gender_id, 'sexual_orientation': sexual_orientation, 'voice_id': voice_id}
                for condition, df in condition_dict.items():
                    for feature in features_corr:
                        corr_list_r = []
                        corr_list_p = []
                        for pc in components_list:
                            pc_correlation_r = stats.pearsonr(df[feature], df[pc])[0]
                            pc_correlation_p = stats.pearsonr(df[feature], df[pc])[1]
                            corr_list_r.append(pc_correlation_r)
                            corr_list_p.append(pc_correlation_p)
                        single_feature_corr_r = pd.DataFrame(corr_list_r, columns={feature})
                        single_feature_corr_p = pd.DataFrame(corr_list_p, columns={feature})
                        pc_corr_r_all = pd.concat([pc_corr_r_all, single_feature_corr_r], axis=1)
                        pc_corr_p_all = pd.concat([pc_corr_p_all, single_feature_corr_p], axis=1)
                    pc_corr_r_all.to_csv(os.path.join(dir,'data_analysis_viz','pc_corr_r_%s.csv'%condition), index=True, encoding='utf-8')
                    pc_corr_p_all.to_csv(os.path.join(dir,'data_analysis_viz','pc_corr_p_%s.csv'%condition), index=True, encoding='utf-8')

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

                plt.savefig(os.path.join(fig_dir, 'ablation','randomforest_grandmean_pca_all.png'), bbox_inches='tight', dpi=300)
                plt.close()

                print ("Saving data as grand_feature_importances_pc.csv")
                grand_PCA.to_csv(os.path.join(dir,'data_analysis_viz','grand_feature_importances_pc.csv'), index=True, encoding='utf-8')


            ## principal component reduction ##
            # save accuracy and number of features on original RF
            print ('Appending grand values...')
            PCA_accuracy_list.append(accuracy_score(test_labels, predictions))
            PCA_n_features_list.append(len(grand_PCA['component']))
            PCA_df_kind_list.append('grand_raw')
            # print(PCA_accuracy_list)
            # print(PCA_n_features_list)
            # print(PCA_df_kind_list)
            # find the min of the features

            idx_max_feature = grand_PCA[['grand_importance']].idxmax()[0]
            max_feature = grand_PCA.iloc[idx_max_feature][0]
            PCA_removed_features_list.append(max_feature)
            print ('Appending max PC: %s'%max_feature)
            # # remove min
            print('Dropping maximum principal component: %s...'%max_feature)
            finalDf = finalDf.drop(max_feature, axis=1)
            # do the RF again
            # save accuracy and number of features
            # repeat til 1 feature left
            # plot x number of features and y accuracy

            if i == 9:
                grand_reduced_df = pd.DataFrame({'accuracy': PCA_accuracy_list, 'n_components': PCA_n_features_list, 'kind': PCA_df_kind_list,'removed_component': PCA_removed_features_list})

                fig = plt.figure(figsize = (100,8))
                plt.title('Model Accuracy by Principal Component Reduction (Elbow Plot)')
                plt.plot(grand_reduced_df['n_components'], grand_reduced_df['accuracy'])
                plt.bar(grand_reduced_df['n_components'], grand_reduced_df['accuracy'])
                plt.xlabel('Number of Principal Components (PCs)')
                plt.ylabel('Model Accuracy')
                # plt.show()

                print("Saving feature ablation for Variable Importances for all clusters as figure...")
                plt.savefig(os.path.join(fig_dir, 'ablation', 'randomforest_PCA_reduction_all_%s.png'%number_of_features), bbox_inches='tight', dpi=300)
                plt.close()

                print ("Saving data as grand_reduced_df.csv")
                grand_reduced_df.to_csv(os.path.join(dir,'data_analysis_viz','grand_reduced_df.csv'), index=True, encoding='utf-8')


        ######### CLUSTERS ########
        ### clusters using grand mean, no additional imputing needed
        # feature_importances_all = grand_features
        # cluster_dict  = {'df_group_0':df_group_0,'df_group_1':df_group_1,'df_group_2':df_group_2,'df_group_3':df_group_3,'df_group_4':df_group_4}
        # for cluster, value in cluster_dict.items():
        #     cluster_number = 'cluster_' + cluster[-1:]
        #
        #     ## PCA
        #     data = value
        #
        #     features = list(data.columns)
        #     feature_number = len(features)
        #
        #     # Separating out the features
        #     x = data.loc[:, features].values
        #     # Separating out the target
        #     y = data.loc[:,[cluster_number]].values
        #     # Standardizing the features
        #     x = StandardScaler().fit_transform(x)
        #
        #     pca = PCA(n_components=10, random_state=random_state)
        #     print("Fitting PCA on %s..."%cluster_number)
        #     principalComponents = pca.fit_transform(x)
        #     pc_columns = []
        #     for j in range(len(principalComponents[1])):
        #         pc_string = 'principal component' + " " + str(j + 1)
        #         pc_columns.append(pc_string)
        #     # principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
        #     principalDf = pd.DataFrame(data = principalComponents, columns = pc_columns)
        #
        #     finalDf = pd.concat([principalDf, data[cluster_number]], axis=1)
        #
        #     print ("Saving data as principal_components.csv")
        #     finalDf.to_csv(os.path.join(dir,'data_analysis_viz','principal_components_%s.csv'%cluster_number), index=True, encoding='utf-8')
        #
        #     exp_var_cluster = pd.DataFrame({"explained_variance": pca.explained_variance_ratio_})
        #     print ("Saving data as pca_explained_variance_%s.csv"%cluster_number)
        #     exp_var_cluster.to_csv(os.path.join(dir,'data_analysis_viz','pca_explained_variance_%s.csv'%cluster_number), index=True, encoding='utf-8')
        #
        #
        #     for i in range(0,10):
        #         ###### BASELINE #######
        #         # Labels are the values we want to predict
        #         labels = np.array(finalDf[cluster_number])
        #         # Remove the labels from the features
        #         # axis 1 refers to the columns
        #         finalDf_array = finalDf.drop('%s'%cluster_number, axis = 1)
        #         number_of_features = len(finalDf.columns)
        #         # Saving feature names for later use
        #         feature_list = list(finalDf_array.columns)
        #         # Convert to numpy array
        #         finalDf_array = np.array(finalDf_array)
        #
        #         # Using Skicit-learn to split data into training and testing sets
        #         from sklearn.model_selection import train_test_split
        #         # Split the data into training and testing sets
        #         train_features, test_features, train_labels, test_labels = train_test_split(finalDf_array, labels, test_size = test_size, random_state = random_state)
        #
        #         # Instantiate model with 1000 decision trees
        #         # rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
        #         rf = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators = n_estimators, random_state = random_state, n_jobs = n_jobs))
        #         # Train the model on training data
        #         print ("Training Random Forest on %s PCs for %s..."%(i, cluster_number))
        #         t = time.time()
        #         rf.fit(train_features, train_labels);
        #         elapsed_rf = time.time() - t
        #         print("Random Forest elapsed time (in sec): %s" %elapsed_rf)
        #
        #         # cross validation
        #         print ("Cross-validation...")
        #         t = time.time()
        #         scores = cross_val_score(rf, train_features, train_labels, cv=cv)
        #         elapsed_rf = time.time() - t
        #         print("CV elapsed time (in sec): %s" %elapsed_rf)
        #         scores
        #         print('Mean CV Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))
        #
        #         # # Use the forest's predict method on the test data
        #         predictions = rf.predict(test_features)
        #
        #         # View the classification report for test data and predictions
        #         print("Accuracy on test data..")
        #         print(accuracy_score(test_labels, predictions))
        #         print("Confusion matrix on test data..")
        #         print(confusion_matrix(test_labels,predictions))
        #         print("Classification report on test data...")
        #         print(classification_report(test_labels, predictions))
        #
        #         # Get numerical feature importances
        #         importances = list(rf.steps[1][1].feature_importances_)
        #         # List of tuples with variable and importance
        #         feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
        #         # Sort the feature importances by most important first
        #         feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
        #         cluster_PCA = pd.DataFrame(feature_importances, columns = {'component': '0','importance_%s'%cluster_number: '1'})
        #         # cluster_PCA_50 = cluster_PCA['importance_%s'%cluster_number].quantile(0.5)
        #         # cluster_PCA_50 = cluster_PCA[cluster_PCA['importance_%s'%cluster_number] > cluster_PCA_50]
        #
        #         # Print out the feature and importances
        #         # [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];
        #         # print(cluster_PCA)
        #
        #         # Import matplotlib for plotting and use magic command for Jupyter Notebooks
        #         # Set the style
        #         # plt.style.use('fivethirtyeight')
        #         # fig = plt.figure(figsize = (100,8))
        #         #
        #         # # list of x locations for plotting
        #         # x_values = list(range(len(importances)))
        #         # # Make a bar chart
        #         # plt.bar(x_values, importances, orientation = 'vertical')
        #         # # Tick labels for x axis
        #         # plt.xticks(x_values, feature_list, rotation='vertical', fontsize = 8)
        #         # # Axis labels and title
        #         # plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances %s'%cluster_number);
        #         #
        #         # print("Saving Variable Importances for %s as figure..."%cluster_number)
        #         # plt.savefig(os.path.join(dir,'figs', 'ablation', 'randomforest_grandmean_%s_PCA_%s_components.png'%(cluster_number,i)), bbox_inches='tight', dpi=300)
        #         # plt.close()
        #
        #         # x = np.cumsum(pca.explained_variance_ratio_)
        #         # plt.plot(x)
        #         # plt.xlabel('Number of components')
        #         # plt.ylabel('Explained variance (%)')
        #         # plt.savefig(os.path.join(dir,'figs', '%s_PCA.png'%cluster_number), bbox_inches='tight', dpi=300)
        #         # plt.close()
        #
        #         # feature_importances_PCA_all = pd.merge(feature_importances_all, cluster_features, on = 'feature', how = 'outer')
        #
        #         if i == 0:
        #             ### correlation ###
        #             components_list = principalDf.columns
        #             df_pc = principalDf
        #             features_corr = features[:368]
        #             pc_corr_r_clusters = pd.DataFrame(components_list, columns={'PC'})
        #             pc_corr_p_clusters = pd.DataFrame(components_list, columns={'PC'})
        #             for feature in features_corr:
        #                 corr_list_r = []
        #                 corr_list_p = []
        #                 for pc in components_list:
        #                     pc_correlation_r = stats.pearsonr(data[feature], df_pc[pc])[0]
        #                     pc_correlation_p = stats.pearsonr(data[feature], df_pc[pc])[1]
        #                     corr_list_r.append(pc_correlation_r)
        #                     corr_list_p.append(pc_correlation_p)
        #                 single_feature_corr_r = pd.DataFrame(corr_list_r, columns={feature})
        #                 single_feature_corr_p = pd.DataFrame(corr_list_p, columns={feature})
        #                 pc_corr_r_clusters = pd.concat([pc_corr_r_clusters, single_feature_corr_r], axis=1)
        #                 pc_corr_p_clusters = pd.concat([pc_corr_p_clusters, single_feature_corr_p], axis=1)
        #
        #             print ("Saving data as cluster_corr_r.csv")
        #             pc_corr_r_clusters.to_csv(os.path.join(dir,'data_analysis_viz','cluster_corr_r_%s.csv'%cluster_number), index=True, encoding='utf-8')
        #
        #             print ("Saving data as cluster_corr_p.csv")
        #             pc_corr_p_clusters.to_csv(os.path.join(dir,'data_analysis_viz','cluster_corr_p_%s.csv'%cluster_number), index=True, encoding='utf-8')
        #
        #         print ('Appending grand values...')
        #         PCA_cluster_accuracy_list.append(accuracy_score(test_labels, predictions))
        #         PCA_cluster_n_features_list.append(len(cluster_PCA['component']))
        #         PCA_cluster_df_kind_list.append(cluster_number)
        #         # print(PCA_cluster_accuracy_list)
        #         # print(PCA_cluster_n_features_list)
        #         # print(PCA_cluster_df_kind_list)
        #         # find the min of the features
        #         idx_max_feature = cluster_PCA[['importance_%s'%cluster_number]].idxmax()[0]
        #         max_feature = cluster_PCA.iloc[idx_max_feature][0]
        #         PCA_cluster_removed_features_list.append(max_feature)
        #         print ('Appending max PC: %s'%max_feature)
        #         # # remove min
        #         print('Dropping maximum principal component: %s...'%max_feature)
        #         finalDf = finalDf.drop(max_feature, axis=1)
        #         # do the RF again
        #         # save accuracy and number of features
        #         # repeat til 1 feature left
        #         # plot x number of features and y accuracy
        #
        #         if i == 9:
        #             cluster_reduced_df = pd.DataFrame({'accuracy': PCA_cluster_accuracy_list, 'n_components': PCA_cluster_n_features_list, 'kind': PCA_cluster_df_kind_list, 'removed_feature': PCA_cluster_removed_features_list})
        #
        #             fig = plt.figure(figsize = (100,8))
        #             plt.title('Model Accuracy by Principal Component Reduction (Elbow Plot)')
        #             plt.plot(cluster_reduced_df['n_components'], cluster_reduced_df['accuracy'])
        #             plt.bar(cluster_reduced_df['n_components'], cluster_reduced_df['accuracy'])
        #             plt.xlabel('Number of Principal Components (PCs)')
        #             plt.ylabel('Model Accuracy')
        #             # plt.show()
        #
        #             print("Saving feature ablation for Variable Importances for all clusters as figure...")
        #             plt.savefig(os.path.join(fig_dir, 'ablation', 'randomforest_PCA_reduction_cluster_%s.png'%number_of_features), bbox_inches='tight', dpi=300)
        #             plt.close()
        #
        #             print ("Saving data as cluster_reduced_df.csv")
        #             cluster_reduced_df.to_csv(os.path.join(dir,'data_analysis_viz','cluster_reduced_df.csv'), index=True, encoding='utf-8')


elapsed_total_time = (time.time() - total_start_time)/60
print("Total elapsed time (in min): %s" %elapsed_total_time)

################################ GRAVEYARD ####################################

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


# # Create our imputer to replace missing values with the mean e.g.,
# # Imputing values shouldn't matter at this stage because the ratings have already been clustered so
# # the effect of a missing values of a fricatives within a given group doesn't change that WAVs membership to a group *in this data*
# # impute for final model
# imp = SimpleImputer(missing_values=np.nan, strategy='mean')
# imp = imp.fit(train_features)
# train_features_imp = imp.transform(train_features)
# imp = imp.fit(test_features)
# test_features_imp = imp.transform(test_features)



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


# # Calculate mean absolute percentage error (MAPE)
# mape = 100 * (errors / test_labels)
# # Calculate and display accuracy
# accuracy = 100 - np.mean(mape)
# print('Accuracy:', round(accuracy, 2), '%.')


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


#### original below, not in loop for raw and PCA ###
# df = df.drop(['Rating','Rating_z_score','kmeans_4_cluster','participant_gender_id','participant_sexual_orientation','participant_voice_id','participant_cis_trans',
#                 'participant_prox_social','participant_prox_affiliation', 'participant_prox_media', 'participant_race','participant_race_hispanic','eng_primary_early','eng_primary_current',
#                 'participant_other_langs','participant_race_free_response','participant_gender_pso_free_response', 'participant_age', 'deaf_hoh', 'Participant',
#                 'survey_experience','survey_feedback','Condition','WAV','color','spectral_S_start','spectral_Z_start','spectral_F_start','spectral_V_start','spectral_JH_start','spectral_SH_start'], axis=1)
#
# number_of_features = len(df.columns)
#
# ## impute for the cross validation, random forest needs values in each, below we are creating one large df with imputed column means from teh entire set, and then 5 cluster groups with the gran imputed means
# imp = SimpleImputer(missing_values=np.nan, strategy='mean')
# imp = imp.fit(df)
# df_imp = imp.transform(df) # this is the big df
# df_imp = pd.DataFrame(df_imp, columns = df.columns)
#
# ## baseline starve the tree ##
# df_imp_baseline = df_imp[['kmeans_5_cluster','rando_baseline_z_score']]
#
# # make cluster dfs with imputed grand means
# df_cluster_dummies = pd.get_dummies(df_imp['kmeans_5_cluster'], prefix='cluster')
# df_grand = pd.concat([df_imp, df_cluster_dummies], axis=1)
# df_sans = df_grand.drop('kmeans_5_cluster', axis=1)
# df_sans = df_sans.rename(columns={'cluster_0.0':'cluster_0', 'cluster_1.0':'cluster_1', 'cluster_2.0':'cluster_2', 'cluster_3.0':'cluster_3', 'cluster_4.0':'cluster_4'})
# df_group_0 = df_sans.drop(['cluster_1','cluster_2','cluster_3','cluster_4'], axis=1)
# df_group_1 = df_sans.drop(['cluster_0','cluster_2','cluster_3','cluster_4'], axis=1)
# df_group_2 = df_sans.drop(['cluster_0','cluster_1','cluster_3','cluster_4'], axis=1)
# df_group_3 = df_sans.drop(['cluster_0','cluster_1','cluster_2','cluster_4'], axis=1)
# df_group_4 = df_sans.drop(['cluster_0','cluster_1','cluster_2','cluster_3'], axis=1)
#
# ## PCA
# data = df_imp
#
# features = list(data.columns)
# feature_number = len(features)
#
# # Separating out the features
# x = data.loc[:, features].values
# # Separating out the target
# y = data.loc[:,['kmeans_5_cluster']].values
# # Standardizing the features
# x = StandardScaler().fit_transform(x)
#
# pca = PCA(n_components=feature_number, random_state=42)
# principalComponents = pca.fit_transform(x)
# pc_columns = []
# for i in range(len(features)):
#     pc_string = 'principal component' + " " + str(i + 1)
#     pc_columns.append(pc_string)
# # principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
# principalDf = pd.DataFrame(data = principalComponents, columns = pc_columns)
#
# finalDf = pd.concat([principalDf, data['kmeans_5_cluster']], axis=1)
#
#
# ###### BASELINE ####### Big ol' df with imputed means on everyone
# # Labels are the values we want to predict
# labels = np.array(finalDf['kmeans_5_cluster'])
# # Remove the labels from the features
# # axis 1 refers to the columns
# finalDf = finalDf.drop('kmeans_5_cluster', axis = 1)
# # Saving feature names for later use
# feature_list = list(finalDf.columns)
# # Convert to numpy array
# finalDf = np.array(finalDf)
#
# # Using Skicit-learn to split data into training and testing sets
# from sklearn.model_selection import train_test_split
# # Split the data into training and testing sets
# train_features, test_features, train_labels, test_labels = train_test_split(finalDf, labels, test_size = 0.75, random_state = 42)
#
# # # # The baseline predictions are the historical averages
# # baseline_preds = test_features[:, feature_list.index('rando')]
# # # # # Baseline errors, and display average baseline error
# # baseline_errors = abs(baseline_preds - test_labels)
# # print('Average baseline error: ', round(np.mean(baseline_errors), 3))
#
# # # Create our imputer to replace missing values with the mean e.g.,
# # # Imputing values shouldn't matter at this stage because the ratings have already been clustered so
# # # the effect of a missing values of a fricatives within a given group doesn't change that WAVs membership to a group *in this data*
# # # impute for final model
# # imp = SimpleImputer(missing_values=np.nan, strategy='mean')
# # imp = imp.fit(train_features)
# # train_features_imp = imp.transform(train_features)
# # imp = imp.fit(test_features)
# # test_features_imp = imp.transform(test_features)
#
# # Import the model we are using
# # Instantiate model with 1000 decision trees
# rf = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators = 5000, random_state = 42, n_jobs = -1, max_features = 10, max_depth = 5))
# # Train the model on training data
# print ("Training Random Forest on %s features for all clusters..."%number_of_features)
# t = time.time()
# rf.fit(train_features, train_labels);
# elapsed_rf = time.time() - t
# print("Random Forest elapsed time (in sec): %s" %elapsed_rf)
# print("Accuracy on train data: {:.2f}".format(rf.score(train_features, train_labels)))
#
# # cross validation
# print ("Cross-validation...")
# # rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
# t = time.time()
# scores = cross_val_score(rf, train_features, train_labels, cv=10)
# elapsed_rf = time.time() - t
# print("CV elapsed time (in sec): %s" %elapsed_rf)
# scores
# print('Mean CV Accuracy: %.3f (%.3f)' % (np.mean(scores), np.std(scores)))
#
# # # accuracy score
# # predictions=rf.predict(test_features)
# # print("Accuracy:", accuracy_score(test_features, predictions))
#
# # # Get and reshape confusion matrix data
# # matrix = confusion_matrix(test_labels, predictions)
# # matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
# #
# # # Build the plot
# # plt.figure(figsize=(16,7))
# # sns.set(font_scale=1.4)
# # sns.heatmap(matrix, annot=True, annot_kws={'size':10},
# #             cmap=plt.cm.Greens, linewidths=0.2)
# #
# # # Add labels to the plot
# # class_names = feature_list
# # tick_marks = np.arange(len(class_names))
# # tick_marks2 = tick_marks + 0.5
# # plt.xticks(tick_marks, class_names, rotation=25)
# # plt.yticks(tick_marks2, class_names, rotation=0)
# # plt.xlabel('Predicted label')
# # plt.ylabel('True label')
# # plt.title('Confusion Matrix for Random Forest Model')
# # plt.show()
#
# # # Use the forest's predict method on the test data
# predictions = rf.predict(test_features)
# # # Calculate the absolute errors
# # errors = abs(predictions - test_labels)
# # # # Print out the mean absolute error (mae)
# # print('Mean Absolute Error:', round(np.mean(errors), 3), 'degrees.')
# # print("Accuracy on test data: {:.2f}".format(rf.score(test_features, test_labels)))
#
# # View the classification report for test data and predictions
# print(confusion_matrix(test_labels,predictions))
# print(accuracy_score(test_labels, predictions))
# print(classification_report(test_labels, predictions))
#
# # # Calculate mean absolute percentage error (MAPE)
# # mape = 100 * (errors / test_labels)
# # # Calculate and display accuracy
# # accuracy = 100 - np.mean(mape)
# # print('Accuracy:', round(accuracy, 2), '%.')
#
# # Get numerical feature importances
# importances = list(rf.steps[1][1].feature_importances_)
# # importances = list(result.importances_mean)
# # List of tuples with variable and importance
# feature_importances = [(feature, round(importance, 5)) for feature, importance in zip(feature_list, importances)]
# # Sort the feature importances by most important first
# feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# grand_features = pd.DataFrame(feature_importances, columns = {'feature': '0','importance': '1'})
# grand_features_50 = grand_features['importance'].quantile(0.5)
# grand_features = grand_features[grand_features['importance'] > grand_features_50]
# # Print out the feature and importances
# [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];
# print(grand_features)
#
# ## make variables to save importances
#
# # Import matplotlib for plotting and use magic command for Jupyter Notebooks
# # Set the style
# # plt.style.use('fivethirtyeight')
# fig = plt.figure(figsize = (100,8))
#
# # grand_features_list = grand_features['feature'].to_list()
#
# # list of x locations for plotting
# x_values = list(range(len(importances)))
# # Make a bar chart
# plt.bar(x_values, importances, orientation = 'vertical')
# # Tick labels for x axis
# plt.xticks(x_values, feature_list, rotation='vertical', fontsize = 8)
# # Axis labels and title
# plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances');
#
# plt.savefig(os.path.join(dir,'figs', 'randomforest_grandmean_PCA.png'), bbox_inches='tight', dpi=300)
# plt.close()
#
#
# # start_time = time.time()
# # result = permutation_importance(
# #     rf, test_features, test_labels, n_repeats=10, random_state=42, n_jobs=5,
# #     scoring = 'f1'
# # )
# # elapsed_time = time.time() - start_time
# # print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")
# #
# # forest_importances = pd.Series(result.importances_mean, index=feature_list)
# #
# # fig, ax = plt.subplots()
# # forest_importances.plt.bar(yerr=result.importances_std, ax=ax)
# # ax.set_title("Feature importances using permutation on full model")
# # ax.set_ylabel("Mean accuracy decrease")
# # fig.tight_layout()
# # plt.show()
#
#
#
# ######### CLUSTERS ########
# ### clusters using grand mean, no additional imputing needed
#
# cluster_dict  = {'df_group_0':df_group_0,'df_group_1':df_group_1,'df_group_2':df_group_2,'df_group_3':df_group_3,'df_group_4':df_group_4}
# for cluster, value in cluster_dict.items():
#     cluster_number = 'cluster_' + cluster[-1:]
#
#     ## PCA
#     data = value
#
#     features = list(data.columns)
#     feature_number = len(features)
#
#     # Separating out the features
#     x = data.loc[:, features].values
#     # Separating out the target
#     y = data.loc[:,[cluster_number]].values
#     # Standardizing the features
#     x = StandardScaler().fit_transform(x)
#
#     pca = PCA(n_components=feature_number, random_state=42)
#     print("Fitting PCA on %s..."%cluster_number)
#     principalComponents = pca.fit_transform(x)
#     pc_columns = []
#     for i in range(len(features)):
#         pc_string = 'principal component' + " " + str(i + 1)
#         pc_columns.append(pc_string)
#     # principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
#     principalDf = pd.DataFrame(data = principalComponents, columns = pc_columns)
#
#     finalDf = pd.concat([principalDf, data[cluster_number]], axis=1)
#
#     ###### BASELINE #######
#     # Labels are the values we want to predict
#     labels = np.array(finalDf[cluster_number])
#     # Remove the labels from the features
#     # axis 1 refers to the columns
#     finalDf = finalDf.drop('%s'%cluster_number, axis = 1)
#     # Saving feature names for later use
#     feature_list = list(finalDf.columns)
#     # Convert to numpy array
#     finalDf = np.array(finalDf)
#
#     # Using Skicit-learn to split data into training and testing sets
#     from sklearn.model_selection import train_test_split
#     # Split the data into training and testing sets
#     train_features, test_features, train_labels, test_labels = train_test_split(finalDf, labels, test_size = 0.75, random_state = 42)
#
#     # # # The baseline predictions are the historical averages
#     # baseline_preds = test_features[:, feature_list.index('rando')]
#     # # # # Baseline errors, and display average baseline error
#     # baseline_errors = abs(baseline_preds - test_labels)
#     # print('Average baseline error: ', round(np.mean(baseline_errors), 3))
#
#     # Instantiate model with 1000 decision trees
#     # rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
#     rf = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators = 5000, random_state = 42, n_jobs = -1, max_features = 10, max_depth = 5))
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
#     cluster_features_50 = cluster_features[cluster_features['importance'] > cluster_features_50]
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
#     plt.savefig(os.path.join(dir,'figs', 'randomforest_grandmean_%s_PCA.png'%cluster_number), bbox_inches='tight', dpi=300)
#     plt.close()
#
#     x = np.cumsum(pca.explained_variance_ratio_)
#     plt.plot(x)
#     plt.xlabel('Number of components')
#     plt.ylabel('Explained variance (%)')
#     plt.savefig(os.path.join(dir,'figs', '%s_PCA.png'%cluster_number), bbox_inches='tight', dpi=300)
#     plt.close()
#
#     components_list = principalDf.columns
#     group_df = value.drop(cluster_number, axis=1)
#     finalDf = pd.concat([principalDf, data[cluster_number]], axis=1)
#     finalDf = finalDf.drop(cluster_number, axis=1)
#     features = features[:368]
#     for feature in features:
#         corr_list = []
#         for pc in components_list:
#             pc_correlation = data[feature].corr(finalDf[pc])
#             corr_list.append(pc2_correlation)
#             single_feature_corr = pd.DataFrame(corr_list, columns={feature})
#
#     feature_correlations = pd.DataFrame({'feature': features, 'correlation_PC2': corr_list})



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
# data = df_imp
#
# features = list(data.columns)
# features = features[:366]
#
# # Separating out the features
# x = data.loc[:, features].values
# # Separating out the target
# y = data.loc[:,['kmeans_5_cluster']].values
# # Standardizing the features
# x = StandardScaler().fit_transform(x)
#
# pca = PCA(n_components=366, random_state=42)
# principalComponents = pca.fit_transform(x)
# pc_columns = []
# for i in range(len(features)):
#     pc_string = 'principal component' + " " + str(i + 1)
#     pc_columns.append(pc_string)
# # principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
# principalDf = pd.DataFrame(data = principalComponents, columns = pc_columns)
#
# finalDf = pd.concat([principalDf, data['kmeans_5_cluster']], axis=1)
#
# fig = plt.figure(figsize = (8,8))
# ax = fig.add_subplot(1,1,1)
# ax.set_xlabel('Principal Component 2', fontsize = 15)
# ax.set_ylabel('Principal Component 39', fontsize = 15)
# ax.set_title('2 component PCA', fontsize = 20)
# targets = [0,1,2,3,4]
# colors = ['r', 'g', 'b', 'y', 'k']
# for target, color in zip(targets,colors):
#     indicesToKeep = finalDf['kmeans_5_cluster'] == target
#     ax.scatter(finalDf.loc[indicesToKeep, 'principal component 2']
#                , finalDf.loc[indicesToKeep, 'principal component 39']
#                , c = color
#                , s = 50)
# ax.legend(targets)
# ax.grid()
# plt.show()
#
# x = np.cumsum(pca.explained_variance_ratio_)
# plt.plot(x)
# plt.xlabel('Number of components')
# plt.ylabel('Explained variance (%)')
# plt.savefig(os.path.join(dir,'figs', 'PCA_5_cluster.png'), bbox_inches='tight', dpi=300)
#
# corr_list = []
# for feature in features:
#     pc2_correlation = data[feature].corr(finalDf['principal component 2'])
#     corr_list.append(pc2_correlation)
#
# feature_correlations = pd.DataFrame({'feature': features, 'correlation_PC2': corr_list})
#
# plt.plot(feature_correlations['correlation_PC2'])
# plt.xlabel('Feature')
# plt.ylabel('Correlation')
# plt.show()
