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
import statsmodels.formula.api as smf
import seaborn as sns
from sklearn.impute import SimpleImputer

# np.set_printoptions(threshold=sys.maxsize)

# set up directory and read in csv
dir = '/Users/bcl/GitHub/queer_speech'
os.chdir(dir)
corr_fname = os.path.join(dir, 'data_analysis_viz', 'grand_corr_r.csv')
grand_ablated_df_fname = os.path.join(dir, 'data_analysis_viz', 'grand_ablated_df.csv')
data_fname = os.path.join(dir, 'data_analysis_viz', 'queer_data.csv')

corr_df = pd.read_csv(corr_fname)
corr_df = corr_df.drop('Unnamed: 0', axis=1) # might need to change column name when it imports in, there's a random addition at the beginning
grand_ablated_df = pd.read_csv(grand_ablated_df_fname)
grand_ablated_df = grand_ablated_df.drop('Unnamed: 0', axis=1) # might need to change column name when it imports in, there's a random addition at the beginning
data = pd.read_csv(data_fname)
data = data.drop('Unnamed: 0', axis=1) # might need to change column name when it imports in, there's a random addition at the beginning

# with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
#     print(finalDf)

## correlation thresholding
ablated_reduced = grand_ablated_df[grand_ablated_df['accuracy'] == 1]
best_features_list = list(ablated_reduced['removed_feature'])
corr_best = corr_df[best_features_list]


# data = data.dropna()
# fill creak nans with 0
# data['percent_creak'] = data['percent_creak'].fillna(0)

# gender_id = data[data['Condition']=='gender_id']
# sexual_orientation = data[data['Condition']=='sexual_orientation']
# voice_id = data[data['Condition']=='voice_id']

df = data
df_temp = df[['Rating','Rating_z_score','kmeans_4_cluster','kmeans_3_cluster','3_rando_classes','4_rando_classes','participant_gender_id','participant_sexual_orientation','participant_voice_id','participant_cis_trans',
                'participant_prox_social','participant_prox_affiliation', 'participant_prox_media', 'participant_race','participant_race_hispanic','eng_primary_early','eng_primary_current',
                'participant_other_langs','participant_race_free_response','participant_gender_pso_free_response', 'participant_age', 'deaf_hoh', 'Participant',
                'survey_experience','survey_feedback','Condition','WAV','color_3_cluster','color_4_cluster','color_5_cluster','spectral_S_start','spectral_Z_start','spectral_F_start','spectral_V_start','spectral_JH_start','spectral_SH_start']]
# df['rando_baseline_z_score'] = stats.zscore(np.round(np.random.uniform(1.0,8.0,len(df)), 1), axis=0)
df = df.drop(['Rating','Rating_z_score','kmeans_4_cluster','kmeans_3_cluster','3_rando_classes','4_rando_classes','participant_gender_id','participant_sexual_orientation','participant_voice_id','participant_cis_trans',
                'participant_prox_social','participant_prox_affiliation', 'participant_prox_media', 'participant_race','participant_race_hispanic','eng_primary_early','eng_primary_current',
                'participant_other_langs','participant_race_free_response','participant_gender_pso_free_response', 'participant_age', 'deaf_hoh', 'Participant',
                'survey_experience','survey_feedback','Condition','WAV','color_3_cluster','color_4_cluster','color_5_cluster','spectral_S_start','spectral_Z_start','spectral_F_start','spectral_V_start','spectral_JH_start','spectral_SH_start',
                'S_avg_dur','Z_avg_dur','F_avg_dur','V_avg_dur','JH_avg_dur','SH_avg_dur'], axis=1)

## impute cluster means for each cluster in the loop
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp = imp.fit(df)
df_imp = imp.transform(df)
df_imp = pd.DataFrame(df_imp, columns = df.columns)
df_imp = pd.concat([df_imp, df_temp], axis=1)


# make cluster dfs with imputed grand means
df_cluster_dummies = pd.get_dummies(df_imp['kmeans_3_cluster'], prefix='cluster')
df_grand = pd.concat([df_imp, df_cluster_dummies], axis=1)
df_sans = df_grand.drop('kmeans_3_cluster', axis=1)
df_sans = df_sans.rename(columns={'cluster_0.0':'cluster_0', 'cluster_1.0':'cluster_1', 'cluster_2.0':'cluster_2'})
df_group_0 = df_sans.drop(['cluster_1','cluster_2'], axis=1)
df_group_1 = df_sans.drop(['cluster_0','cluster_2'], axis=1)
df_group_2 = df_sans.drop(['cluster_0','cluster_1'], axis=1)
# df_group_3 = df_sans.drop(['cluster_0','cluster_1','cluster_2','cluster_4'], axis=1)
# df_group_4 = df_sans.drop(['cluster_0','cluster_1','cluster_2','cluster_3'], axis=1)

gender_id = df_imp[df_imp['Condition']=='gender_id']
sexual_orientation = df_imp[df_imp['Condition']=='sexual_orientation']
voice_id = df_imp[df_imp['Condition']=='voice_id']
number_participants = df_imp['Participant'].nunique()

### linear regression ###

## grand ##
# feature_list = ['F0_mean','IH_sF1_mean','IH_sF2_mean','IH_sF3_mean','IH_sF4_mean','IH_sF1_mean_dist','IH_sF2_mean_dist','IH_sF3_mean_dist','IH_sF4_mean_dist',
#                 'AY_sF1_mean_first', 'AY_sF2_mean_first', 'AY_sF3_mean_first', 'AY_sF4_mean_first', 'AY_sF1_mean_third', 'AY_sF2_mean_third', 'AY_sF3_mean_third', 'AY_sF4_mean_third',
#                 'vowel_avg_dur', 'percent_creak', 'spectral_S_duration','spectral_S_cog','spectral_S_skew']

feature_list = ['F0_mean',
 'F0_std',
 'F0_range',
 'F0_90',
 'F0_10',
 'AA_avg_dur',
 'AE_avg_dur',
 'AH_avg_dur',
 'AO_avg_dur',
 'AR_avg_dur',
 'AW_avg_dur',
 'AX_avg_dur',
 'AY_avg_dur',
 'B_avg_dur',
 'CH_avg_dur',
 'D_avg_dur',
 'DH_avg_dur',
 'EH_avg_dur',
 'EL_avg_dur',
 'ER_avg_dur',
 'EY_avg_dur',
 'G_avg_dur',
 'H_avg_dur',
 'HH_avg_dur',
 'IH_avg_dur',
 'IY_avg_dur',
 'K_avg_dur',
 'L_avg_dur',
 'M_avg_dur',
 'N_avg_dur',
 'NG_avg_dur',
 'OW_avg_dur',
 'OY_avg_dur',
 'P_avg_dur',
 'R_avg_dur',
 'T_avg_dur',
 'TH_avg_dur',
 'UH_avg_dur',
 'UW_avg_dur',
 'W_avg_dur',
 'Y_avg_dur',
 'ZH_avg_dur',
 'vowel_avg_dur',
 'AA_sF1_mean',
 'AE_sF1_mean',
 'AH_sF1_mean',
 'AO_sF1_mean',
 'AX_sF1_mean',
 'EH_sF1_mean',
 'IH_sF1_mean',
 'IY_sF1_mean',
 'UH_sF1_mean',
 'UW_sF1_mean',
 'AA_sF1_min',
 'AE_sF1_min',
 'AH_sF1_min',
 'AO_sF1_min',
 'AX_sF1_min',
 'EH_sF1_min',
 'IH_sF1_min',
 'IY_sF1_min',
 'UH_sF1_min',
 'UW_sF1_min',
 'AA_sF1_max',
 'AE_sF1_max',
 'AH_sF1_max',
 'AO_sF1_max',
 'AX_sF1_max',
 'EH_sF1_max',
 'IH_sF1_max',
 'IY_sF1_max',
 'UH_sF1_max',
 'UW_sF1_max',
 'AA_sF2_mean',
 'AE_sF2_mean',
 'AH_sF2_mean',
 'AO_sF2_mean',
 'AX_sF2_mean',
 'EH_sF2_mean',
 'IH_sF2_mean',
 'IY_sF2_mean',
 'UH_sF2_mean',
 'UW_sF2_mean',
 'AA_sF2_min',
 'AE_sF2_min',
 'AH_sF2_min',
 'AO_sF2_min',
 'AX_sF2_min',
 'EH_sF2_min',
 'IH_sF2_min',
 'IY_sF2_min',
 'UH_sF2_min',
 'UW_sF2_min',
 'AA_sF2_max',
 'AE_sF2_max',
 'AH_sF2_max',
 'AO_sF2_max',
 'AX_sF2_max',
 'EH_sF2_max',
 'IH_sF2_max',
 'IY_sF2_max',
 'UH_sF2_max',
 'UW_sF2_max',
 'AA_sF3_mean',
 'AE_sF3_mean',
 'AH_sF3_mean',
 'AO_sF3_mean',
 'AX_sF3_mean',
 'EH_sF3_mean',
 'IH_sF3_mean',
 'IY_sF3_mean',
 'UH_sF3_mean',
 'UW_sF3_mean',
 'AA_sF3_min',
 'AE_sF3_min',
 'AH_sF3_min',
 'AO_sF3_min',
 'AX_sF3_min',
 'EH_sF3_min',
 'IH_sF3_min',
 'IY_sF3_min',
 'UH_sF3_min',
 'UW_sF3_min',
 'AA_sF3_max',
 'AE_sF3_max',
 'AH_sF3_max',
 'AO_sF3_max',
 'AX_sF3_max',
 'EH_sF3_max',
 'IH_sF3_max',
 'IY_sF3_max',
 'UH_sF3_max',
 'UW_sF3_max',
 'AA_sF4_mean',
 'AE_sF4_mean',
 'AH_sF4_mean',
 'AO_sF4_mean',
 'AX_sF4_mean',
 'EH_sF4_mean',
 'IH_sF4_mean',
 'IY_sF4_mean',
 'UH_sF4_mean',
 'UW_sF4_mean',
 'AA_sF4_min',
 'AE_sF4_min',
 'AH_sF4_min',
 'AO_sF4_min',
 'AX_sF4_min',
 'EH_sF4_min',
 'IH_sF4_min',
 'IY_sF4_min',
 'UH_sF4_min',
 'UW_sF4_min',
 'AA_sF4_max',
 'AE_sF4_max',
 'AH_sF4_max',
 'AO_sF4_max',
 'AX_sF4_max',
 'EH_sF4_max',
 'IH_sF4_max',
 'IY_sF4_max',
 'UH_sF4_max',
 'UW_sF4_max',
 'AW_sF1_mean_first',
 'AW_sF1_min_first',
 'AW_sF1_max_first',
 'AW_sF1_mean_third',
 'AW_sF1_min_third',
 'AW_sF1_max_third',
 'AW_sF2_mean_first',
 'AW_sF2_min_first',
 'AW_sF2_max_first',
 'AW_sF2_mean_third',
 'AW_sF2_min_third',
 'AW_sF2_max_third',
 'AW_sF3_mean_first',
 'AW_sF3_min_first',
 'AW_sF3_max_first',
 'AW_sF3_mean_third',
 'AW_sF3_min_third',
 'AW_sF3_max_third',
 'AW_sF4_mean_first',
 'AW_sF4_min_first',
 'AW_sF4_max_first',
 'AW_sF4_mean_third',
 'AW_sF4_min_third',
 'AW_sF4_max_third',
 'AY_sF1_mean_first',
 'AY_sF1_min_first',
 'AY_sF1_max_first',
 'AY_sF1_mean_third',
 'AY_sF1_min_third',
 'AY_sF1_max_third',
 'AY_sF2_mean_first',
 'AY_sF2_min_first',
 'AY_sF2_max_first',
 'AY_sF2_mean_third',
 'AY_sF2_min_third',
 'AY_sF2_max_third',
 'AY_sF3_mean_first',
 'AY_sF3_min_first',
 'AY_sF3_max_first',
 'AY_sF3_mean_third',
 'AY_sF3_min_third',
 'AY_sF3_max_third',
 'AY_sF4_mean_first',
 'AY_sF4_min_first',
 'AY_sF4_max_first',
 'AY_sF4_mean_third',
 'AY_sF4_min_third',
 'AY_sF4_max_third',
 'EY_sF1_mean_first',
 'EY_sF1_min_first',
 'EY_sF1_max_first',
 'EY_sF1_mean_third',
 'EY_sF1_min_third',
 'EY_sF1_max_third',
 'EY_sF2_mean_first',
 'EY_sF2_min_first',
 'EY_sF2_max_first',
 'EY_sF2_mean_third',
 'EY_sF2_min_third',
 'EY_sF2_max_third',
 'EY_sF3_mean_first',
 'EY_sF3_min_first',
 'EY_sF3_max_first',
 'EY_sF3_mean_third',
 'EY_sF3_min_third',
 'EY_sF3_max_third',
 'EY_sF4_mean_first',
 'EY_sF4_min_first',
 'EY_sF4_max_first',
 'EY_sF4_mean_third',
 'EY_sF4_min_third',
 'EY_sF4_max_third',
 'OW_sF1_mean_first',
 'OW_sF1_min_first',
 'OW_sF1_max_first',
 'OW_sF1_mean_third',
 'OW_sF1_min_third',
 'OW_sF1_max_third',
 'OW_sF2_mean_first',
 'OW_sF2_min_first',
 'OW_sF2_max_first',
 'OW_sF2_mean_third',
 'OW_sF2_min_third',
 'OW_sF2_max_third',
 'OW_sF3_mean_first',
 'OW_sF3_min_first',
 'OW_sF3_max_first',
 'OW_sF3_mean_third',
 'OW_sF3_min_third',
 'OW_sF3_max_third',
 'OW_sF4_mean_first',
 'OW_sF4_min_first',
 'OW_sF4_max_first',
 'OW_sF4_mean_third',
 'OW_sF4_min_third',
 'OW_sF4_max_third',
 'OY_sF1_mean_first',
 'OY_sF1_min_first',
 'OY_sF1_max_first',
 'OY_sF1_mean_third',
 'OY_sF1_min_third',
 'OY_sF1_max_third',
 'OY_sF2_mean_first',
 'OY_sF2_min_first',
 'OY_sF2_max_first',
 'OY_sF2_mean_third',
 'OY_sF2_min_third',
 'OY_sF2_max_third',
 'OY_sF3_mean_first',
 'OY_sF3_min_first',
 'OY_sF3_max_first',
 'OY_sF3_mean_third',
 'OY_sF3_min_third',
 'OY_sF3_max_third',
 'OY_sF4_mean_first',
 'OY_sF4_min_first',
 'OY_sF4_max_first',
 'OY_sF4_mean_third',
 'OY_sF4_min_third',
 'OY_sF4_max_third',
 'AA_sF1_mean_dist',
 'AA_sF2_mean_dist',
 'AA_sF3_mean_dist',
 'AA_sF4_mean_dist',
 'AE_sF1_mean_dist',
 'AE_sF2_mean_dist',
 'AE_sF3_mean_dist',
 'AE_sF4_mean_dist',
 'AH_sF1_mean_dist',
 'AH_sF2_mean_dist',
 'AH_sF3_mean_dist',
 'AH_sF4_mean_dist',
 'AO_sF1_mean_dist',
 'AO_sF2_mean_dist',
 'AO_sF3_mean_dist',
 'AO_sF4_mean_dist',
 'AX_sF1_mean_dist',
 'AX_sF2_mean_dist',
 'AX_sF3_mean_dist',
 'AX_sF4_mean_dist',
 'EH_sF1_mean_dist',
 'EH_sF2_mean_dist',
 'EH_sF3_mean_dist',
 'EH_sF4_mean_dist',
 'IH_sF1_mean_dist',
 'IH_sF2_mean_dist',
 'IH_sF3_mean_dist',
 'IH_sF4_mean_dist',
 'IY_sF1_mean_dist',
 'IY_sF2_mean_dist',
 'IY_sF3_mean_dist',
 'IY_sF4_mean_dist',
 'UH_sF1_mean_dist',
 'UH_sF2_mean_dist',
 'UH_sF3_mean_dist',
 'UH_sF4_mean_dist',
 'UW_sF1_mean_dist',
 'UW_sF2_mean_dist',
 'UW_sF3_mean_dist',
 'UW_sF4_mean_dist',
 'spectral_S_duration',
 'spectral_S_intensity',
 'spectral_S_cog',
 'spectral_S_sdev',
 'spectral_S_skew',
 'spectral_S_kurt',
 'spectral_Z_duration',
 'spectral_Z_intensity',
 'spectral_Z_cog',
 'spectral_Z_sdev',
 'spectral_Z_skew',
 'spectral_Z_kurt',
 'spectral_F_duration',
 'spectral_F_intensity',
 'spectral_F_cog',
 'spectral_F_sdev',
 'spectral_F_skew',
 'spectral_F_kurt',
 'spectral_V_duration',
 'spectral_V_intensity',
 'spectral_V_cog',
 'spectral_V_sdev',
 'spectral_V_skew',
 'spectral_V_kurt',
 'spectral_SH_duration',
 'spectral_SH_intensity',
 'spectral_SH_cog',
 'spectral_SH_sdev',
 'spectral_SH_skew',
 'spectral_SH_kurt',
 'spectral_JH_duration',
 'spectral_JH_intensity',
 'spectral_JH_cog',
 'spectral_JH_sdev',
 'spectral_JH_skew',
 'spectral_JH_kurt',
 'spectral_TH_start',
 'spectral_TH_duration',
 'spectral_TH_intensity',
 'spectral_TH_cog',
 'spectral_TH_sdev',
 'spectral_TH_skew',
 'spectral_TH_kurt',
 'spectral_DH_start',
 'spectral_DH_duration',
 'spectral_DH_intensity',
 'spectral_DH_cog',
 'spectral_DH_sdev',
 'spectral_DH_skew',
 'spectral_DH_kurt',
 'percent_creak',
 'spectral_S_start',
 'spectral_Z_start',
 'spectral_F_start',
 'spectral_V_start',
 'spectral_JH_start',
 'spectral_SH_start']

model_list=[]
condition_dict = {'gender_id': gender_id, 'sexual_orientation': sexual_orientation, 'voice_id':voice_id}

# for feature in feature_list: # loop for making a list of the vowel spectral features for each vowel--matches columns in spreadsheet
#     # concatenate strings with '_'
#     model_string = feature + " ~ " + "C(kmeans_5_cluster)"
#     # append to list
#     model_list.append(model_string)
#
# for model in model_list:
#             mod = smf.ols(formula=model, data=df_imp)
#             res = mod.fit()
#             print(res.summary())

for feature in feature_list: # loop for making a list of the vowel spectral features for each vowel--matches columns in spreadsheet
    # concatenate strings with '_'
    model_string = "Rating_z_score" + " ~ " + feature
    # append to list
    model_list.append(model_string)

for condition, df in condition_dict.items():
    coef_list = []
    t_list = []
    p_list = []
    condition_list = []
    mdl_list = []
    feat_list = []
    for model in model_list:
        print(condition)
        mod = smf.ols(formula=model, data=df)
        res = mod.fit()
        coef_list.append(res.params[1])
        t_list.append(res.tvalues[1])
        p_list.append(res.pvalues[1])
        condition_list.append(condition)
        mdl_list.append(model)
        feat_list.append(model[17:])
        print(res.summary())
    effect_size = pd.DataFrame({'coef':coef_list, 'p': p_list, 't':t_list, 'condition': condition_list, 'model':mdl_list, 'feature':feat_list})
    effect_size.to_csv(os.path.join(dir,'data_analysis_viz','effect_sizes_all_%s.csv'%condition), index=True, encoding='utf-8')


# super_model = 'Rating_z_score ~ F0_mean + IH_sF1_mean+IH_sF2_mean+IH_sF3_mean+IH_sF4_mean+IH_sF1_mean_dist+IH_sF2_mean_dist+IH_sF3_mean_dist+IH_sF4_mean_dist+AY_sF1_mean_first+ AY_sF2_mean_first+ AY_sF3_mean_first+ AY_sF4_mean_first+ AY_sF1_mean_third+ AY_sF2_mean_third+ AY_sF3_mean_third+ AY_sF4_mean_third+vowel_avg_dur+ percent_creak+ spectral_S_duration+spectral_S_cog + spectral_S_skew'
#
# for condition, df in condition_dict.items():
#     print(condition)
#     mod = smf.ols(formula=super_model, data=df)
#     res = mod.fit()
#     print(res.summary())



# res.params
# res.bse
# res.tvalues
# res.pvalues

# group_dict = {'sm': df_group_2, 'qm': df_group_0, 'qn': df_group_4, 'qw': df_group_3, 'sw': df_group_1}
# cluster_list = ['cluster_0','cluster_1','cluster_2','cluster_3','cluster_4']
# ## cluster ##
# for feature in feature_list: # loop for making a list of the vowel spectral features for each vowel--matches columns in spreadsheet
#     for group, df in group_dict.items():
#         for cluster in cluster_list:
#             model_list=[]
#             if cluster in list(df.columns):
#                 model_string = feature + " ~ " + "C(%s)"%cluster
#                 # append to list
#                 model_list.append(model_string)
#             for model in model_list:
#                mod = smf.ols(formula=model, data=df)
#                res = mod.fit()
#                print(res.summary())
#
#
# ## PCs ##
# pca_fname = os.path.join(dir, 'data_analysis_viz','principal_components.csv')
# pca = pd.read_csv(pca_fname)
#
# pc_list = ['Principal Component 1', 'Principal Component 2',
#        'Principal Component 3', 'Principal Component 4',
#        'Principal Component 5', 'Principal Component 6',
#        'Principal Component 7', 'Principal Component 8',
#        'Principal Component 9', 'Principal Component 10']

# X = df_imp[['kmeans_5_cluster']]
# Y = df_imp['percent_creak']
#
# X = sm.add_constant(X) # adding a constant
#
# model = sm.OLS(Y, X).fit()
# predictions = model.predict(X)
#
# print_model = model.summary()
# print(print_model)

# ###### BASELINE #######
# # Labels are the values we want to predict
# labels = np.array(df_group_4['cluster_4'])
# # Remove the labels from the features
# # axis 1 refers to the columns
# df_group_4 = df_group_4.drop('cluster_4', axis = 1)
# # Saving feature names for later use
# feature_list = list(df_group_4.columns)
# # Convert to numpy array
# # df_group_0 = np.array(df_group_0)
#
# # Using Skicit-learn to split data into training and testing sets
# from sklearn.model_selection import train_test_split
# # Split the data into training and testing sets
# Xtrain, Xtest, ytrain, ytest = train_test_split(df_group_4, labels, test_size = 0.75, random_state = 42)
#
# ## logistic regressions
# # defining the dependent and independent variables
# # Xtrain = df
# # ytrain = df[['kmeans_5_cluster']]
#
# # building the model and fitting the data
# # loading the training dataset
# # df = pd.read_csv('logit_train1.csv', index_col = 0)
#
# ## check variable correlations
# # fig, axes = plt.subplots()
# # fig.set_size_inches(200,200)
# # sns.clustermap(Xtrain)
# # plt.savefig(os.path.join(dir,'figs', 'logit_cluster_corr.png'), bbox_inches='tight', dpi=300)
# # plt.close()
#
# # plt.show()
#
# log_reg = sm.Logit(ytrain, Xtrain)
# result = log_reg.fit(maxiter=500, method='bfgs')
#
# print(result.summary())
#
# # loading the testing dataset
# # df = pd.read_csv('logit_test1.csv', index_col = 0)
#
# # defining the dependent and independent variables
# # Xtest = df
# # ytest = df['kmeans_5_cluster']
#
# # performing predictions on the test datdaset
# yhat = log_reg.predict(Xtest)
# prediction = list(map(round, yhat))
#
# # comparing original and predicted values of y
# print('Actual values', list(ytest.values))
# print('Predictions :', prediction)
#
# from sklearn.metrics import (confusion_matrix, accuracy_score)
# # confusion matrix
# cm = confusion_matrix(ytest, prediction)
# print ("Confusion Matrix : \n", cm)
#
# # accuracy score of the model
# print('Test accuracy = ', accuracy_score(ytest, prediction))
