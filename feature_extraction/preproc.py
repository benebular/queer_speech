## raw data organization for queer speech
# author: Ben lang
# e: blang@ucsd.edu

import numpy as np
import pandas as pd
import time
import seaborn as sns
import os
import os.path as op
import sys
import matplotlib.pyplot as plt
import re
import glob
import scipy.stats as stats
np.set_printoptions(threshold=sys.maxsize)

# set up directory and read in csv
dir = '/Users/bcl/Documents/GitHub/queer_speech'
os.chdir(dir)
vs_fname = os.path.join(dir,'feature_extraction','vs_ms_output.csv')
matches_fname = os.path.join(dir, 'qualtrics_data/mark1_jan28/','matches_queer_speech.csv')
queer_ratings_fname = os.path.join(dir, 'qualtrics_data/mark1_jan28/','matched_queer.csv')
spectral_S_fname = os.path.join(dir,'feature_extraction','logfile_S.csv')
spectral_SH_fname = os.path.join(dir,'feature_extraction','logfile_SH.csv')
spectral_Z_fname = os.path.join(dir,'feature_extraction','logfile_Z.csv')
spectral_JH_fname = os.path.join(dir,'feature_extraction','logfile_JH.csv')
spectral_F_fname = os.path.join(dir,'feature_extraction','logfile_F.csv')
spectral_V_fname = os.path.join(dir,'feature_extraction','logfile_V.csv')

vs = pd.read_csv(vs_fname)
matches = pd.read_csv(matches_fname)
queer_ratings = pd.read_csv(queer_ratings_fname)
spectral_S = pd.read_csv(spectral_S_fname, header=None)
spectral_SH = pd.read_csv(spectral_SH_fname, header=None)
spectral_Z = pd.read_csv(spectral_Z_fname, header=None)
spectral_JH = pd.read_csv(spectral_JH_fname, header=None)
spectral_F = pd.read_csv(spectral_F_fname, header=None)
spectral_V = pd.read_csv(spectral_V_fname, header=None)
vs.rename(columns = {'Filename':'WAV'}, inplace = True)
# view all labels in each file
with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(vs[['WAV','Label']].head(5))

# remove file extension .mat from strings to match other columns
new_names = []
for i in vs['WAV']:
    x = i[:-4]
    new_names.append(x)
vs['WAV'] = new_names

## rename cols in spectral moment dfs
spectral_dfs = [spectral_S,spectral_SH,spectral_Z,spectral_JH,spectral_F,spectral_V]
for i in spectral_dfs:
    i.columns = ['WAV','start','duration','intensity','cog','sdev','skew','kurt','delete_me'] # add in colnames
    i['WAV'] = i['WAV'].str.replace(r'\.wav', '') # get rid of .wav suffix
    i['WAV'] = i['WAV'].str.replace(r'\ ', '') # get rid of .wav suffix
    del i['delete_me'] # delete random NaN column from raw data
    # the praat script puts the colmn name in as the nan value instead of a nan or a 0 so the following line changes all of those to nans for analysis
    i[['start','duration','intensity','cog','sdev','skew','kurt']] = i[['start','duration','intensity','cog','sdev','skew','kurt']].apply(pd.to_numeric,errors='coerce') # change all copied strings in data to NaNs

# change col names to unique spectral segments
spectral_S = spectral_S.add_prefix('spectral_S_')
spectral_S = spectral_S.rename(columns={'spectral_S_WAV':'WAV'})
spectral_Z = spectral_Z.add_prefix('spectral_Z_')
spectral_Z = spectral_Z.rename(columns={'spectral_Z_WAV':'WAV'})
spectral_F = spectral_F.add_prefix('spectral_F_')
spectral_F = spectral_F.rename(columns={'spectral_F_WAV':'WAV'})
spectral_V = spectral_V.add_prefix('spectral_V_')
spectral_V = spectral_V.rename(columns={'spectral_V_WAV':'WAV'})
spectral_SH = spectral_SH.add_prefix('spectral_SH_')
spectral_SH = spectral_SH.rename(columns={'spectral_SH_WAV':'WAV'})
spectral_JH = spectral_JH.add_prefix('spectral_JH_')
spectral_JH = spectral_JH.rename(columns={'spectral_JH_WAV':'WAV'})

# average over spectral segments, leaves nans in place for files without a spectral item
spectral_S = spectral_S.groupby('WAV', as_index=False).mean()
spectral_Z = spectral_Z.groupby('WAV', as_index=False).mean()
spectral_F = spectral_F.groupby('WAV', as_index=False).mean()
spectral_V = spectral_V.groupby('WAV', as_index=False).mean()
spectral_SH = spectral_SH.groupby('WAV', as_index=False).mean()
spectral_JH = spectral_JH.groupby('WAV', as_index=False).mean()

# segment durations
vs['duration'] = vs['seg_End'] - vs['seg_Start']
duration_pivot = vs.pivot_table(index = ['WAV'], columns = 'Label', values = 'duration')
duration_pivot_mask = vs.pivot_table(index = ['WAV'], columns = 'Label', values = 'duration', aggfunc=lambda x: len(x.unique()))
duration = (duration_pivot/duration_pivot_mask).add_suffix('_avg_dur')

# collect average strF0 across single utterances by speaker
window = 50
F0_avg_temp = vs.groupby(['WAV'], as_index=False)['strF0'].rolling(window=window).mean().to_frame()
F0_avg = F0_avg_temp.groupby(['WAV'], as_index=False)['strF0'].mean().rename(columns = {'strF0':'F0_mean'})
F0_avg = pd.concat([pd.DataFrame({'WAV':duration_pivot.index}), F0_avg], axis = 1)

F0_std_temp = vs.groupby(['WAV'], as_index=False)['strF0'].rolling(window=window).mean().to_frame()
F0_std = F0_std_temp.groupby(['WAV'], as_index=False)['strF0'].std().rename(columns = {'strF0':'F0_std'})
F0_std = pd.concat([pd.DataFrame({'WAV':duration_pivot.index}), F0_std], axis = 1)

# collect 90th and 10th percentiles for min, max, and range of F0
F0_90 = F0_avg_temp.groupby(['WAV'], as_index=False)['strF0'].quantile(0.9).rename(columns = {'strF0':'F0_90'})
F0_90 = pd.concat([pd.DataFrame({'WAV':duration_pivot.index}), F0_90], axis = 1)
F0_10 = F0_avg_temp.groupby(['WAV'], as_index=False)['strF0'].quantile(0.1).rename(columns = {'strF0':'F0_10'})
F0_10 = pd.concat([pd.DataFrame({'WAV':duration_pivot.index}), F0_10], axis = 1)
F0_subtraction = F0_90['F0_90'] - F0_10['F0_10']
F0_range = pd.DataFrame({'WAV':F0_avg['WAV'],'F0_range':F0_subtraction})

# concatenate all into ratings df
ratings_all = pd.merge(queer_ratings, F0_avg, on='WAV')
ratings_all = pd.merge(ratings_all, F0_std, on='WAV')
ratings_all = pd.merge(ratings_all, F0_range, on='WAV')
ratings_all = pd.merge(ratings_all, F0_90, on='WAV')
ratings_all = pd.merge(ratings_all, F0_10, on='WAV')
ratings_all = pd.merge(ratings_all, duration, on='WAV')
ratings_all = pd.merge(ratings_all, spectral_S, on='WAV')
ratings_all = pd.merge(ratings_all, spectral_Z, on='WAV')
ratings_all = pd.merge(ratings_all, spectral_F, on='WAV')
ratings_all = pd.merge(ratings_all, spectral_V, on='WAV')
ratings_all = pd.merge(ratings_all, spectral_SH, on='WAV')
ratings_all = pd.merge(ratings_all, spectral_JH, on='WAV')


############## VOWEL MEASUREMENTS ##############
vowel_labels = ['AA','AE','AH','AO','AW','AX','AY','EH','EY','IH','IY','OW','OY','UH','UW']
formant_bandwidth_label = ['F1','F2','F3','F4','B1','B2','B3','B4']
vowel_spectral_names = []
for vowel in vowel_labels: # loop for making a list of the vowel spectral features for each vowel--matches columns in spreadsheet
    for fblabel in formant_bandwidth_label:
        # concatenate strings with '_'
        vowel_string = vowel + "_s" + fblabel + "_mean"
        # append to list
        vowel_spectral_names.append(vowel_string)
        vowel_string = vowel + "_s" + fblabel + "_min"
        vowel_spectral_names.append(vowel_string)
        vowel_string = vowel + "_s" + fblabel + "_max"
        vowel_spectral_names.append(vowel_string)

# mean, min, max formants and bandwidths for individual vowels
formant_labels = ['sF1','sF2','sF3','sF4','sB1','sB2','sB3','sB4']
for val in formant_labels:
    vowels_pivot_mean = vs.pivot_table(index = ['WAV'], columns = 'Label', values = val, aggfunc=np.mean).add_suffix('_%s_mean'%val)
    vowels_pivot_min = vs.pivot_table(index = ['WAV'], columns = 'Label', values = val, aggfunc=np.min).add_suffix('_%s_min'%val)
    vowels_pivot_max = vs.pivot_table(index = ['WAV'], columns = 'Label', values = val, aggfunc=np.max).add_suffix('_%s_max'%val)

    # vowels_pivot_mask = vs.pivot_table(index = ['WAV'], columns = 'Label', values = val, aggfunc=lambda x: len(x.unique()))
    # vowels = (vowels_pivot/vowels_pivot_mask).add_suffix('_%s'%val)
    ratings_all = pd.merge(ratings_all, vowels_pivot_mean, on='WAV')
    ratings_all = pd.merge(ratings_all, vowels_pivot_min, on='WAV')
    ratings_all = pd.merge(ratings_all, vowels_pivot_max, on='WAV')

# mean vowel duration across entire utterance (Pierrehumbert)
vowel_avg_duration_names = []
for vowel in vowel_labels:
    vowel_string = vowel + '_avg_dur'
    vowel_avg_duration_names.append(vowel_string)

vowel_avg_duration = pd.DataFrame({'WAV':duration_pivot.index})
for vowel in vowel_avg_duration_names:
    vowel_avg = ratings_all.groupby(['WAV'], as_index=False)[vowel].mean()
    vowel_avg_duration = pd.merge(vowel_avg_duration, vowel_avg, on='WAV', how='outer')
vowel_avg_duration['vowel_avg_dur'] = vowel_avg_duration.mean(axis=1)
vowel_avg_duration = vowel_avg_duration[['WAV','vowel_avg_dur']]
ratings_all = pd.merge(ratings_all, vowel_avg_duration, on='WAV')

# dispersion (average of euclidean distances within a speaker, centered on IH or AH) # dykingout_notlikeme needs to be separate
vowel_labels_euc = ['AA','AE','AH','AO','AW','AX','AY','EH','EY','IY','OW','OY','UH','UW'] # does not have 'IH', held out as center of vowel space
formant_bandwidth_label_euc = ['F1','F2','F3','F4']
euc_center_IH_list = ['IH_sF1_mean','IH_sF2_mean','IH_sF3_mean','IH_sF4_mean']
euc_center_IH = ratings_all.groupby('WAV', as_index=False)[euc_center_IH_list].mean().set_index('WAV').dropna()

# WAV_list = euc_center_IH['WAV'].to_list()
# euc_center_all = ratings_all.groupby('WAV', as_index=False)[vowel_spectral_names_euc].mean()

# vowel_spectral_names_euc = []
dist_list = []
WAV_list = []
dispersion_strings =[]
for vowel in vowel_labels_euc: # loop for making a list of the vowel spectral features for each vowel--matches columns in spreadsheet
    vowel_spectral_names_short = []
    for fblabel in formant_bandwidth_label_euc:
        # concatenate strings with '_'
        vowel_string = vowel + "_s" + fblabel + "_mean"
        # append to list
        vowel_spectral_names_short.append(vowel_string)
    # create a df of just one vowel and formant combination
    euc_center_short = ratings_all.groupby('WAV', as_index=False)[vowel_spectral_names_short].mean().set_index('WAV').dropna() # creates
    # merge the single vowel and formant combination with the baseline vowel, in this case IH, and only merge on rows where IH and the vowel are in the same file, rest will need to happen with AH or IY
    combined_euc = pd.merge(euc_center_short, euc_center_IH, on='WAV')
    # split the dfs out again so you can compute distance on their values
    new_euc_byvowel = combined_euc[vowel_spectral_names_short]
    new_euc_IH = combined_euc[euc_center_IH_list]
    # loop through the rows of the dfs (which are now the same length)
    for i in range(len(new_euc_IH)):
        # compute euclidean distance between individual vowels
        dist = np.linalg.norm(new_euc_IH.iloc[i].values-new_euc_byvowel.iloc[i].values)
        # get the WAV for vowel being compared
        WAV_compared = new_euc_IH.index[i]
        # grab first vowel string from both frames
        IH_string = new_euc_IH.columns[0]
        byvowel_string = new_euc_byvowel.columns[0]
        IH_string = IH_string[0:2]
        byvowel_string = byvowel_string[0:2]
        # concatenate it
        vowel_pair = IH_string + "_to_" + byvowel_string
        #append things to lists in order
        dist_list.append(dist)
        WAV_list.append(WAV_compared)
        dispersion_strings.append(vowel_pair)

# make a df with the three lists from the dispersion measures
distance_all_IH = pd.DataFrame({"WAV": WAV_list, "vowel_distance_pair": dispersion_strings, "vowel_distance_value": dist_list})
# sort for easy viewing
distance_IH = distance_all_IH.sort_values(by='WAV')


# vowel_spectral_names_euc = []
vowel_labels_euc = ['AA','AE','IH','AO','AW','AX','AY','EH','EY','IY','OW','OY','UH','UW'] # does not have 'AH', held out as center of vowel space
euc_center_AH_list = ['AH_sF1_mean','AH_sF2_mean','AH_sF3_mean','AH_sF4_mean']
euc_center_AH = ratings_all.groupby('WAV', as_index=False)[euc_center_AH_list].mean().set_index('WAV').dropna()
dist_list = []
WAV_list = []
dispersion_strings =[]
for vowel in vowel_labels_euc: # loop for making a list of the vowel spectral features for each vowel--matches columns in spreadsheet
    vowel_spectral_names_short = []
    for fblabel in formant_bandwidth_label_euc:
        # concatenate strings with '_'
        vowel_string = vowel + "_s" + fblabel + "_mean"
        # append to list
        vowel_spectral_names_short.append(vowel_string)
    # create a df of just one vowel and formant combination
    euc_center_short = ratings_all.groupby('WAV', as_index=False)[vowel_spectral_names_short].mean().set_index('WAV').dropna() # creates
    # merge the single vowel and formant combination with the baseline vowel, in this case IH, and only merge on rows where IH and the vowel are in the same file, rest will need to happen with AH or IY
    combined_euc = pd.merge(euc_center_short, euc_center_AH, on='WAV')
    # split the dfs out again so you can compute distance on their values
    new_euc_byvowel = combined_euc[vowel_spectral_names_short]
    new_euc_AH = combined_euc[euc_center_AH_list]
    # loop through the rows of the dfs (which are now the same length)
    for i in range(len(new_euc_AH)):
        # compute euclidean distance between individual vowels
        dist = np.linalg.norm(new_euc_AH.iloc[i].values-new_euc_byvowel.iloc[i].values)
        # get the WAV for vowel being compared
        WAV_compared = new_euc_AH.index[i]
        # grab first vowel string from both frames
        AH_string = new_euc_AH.columns[0]
        byvowel_string = new_euc_byvowel.columns[0]
        AH_string = AH_string[0:2]
        byvowel_string = byvowel_string[0:2]
        # concatenate it
        vowel_pair = AH_string + "_to_" + byvowel_string
        #append things to lists in order
        dist_list.append(dist)
        WAV_list.append(WAV_compared)
        dispersion_strings.append(vowel_pair)

# make a df with the three lists from the dispersion measures
distance_all_AH = pd.DataFrame({"WAV": WAV_list, "vowel_distance_pair": dispersion_strings, "vowel_distance_value": dist_list})
# sort for easy viewing
distance_AH = distance_all_AH.sort_values(by='WAV')

# dykingout_notlikeme
vowel_labels_euc = ['AA','AE','AH','AO','AW','AX','AY','EH','EY','OW','OY','UH','UW'] # does not have 'IY', held out as center of vowel space
euc_center_IY_list = ['IY_sF1_mean','IY_sF2_mean','IY_sF3_mean','IY_sF4_mean']
euc_center_IY = ratings_all.groupby('WAV', as_index=False)[euc_center_IY_list].mean().set_index('WAV').dropna()
dist_list = []
WAV_list = []
dispersion_strings =[]
for vowel in vowel_labels_euc: # loop for making a list of the vowel spectral features for each vowel--matches columns in spreadsheet
    vowel_spectral_names_short = []
    for fblabel in formant_bandwidth_label_euc:
        # concatenate strings with '_'
        vowel_string = vowel + "_s" + fblabel + "_mean"
        # append to list
        vowel_spectral_names_short.append(vowel_string)
    # create a df of just one vowel and formant combination
    euc_center_short = ratings_all.groupby('WAV', as_index=False)[vowel_spectral_names_short].mean().set_index('WAV').dropna() # creates
    # merge the single vowel and formant combination with the baseline vowel, in this case IH, and only merge on rows where IH and the vowel are in the same file, rest will need to happen with AH or IY
    combined_euc = pd.merge(euc_center_short, euc_center_IY, on='WAV')
    # split the dfs out again so you can compute distance on their values
    new_euc_byvowel = combined_euc[vowel_spectral_names_short]
    new_euc_IY = combined_euc[euc_center_IY_list]
    # loop through the rows of the dfs (which are now the same length)
    for i in range(len(new_euc_IY)):
        # compute euclidean distance between individual vowels
        dist = np.linalg.norm(new_euc_IY.iloc[i].values-new_euc_byvowel.iloc[i].values)
        # get the WAV for vowel being compared
        WAV_compared = new_euc_IY.index[i]
        # grab first vowel string from both frames
        IY_string = new_euc_IY.columns[0]
        byvowel_string = new_euc_byvowel.columns[0]
        IY_string = IY_string[0:2]
        byvowel_string = byvowel_string[0:2]
        # concatenate it
        vowel_pair = IY_string + "_to_" + byvowel_string
        #append things to lists in order
        dist_list.append(dist)
        WAV_list.append(WAV_compared)
        dispersion_strings.append(vowel_pair)

# make a df with the three lists from the dispersion measures
distance_all_IY = pd.DataFrame({"WAV": WAV_list, "vowel_distance_pair": dispersion_strings, "vowel_distance_value": dist_list})
# sort for easy viewing
distance_IY = distance_all_IY.sort_values(by='WAV')


# distance_all = pd.merge(distance_IH, distance_AH, on='WAV', how='right')

distance_interim_AH = (distance_IH.merge(distance_AH, on='WAV', how='right', indicator=True)
     .query('_merge == "right_only"')
     .drop('_merge', 1)
     .dropna(axis=1, how='all')
     .rename(columns={'vowel_distance_pair_y': 'vowel_distance_pair', 'vowel_distance_value_y': 'vowel_distance_value'}))

distance_AH_IH = pd.concat([distance_IH, distance_interim_AH], axis=0)

distance_interim_IY = (distance_AH_IH.merge(distance_IY, on='WAV', how='right', indicator=True)
     .query('_merge == "right_only"')
     .drop('_merge', 1)
     .dropna(axis=1, how='all')
     .rename(columns={'vowel_distance_pair_y': 'vowel_distance_pair', 'vowel_distance_value_y': 'vowel_distance_value'}))

distance_all = pd.concat([distance_AH_IH, distance_interim_IY], axis=0)

# save so you can double check which vowels are missing for certain speakers and make a decision about reducing spaces or norming this all somehow
distance_all.to_csv(os.path.join(dir,'feature_extraction','distance_all.csv'), index=True, encoding='utf-8')

dispersion_all = distance_all.groupby('WAV', as_index=False)['vowel_distance_value'].mean().rename(columns={"vowel_distance_value":'dispersion'})

## sanity check##
assert len(dist_list) == len(WAV_list) == len(dispersion_strings)
## verify that length of dispersion is the same length as the number of vowel tokens in the data, this will throw an error if something went wrong
# if this goes fine, it means dispersion_all has a single value that is the average distance between IH and all the other vowels in the speaker's vowel space, when IH and other vowels are present
# for IH, it should be 56, which means another 10 need to be found, 1 will need to be done on its own
# assert ratings_all['IH_sF1_mean'].nunique() == len(dispersion_all)
distance_all['WAV'].nunique() # must be 66

# merge dispersion
ratings_all = pd.merge(ratings_all, dispersion_all, on='WAV', how='outer')


# with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
#     print(distance_all)

# creak (Podesva)
print ('Extracting creak...')
dir = '/Users/bcl/Documents/MATLAB/covarep/output'
os.chdir(dir)
creak_WAV_list = []
creak_percent_list =[]
for name in glob.glob(os.path.join(dir,'*')):
    # get corresponding WAV name
    creak_fname = str(name)
    creak_WAV_name = re.search(r'creak_(.*?).csv', creak_fname).group(1)
    creak_WAV_list.append(creak_WAV_name)
    # read in the creak csv for a WAV
    creak = pd.read_csv(name, header=None, nrows=1) # this takes time per file because they are ms-by-ms and so it's like 40000 lines
    # get all the values that have a 0 or 1 and ignore the NaNs at the beginning
    total_creaks = np.count_nonzero(~np.isnan(creak))
    # get all the hits where there is creak
    creak_hits = np.count_nonzero(creak == 1)
    # divide and get percentage
    percent_creak = creak_hits/total_creaks
    # append the values in order
    creak_percent_list.append(percent_creak)

creak_data = pd.DataFrame({'WAV':creak_WAV_list,'percent_creak':creak_percent_list}).sort_values(by='WAV')
creak_data['percent_creak'] = creak_data['percent_creak'].mul(100,axis=0)
creak_data['percent_creak'] = creak_data['percent_creak'].replace(0,np.nan)
ratings_all = pd.merge(ratings_all, creak_data, on='WAV')

## z-score Ratings columns
ratings_all['Rating_z_score'] = stats.zscore(ratings_all['Rating'], axis=0)

## random number column for sanity checks
ratings_all['rando'] = np.random.rand(len(ratings_all), 1)

### EXPORT ###
dir = '/Users/bcl/Documents/GitHub/queer_speech'
sc_WAV_count = ratings_all['WAV'].nunique()
print("Sanity check: There are %s unique WAV files in ratings_all."%sc_WAV_count)
sc_participant_count = ratings_all['Participant'].nunique()
print("Sanity check: There are %s unique participants in ratings_all."%sc_participant_count)
print ("Saving ratings_all as ratings_features_all.csv")
ratings_all.to_csv(os.path.join(dir,'feature_extraction','ratings_features_all.csv'), index=True, encoding='utf-8')

### graveyard

# make a list of all the features in the VS output
# vs_features = ['strF0','sF1','sF2','sF3','sF4','sB1','sB2','sB3','sB4']
# get rid of indexes or random columns to make list of features from VS ouput
# vs_colnames = pd.Series(vs.columns.values.tolist())
# vs_colnames = vs_colnames.drop(labels=[0, 1, 72], axis=0)

# merge_list = [strF0_avg, strF0_std]
# for idx in range(len(merge_list)):
#     i = merge_list[idx]
#     if ratings_all is None:
#         ratings_all = pd.merge(queer_ratings, i, on='WAV')
#         mid = ratings_all['WAV']
#         ratings_all.drop(labels=['WAV'], axis=1, inplace = True)
#         ratings_all.insert(len(ratings_all.columns), 'WAV', mid)
#     else:
#         ratings_all = pd.merge(ratings_all, i, on='WAV')
#         mid = ratings_all['WAV']
#         ratings_all.drop(labels=['WAV'], axis=1, inplace = True)
#         ratings_all.insert(len(ratings_all.columns), 'WAV', mid)
#
# with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
#     print(ratings_all.head(5))


# for i in range(len(vs['Filename'])):
#     strF0 = vs['strF0'][i]
#     print(strF0)

# use the creak detector
# extract each strF0 per ms, cut off outliers (top 90%), min, max
# smoothed out contour but just calculate the range on the smoothed contour
# fit loess curve to the pitch range


# make lists of relevant segments for certain features
# vowels = ['IY','AA','IH','AX','AH','EH','EY','UW','AE','AO','OW','AY','AW','OY','IX']
# sonorants = ['IY','AA','IH','AX','AH','EH','EY','UW','AE','AO','OW','AY','AW','ER','EL','N','NG','M','R','W','L','OY','Y']
#
# vs_features_avg = pd.DataFrame()
#
# for i in vs_features:
# if vowel in vowels:
#     vs_pivot = vs.pivot_table(index = 'Filename', columns = 'Label', values = 'strF0')
#     vs_append = pd.DataFrame(vs_pivot.mean(axis=1))
#     pd.concat([vs_features_avg,vs_append], axis=1)
#
#
# for i in vs_colnames:
#     vs_pivot = vs.pivot_table(index = 'Filename', columns = 'Label', values = i)
#     vs_pivot.index.name = None
#     vs_pivot.index = pd.Series(vs_pivot.index).astype(str)
#     vs_append = pd.Series(vs_pivot.index).astype(str)
#
#     print(vs_pivot)
