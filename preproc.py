## raw data organization for queer speech
# author: Ben lang
# e: blang@ucsd.edu

import numpy as np
import pandas as pd
import time
import os
import os.path as op
import sys
import matplotlib.pyplot as plt
np.set_printoptions(threshold=sys.maxsize)

# set up directory and read in csv
dir = '/Users/bcl/Documents/GitHub/queer_speech'
os.chdir(dir)
vs_fname = os.path.join(dir,'feature_extraction','vs_ms_output.csv')
matches_fname = os.path.join(dir, 'qualtrics_data/mark1_jan28/','matches_queer_speech.csv')
queer_ratings_fname = os.path.join(dir, 'qualtrics_data/mark1_jan28/','matched_queer.csv')

vs = pd.read_csv(vs_fname)
matches = pd.read_csv(matches_fname)
queer_ratings = pd.read_csv(queer_ratings_fname)
vs.rename(columns = {'Filename':'WAV'}, inplace = True)
# view all labels in each file
# with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
#     print(vs[['Filename','Label']])

# make a list of all the features in the VS output
vs_features = ['strF0','sF1','sF2','sF3','sF4','sB1','sB2','sB3','sB4']
#get rid of indexes or random columns to make list of features from VS ouput
# vs_colnames = pd.Series(vs.columns.values.tolist())
# vs_colnames = vs_colnames.drop(labels=[0, 1, 72], axis=0)

# remove file extension .mat from strings to match other columns
new_names = []
for i in vs['WAV']:
    x = i[:-4]
    new_names.append(x)
vs['WAV'] = new_names

# collect average strF0 across single utterances by speaker
strF0_avg = vs.groupby(['WAV'], as_index=False)['strF0'].mean()

# collect 90th and 10th percentiles for min, max, and range of F0
# strF0_range_temp = vs.groupby(['Filename'], as_index=False)['strF0']
# strF0_range = np.percentile(vs['strF0'], 90)


# concatenate dfs into ratings df
ratings_all = pd.merge(queer_ratings, strF0_avg, on='WAV')

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
