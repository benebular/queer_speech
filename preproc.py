## raw data organization for queer speech
# author: Ben lang
# e: blang@ucsd.edu

import numpy as np
import pandas as pd
import time
import seaborn as sns
import ptitprince as pt
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
with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(vs[['WAV','Label']].head(5))

# remove file extension .mat from strings to match other columns
new_names = []
for i in vs['WAV']:
    x = i[:-4]
    new_names.append(x)
vs['WAV'] = new_names

# segment durations
vs['duration'] = vs['seg_End'] - vs['seg_Start']
duration_pivot = vs.pivot_table(index = ['WAV'], columns = 'Label', values = 'duration')
duration_pivot_mask = vs.pivot_table(index = ['WAV'], columns = 'Label', values = 'duration', aggfunc=lambda x: len(x.unique()))
duration = (duration_pivot/duration_pivot_mask).add_suffix('_dur')

# collect average strF0 across single utterances by speaker
F0_avg = vs.groupby(['WAV'], as_index=False)['strF0'].mean().rename(columns = {'strF0':'F0_mean'})
F0_std = vs.groupby(['WAV'], as_index=False)['strF0'].std().rename(columns = {'strF0':'F0_std'})

# collect 90th and 10th percentiles for min, max, and range of F0
F0_90 = vs.groupby(['WAV'], as_index=False)['strF0'].quantile(0.9).rename(columns = {'strF0':'F0_mean'})
F0_10 = vs.groupby(['WAV'], as_index=False)['strF0'].quantile(0.1).rename(columns = {'strF0':'F0_mean'})
F0_subtraction = F0_90['F0_mean'] - F0_10['F0_mean']
F0_range = pd.DataFrame({'WAV':F0_10['WAV'],'F0_range':F0_subtraction})

# concatenate dfs into ratings df
ratings_all = pd.merge(queer_ratings, F0_avg, on='WAV')
ratings_all = pd.merge(ratings_all, F0_std, on='WAV')
ratings_all = pd.merge(ratings_all, F0_range, on='WAV')
ratings_all = pd.merge(ratings_all, duration, on='WAV')

with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(F0_90)


### EXPORT ###
ratings_all.to_csv('ratings_features_all.csv', index=True, encoding='utf-8')


### Plotting
#adding the boxplot with quartiles
f, ax = plt.subplots(figsize=(7, 5))
dy="WAV"; dx="F0_mean"; ort="h"; pal = sns.color_palette(n_colors=1)
ax=pt.half_violinplot( x = dx, y = dy, data = ratings_all, bw = .2, cut = 0., palette = pal,
                      scale = "area", width = .6, inner = None, orient = ort)
ax=sns.stripplot( x = dx, y = dy, data = ratings_all, palette = pal, edgecolor = "white",
                 size = 3, jitter = 1, zorder = 0, orient = ort)
ax=sns.boxplot( x = dx, y = dy, data = ratings_all, color = "black", width = .15, zorder = 10,\
            showcaps = True, boxprops = {'facecolor':'none', "zorder":10},\
            showfliers=True, whiskerprops = {'linewidth':2, "zorder":10},\
               saturation = 1, orient = ort)

ax = sns.barplot(x=dy, y=dx, data=ratings_all, capsize=.1)
plt.boxplot(ratings_all['WAV'].drop_duplicates(),ratings_all['F0_mean'].drop_duplicates())
plt.show()

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
