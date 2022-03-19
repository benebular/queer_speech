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


# formants for vowels
formant_labels = ['sF1','sF2','sF3','sF4','sB1','sB2','sB3','sB4']
for val in formant_labels:
    vowels_pivot = vs.pivot_table(index = ['WAV'], columns = 'Label', values = val).add_suffix('_%s'%val)
    # vowels_pivot_mask = vs.pivot_table(index = ['WAV'], columns = 'Label', values = val, aggfunc=lambda x: len(x.unique()))
    # vowels = (vowels_pivot/vowels_pivot_mask).add_suffix('_%s'%val)
    ratings_all = pd.merge(ratings_all, vowels_pivot, on='WAV')

# vowel center measurements
vowel_labels = ['AA','AE','AH','AO','AW','AX','AY','EH','EY','IH','IY','OW','OY','UH','UW']
# vowels_B4 = vowels_pivot[['AA','AE','AH','AO','AW','AX','AY','EH','EY','IH','IY','OW','OY','UH','UW']]

# with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
#     print(vowels_pivot[['AA','AE','AH','AO','AW','AX','AY','EH','EY','IH','IY','OW','OY','UH','UW']])

### EXPORT ###
sc_WAV_count = ratings_all['WAV'].nunique()
print("Sanity check: There are %s unique WAV files in ratings_all."%sc_WAV_count)
print ("Saving ratings_all as ratings_features_all.csv")
ratings_all.to_csv(os.path.join(dir,'feature_extraction','ratings_features_all.csv'), index=True, encoding='utf-8')


'------------------------------------------------------------------------------'

### Plotting
gender_id = ratings_all[ratings_all['Condition']=='gender_id']
sexual_orientation = ratings_all[ratings_all['Condition']=='sexual_orientation']
voice_id = ratings_all[ratings_all['Condition']=='voice_id']

# F0, adding the boxplot with quartiles
plot_F0_mean = pd.DataFrame({'group':'F0', 'F0': ratings_all['F0_mean']}).drop_duplicates()
plot_F0_90 = pd.DataFrame({'group':'F0_90', 'F0': ratings_all['F0_90']}).drop_duplicates()
plot_F0_10 = pd.DataFrame({'group':'F0_10', 'F0': ratings_all['F0_10']}).drop_duplicates()
plot_F0 = pd.concat([plot_F0_10, plot_F0_mean, plot_F0_90])

dx="group"; dy="F0"; ort="h"; pal = sns.color_palette(n_colors=3); sigma = .2
f, ax = plt.subplots(figsize=(7, 5))
pt.RainCloud(x = dx, y = dy, data = plot_F0, palette = pal, bw = sigma,
                 width_viol = .6, ax = ax, orient = ort)

plt.title("10th percentile; Average F0; 90th percentile, by speaker (across entire utterance)")
# plt.show()
plt.savefig(os.path.join(dir,'figs', 'F0_raincloud.png'), bbox_inches='tight', dpi=300)

# # proximity, gender_id, adding the boxplot with quartiles
# plot_gender_id_prox_social = pd.DataFrame({'group':'prox_social', 'Rating': gender_id['Rating']}).drop_duplicates()
# plot_gender_id_prox_affiliation = pd.DataFrame({'group':'prox_affiliation', 'Rating': gender_id['Rating']}).drop_duplicates()
# plot_gender_id_prox_media = pd.DataFrame({'group':'prox_media', 'Rating': gender_id['Rating']}).drop_duplicates()
# plot_gender_id_prox = pd.concat([plot_gender_id_prox_social, plot_gender_id_prox_affiliation, plot_gender_id_prox_media])
#
# dx="group"; dy="Rating"; ort="h"; pal = sns.color_palette(n_colors=3); sigma = .2
# f, ax = plt.subplots(figsize=(7, 5))
# pt.RainCloud(x = dx, y = dy, data = plot_gender_id_prox, palette = pal, bw = sigma,
#                  width_viol = .6, ax = ax, orient = ort)
#
# plt.title("Ratings of Gender Identity by Proximity")
# # plt.show()
# plt.savefig(os.path.join(dir,'figs', 'gender_id_prox_raincloud.png'), bbox_inches='tight', dpi=300)

# Ratings by Condition
plot_gender_id_rating = pd.DataFrame({'group':'Gender Identity', 'Rating': gender_id['Rating']}).drop_duplicates()
plot_sexual_orientation_rating = pd.DataFrame({'group':'PSO', 'Rating': sexual_orientation['Rating']}).drop_duplicates()
plot_voice_id_rating = pd.DataFrame({'group':'Voice Typicality', 'Rating': voice_id['Rating']}).drop_duplicates()
plot_conditions = pd.concat([plot_sexual_orientation_rating, plot_gender_id_rating, plot_voice_id_rating])

dx="group"; dy="Rating"; ort="h"; pal = sns.color_palette(n_colors=3); sigma = .2
f, ax = plt.subplots(figsize=(7, 5))
pt.RainCloud(x = dx, y = dy, data = plot_conditions, palette = pal, bw = sigma,
                 width_viol = .6, ax = ax, orient = ort)

plt.title("Avg Ratings Distribution by Condition")
# plt.show()
plt.savefig(os.path.join(dir,'figs', 'ratingsbycondition_raincloud.png'), bbox_inches='tight', dpi=300)


### F0 AVG ####create scatterplot with regression line and confidence interval lines
gender_id_avg_rating = gender_id.groupby('WAV', as_index=False)['Rating'].mean()
gender_id_avg_F0 = gender_id.groupby('WAV', as_index=False)['F0_mean'].mean()
gender_id_F0 = pd.merge(gender_id_avg_rating, gender_id_avg_F0, on='WAV')

sexual_orientation_avg_rating = sexual_orientation.groupby('WAV', as_index=False)['Rating'].mean()
sexual_orientation_avg_F0 = sexual_orientation.groupby('WAV', as_index=False)['F0_mean'].mean()
sexual_orientation_F0 = pd.merge(sexual_orientation_avg_rating, sexual_orientation_avg_F0, on='WAV')

voice_id_rating = voice_id.groupby('WAV', as_index=False)['Rating'].mean()
voice_id_avg_F0 = voice_id.groupby('WAV', as_index=False)['F0_mean'].mean()
voice_id_F0 = pd.merge(voice_id_rating, voice_id_avg_F0, on='WAV')

# plot
fig, axes = plt.subplots(1, 3)
fig.subplots_adjust(hspace=0.5)
fig.set_size_inches(16, 6)
fig.suptitle("Avg F0 by Rating", fontsize=20, fontweight='bold')
fig.subplots_adjust( top = 0.85 )

axes[0].set_title('Gender Identity')
axes[0].set_xlim(1,7)
sns.regplot(data=gender_id_F0, x='Rating', y='F0_mean', ax=axes[0], color='#d55e00')
axes[0].set_xlabel('Rating (1-Male, 7-Female)')
axes[0].set_ylabel('Avg F0')

axes[1].set_title('Sexual Orientation')
axes[1].set_xlim(1,7)
sns.regplot(data=sexual_orientation_F0, x='Rating', y='F0_mean', ax=axes[1], color='#0072b2')
axes[1].set_xlabel('Rating (1-Homo, 7-Het)')
axes[1].set_ylabel('')

axes[2].set_title('Voice Identity')
axes[2].set_xlim(1,7)
sns.regplot(data=voice_id_F0, x='Rating', y='F0_mean', ax=axes[2], color='#009e73')
axes[2].set_xlabel('Rating (1-Masc, 7-Femme)')
axes[2].set_ylabel('')

# plt.show()
plt.savefig(os.path.join(dir,'figs', 'F0_avgbycondition.png'), bbox_inches='tight', dpi=300)
plt.clf()

# overlay
fig, axes = plt.subplots()
fig.set_size_inches(18, 10)
axes.set_title('Avg F0 by Condition Rating', fontsize=20, fontweight='bold')
axes.set_xlim(1,7)
sns.regplot(data=gender_id_F0, x='Rating', y='F0_mean', color='#d55e00')
sns.regplot(data=sexual_orientation_F0, x='Rating', y='F0_mean', color='#0072b2')
sns.regplot(data=voice_id_F0, x='Rating', y='F0_mean', color='#009e73')
axes.set_xlabel('Rating (1-7)')
axes.set_ylabel('Avg F0')
plt.legend(labels=['Gender ID','Sexual Orientation','Voice ID'])
plt.savefig(os.path.join(dir,'figs', 'F0_avgbycondition_overlay.png'), bbox_inches='tight', dpi=300)
# plt.show()


#### COG ####
gender_id_avg_rating = gender_id.groupby('WAV', as_index=False)['Rating'].mean()
gender_id_avg_F0 = gender_id.groupby('WAV', as_index=False)['F0_mean'].mean()
gender_id_F0 = pd.merge(gender_id_avg_rating, gender_id_avg_F0, on='WAV')

sexual_orientation_avg_rating = sexual_orientation.groupby('WAV', as_index=False)['Rating'].mean()
sexual_orientation_avg_F0 = sexual_orientation.groupby('WAV', as_index=False)['F0_mean'].mean()
sexual_orientation_F0 = pd.merge(sexual_orientation_avg_rating, sexual_orientation_avg_F0, on='WAV')

voice_id_rating = voice_id.groupby('WAV', as_index=False)['Rating'].mean()
voice_id_avg_F0 = voice_id.groupby('WAV', as_index=False)['F0_mean'].mean()
voice_id_F0 = pd.merge(voice_id_rating, voice_id_avg_F0, on='WAV')

# plot
fig, axes = plt.subplots(1, 3)
fig.subplots_adjust(hspace=0.5)
fig.set_size_inches(16, 6)
fig.suptitle("Avg F0 by Rating", fontsize=20, fontweight='bold')
fig.subplots_adjust( top = 0.85 )

axes[0].set_title('Gender Identity')
axes[0].set_xlim(1,7)
sns.regplot(data=gender_id_F0, x='Rating', y='F0_mean', ax=axes[0], color='#d55e00')
axes[0].set_xlabel('Rating (1-Male, 7-Female)')
axes[0].set_ylabel('Avg F0')

axes[1].set_title('Sexual Orientation')
axes[1].set_xlim(1,7)
sns.regplot(data=sexual_orientation_F0, x='Rating', y='F0_mean', ax=axes[1], color='#0072b2')
axes[1].set_xlabel('Rating (1-Homo, 7-Het)')
axes[1].set_ylabel('')

axes[2].set_title('Voice Identity')
axes[2].set_xlim(1,7)
sns.regplot(data=voice_id_F0, x='Rating', y='F0_mean', ax=axes[2], color='#009e73')
axes[2].set_xlabel('Rating (1-Masc, 7-Femme)')
axes[2].set_ylabel('')

# plt.show()
plt.savefig(os.path.join(dir,'figs', 'F0_avgbycondition.png'), bbox_inches='tight', dpi=300)
plt.clf()

# overlay
fig, axes = plt.subplots()
fig.set_size_inches(18, 10)
axes.set_title('Avg F0 by Condition Rating', fontsize=20, fontweight='bold')
axes.set_xlim(1,7)
sns.regplot(data=gender_id_F0, x='Rating', y='F0_mean', color='#d55e00')
sns.regplot(data=sexual_orientation_F0, x='Rating', y='F0_mean', color='#0072b2')
sns.regplot(data=voice_id_F0, x='Rating', y='F0_mean', color='#009e73')
axes.set_xlabel('Rating (1-7)')
axes.set_ylabel('Avg F0')
plt.legend(labels=['Gender ID','Sexual Orientation','Voice ID'])
plt.savefig(os.path.join(dir,'figs', 'F0_avgbycondition_overlay.png'), bbox_inches='tight', dpi=300)
# plt.show()



#### proximity
# social
gender_id_avg_rating = gender_id.groupby('Participant', as_index=False)['Rating'].mean()
gender_id_prox_social = pd.DataFrame({'Participant':gender_id['Participant'], 'prox_social': gender_id['participant_prox_social']}).drop_duplicates()
gender_id_social = pd.merge(gender_id_avg_rating, gender_id_prox_social, on='Participant')

sexual_orientation_avg_rating = sexual_orientation.groupby('Participant', as_index=False)['Rating'].mean()
sexual_orientation_prox_social = pd.DataFrame({'Participant':sexual_orientation['Participant'], 'prox_social': sexual_orientation['participant_prox_social']}).drop_duplicates()
sexual_orientation_social = pd.merge(sexual_orientation_avg_rating, sexual_orientation_prox_social, on='Participant')

voice_id_avg_rating = voice_id.groupby('Participant', as_index=False)['Rating'].mean()
voice_id_prox_social = pd.DataFrame({'Participant':voice_id['Participant'], 'prox_social': voice_id['participant_prox_social']}).drop_duplicates()
voice_id_social = pd.merge(voice_id_avg_rating, voice_id_prox_social, on='Participant')

# affiliation
gender_id_prox_affiliation = pd.DataFrame({'Participant':gender_id['Participant'], 'prox_affiliation': gender_id['participant_prox_affiliation']}).drop_duplicates()
sexual_orientation_prox_affiliation = pd.DataFrame({'Participant':sexual_orientation['Participant'], 'prox_affiliation': sexual_orientation['participant_prox_affiliation']}).drop_duplicates()
voice_id_prox_affiliation = pd.DataFrame({'Participant':voice_id['Participant'], 'prox_affiliation': voice_id['participant_prox_affiliation']}).drop_duplicates()

gender_id_affiliation = pd.merge(gender_id_avg_rating, gender_id_prox_affiliation, on='Participant')
sexual_orientation_affiliation = pd.merge(sexual_orientation_avg_rating, sexual_orientation_prox_affiliation, on='Participant')
voice_id_affiliation = pd.merge(voice_id_avg_rating, voice_id_prox_affiliation, on='Participant')

# media
gender_id_prox_media = pd.DataFrame({'Participant':gender_id['Participant'], 'prox_media': gender_id['participant_prox_media']}).drop_duplicates()
sexual_orientation_prox_media = pd.DataFrame({'Participant':sexual_orientation['Participant'], 'prox_media': sexual_orientation['participant_prox_media']}).drop_duplicates()
voice_id_prox_media = pd.DataFrame({'Participant':voice_id['Participant'], 'prox_media': voice_id['participant_prox_media']}).drop_duplicates()

gender_id_media = pd.merge(gender_id_avg_rating, gender_id_prox_media, on='Participant')
sexual_orientation_media = pd.merge(sexual_orientation_avg_rating, sexual_orientation_prox_media, on='Participant')
voice_id_media = pd.merge(voice_id_avg_rating, voice_id_prox_media, on='Participant')

# plot
fig, axes = plt.subplots(3, 3)
fig.subplots_adjust(hspace=0.5)
fig.set_size_inches(16, 6)
fig.suptitle("Rating by Proximity to LGBTQ+ Community", fontsize=20, fontweight='bold')
fig.subplots_adjust( top = 0.85 )

# social
axes[0,0].set_title('Gender Identity')
axes[0,0].set_xlim(0,100)
axes[0,0].set_ylim(1,7)
sns.regplot(data=gender_id_social, x='prox_social', y='Rating', ax=axes[0,0], color='#d55e00')
axes[0,0].set_xlabel('Percent of Social Circle')
axes[0,0].set_ylabel('')

axes[0,1].set_title('Sexual Orientation')
axes[0,1].set_xlim(0,100)
axes[0,1].set_ylim(1,7)
sns.regplot(data=sexual_orientation_social, x='prox_social', y='Rating', ax=axes[0,1], color='#0072b2')
axes[0,1].set_xlabel('Percent of Social Circle')
axes[0,1].set_ylabel('')

axes[0,2].set_title('Voice Identity')
axes[0,2].set_xlim(0,100)
axes[0,2].set_ylim(1,7)
sns.regplot(data=voice_id_social, x='prox_social', y='Rating', ax=axes[0,2], color='#009e73')
axes[0,2].set_xlabel('Percent of Social Circle')
axes[0,2].set_ylabel('')

# affiliation
# axes[1,0].set_title('Gender Identity')
axes[1,0].set_xlim(0,100)
axes[1,0].set_ylim(1,7)
sns.regplot(data=gender_id_affiliation, x='prox_affiliation', y='Rating', ax=axes[1,0], color='#d55e00')
axes[1,0].set_xlabel('Percent of Affiliation')
axes[1,0].set_ylabel('Rating (1-Male, 7-Female)')

# axes[1,1].set_title('Sexual Orientation')
axes[1,1].set_xlim(0,100)
axes[1,1].set_ylim(1,7)
sns.regplot(data=sexual_orientation_affiliation, x='prox_affiliation', y='Rating', ax=axes[1,1], color='#0072b2')
axes[1,1].set_xlabel('Percent of Affiliation')
axes[1,1].set_ylabel('Rating (1-Homo, 7-Het)')

# axes[1,2].set_title('Voice Identity')
axes[1,2].set_xlim(0,100)
axes[1,2].set_ylim(1,7)
sns.regplot(data=voice_id_affiliation, x='prox_affiliation', y='Rating', ax=axes[1,2], color='#009e73')
axes[1,2].set_xlabel('Percent of Affiliation')
axes[1,2].set_ylabel('Rating (1-Masc, 7-Femme)')

# media
# axes[2,0].set_title('Gender Identity')
axes[2,0].set_xlim(0,100)
axes[2,0].set_ylim(1,7)
sns.regplot(data=gender_id_media, x='prox_media', y='Rating', ax=axes[2,0], color='#d55e00')
axes[2,0].set_xlabel('Percent of Media Consumed')
axes[2,0].set_ylabel('')

# axes[2,1].set_title('Sexual Orientation')
axes[2,1].set_xlim(0,100)
axes[2,1].set_ylim(1,7)
sns.regplot(data=sexual_orientation_media, x='prox_media', y='Rating', ax=axes[2,1], color='#0072b2')
axes[2,1].set_xlabel('Percent of Media Consumed')
axes[2,1].set_ylabel('')

# axes[2,2].set_title('Voice Identity')
axes[2,2].set_xlim(0,100)
axes[2,2].set_ylim(1,7)
sns.regplot(data=voice_id_media, x='prox_media', y='Rating', ax=axes[2,2], color='#009e73')
axes[2,2].set_xlabel('Percent of Media Consumed')
axes[2,2].set_ylabel('')

# plt.show()
plt.savefig(os.path.join(dir,'figs', 'proximity_social.png'), bbox_inches='tight', dpi=300)
plt.clf()

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
