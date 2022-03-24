## plotting different thigns after features extraction for queer queer_speech
# author: Ben Lang, blang@ucsd.edu

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
ratings_fname = os.path.join(dir, 'feature_extraction','ratings_features_all.csv')
ratings_all = pd.read_csv(ratings_fname)
ratings_all = ratings_all.drop('Unnamed: 0', axis=1) # might need to change column name when it imports in, there's a random addition at the beginning
sc_WAV_count = ratings_all['WAV'].nunique()
print("Sanity check: There are %s unique WAV files in ratings_all."%sc_WAV_count)
with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(ratings_all.head(5))


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


######################################################
#### COG ####
gender_id_avg_rating = gender_id.groupby('WAV', as_index=False)['Rating'].mean()
gender_id_avg_S_cog = gender_id.groupby('WAV', as_index=False)['spectral_S_cog'].mean()
gender_id_S_cog = pd.merge(gender_id_avg_rating, gender_id_avg_S_cog, on='WAV')

sexual_orientation_avg_rating = sexual_orientation.groupby('WAV', as_index=False)['Rating'].mean()
sexual_orientation_avg_S_cog = sexual_orientation.groupby('WAV', as_index=False)['spectral_S_cog'].mean()
sexual_orientation_S_cog = pd.merge(sexual_orientation_avg_rating, sexual_orientation_avg_S_cog, on='WAV')

voice_id_rating = voice_id.groupby('WAV', as_index=False)['Rating'].mean()
voice_id_avg_S_cog = voice_id.groupby('WAV', as_index=False)['spectral_S_cog'].mean()
voice_id_S_cog = pd.merge(voice_id_rating, voice_id_avg_S_cog, on='WAV')

# plot
fig, axes = plt.subplots(1, 3)
fig.subplots_adjust(hspace=0.5)
fig.set_size_inches(16, 6)
fig.suptitle("Avg /s/ COG by Rating", fontsize=20, fontweight='bold')
fig.subplots_adjust( top = 0.85 )

axes[0].set_title('Gender Identity')
axes[0].set_xlim(1,7)
sns.regplot(data=gender_id_S_cog, x='Rating', y='spectral_S_cog', ax=axes[0], color='#d55e00')
axes[0].set_xlabel('Rating (1-Male, 7-Female)')
axes[0].set_ylabel('Avg /s/ COG')

axes[1].set_title('Sexual Orientation')
axes[1].set_xlim(1,7)
sns.regplot(data=sexual_orientation_S_cog, x='Rating', y='spectral_S_cog', ax=axes[1], color='#0072b2')
axes[1].set_xlabel('Rating (1-Homo, 7-Het)')
axes[1].set_ylabel('')

axes[2].set_title('Voice Identity')
axes[2].set_xlim(1,7)
sns.regplot(data=voice_id_S_cog, x='Rating', y='spectral_S_cog', ax=axes[2], color='#009e73')
axes[2].set_xlabel('Rating (1-Masc, 7-Femme)')
axes[2].set_ylabel('')

# plt.show()
plt.savefig(os.path.join(dir,'figs', 'S_COG_avgbycondition.png'), bbox_inches='tight', dpi=300)
plt.clf()

# overlay
fig, axes = plt.subplots()
fig.set_size_inches(18, 10)
axes.set_title('Avg /s/ COG by Condition Rating', fontsize=20, fontweight='bold')
axes.set_xlim(1,7)
sns.regplot(data=gender_id_S_cog, x='Rating', y='spectral_S_cog', color='#d55e00')
sns.regplot(data=sexual_orientation_S_cog, x='Rating', y='spectral_S_cog', color='#0072b2')
sns.regplot(data=voice_id_S_cog, x='Rating', y='spectral_S_cog', color='#009e73')
axes.set_xlabel('Rating (1-7)')
axes.set_ylabel('Avg /s/ COG')
plt.legend(labels=['Gender ID','Sexual Orientation','Voice ID'])
plt.savefig(os.path.join(dir,'figs', 'S_COG_avgbycondition_overlay.png'), bbox_inches='tight', dpi=300)
# plt.show()

##### regression indiv and overlay loop over all features ########
# need to add in the number of WAV files for each since there's variability
vowel_labels = ['AA','AE','AH','AO','AW','AX','AY','EH','EY','IH','IY','OW','OY','UH','UW']
formant_bandwidth_label = ['F1','F2','F3','F4','B1','B2','B3','B4']
vowel_spectral_names = []
for vowel in vowel_labels:
    for fblabel in formant_bandwidth_label:
        # concatenate strings with '_'
        vowel_string = vowel + "_s" + fblabel
        # append to list
        vowel_spectral_names.append(vowel_string)


features_to_plot = ['F0_mean','F0_range','F0_std','spectral_S_duration','spectral_S_intensity','spectral_S_cog','spectral_S_sdev', 'spectral_S_skew','spectral_S_kurt',
                                                    'spectral_Z_duration','spectral_Z_intensity','spectral_Z_cog','spectral_Z_sdev', 'spectral_Z_skew','spectral_Z_kurt',
                                                    'spectral_F_duration','spectral_F_intensity','spectral_F_cog','spectral_F_sdev', 'spectral_F_skew','spectral_F_kurt',
                                                    'spectral_V_duration','spectral_V_intensity','spectral_V_cog','spectral_V_sdev', 'spectral_V_skew','spectral_V_kurt',
                                                    'spectral_SH_duration','spectral_SH_intensity','spectral_SH_cog','spectral_SH_sdev', 'spectral_SH_skew','spectral_SH_kurt',
                                                    'spectral_JH_duration','spectral_JH_intensity','spectral_JH_cog','spectral_JH_sdev', 'spectral_JH_skew','spectral_JH_kurt']

features_to_plot = features_to_plot + vowel_spectral_names

for feature in features_to_plot:
    gender_id_avg_rating = gender_id.groupby('WAV', as_index=False)['Rating'].mean()
    gender_id_avg_feature = gender_id.groupby('WAV', as_index=False)[feature].mean()
    gender_id_feature = pd.merge(gender_id_avg_rating, gender_id_avg_feature, on='WAV')

    sexual_orientation_avg_rating = sexual_orientation.groupby('WAV', as_index=False)['Rating'].mean()
    sexual_orientation_avg_feature = sexual_orientation.groupby('WAV', as_index=False)[feature].mean()
    sexual_orientation_feature = pd.merge(sexual_orientation_avg_rating, sexual_orientation_avg_feature, on='WAV')

    voice_id_rating = voice_id.groupby('WAV', as_index=False)['Rating'].mean()
    voice_id_avg_feature = voice_id.groupby('WAV', as_index=False)[feature].mean()
    voice_id_feature = pd.merge(voice_id_rating, voice_id_avg_feature, on='WAV')

    ## add something in here that counts the number of WAV with a value, then assign a subplot value to graphs that says "N=inserted value" so that we know how many of each sound file had a certain feature

    fig, axes = plt.subplots(1, 3)
    fig.subplots_adjust(hspace=0.5)
    fig.set_size_inches(16, 6)
    fig.suptitle("Avg %s by Rating"%feature, fontsize=20, fontweight='bold')
    fig.subplots_adjust( top = 0.85 )

    axes[0].set_title('Gender Identity')
    axes[0].set_xlim(1,7)
    sns.regplot(data=gender_id_feature, x='Rating', y=feature, ax=axes[0], color='#648FFF') # #648FFF d55e00
    axes[0].set_xlabel('Rating (1-Male, 7-Female)')
    axes[0].set_ylabel('Avg %s'%feature)

    axes[1].set_title('Sexual Orientation')
    axes[1].set_xlim(1,7)
    sns.regplot(data=sexual_orientation_feature, x='Rating', y=feature, ax=axes[1], color='#785EF0') # #785EF0 0072b2
    axes[1].set_xlabel('Rating (1-Homo, 7-Het)')
    axes[1].set_ylabel('')

    axes[2].set_title('Voice Identity')
    axes[2].set_xlim(1,7)
    sns.regplot(data=voice_id_feature, x='Rating', y=feature, ax=axes[2], color='#DC267F') # #DC267F 009e73
    axes[2].set_xlabel('Rating (1-Masc, 7-Femme)')
    axes[2].set_ylabel('')

    # plt.show()
    plt.savefig(os.path.join(dir,'figs', '%s_avgbycondition.png'%feature), bbox_inches='tight', dpi=300)
    plt.close()



#### proximity ###############################
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
