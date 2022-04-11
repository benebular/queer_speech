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
import matplotlib
import matplotlib.pyplot as plt
np.set_printoptions(threshold=sys.maxsize)

# set up directory and read in csv
dir = '/Users/bcl/Documents/GitHub/queer_speech'
os.chdir(dir)
ratings_fname = os.path.join(dir, 'feature_extraction','queer_data.csv')
ratings_all = pd.read_csv(ratings_fname)
ratings_all = ratings_all.drop('Unnamed: 0', axis=1) # might need to change column name when it imports in, there's a random addition at the beginning
sc_WAV_count = ratings_all['WAV'].nunique()
print("Sanity check: There are %s unique WAV files in ratings_all."%sc_WAV_count)
with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(ratings_all.head(5))


### Plotting
## assign colors based on clusters from classification
# import random
# get_colors = lambda n: list(map(lambda i: "#" + "%06x" % random.randint(0, 0xFFFFFF),range(n)))
# colors = pd.DataFrame(get_colors(5), columns={'color'}) # sample return:  ['#8af5da', '#fbc08c', '#b741d0', '#e599f1', '#bbcb59', '#a2a6c0']
# ratings_all = pd.merge(ratings_all, colors, on = 'kmeans_5_cluster', how = "outer")
colors = pd.DataFrame({"color": ['#648FFF','#785EF0','#DC267F','#FE6100','#FFB000']})
colors['kmeans_5_cluster'] = [0,1,2,3,4]
ratings_all['percent_creak'] = ratings_all['percent_creak'].replace(0, np.nan)
gender_id = ratings_all[ratings_all['Condition']=='gender_id']
sexual_orientation = ratings_all[ratings_all['Condition']=='sexual_orientation']
voice_id = ratings_all[ratings_all['Condition']=='voice_id']
number_participants = ratings_all['Participant'].nunique()
# cluster checks
cluster_id = ratings_all.pivot_table(index = 'kmeans_5_cluster', columns = 'Condition', values = ['Rating_z_score'])

# F0, adding the boxplot with quartiles
plot_F0_mean = pd.DataFrame({'group':'F0', 'F0': ratings_all['F0_mean']}).drop_duplicates()
plot_F0_90 = pd.DataFrame({'group':'F0_90', 'F0': ratings_all['F0_90']}).drop_duplicates()
plot_F0_10 = pd.DataFrame({'group':'F0_10', 'F0': ratings_all['F0_10']}).drop_duplicates()
plot_F0 = pd.concat([plot_F0_10, plot_F0_mean, plot_F0_90])

dx="group"; dy="F0"; ort="h"; pal = sns.color_palette(n_colors=3); sigma = .2
f, ax = plt.subplots(figsize=(7, 5))
pt.RainCloud(x = dx, y = dy, data = plot_F0, palette = pal, bw = sigma,
                 width_viol = .6, ax = ax, orient = ort)

plt.title("10th percentile; Average F0; 90th percentile, by speaker (across entire utterance), %s Participants"%number_participants)
# plt.show()
plt.savefig(os.path.join(dir,'figs', 'F0_raincloud.png'), bbox_inches='tight', dpi=300)
plt.close()

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
# plot_gender_id_rating = pd.DataFrame({'group':'Gender Identity', 'Rating_z_score': gender_id['Rating_z_score']}).drop_duplicates()
# plot_sexual_orientation_rating = pd.DataFrame({'group':'PSO', 'Rating_z_score': sexual_orientation['Rating_z_score']}).drop_duplicates()
# plot_voice_id_rating = pd.DataFrame({'group':'Voice Typicality', 'Rating_z_score': voice_id['Rating_z_score']}).drop_duplicates()
# plot_conditions = pd.concat([plot_sexual_orientation_rating, plot_gender_id_rating, plot_voice_id_rating])
#
# dx="group"; dy="Rating_z_score"; ort="h"; pal = sns.color_palette(n_colors=3); sigma = .2
# f, ax = plt.subplots(figsize=(7, 5))
# pt.RainCloud(x = dx, y = dy, data = plot_conditions, palette = pal, bw = sigma,
#                  width_viol = .6, ax = ax, orient = ort)
# 
# plt.title("Avg Ratings Distribution by Condition, %s Participants"%number_participants)
# # plt.show()
# plt.savefig(os.path.join(dir,'figs', 'ratingsbycondition_raincloud.png'), bbox_inches='tight', dpi=300)
# plt.close()


### F0 AVG ####create scatterplot with regression line and confidence interval lines
gender_id_avg_rating = gender_id.groupby('WAV', as_index=False)['Rating_z_score'].mean()
gender_id_avg_F0 = gender_id.groupby('WAV', as_index=False)['F0_mean'].mean()
gender_id_color = gender_id[['WAV','color']].drop_duplicates().reset_index().drop(columns='index')
gender_id_F0 = pd.merge(gender_id_avg_rating, gender_id_avg_F0, on='WAV')
gender_id_F0 = pd.merge(gender_id_F0, gender_id_color, on='WAV')

sexual_orientation_avg_rating = sexual_orientation.groupby('WAV', as_index=False)['Rating_z_score'].mean()
sexual_orientation_avg_F0 = sexual_orientation.groupby('WAV', as_index=False)['F0_mean'].mean()
sexual_orientation_color = sexual_orientation[['WAV','color']].drop_duplicates().reset_index().drop(columns='index')
sexual_orientation_F0 = pd.merge(sexual_orientation_avg_rating, sexual_orientation_avg_F0, on='WAV')
sexual_orientation_F0 = pd.merge(sexual_orientation_F0, sexual_orientation_color, on='WAV')

voice_id_rating = voice_id.groupby('WAV', as_index=False)['Rating_z_score'].mean()
voice_id_avg_F0 = voice_id.groupby('WAV', as_index=False)['F0_mean'].mean()
voice_id_color = voice_id[['WAV','color']].drop_duplicates().reset_index().drop(columns='index')
voice_id_F0 = pd.merge(voice_id_rating, voice_id_avg_F0, on='WAV')
voice_id_F0 = pd.merge(voice_id_F0, voice_id_color, on='WAV')

# plot
fig, axes = plt.subplots(1, 3)
fig.subplots_adjust(hspace=0.5)
fig.set_size_inches(16, 6)
fig.suptitle("Avg F0 by Rating, %s Participants"%number_participants, fontsize=20, fontweight='bold')
fig.subplots_adjust( top = 0.85 )

axes[0].set_title('Gender Identity')
axes[0].set_xlim(-1.5,1.5)
sns.regplot(data=gender_id_F0, x='Rating_z_score', y='F0_mean', ax=axes[0], marker = '+', scatter_kws={'facecolors':gender_id_F0['color']}, line_kws={"color": "#648FFF"})
axes[0].set_xlabel('Rating (-: Male, +: Female)')
axes[0].set_ylabel('Avg F0')

axes[1].set_title('Sexual Orientation')
axes[1].set_xlim(-1.5,1.5)
sns.regplot(data=sexual_orientation_F0, x='Rating_z_score', y='F0_mean', ax=axes[1], marker = '+', scatter_kws={'facecolors':sexual_orientation_F0['color']}, line_kws={"color": "#785EF0"})
axes[1].set_xlabel('Rating (-: Homo, +: Het)')
axes[1].set_ylabel('')

axes[2].set_title('Voice Identity')
axes[2].set_xlim(-1.5,1.5)
sns.regplot(data=voice_id_F0, x='Rating_z_score', y='F0_mean', ax=axes[2], marker = '+', scatter_kws={'facecolors':voice_id_F0['color']}, line_kws={"color": "#DC267F"})
axes[2].set_xlabel('Rating (-: Masc, +: Femme)')
axes[2].set_ylabel('')

scatter1_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c=colors['color'][0], marker = 'o')
scatter2_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c=colors['color'][1], marker = 'o')
scatter3_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c=colors['color'][2], marker = 'o')
scatter4_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c=colors['color'][3], marker = 'o')
scatter5_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c=colors['color'][4], marker = 'o')
axes[0].legend([scatter1_proxy, scatter5_proxy, scatter3_proxy, scatter4_proxy, scatter2_proxy], ['straight men','queer men','queer NB, men and women','queer women','straight women'], numpoints = 1, loc = 'upper left')


# plt.show()
plt.savefig(os.path.join(dir,'figs', 'F0_avgbycondition.png'), bbox_inches='tight', dpi=300)
plt.close()

# overlay
fig, axes = plt.subplots()
fig.set_size_inches(18, 10)
axes.set_title('Avg F0 by Condition Rating, %s Participants'%number_participants, fontsize=20, fontweight='bold')
axes.set_xlim(-1.5,1.5)
sns.regplot(data=gender_id_F0, x='Rating_z_score', y='F0_mean', marker = '+', scatter_kws={'facecolors':gender_id_F0['color']}, line_kws={"color": "#648FFF"})
sns.regplot(data=sexual_orientation_F0, x='Rating_z_score', y='F0_mean', marker = '+', scatter_kws={'facecolors':sexual_orientation_F0['color']}, line_kws={"color": "#785EF0"})
sns.regplot(data=voice_id_F0, x='Rating_z_score', y='F0_mean', marker = '+', scatter_kws={'facecolors':voice_id_F0['color']}, line_kws={"color": "#DC267F"})
axes.set_xlabel('Rating (-1.5-1.5)')
axes.set_ylabel('Avg F0')
scatter1_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c=colors['color'][0], marker = 'o')
scatter2_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c=colors['color'][1], marker = 'o')
scatter3_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c=colors['color'][2], marker = 'o')
scatter4_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c=colors['color'][3], marker = 'o')
scatter5_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c=colors['color'][4], marker = 'o')
axes.legend([scatter1_proxy, scatter5_proxy, scatter3_proxy, scatter4_proxy, scatter2_proxy], ['straight men','queer men','queer NB, men and women','queer women','straight women'], numpoints = 1, loc = 'upper left')
plt.savefig(os.path.join(dir,'figs', 'F0_avgbycondition_overlay.png'), bbox_inches='tight', dpi=300)
plt.close()
# plt.show()

##### regression indiv and overlay loop over all features ########
# need to add in the number of WAV files for each since there's variability
vowel_labels = ['AA','AE','AH','AO','AW','AX','AY','EH','EY','IH','IY','OW','OY','UH','UW']
consonant_labels = ['AR','B','CH','D','DH','EL','ER','F','G','H','HH','JH','K','L','M','N','NG','P','R','S','SH','T','TH','V','W','Y','Z','ZH']
formant_bandwidth_label = ['F1','F2','F3','F4']
vowel_spectral_names = []
vowel_spectral_names_mean = []
vowel_spectral_names_min = []
vowel_spectral_names_max = []
vowel_spectral_names_dist = []
consonant_spectral_names = [] # without the mean, min, max suffix
for vowel in vowel_labels: # loop for making a list of the vowel spectral features for each vowel--matches columns in spreadsheet
    for fblabel in formant_bandwidth_label:
        # concatenate strings with '_'
        vowel_string = vowel + "_s" + fblabel + "_mean"
        # append to list
        vowel_spectral_names.append(vowel_string)
        vowel_spectral_names_mean.append(vowel_string)
        vowel_string = vowel + "_s" + fblabel + "_min"
        vowel_spectral_names.append(vowel_string)
        vowel_spectral_names_min.append(vowel_string)
        vowel_string = vowel + "_s" + fblabel + "_max"
        vowel_spectral_names.append(vowel_string)
        vowel_spectral_names_max.append(vowel_string)
        vowel_string = vowel + "_s" + fblabel + "_mean_dist"
        vowel_spectral_names.append(vowel_string)
        vowel_spectral_names_dist.append(vowel_string)


for consonant in consonant_labels: # loop for making a list of the vowel spectral features for each vowel--matches columns in spreadsheet
    for fblabel in formant_bandwidth_label:
        # concatenate strings with '_'
        consonant_string = consonant + "_s" + fblabel + "_mean"
        # append to list
        consonant_spectral_names.append(consonant_string)
        consonant_string = consonant + "_s" + fblabel + "_min"
        consonant_spectral_names.append(consonant_string)
        consonant_string = consonant + "_s" + fblabel + "_max"
        consonant_spectral_names.append(consonant_string)


features_to_plot = ['F0_mean','F0_range','F0_std','spectral_S_duration','spectral_S_intensity','spectral_S_cog','spectral_S_sdev', 'spectral_S_skew','spectral_S_kurt',
                                                    'spectral_Z_duration','spectral_Z_intensity','spectral_Z_cog','spectral_Z_sdev', 'spectral_Z_skew','spectral_Z_kurt',
                                                    'spectral_F_duration','spectral_F_intensity','spectral_F_cog','spectral_F_sdev', 'spectral_F_skew','spectral_F_kurt',
                                                    'spectral_V_duration','spectral_V_intensity','spectral_V_cog','spectral_V_sdev', 'spectral_V_skew','spectral_V_kurt',
                                                    'spectral_SH_duration','spectral_SH_intensity','spectral_SH_cog','spectral_SH_sdev', 'spectral_SH_skew','spectral_SH_kurt',
                                                    'spectral_JH_duration','spectral_JH_intensity','spectral_JH_cog','spectral_JH_sdev', 'spectral_JH_skew','spectral_JH_kurt',
                                                    'percent_creak','vowel_avg_dur','rando_baseline_z_score']

features_to_plot = features_to_plot + vowel_spectral_names

for feature in features_to_plot:
    gender_id_avg_rating = gender_id.groupby('WAV', as_index=False)['Rating_z_score'].mean()
    gender_id_avg_feature = gender_id.groupby('WAV', as_index=False)[feature].mean()
    gender_id_color = gender_id[['WAV','color']].drop_duplicates().reset_index().drop(columns='index')
    gender_id_feature = pd.merge(gender_id_avg_rating, gender_id_avg_feature, on='WAV')
    gender_id_feature = pd.merge(gender_id_feature, gender_id_color, on='WAV').dropna()

    sexual_orientation_avg_rating = sexual_orientation.groupby('WAV', as_index=False)['Rating_z_score'].mean()
    sexual_orientation_avg_feature = sexual_orientation.groupby('WAV', as_index=False)[feature].mean()
    sexual_orientation_color = sexual_orientation[['WAV','color']].drop_duplicates().reset_index().drop(columns='index')
    sexual_orientation_feature = pd.merge(sexual_orientation_avg_rating, sexual_orientation_avg_feature, on='WAV')
    sexual_orientation_feature = pd.merge(sexual_orientation_feature, sexual_orientation_color, on='WAV').dropna()

    voice_id_rating = voice_id.groupby('WAV', as_index=False)['Rating_z_score'].mean()
    voice_id_avg_feature = voice_id.groupby('WAV', as_index=False)[feature].mean()
    voice_id_color = voice_id[['WAV','color']].drop_duplicates().reset_index().drop(columns='index')
    voice_id_feature = pd.merge(voice_id_rating, voice_id_avg_feature, on='WAV')
    voice_id_feature = pd.merge(voice_id_feature, voice_id_color, on='WAV').dropna()

    # assert gender_id_feature[feature].nunique() == sexual_orientation_feature[feature].nunique() == voice_id_feature[feature].nunique()
    feature_number = gender_id_feature[feature].nunique()

    fig, axes = plt.subplots(1, 3)
    fig.subplots_adjust(hspace=0.5)
    fig.set_size_inches(16, 6)
    fig.suptitle("(Avg) %s by Rating, %s Tokens, %s Participants"%(feature, feature_number, number_participants), fontsize=20, fontweight='bold')
    fig.subplots_adjust( top = 0.85 )

    axes[0].set_title('Gender Identity')
    axes[0].set_xlim(-1.5,1.5)
    sns.regplot(data=gender_id_feature, x='Rating_z_score', y=feature, ax=axes[0], marker = '+', scatter_kws={'facecolors':gender_id_feature['color']}, line_kws={"color": "#648FFF"}) # #648FFF d55e00
    axes[0].set_xlabel('Rating (-: Male, +: Female)')
    axes[0].set_ylabel('Avg %s'%feature)

    axes[1].set_title('Sexual Orientation')
    axes[1].set_xlim(-1.5,1.5)
    sns.regplot(data=sexual_orientation_feature, x='Rating_z_score', y=feature, ax=axes[1], marker = '+', scatter_kws={'facecolors':sexual_orientation_feature['color']}, line_kws={"color": "#785EF0"}) # #785EF0 0072b2
    axes[1].set_xlabel('Rating (-: Homo, +: Het)')
    axes[1].set_ylabel('')

    axes[2].set_title('Voice Identity')
    axes[2].set_xlim(-1.5,1.5)
    sns.regplot(data=voice_id_feature, x='Rating_z_score', y=feature, ax=axes[2], marker = '+', scatter_kws={'facecolors':voice_id_feature['color']}, line_kws={"color": "#DC267F"}) # #DC267F 009e73
    axes[2].set_xlabel('Rating (-: Masc, +: Femme)')
    axes[2].set_ylabel('')

    scatter1_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c=colors['color'][0], marker = 'o')
    scatter2_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c=colors['color'][1], marker = 'o')
    scatter3_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c=colors['color'][2], marker = 'o')
    scatter4_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c=colors['color'][3], marker = 'o')
    scatter5_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c=colors['color'][4], marker = 'o')
    axes[0].legend([scatter1_proxy, scatter5_proxy, scatter3_proxy, scatter4_proxy, scatter2_proxy], ['straight men','queer men','queer NB, men and women','queer women','straight women'], numpoints = 1, loc = 'upper left')


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
fig.suptitle("Rating by Proximity to LGBTQ+ Community, %s Participants"%number_participants, fontsize=20, fontweight='bold')
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
plt.close()
