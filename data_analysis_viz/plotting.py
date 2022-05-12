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
import scipy.stats as stats
import matplotlib
import matplotlib.pyplot as plt
import string
np.set_printoptions(threshold=sys.maxsize)

# set up directory and read in csv
dir = '/Users/bcl/GitHub/queer_speech'
fig_dir = '/Volumes/GoogleDrive/My Drive/Comps/figs'
os.chdir(dir)
ratings_fname = os.path.join(dir, 'data_analysis_viz','queer_data.csv')
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
# colors = pd.DataFrame({"color_5_cluster": ['#FFB000','#785EF0','#648FFF','#FE6100','#DC267F']})
# colors['kmeans_5_cluster'] = [0,1,2,3,4]
ratings_all['percent_creak'] = ratings_all['percent_creak'].replace(0, np.nan)
gender_id = ratings_all[ratings_all['Condition']=='gender_id']
sexual_orientation = ratings_all[ratings_all['Condition']=='sexual_orientation']
voice_id = ratings_all[ratings_all['Condition']=='voice_id']
number_participants = ratings_all['Participant'].nunique()

### correlation between PGI and PGV
stats.pearsonr(gender_id['Rating_z_score'], voice_id['Rating_z_score'])


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
plt.savefig(os.path.join(fig_dir, 'F0_raincloud.png'), bbox_inches='tight', dpi=300)
plt.close()

##### regression indiv loop over all features ########
# need to add in the number of WAV files for each since there's variability
cluster_colors = ['color_3_cluster','color_4_cluster','color_5_cluster']
vowel_labels = ['AA','AE','AH','AO','AX','EH','IH','IY','UH','UW']
diph_list = ['AW','AY','EY','OW','OY']
consonant_labels = ['AR','B','CH','D','DH','EL','ER','F','G','H','HH','JH','K','L','M','N','NG','P','R','S','SH','T','TH','V','W','Y','Z','ZH']
formant_bandwidth_label = ['F1','F2','F3','F4']
vowel_spectral_names = []
vowel_spectral_names_mean = []
vowel_spectral_names_min = []
vowel_spectral_names_max = []
vowel_spectral_names_dist = []
consonant_spectral_names = [] # without the mean, min, max suffix
diph_spectral_names = []
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

split_list = ['_first','_third']
for diph in diph_list:
    for split in split_list: # loop for making a list of the vowel spectral features for each vowel--matches columns in spreadsheet
        for fblabel in formant_bandwidth_label:
            # concatenate strings with '_'
            diph_string = diph + "_s" + fblabel + "_mean" + split
            # append to list
            diph_spectral_names.append(diph_string)
            diph_string = diph + "_s" + fblabel + "_min" + split
            diph_spectral_names.append(diph_string)
            diph_string = diph + "_s" + fblabel + "_max" + split
            diph_spectral_names.append(diph_string)



features_to_plot = ['F0_mean','F0_range','F0_std','spectral_S_duration','spectral_S_intensity','spectral_S_cog','spectral_S_sdev', 'spectral_S_skew','spectral_S_kurt',
                                                    'spectral_Z_duration','spectral_Z_intensity','spectral_Z_cog','spectral_Z_sdev', 'spectral_Z_skew','spectral_Z_kurt',
                                                    'spectral_F_duration','spectral_F_intensity','spectral_F_cog','spectral_F_sdev', 'spectral_F_skew','spectral_F_kurt',
                                                    'spectral_V_duration','spectral_V_intensity','spectral_V_cog','spectral_V_sdev', 'spectral_V_skew','spectral_V_kurt',
                                                    'spectral_SH_duration','spectral_SH_intensity','spectral_SH_cog','spectral_SH_sdev', 'spectral_SH_skew','spectral_SH_kurt',
                                                    'spectral_JH_duration','spectral_JH_intensity','spectral_JH_cog','spectral_JH_sdev', 'spectral_JH_skew','spectral_JH_kurt',
                                                    'spectral_TH_duration','spectral_TH_intensity','spectral_TH_cog','spectral_TH_sdev', 'spectral_TH_skew','spectral_TH_kurt',
                                                    'spectral_DH_duration','spectral_DH_intensity','spectral_DH_cog','spectral_DH_sdev', 'spectral_DH_skew','spectral_DH_kurt',
                                                    'percent_creak','vowel_avg_dur','rando_baseline_z_score']

features_to_plot = features_to_plot + vowel_spectral_names + diph_spectral_names

cluster_color_dict = {'kmeans_3_cluster':'color_3_cluster', 'kmeans_4_cluster':'color_4_cluster', 'kmeans_5_cluster':'color_5_cluster'}

for feature in features_to_plot:
    for cluster, color in cluster_color_dict.items():
        gender_id_avg_rating = gender_id.groupby('WAV', as_index=False)['Rating_z_score'].mean()
        gender_id_avg_feature = gender_id.groupby('WAV', as_index=False)[feature].mean()
        gender_id_cluster = gender_id.groupby('WAV', as_index=False)[cluster].mean()

        gender_id_color = gender_id[['WAV',color]].drop_duplicates().reset_index().drop(columns='index')
        gender_id_feature = pd.merge(gender_id_avg_rating, gender_id_avg_feature, on='WAV')
        gender_id_feature = pd.merge(gender_id_feature, gender_id_color, on='WAV').dropna()
        gender_id_feature = pd.merge(gender_id_feature, gender_id_cluster, on='WAV')


        sexual_orientation_avg_rating = sexual_orientation.groupby('WAV', as_index=False)['Rating_z_score'].mean()
        sexual_orientation_avg_feature = sexual_orientation.groupby('WAV', as_index=False)[feature].mean()
        sexual_orientation_cluster = sexual_orientation.groupby('WAV', as_index=False)[cluster].mean()

        sexual_orientation_color = sexual_orientation[['WAV',color]].drop_duplicates().reset_index().drop(columns='index')
        sexual_orientation_feature = pd.merge(sexual_orientation_avg_rating, sexual_orientation_avg_feature, on='WAV')
        sexual_orientation_feature = pd.merge(sexual_orientation_feature, sexual_orientation_color, on='WAV').dropna()
        sexual_orientation_feature = pd.merge(sexual_orientation_feature, sexual_orientation_cluster, on='WAV')


        voice_id_rating = voice_id.groupby('WAV', as_index=False)['Rating_z_score'].mean()
        voice_id_avg_feature = voice_id.groupby('WAV', as_index=False)[feature].mean()
        voice_id_cluster = voice_id.groupby('WAV', as_index=False)[cluster].mean()

        voice_id_color = voice_id[['WAV',color]].drop_duplicates().reset_index().drop(columns='index')
        voice_id_feature = pd.merge(voice_id_rating, voice_id_avg_feature, on='WAV')
        voice_id_feature = pd.merge(voice_id_feature, voice_id_color, on='WAV').dropna()
        voice_id_feature = pd.merge(voice_id_feature, voice_id_cluster, on='WAV')

        if cluster == 'kmeans_3_cluster':
            color_dict = {0:'#785EF0', 1:'#DC267F', 2:'#648FFF'}
            markers_dict = {0: "P", 1: "o", 2: "v"}
        if cluster == 'kmeans_4_cluster':
            color_dict = {0:'#DC267F', 1:'#785EF0', 2:'#648FFF', 3:'#FFB000'}
            markers_dict = {0: "o", 1: "P", 2: "v", 3: "D"}
        if cluster == 'kmeans_5_cluster':
            color_dict = {0:'#FFB000', 1:'#785EF0', 2:'#648FFF', 3:'#FE6100', 4:'#DC267F'}
            markers_dict = {0: "D", 1: "P", 2: "v", 3: "H", 4: "o"}

        # assert gender_id_feature[feature].nunique() == sexual_orientation_feature[feature].nunique() == voice_id_feature[feature].nunique()
        feature_number = gender_id_feature[feature].nunique()

        fig, axes = plt.subplots(1, 3)
        fig.subplots_adjust(hspace=0.5)
        fig.set_size_inches(16, 6)
        fig.suptitle("(Avg) %s by Rating, %s Tokens, %s Participants"%(feature, feature_number, number_participants), fontsize=20, fontweight='bold')
        fig.subplots_adjust( top = 0.85 )

        axes[0].set_title('Gender Identity')
        axes[0].set_xlim(-1.5,1.5)
        sns.scatterplot(data=gender_id_feature, x='Rating_z_score', y=feature, ax=axes[0], hue = gender_id_feature[cluster], style = gender_id_feature[cluster], palette = color_dict, markers=markers_dict)
        sns.regplot(data=gender_id_feature, x='Rating_z_score', y=feature, ax=axes[0], scatter = False, line_kws={"color": "k"}) # #648FFF d55e00
        axes[0].set_xlabel('Rating (-: Male, +: Female)')
        axes[0].set_ylabel('Avg %s'%feature)

        axes[1].set_title('Sexual Orientation')
        axes[1].set_xlim(-1.5,1.5)
        sns.scatterplot(data=sexual_orientation_feature, x='Rating_z_score', y=feature, ax=axes[1], hue = sexual_orientation_feature[cluster], style = sexual_orientation_feature[cluster], palette = color_dict,  markers=markers_dict)
        sns.regplot(data=sexual_orientation_feature, x='Rating_z_score', y=feature, ax=axes[1], scatter = False, line_kws={"color": "k"}) # #785EF0 0072b2
        axes[1].set_xlabel('Rating (-: Homo, +: Het)')
        axes[1].set_ylabel('')

        axes[2].set_title('Voice Identity')
        axes[2].set_xlim(-1.5,1.5)
        sns.scatterplot(data=voice_id_feature, x='Rating_z_score', y=feature, ax=axes[2], hue = voice_id_feature[cluster], style = voice_id_feature[cluster], palette = color_dict,  markers=markers_dict)
        sns.regplot(data=voice_id_feature, x='Rating_z_score', y=feature, ax=axes[2], scatter = False, line_kws={"color": "k"}) # #DC267F 009e73
        axes[2].set_xlabel('Rating (-: Masc, +: Femme)')
        axes[2].set_ylabel('')

        if color == 'color_3_cluster':
            colors = pd.DataFrame({"color_3_cluster": ['#785EF0','#DC267F','#648FFF']})
            colors['kmeans_3_cluster'] = [0,1,2]
            scatter1_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c=colors['color_3_cluster'][0], marker = 'P') #^
            scatter2_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c=colors['color_3_cluster'][1], marker = 'o') #p
            scatter3_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c=colors['color_3_cluster'][2], marker = 'v') #+
            axes[0].legend([scatter3_proxy, scatter2_proxy, scatter1_proxy], ['SM','QE','SW'], numpoints = 1, loc = 'upper left')
            axes[1].get_legend().remove()
            axes[2].get_legend().remove()

            # plt.show()
            plt.savefig(os.path.join(fig_dir, '3_cluster', '%s_avgbycondition_3_clusters.png'%feature), bbox_inches='tight', dpi=300)
            plt.close()

        if color == 'color_4_cluster':
            colors = pd.DataFrame({"color_4_cluster": ['#DC267F','#785EF0','#648FFF','#FFB000']})
            colors['kmeans_4_cluster'] = [0,1,2,3]
            scatter1_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c=colors['color_4_cluster'][0], marker = 'o')
            scatter2_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c=colors['color_4_cluster'][1], marker = 'P')
            scatter3_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c=colors['color_4_cluster'][2], marker = 'v')
            scatter4_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c=colors['color_4_cluster'][3], marker = 'D')
            axes[0].legend([scatter3_proxy, scatter4_proxy, scatter1_proxy, scatter2_proxy], ['SM','QM','QE','SW'], numpoints = 1, loc = 'upper left')
            axes[1].get_legend().remove()
            axes[2].get_legend().remove()


            # plt.show()
            plt.savefig(os.path.join(fig_dir, '4_cluster', '%s_avgbycondition_4_clusters.png'%feature), bbox_inches='tight', dpi=300)
            plt.close()


        if color == 'color_5_cluster':
            colors = pd.DataFrame({"color_5_cluster": ['#FFB000','#785EF0','#648FFF','#FE6100','#DC267F']})
            colors['kmeans_5_cluster'] = [0,1,2,3,4]
            scatter1_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c=colors['color_5_cluster'][0], marker = 'D')
            scatter2_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c=colors['color_5_cluster'][1], marker = 'P')
            scatter3_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c=colors['color_5_cluster'][2], marker = 'v')
            scatter4_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c=colors['color_5_cluster'][3], marker = 'H')
            scatter5_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c=colors['color_5_cluster'][4], marker = 'o')
            axes[0].legend([scatter3_proxy, scatter1_proxy, scatter5_proxy, scatter4_proxy, scatter2_proxy], ['SM','QM','QE','QW','SW'], numpoints = 1, loc = 'upper left')
            axes[1].get_legend().remove()
            axes[2].get_legend().remove()


            # plt.show()
            plt.savefig(os.path.join(fig_dir, '5_cluster', '%s_avgbycondition_5_clusters.png'%feature), bbox_inches='tight', dpi=300)
            plt.close()

#### F0 for 3, 4, 5 clusters in same graph  for PS ####

feature= 'F0_mean'
feature_number = sexual_orientation['F0_mean'].nunique()

sexual_orientation_avg_rating = sexual_orientation.groupby('WAV', as_index=False)['Rating_z_score'].mean()
sexual_orientation_avg_feature = sexual_orientation.groupby('WAV', as_index=False)[feature].mean()
sexual_orientation_cluster_3 = sexual_orientation.groupby('WAV', as_index=False)['kmeans_3_cluster'].mean()
sexual_orientation_cluster_4 = sexual_orientation.groupby('WAV', as_index=False)['kmeans_4_cluster'].mean()
sexual_orientation_cluster_5 = sexual_orientation.groupby('WAV', as_index=False)['kmeans_5_cluster'].mean()


sexual_orientation_color_3 = sexual_orientation[['WAV','color_3_cluster']].drop_duplicates().reset_index().drop(columns='index')
sexual_orientation_color_4 = sexual_orientation[['WAV','color_4_cluster']].drop_duplicates().reset_index().drop(columns='index')
sexual_orientation_color_5 = sexual_orientation[['WAV','color_5_cluster']].drop_duplicates().reset_index().drop(columns='index')

sexual_orientation_feature = pd.merge(sexual_orientation_avg_rating, sexual_orientation_avg_feature, on='WAV')
sexual_orientation_feature = pd.merge(sexual_orientation_feature, sexual_orientation_color_3, on='WAV').dropna()
sexual_orientation_feature = pd.merge(sexual_orientation_feature, sexual_orientation_color_4, on='WAV').dropna()
sexual_orientation_feature = pd.merge(sexual_orientation_feature, sexual_orientation_color_5, on='WAV').dropna()

sexual_orientation_feature = pd.merge(sexual_orientation_feature, sexual_orientation_cluster_3, on='WAV')
sexual_orientation_feature = pd.merge(sexual_orientation_feature, sexual_orientation_cluster_4, on='WAV')
sexual_orientation_feature = pd.merge(sexual_orientation_feature, sexual_orientation_cluster_5, on='WAV')

color_dict_3 = {0:'#785EF0', 1:'#DC267F', 2:'#648FFF'}
markers_dict_3 = {0: "P", 1: "o", 2: "v"}
color_dict_4 = {0:'#DC267F', 1:'#785EF0', 2:'#648FFF', 3:'#FFB000'}
markers_dict_4 = {0: "o", 1: "P", 2: "v", 3: "D"}
color_dict_5 = {0:'#FFB000', 1:'#785EF0', 2:'#648FFF', 3:'#FE6100', 4:'#DC267F'}
markers_dict_5 = {0: "D", 1: "P", 2: "v", 3: "H", 4: "o"}

fig, axes = plt.subplots(1, 3)
fig.subplots_adjust(hspace=0.5)
fig.set_size_inches(16, 6)
fig.suptitle("Mean f0 by PS Rating for 3, 4 and 5 Clusters", fontsize=20, fontweight='bold')
fig.subplots_adjust( top = 0.85 )

axes[0].set_title('Sexual Orientation (3 Clusters)')
axes[0].set_xlim(-1.5,1.5)
sns.scatterplot(data=sexual_orientation_feature, x='Rating_z_score', y='F0_mean', ax=axes[0], hue = sexual_orientation_feature['kmeans_3_cluster'], style = sexual_orientation_feature['kmeans_3_cluster'], palette = color_dict_3,  markers=markers_dict_3)
sns.regplot(data=sexual_orientation_feature, x='Rating_z_score', y='F0_mean', ax=axes[0], scatter = False, line_kws={"color": "k"}) # #785EF0 0072b2
axes[0].set_xlabel('Rating (-: Homo, +: Het)')
axes[0].set_ylabel('Mean f0')

axes[1].set_title('Sexual Orientation (4 Clusters)')
axes[1].set_xlim(-1.5,1.5)
sns.scatterplot(data=sexual_orientation_feature, x='Rating_z_score', y='F0_mean', ax=axes[1], hue = sexual_orientation_feature['kmeans_4_cluster'], style = sexual_orientation_feature['kmeans_4_cluster'], palette = color_dict_4,  markers=markers_dict_4)
sns.regplot(data=sexual_orientation_feature, x='Rating_z_score', y='F0_mean', ax=axes[1], scatter = False, line_kws={"color": "k"}) # #785EF0 0072b2
axes[1].set_xlabel('Rating (-: Homo, +: Het)')
axes[1].set_ylabel(' ')

axes[2].set_title('Sexual Orientation (5 Clusters)')
axes[2].set_xlim(-1.5,1.5)
sns.scatterplot(data=sexual_orientation_feature, x='Rating_z_score', y='F0_mean', ax=axes[2], hue = sexual_orientation_feature['kmeans_5_cluster'], style = sexual_orientation_feature['kmeans_5_cluster'], palette = color_dict_5,  markers=markers_dict_5)
sns.regplot(data=sexual_orientation_feature, x='Rating_z_score', y='F0_mean', ax=axes[2], scatter = False, line_kws={"color": "k"}) # #785EF0 0072b2
axes[2].set_xlabel('Rating (-: Homo, +: Het)')
axes[2].set_ylabel(' ')

colors = pd.DataFrame({"color_3_cluster": ['#785EF0','#DC267F','#648FFF']})
colors['kmeans_3_cluster'] = [0,1,2]
scatter1_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c=colors['color_3_cluster'][0], marker = 'P') #^
scatter2_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c=colors['color_3_cluster'][1], marker = 'o') #p
scatter3_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c=colors['color_3_cluster'][2], marker = 'v') #+
axes[0].legend([scatter3_proxy, scatter2_proxy, scatter1_proxy], ['SM','QE','SW'], numpoints = 1, loc = 'upper left', size = 6)

colors = pd.DataFrame({"color_4_cluster": ['#DC267F','#785EF0','#648FFF','#FFB000']})
colors['kmeans_4_cluster'] = [0,1,2,3]
scatter1_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c=colors['color_4_cluster'][0], marker = 'o')
scatter2_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c=colors['color_4_cluster'][1], marker = 'P')
scatter3_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c=colors['color_4_cluster'][2], marker = 'v')
scatter4_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c=colors['color_4_cluster'][3], marker = 'D')
axes[1].legend([scatter3_proxy, scatter4_proxy, scatter1_proxy, scatter2_proxy], ['SM','QM','QE','SW'], numpoints = 1, loc = 'upper left', size = 6)

colors = pd.DataFrame({"color_5_cluster": ['#FFB000','#785EF0','#648FFF','#FE6100','#DC267F']})
colors['kmeans_5_cluster'] = [0,1,2,3,4]
scatter1_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c=colors['color_5_cluster'][0], marker = 'D')
scatter2_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c=colors['color_5_cluster'][1], marker = 'P')
scatter3_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c=colors['color_5_cluster'][2], marker = 'v')
scatter4_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c=colors['color_5_cluster'][3], marker = 'H')
scatter5_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c=colors['color_5_cluster'][4], marker = 'o')
axes[2].legend([scatter3_proxy, scatter1_proxy, scatter5_proxy, scatter4_proxy, scatter2_proxy], ['SM','QM','QE','QW','SW'], numpoints = 1, loc = 'upper left', size = 6)

        # plt.show()
plt.savefig(os.path.join(fig_dir, 'F0_cluster_comparison.png'), bbox_inches='tight', dpi=300)
plt.close()


### PC  + FEATURE ###
pca_fname = os.path.join(dir, 'data_analysis_viz','principal_components.csv')
pca0_fname = os.path.join(dir, 'data_analysis_viz','principal_components_cluster_0.csv')
pca1_fname = os.path.join(dir, 'data_analysis_viz','principal_components_cluster_1.csv')
pca2_fname = os.path.join(dir, 'data_analysis_viz','principal_components_cluster_2.csv')
pca3_fname = os.path.join(dir, 'data_analysis_viz','principal_components_cluster_3.csv')
pca4_fname = os.path.join(dir, 'data_analysis_viz','principal_components_cluster_4.csv')
corr_fname = os.path.join(dir, 'data_analysis_viz','grand_corr_r.csv')
corr_gender_id_fname = os.path.join(dir, 'data_analysis_viz','pc_corr_r_gender_id.csv')
corr_sexual_orientation_fname = os.path.join(dir, 'data_analysis_viz','pc_corr_r_sexual_orientation.csv')
corr_voice_id_fname = os.path.join(dir, 'data_analysis_viz','pc_corr_r_voice_id.csv')
corr_p_gender_id_fname = os.path.join(dir, 'data_analysis_viz','pc_corr_p_gender_id.csv')
corr_p_sexual_orientation_fname = os.path.join(dir, 'data_analysis_viz','pc_corr_p_sexual_orientation.csv')
corr_p_voice_id_fname = os.path.join(dir, 'data_analysis_viz','pc_corr_p_voice_id.csv')
grand_ablated_df_fname = os.path.join(dir, 'data_analysis_viz', 'grand_ablated_df.csv')

pca = pd.read_csv(pca_fname)
pca0 = pd.read_csv(pca0_fname)
pca1 = pd.read_csv(pca1_fname)
pca2 = pd.read_csv(pca2_fname)
pca3 = pd.read_csv(pca3_fname)
pca4 = pd.read_csv(pca4_fname)
corr_df = pd.read_csv(corr_fname)
corr_df = corr_df.drop('Unnamed: 0', axis=1) # might need to change column name when it imports in, there's a random addition at the beginning
corr_gender_id = pd.read_csv(corr_gender_id_fname)
corr_gender_id = corr_gender_id.drop('Unnamed: 0', axis=1) # might need to change column name when it imports in, there's a random addition at the beginning
corr_sexual_orientation = pd.read_csv(corr_sexual_orientation_fname)
corr_sexual_orientation = corr_sexual_orientation.drop('Unnamed: 0', axis=1) # might need to change column name when it imports in, there's a random addition at the beginning
corr_voice_id = pd.read_csv(corr_voice_id_fname)
corr_voice_id = corr_voice_id.drop('Unnamed: 0', axis=1) # might need to change column name when it imports in, there's a random addition at the beginning

corr_p_gender_id = pd.read_csv(corr_p_gender_id_fname)
corr_p_gender_id = corr_p_gender_id.drop('Unnamed: 0', axis=1) # might need to change column name when it imports in, there's a random addition at the beginning
corr_p_sexual_orientation = pd.read_csv(corr_p_sexual_orientation_fname)
corr_p_sexual_orientation = corr_p_sexual_orientation.drop('Unnamed: 0', axis=1) # might need to change column name when it imports in, there's a random addition at the beginning
corr_p_voice_id = pd.read_csv(corr_p_voice_id_fname)
corr_p_voice_id = corr_p_voice_id.drop('Unnamed: 0', axis=1) # might need to change column name when it imports in, there's a random addition at the beginning



# grand_ablated_df = pd.read_csv(grand_ablated_df_fname)
# grand_ablated_df = grand_ablated_df.drop('Unnamed: 0', axis=1) # might need to change column name when it imports in, there's a random addition at the beginning

### feature by condition correlations ###
ratings_temp = ratings_all[['Rating','kmeans_4_cluster','kmeans_3_cluster','3_rando_classes','4_rando_classes','participant_gender_id','participant_sexual_orientation','participant_voice_id','participant_cis_trans',
                'participant_prox_social','participant_prox_affiliation', 'participant_prox_media', 'participant_race','participant_race_hispanic','eng_primary_early','eng_primary_current',
                'participant_other_langs','participant_race_free_response','participant_gender_pso_free_response', 'participant_age', 'deaf_hoh', 'Participant',
                'survey_experience','survey_feedback','Condition','WAV','color_3_cluster','color_4_cluster','color_5_cluster','spectral_S_start','spectral_Z_start','spectral_F_start','spectral_V_start','spectral_JH_start','spectral_SH_start']]

ratings_nums = ratings_all.drop(['Rating','kmeans_4_cluster','kmeans_3_cluster','3_rando_classes','4_rando_classes','participant_gender_id','participant_sexual_orientation','participant_voice_id','participant_cis_trans',
                'participant_prox_social','participant_prox_affiliation', 'participant_prox_media', 'participant_race','participant_race_hispanic','eng_primary_early','eng_primary_current',
                'participant_other_langs','participant_race_free_response','participant_gender_pso_free_response', 'participant_age', 'deaf_hoh', 'Participant',
                'survey_experience','survey_feedback','Condition','WAV','color_3_cluster','color_4_cluster','color_5_cluster','spectral_S_start','spectral_Z_start','spectral_F_start','spectral_V_start','spectral_JH_start','spectral_SH_start',
                'S_avg_dur','Z_avg_dur','F_avg_dur','V_avg_dur','JH_avg_dur','SH_avg_dur'], axis=1)

## impute for the cross validation, random forest needs values in each, below we are creating one large df with imputed column means from teh entire set, and then 5 cluster groups with the gran imputed means
from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp = imp.fit(ratings_nums)
ratings_imp = imp.transform(ratings_nums) # this is the big df
ratings_imp = pd.DataFrame(ratings_imp, columns = ratings_nums.columns)

ratings_all = pd.concat([ratings_imp, ratings_temp], axis=1)
gender_id = ratings_all[ratings_all['Condition']=='gender_id']
sexual_orientation = ratings_all[ratings_all['Condition']=='sexual_orientation']
voice_id = ratings_all[ratings_all['Condition']=='voice_id']

condition_dict = {'all': ratings_all, 'gender_id': gender_id, 'sexual_orientation': sexual_orientation, 'voice_id': voice_id}

specific_features_to_plot = ['F0_mean','IH_sF1_mean','IH_sF2_mean','IH_sF3_mean','IH_sF4_mean','IH_sF1_mean_dist', 'IH_sF2_mean_dist', 'IH_sF3_mean_dist', 'IH_sF4_mean_dist',
                    'AY_sF1_mean_first', 'AY_sF2_mean_first', 'AY_sF3_mean_first', 'AY_sF4_mean_first', 'AY_sF1_mean_third', 'AY_sF2_mean_third', 'AY_sF3_mean_third', 'AY_sF4_mean_third',
                    'vowel_avg_dur','percent_creak','spectral_S_cog','spectral_S_duration','spectral_S_skew']


corr_r = pd.DataFrame({'split':['all','gender_id','sexual_orientation','voice_id']})
corr_p = pd.DataFrame({'split':['all','gender_id','sexual_orientation','voice_id']})
for feature in specific_features_to_plot:
    corr_list_r = []
    corr_list_p = []
    for condition, df in condition_dict.items():
        pc_correlation_r = stats.pearsonr(df['Rating_z_score'], df[feature])[0]
        pc_correlation_p = stats.pearsonr(df['Rating_z_score'], df[feature])[1]
        corr_list_r.append(pc_correlation_r)
        corr_list_p.append(pc_correlation_p)
    single_feature_corr_r = pd.DataFrame(corr_list_r, columns={'%s'%(feature)})
    single_feature_corr_p = pd.DataFrame(corr_list_p, columns={'%s'%(feature)})
    corr_r = pd.concat([corr_r, single_feature_corr_r], axis=1)
    corr_p = pd.concat([corr_p, single_feature_corr_p], axis=1)

corr_r.to_csv(os.path.join(dir,'data_analysis_viz','grand_corr_r_raw_features.csv'), index=True, encoding='utf-8')
corr_p.to_csv(os.path.join(dir,'data_analysis_viz','grand_corr_p_raw_features.csv'), index=True, encoding='utf-8')


specific_features_to_plot = ['F0_mean','IH_sF1_mean','IH_sF2_mean','IH_sF3_mean','IH_sF4_mean','IH_sF1_mean_dist', 'IH_sF2_mean_dist', 'IH_sF3_mean_dist', 'IH_sF4_mean_dist',
                    'AY_sF1_mean_first', 'AY_sF2_mean_first', 'AY_sF3_mean_first', 'AY_sF4_mean_first', 'AY_sF1_mean_third', 'AY_sF2_mean_third', 'AY_sF3_mean_third', 'AY_sF4_mean_third',
                    'vowel_avg_dur','percent_creak','spectral_S_cog','spectral_S_duration','spectral_S_skew']

# pc_list = ['Principal Component 1', 'Principal Component 2',
#        'Principal Component 3', 'Principal Component 4',
#        'Principal Component 5', 'Principal Component 6',
#        'Principal Component 7', 'Principal Component 8',
#        'Principal Component 9', 'Principal Component 10']

pc_list = ['Principal Component 2', 'Principal Component 4']

corr_best = corr_df[specific_features_to_plot]
components = corr_df['PC']
corr_best = pd.concat([components,corr_best], axis=1)

## r values
corr_best_gender_id = corr_gender_id[specific_features_to_plot]
components = corr_gender_id['PC']
corr_best_gender_id = pd.concat([components,corr_best_gender_id], axis=1)

corr_best_sexual_orientation = corr_sexual_orientation[specific_features_to_plot]
components = corr_sexual_orientation['PC']
corr_best_sexual_orientation = pd.concat([components,corr_best_sexual_orientation], axis=1)

corr_best_voice_id = corr_voice_id[specific_features_to_plot]
components = corr_voice_id['PC']
corr_best_voice_id = pd.concat([components,corr_best_voice_id], axis=1)

## p values
corr_best_gender_id_p = corr_p_gender_id[specific_features_to_plot]
components = corr_p_gender_id['PC']
corr_best_gender_id_p = pd.concat([components,corr_best_gender_id_p], axis=1)

corr_best_sexual_orientation_p = corr_p_sexual_orientation[specific_features_to_plot]
components = corr_p_sexual_orientation['PC']
corr_best_sexual_orientation_p = pd.concat([components,corr_best_sexual_orientation_p], axis=1)

corr_best_voice_id_p = corr_p_voice_id[specific_features_to_plot]
components = corr_p_voice_id['PC']
corr_best_voice_id_p = pd.concat([components,corr_best_voice_id_p], axis=1)



data = ratings_all

pca = pca.drop('kmeans_5_cluster', axis=1)
data = pd.concat([data, pca], axis=1)

gender_id = data[data['Condition']=='gender_id']
sexual_orientation = data[data['Condition']=='sexual_orientation']
voice_id = data[data['Condition']=='voice_id']

cluster_color_dict = {'kmeans_3_cluster':'color_3_cluster', 'kmeans_4_cluster':'color_4_cluster', 'kmeans_5_cluster':'color_5_cluster'}

for pc in pc_list:
    for feature in specific_features_to_plot:
        for cluster, color in cluster_color_dict.items():
            data_avg_rating = data.groupby('WAV', as_index=False)['Rating_z_score'].mean()
            data_avg_feature = data.groupby('WAV', as_index=False)[feature].mean()
            data_cluster = data.groupby('WAV', as_index=False)[cluster].mean()
            data_pc = data.groupby('WAV', as_index=False)[pc].mean()

            data_feature = pd.merge(data_avg_rating, data_avg_feature, on='WAV')
            data_feature = pd.merge(data_feature, data_cluster, on='WAV')
            data_feature = pd.merge(data_feature, data_pc, on='WAV')


            gender_id_avg_rating = gender_id.groupby('WAV', as_index=False)['Rating_z_score'].mean()
            gender_id_avg_feature = gender_id.groupby('WAV', as_index=False)[feature].mean()
            gender_id_cluster = gender_id.groupby('WAV', as_index=False)[cluster].mean()
            gender_id_pc = gender_id.groupby('WAV', as_index=False)[pc].mean()

            gender_id_color = gender_id[['WAV',color]].drop_duplicates().reset_index().drop(columns='index')
            gender_id_feature = pd.merge(gender_id_avg_rating, gender_id_avg_feature, on='WAV')
            gender_id_feature = pd.merge(gender_id_feature, gender_id_color, on='WAV').dropna()
            gender_id_feature = pd.merge(gender_id_feature, gender_id_cluster, on='WAV')
            gender_id_feature = pd.merge(gender_id_feature, gender_id_pc, on='WAV')


            sexual_orientation_avg_rating = sexual_orientation.groupby('WAV', as_index=False)['Rating_z_score'].mean()
            sexual_orientation_avg_feature = sexual_orientation.groupby('WAV', as_index=False)[feature].mean()
            sexual_orientation_cluster = sexual_orientation.groupby('WAV', as_index=False)[cluster].mean()
            sexual_orientation_pc = sexual_orientation.groupby('WAV', as_index=False)[pc].mean()

            sexual_orientation_color = sexual_orientation[['WAV',color]].drop_duplicates().reset_index().drop(columns='index')
            sexual_orientation_feature = pd.merge(sexual_orientation_avg_rating, sexual_orientation_avg_feature, on='WAV')
            sexual_orientation_feature = pd.merge(sexual_orientation_feature, sexual_orientation_color, on='WAV').dropna()
            sexual_orientation_feature = pd.merge(sexual_orientation_feature, sexual_orientation_cluster, on='WAV')
            sexual_orientation_feature = pd.merge(sexual_orientation_feature, sexual_orientation_pc, on='WAV')


            voice_id_rating = voice_id.groupby('WAV', as_index=False)['Rating_z_score'].mean()
            voice_id_avg_feature = voice_id.groupby('WAV', as_index=False)[feature].mean()
            voice_id_cluster = voice_id.groupby('WAV', as_index=False)[cluster].mean()
            voice_id_pc = voice_id.groupby('WAV', as_index=False)[pc].mean()

            voice_id_color = voice_id[['WAV',color]].drop_duplicates().reset_index().drop(columns='index')
            voice_id_feature = pd.merge(voice_id_rating, voice_id_avg_feature, on='WAV')
            voice_id_feature = pd.merge(voice_id_feature, voice_id_color, on='WAV').dropna()
            voice_id_feature = pd.merge(voice_id_feature, voice_id_cluster, on='WAV')
            voice_id_feature = pd.merge(voice_id_feature, voice_id_pc, on='WAV')

            if cluster == 'kmeans_3_cluster':
                color_dict = {0:'#785EF0', 1:'#DC267F', 2:'#648FFF'}
                markers_dict = {0: "P", 1: "o", 2: "v"}
            if cluster == 'kmeans_4_cluster':
                color_dict = {0:'#DC267F', 1:'#785EF0', 2:'#648FFF', 3:'#FFB000'}
                markers_dict = {0: "o", 1: "P", 2: "v", 3: "D"}
            if cluster == 'kmeans_5_cluster':
                color_dict = {0:'#FFB000', 1:'#785EF0', 2:'#648FFF', 3:'#FE6100', 4:'#DC267F'}
                markers_dict = {0: "D", 1: "P", 2: "v", 3: "H", 4: "o"}

            # assert gender_id_feature[feature].nunique() == sexual_orientation_feature[feature].nunique() == voice_id_feature[feature].nunique()
            feature_number = gender_id_feature[feature].nunique()

            ## correlation value for PC and F0 across whole data set
            r_value = corr_best.loc[corr_best['PC'] == pc][feature]
            r_fig = np.round(r_value.values[0], 2)

            ## correlation value for PC and F0 by condition
            r_value_gender_id_pc = corr_best_gender_id.loc[corr_best_gender_id['PC'] == pc][feature]
            r_fig_gender_id_pc = np.round(r_value_gender_id_pc.values[0], 2)

            r_value_sexual_orientation_pc = corr_best_sexual_orientation.loc[corr_best_sexual_orientation['PC'] == pc][feature]
            r_fig_sexual_orientation_pc = np.round(r_value_sexual_orientation_pc.values[0], 2)

            r_value_voice_id_pc = corr_best_voice_id.loc[corr_best_voice_id['PC'] == pc][feature]
            r_fig_voice_id_pc = np.round(r_value_voice_id_pc.values[0], 2)

            ## p values
            p_value_gender_id_pc = corr_best_gender_id_p.loc[corr_best_gender_id_p['PC'] == pc][feature]
            p_fig_gender_id_pc_temp = np.round(p_value_gender_id_pc.values[0], 2)
            if p_fig_gender_id_pc_temp < 0.05:
                p_fig_gender_id_pc = 'p < 0.05'
            if p_fig_gender_id_pc_temp > 0.05:
                p_fig_gender_id_pc = 'n.s.'

            p_value_sexual_orientation_pc = corr_best_sexual_orientation_p.loc[corr_best_sexual_orientation_p['PC'] == pc][feature]
            p_fig_sexual_orientation_pc_temp = np.round(p_value_sexual_orientation_pc.values[0], 2)
            if p_fig_sexual_orientation_pc_temp < 0.05:
                p_fig_sexual_orientation_pc = 'p < 0.05'
            if p_fig_sexual_orientation_pc_temp > 0.05:
                p_fig_sexual_orientation_pc = 'n.s.'

            p_value_voice_id_pc = corr_best_voice_id_p.loc[corr_best_voice_id_p['PC'] == pc][feature]
            p_fig_voice_id_pc_temp = np.round(p_value_voice_id_pc.values[0], 2)
            if p_fig_voice_id_pc_temp < 0.05:
                p_fig_voice_id_pc = 'p < 0.05'
            if p_fig_voice_id_pc_temp > 0.05:
                p_fig_voice_id_pc = 'n.s.'


            ## correlation value for raw feature by condition
            r_value_gender_id = corr_r.loc[corr_r['split'] == 'gender_id'][feature]
            r_fig_gender_id = np.round(r_value_gender_id.values[0], 2)

            r_value_sexual_orientation = corr_r.loc[corr_r['split'] == 'sexual_orientation'][feature]
            r_fig_sexual_orientation = np.round(r_value_sexual_orientation.values[0], 2)

            r_value_voice_id = corr_r.loc[corr_r['split'] == 'voice_id'][feature]
            r_fig_voice_id = np.round(r_value_voice_id.values[0], 2)

            ## p values
            p_value_gender_id = corr_p.loc[corr_p['split'] == 'gender_id'][feature]
            p_fig_gender_id_temp = np.round(p_value_gender_id.values[0], 2)
            if p_fig_gender_id_temp < 0.05:
                p_fig_gender_id = 'p < 0.05'
            if p_fig_gender_id_temp > 0.05:
                p_fig_gender_id = 'n.s.'

            p_value_sexual_orientation = corr_p.loc[corr_p['split'] == 'sexual_orientation'][feature]
            p_fig_sexual_orientation_temp = np.round(p_value_sexual_orientation.values[0], 2)
            if p_fig_sexual_orientation_temp < 0.05:
                p_fig_sexual_orientation = 'p < 0.05'
            if p_fig_sexual_orientation_temp > 0.05:
                p_fig_sexual_orientation = 'n.s.'

            p_value_voice_id = corr_p.loc[corr_p['split'] == 'voice_id'][feature]
            p_fig_voice_id_temp = np.round(p_value_voice_id.values[0], 2)
            if p_fig_voice_id_temp < 0.05:
                p_fig_voice_id = 'p < 0.05'
            if p_fig_voice_id_temp > 0.05:
                p_fig_voice_id = 'n.s.'

            props = dict(boxstyle='round', facecolor='w')

            sns.set()
            fig, axes = plt.subplots(3, 3, figsize = (20,20))
            fig.subplots_adjust(hspace=0.5)
            fig.suptitle("Correlation between %s, \n %s, and Participant Ratings"%(pc, feature), fontsize=30, fontweight='bold', y = 0.75, x = 0.65)
            fig.subplots_adjust(top = 0.85)
            axes[0, 1].axis('off')
            axes[0, 2].axis('off')

            ### pcs
            axes[0,0].set_title('All Conditions', fontsize=18)
            axes[0,0].set_xlim(-1.5,1.5)
            # sns.scatterplot(data=gender_id_feature, x='Rating_z_score', y=pc, ax=axes[0,0])
            sns.regplot(data=data_feature, x='Rating_z_score', y=pc, ax=axes[0,0], scatter_kws={"color": "green", 's': 50, 'edgecolors': 'w'}, line_kws={"color": "k"}) # #648FFF d55e00
            axes[0,0].text(0.50, 0.95, 'Pearson r = %s, p < 0.05'%r_fig, transform=axes[0,0].transAxes, fontsize=14,
                    verticalalignment='top', bbox=props)
            axes[0,0].text(-0.1, 1.1, string.ascii_uppercase[0], transform=axes[0,0].transAxes,
                    size=20, weight='bold')
            axes[0,0].set_xlabel('Rating', fontsize=18)
            axes[0,0].set_ylabel('Dimensionless', fontsize=18)

            axes[1,0].set_title('Gender Identity', fontsize = 18)
            axes[1,0].set_xlim(-1.5,1.5)
            # sns.scatterplot(data=gender_id_feature, x='Rating_z_score', y=pc, ax=axes[0,0])
            sns.regplot(data=gender_id_feature, x='Rating_z_score', y=pc, ax=axes[1,0], scatter_kws={"color": "green", 's': 50, 'edgecolors': 'w'}, line_kws={"color": "k"}) # #648FFF d55e00
            axes[1,0].text(0.50, 0.95, 'Pearson r = %s, %s'%(r_fig_gender_id_pc, p_fig_gender_id_pc), transform=axes[1,0].transAxes, fontsize=14,
                    verticalalignment='top', bbox=props)
            axes[1,0].text(-0.1, 1.1, string.ascii_uppercase[1], transform=axes[1,0].transAxes,
                    size=20, weight='bold')
            axes[1,0].set_xlabel('Rating (-: Male, +: Female)', fontsize = 18)
            axes[1,0].set_ylabel('Dimensionless', fontsize = 18)

            axes[1,1].set_title('Sexual Orientation', fontsize = 18)
            axes[1,1].set_xlim(-1.5,1.5)
            # sns.scatterplot(data=sexual_orientation_feature, x='Rating_z_score', y=pc, ax=axes[0,1])
            sns.regplot(data=sexual_orientation_feature, x='Rating_z_score', y=pc, ax=axes[1,1], scatter_kws={"color": "green", 's': 50, 'edgecolors': 'w'}, line_kws={"color": "k"}) # #785EF0 0072b2
            axes[1,1].text(0.50, 0.95, 'Pearson r = %s, %s'%(r_fig_sexual_orientation_pc, p_fig_sexual_orientation_pc), transform=axes[1,1].transAxes, fontsize=14,
                    verticalalignment='top', bbox=props)
            axes[1,1].set_xlabel('Rating (-: Homo, +: Het)', fontsize = 18)
            axes[1,1].set_ylabel('')

            axes[1,2].set_title('Voice Identity', fontsize = 18)
            axes[1,2].set_xlim(-1.5,1.5)
            # sns.scatterplot(data=voice_id_feature, x='Rating_z_score', y=pc, ax=axes[0,2])
            sns.regplot(data=voice_id_feature, x='Rating_z_score', y=pc, ax=axes[1,2], scatter_kws={"color": "green", 's': 50, 'edgecolors': 'w'}, line_kws={"color": "k"}) # #DC267F 009e73
            axes[1,2].text(0.50, 0.95, 'Pearson r = %s, %s'%(r_fig_voice_id_pc, p_fig_voice_id_pc), transform=axes[1,2].transAxes, fontsize=14,
                    verticalalignment='top', bbox=props)
            axes[1,2].set_xlabel('Rating (-: Masc, +: Femme)', fontsize = 18)
            axes[1,2].set_ylabel('')

            ### individual features
            axes[2,0].set_title('Gender Identity', fontsize = 18)
            axes[2,0].set_xlim(-1.5,1.5)
            sns.scatterplot(data=gender_id_feature, x='Rating_z_score', y=feature, ax=axes[2,0], hue = gender_id_feature[cluster], style = gender_id_feature[cluster], palette = color_dict, markers=markers_dict, s=50)
            sns.regplot(data=gender_id_feature, x='Rating_z_score', y=feature, ax=axes[2,0], scatter=False, line_kws={"color": "k"}) # #648FFF d55e00
            axes[2,0].text(0.50, 0.95, 'Pearson r = %s, %s'%(r_fig_gender_id, p_fig_gender_id), transform=axes[2,0].transAxes, fontsize=14,
                    verticalalignment='top', bbox=props)
            axes[2,0].text(-0.1, 1.1, string.ascii_uppercase[2], transform=axes[2,0].transAxes,
                    size=20, weight='bold')
            axes[2,0].set_xlabel('Rating (-: Male, +: Female)', fontsize = 18)
            axes[2,0].set_ylabel('Avg %s'%feature, fontsize = 18)

            axes[2,1].set_title('Sexual Orientation', fontsize = 18)
            axes[2,1].set_xlim(-1.5,1.5)
            sns.scatterplot(data=sexual_orientation_feature, x='Rating_z_score', y=feature, ax=axes[2,1], hue = sexual_orientation_feature[cluster], style = sexual_orientation_feature[cluster], palette = color_dict, markers=markers_dict, s=50)
            sns.regplot(data=sexual_orientation_feature, x='Rating_z_score', y=feature, ax=axes[2,1], scatter=False, line_kws={"color": "k"}) # #785EF0 0072b2
            axes[2,1].text(0.50, 0.95, 'Pearson r = %s, %s'%(r_fig_sexual_orientation, p_fig_sexual_orientation), transform=axes[2,1].transAxes, fontsize=14,
                    verticalalignment='top', bbox=props)
            axes[2,1].set_xlabel('Rating (-: Homo, +: Het)', fontsize = 18)
            axes[2,1].set_ylabel('')

            axes[2,2].set_title('Voice Identity', fontsize = 18)
            axes[2,2].set_xlim(-1.5,1.5)
            sns.scatterplot(data=voice_id_feature, x='Rating_z_score', y=feature, ax=axes[2,2], hue = voice_id_feature[cluster], style = voice_id_feature[cluster], palette = color_dict, markers=markers_dict, s=50)
            sns.regplot(data=voice_id_feature, x='Rating_z_score', y=feature, ax=axes[2,2], scatter=False, line_kws={"color": "k"}) # #DC267F 009e73
            axes[2,2].text(0.50, 0.95, 'Pearson r = %s, %s'%(r_fig_voice_id, p_fig_voice_id), transform=axes[2,2].transAxes, fontsize=14,
                    verticalalignment='top', bbox=props)
            axes[2,2].set_xlabel('Rating (-: Masc, +: Femme)', fontsize = 18)
            axes[2,2].set_ylabel('')

            if color == 'color_3_cluster':
                colors = pd.DataFrame({"color_3_cluster": ['#785EF0','#DC267F','#648FFF']})
                colors['kmeans_3_cluster'] = [0,1,2]
                scatter1_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c=colors['color_3_cluster'][0], marker = 'P') #^
                scatter2_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c=colors['color_3_cluster'][1], marker = 'o') #p
                scatter3_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c=colors['color_3_cluster'][2], marker = 'v') #+
                axes[2,0].get_legend().remove()
                axes[2,1].legend([scatter3_proxy, scatter2_proxy, scatter1_proxy], ['SM','QE','SW'], numpoints = 1, loc = 'upper left', fontsize=14, facecolor = 'w')
                axes[2,2].get_legend().remove()

                # plt.show()
                plt.xticks(fontsize=12)
                plt.savefig(os.path.join(fig_dir, '3_cluster', 'pc_corr', '%s_avgbycondition_3_clusters_corr_%s.png'%(feature,pc)), bbox_inches='tight', dpi=300)
                plt.close()

            if color == 'color_4_cluster':
                colors = pd.DataFrame({"color_4_cluster": ['#DC267F','#785EF0','#648FFF','#FFB000']})
                colors['kmeans_4_cluster'] = [0,1,2,3]
                scatter1_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c=colors['color_4_cluster'][0], marker = 'o')
                scatter2_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c=colors['color_4_cluster'][1], marker = 'P')
                scatter3_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c=colors['color_4_cluster'][2], marker = 'v')
                scatter4_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c=colors['color_4_cluster'][3], marker = 'D')
                axes[2,0].get_legend().remove()
                axes[2,1].legend([scatter3_proxy, scatter4_proxy, scatter1_proxy, scatter2_proxy], ['SM','QM','QE','SW'], numpoints = 1, loc = 'upper left', fontsize=14, facecolor = 'w')
                axes[2,2].get_legend().remove()


                # plt.show()
                plt.xticks(fontsize=12)
                plt.savefig(os.path.join(fig_dir, '4_cluster', 'pc_corr', '%s_avgbycondition_4_clusters_corr_%s.png'%(feature,pc)), bbox_inches='tight', dpi=300)
                plt.close()


            if color == 'color_5_cluster':
                colors = pd.DataFrame({"color_5_cluster": ['#FFB000','#785EF0','#648FFF','#FE6100','#DC267F']})
                colors['kmeans_5_cluster'] = [0,1,2,3,4]
                scatter1_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c=colors['color_5_cluster'][0], marker = 'D')
                scatter2_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c=colors['color_5_cluster'][1], marker = 'P')
                scatter3_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c=colors['color_5_cluster'][2], marker = 'v')
                scatter4_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c=colors['color_5_cluster'][3], marker = 'H')
                scatter5_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c=colors['color_5_cluster'][4], marker = 'o')
                axes[2,0].get_legend().remove()
                axes[2,1].legend([scatter3_proxy, scatter1_proxy, scatter5_proxy, scatter4_proxy, scatter2_proxy], ['SM','QM','QE','QW','SW'], numpoints = 1, loc = 'upper left', fontsize=14, facecolor = 'w')
                axes[2,2].get_legend().remove()


                # plt.show()
                plt.xticks(fontsize=12)
                plt.savefig(os.path.join(fig_dir, '5_cluster', 'pc_corr', '%s_avgbycondition_5_clusters_corr_%s.png'%(feature,pc)), bbox_inches='tight', dpi=300)
                plt.close()


### cluster identity flags ###
sns.set()
condition_means = ratings_all.pivot_table(index='kmeans_5_cluster', columns = 'Condition', values = 'Rating_z_score')
n_per_cluster = ratings_all.groupby(['kmeans_5_cluster'], as_index=False)['WAV'].nunique()

# year = [2014, 2015, 2016, 2017, 2018, 2019]
PG_condition = ['PGI']
PS_condition = ['PSO']
PV_condition = ['PGV']
# issues_addressed = [10, 14, 0, 10, 15, 15]
# issues_pending = [5, 10, 50, 2, 0, 10]

cluster_list = {0:'#FFB000', 1:'#785EF0', 2:'#648FFF', 3:'#FE6100', 4:'#DC267F'}
for cluster, color in cluster_list.items():

    PG = condition_means['gender_id'][cluster]
    PS = condition_means['sexual_orientation'][cluster]
    PV = condition_means['voice_id'][cluster]

    labels_left = ['more man-like','same gender','more masculine-sounding']
    labels_right = ['more woman-like','opposite gender','more feminine-sounding']
    labels_center = ['nb','any gender','neither']

    fig, ax = plt.subplots(figsize = (20,8))
    b3 = plt.barh(PV_condition, PV, color=color)
    b2 = plt.barh(PS_condition, PS, color=color)
    b1 = plt.barh(PG_condition, PG, color=color)
    for bar, label_right, label_left, label_center in zip(ax.patches, labels_right, labels_left, labels_center):
        ax.text(0.8, bar.get_y()+bar.get_height()/2, label_right, color = 'black', ha = 'left', va = 'center', fontsize = 30)
        ax.text(-1.4, bar.get_y()+bar.get_height()/2, label_left, color = 'black', ha = 'left', va = 'center', fontsize = 30)
        ax.text(-0.1, bar.get_y()+bar.get_height()/2, label_center, color = 'black', ha = 'left', va = 'center', fontsize = 30)

    # plt.axvline(x=0, linestyle='--', color='lightgray')
    # plt.legend([b1], ["Completed", "Pending"], title="Issues", loc="upper right")
    ax.set_xlim([-1.5,1.5])
    ax.set_xticks(np.arange(-1.5, 2, step=0.5))
    ax.set_xlabel('Rating (standardized)',fontsize=30)
    ax.set_ylabel('Condition',fontsize=30)
    ax.tick_params(axis='both', which='major', labelsize=30)
    ax.tick_params(axis='both', which='minor', labelsize=30)
    ax.set_facecolor("white")
    if cluster == 0:
        ax.set_title('QM', fontsize=40)
        plt.savefig(os.path.join(fig_dir, 'QM_flag.png'), bbox_inches='tight', dpi=300)
        plt.close()
    if cluster == 1:
        ax.set_title('SW', fontsize=40)
        plt.savefig(os.path.join(fig_dir, 'SW_flag.png'), bbox_inches='tight', dpi=300)
        plt.close()

    if cluster == 2:
        ax.set_title('SM', fontsize=40)
        plt.savefig(os.path.join(fig_dir, 'SM_flag.png'), bbox_inches='tight', dpi=300)
        plt.close()

    if cluster == 3:
        ax.set_title('QW', fontsize=40)
        plt.savefig(os.path.join(fig_dir, 'QW_flag.png'), bbox_inches='tight', dpi=300)
        plt.close()

    if cluster == 4:
        ax.set_title('QE', fontsize=40)
        plt.savefig(os.path.join(fig_dir, 'QE_flag.png'), bbox_inches='tight', dpi=300)
        plt.close()


### heatmap ###
corr_fname = os.path.join(dir, 'data_analysis_viz','grand_corr_r.csv')
grand_ablated_df_fname = os.path.join(dir, 'data_analysis_viz', 'grand_ablated_df.csv')
cluster_ablated_df_fname = os.path.join(dir, 'data_analysis_viz', 'cluster_ablated_df.csv')
corr_cluster0_fname = os.path.join(dir, 'data_analysis_viz','cluster_corr_r_cluster_0.csv')
corr_cluster1_fname = os.path.join(dir, 'data_analysis_viz','cluster_corr_r_cluster_1.csv')
corr_cluster2_fname = os.path.join(dir, 'data_analysis_viz','cluster_corr_r_cluster_2.csv')
corr_cluster3_fname = os.path.join(dir, 'data_analysis_viz','cluster_corr_r_cluster_3.csv')
corr_cluster4_fname = os.path.join(dir, 'data_analysis_viz','cluster_corr_r_cluster_4.csv')

corr_df = pd.read_csv(corr_fname)
corr_df = corr_df.drop('Unnamed: 0', axis=1) # might need to change column name when it imports in, there's a random addition at the beginning

cluster0_corr_df = pd.read_csv(corr_cluster0_fname)
cluster0_corr_df = cluster0_corr_df.drop('Unnamed: 0', axis=1) # might need to change column name when it imports in, there's a random addition at the beginning

cluster1_corr_df = pd.read_csv(corr_cluster1_fname)
cluster1_corr_df = cluster1_corr_df.drop('Unnamed: 0', axis=1) # might need to change column name when it imports in, there's a random addition at the beginning

cluster2_corr_df = pd.read_csv(corr_cluster2_fname)
cluster2_corr_df = cluster2_corr_df.drop('Unnamed: 0', axis=1) # might need to change column name when it imports in, there's a random addition at the beginning

cluster3_corr_df = pd.read_csv(corr_cluster3_fname)
cluster3_corr_df = cluster3_corr_df.drop('Unnamed: 0', axis=1) # might need to change column name when it imports in, there's a random addition at the beginning

cluster4_corr_df = pd.read_csv(corr_cluster4_fname)
cluster4_corr_df = cluster4_corr_df.drop('Unnamed: 0', axis=1) # might need to change column name when it imports in, there's a random addition at the beginning

grand_ablated_df = pd.read_csv(grand_ablated_df_fname)
grand_ablated_df = grand_ablated_df.drop('Unnamed: 0', axis=1) # might need to change column name when it imports in, there's a random addition at the beginning

cluster_ablated_df = pd.read_csv(cluster_ablated_df_fname)
cluster_ablated_df = cluster_ablated_df.drop('Unnamed: 0', axis=1) # might need to change column name when it imports in, there's a random addition at the beginning

## correlation thresholding
ablated_reduced = grand_ablated_df[grand_ablated_df['accuracy'] == 1]
best_features_list = list(ablated_reduced['removed_feature'])

corr_best = corr_df[best_features_list]

# important_features = ['PC','percent_creak','F0_mean','spectral_S_cog','spectral_S_skew','F0_10','F0_90','F0_std','rando']
# corr_df = corr_df[important_features]

features = list(corr_best.columns)
components = list(corr_df['PC'])

corr_df_array = np.round(np.array(corr_best), 2)

fig, ax = plt.subplots(figsize = (100,8))
im = ax.imshow(corr_df_array, cmap='viridis', interpolation='nearest')

ax.set_xticks(np.arange(len(features)), labels=features, fontsize=20)
ax.set_yticks(np.arange(len(components)), labels=components, fontsize=20)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(len(components)):
    for j in range(len(features)):
        text = ax.text(j, i, corr_df_array[i, j],
                       ha="center", va="center", color="w")

ax.set_title('Correlation between Important Features and PCs for Grand Features', fontsize=28)
# ax.legend()
# fig.tight_layout()

plt.colorbar(im)
plt.savefig(os.path.join(fig_dir, 'heatmap_important_features_PC_corr.png'), bbox_inches='tight', dpi=300)
plt.close()
# plt.show()

### cluster heatmaps ###

cluster0_ablated_df = cluster_ablated_df[cluster_ablated_df['kind'] == 'cluster_0']
cluster1_ablated_df = cluster_ablated_df[cluster_ablated_df['kind'] == 'cluster_1']
cluster2_ablated_df = cluster_ablated_df[cluster_ablated_df['kind'] == 'cluster_2']
cluster3_ablated_df = cluster_ablated_df[cluster_ablated_df['kind'] == 'cluster_3']
cluster4_ablated_df = cluster_ablated_df[cluster_ablated_df['kind'] == 'cluster_4']

cluster_dict  = {'cluster_0':cluster0_ablated_df,'cluster_1':cluster1_ablated_df,'cluster_2':cluster2_ablated_df,'cluster_3':cluster3_ablated_df,'cluster_4':cluster4_ablated_df}
# loop_number = loop_number + 1
for cluster, value in cluster_dict.items():

    ablated_reduced = value[value['accuracy'] == 1]
    best_features_list = list(ablated_reduced['removed_feature'])
    if cluster == 'cluster_0':
        corr_best = cluster0_corr_df[best_features_list]
    if cluster == 'cluster_1':
        corr_best = cluster1_corr_df[best_features_list]
    if cluster == 'cluster_2':
        corr_best = cluster2_corr_df[best_features_list]
    if cluster == 'cluster_3':
        corr_best = cluster3_corr_df[best_features_list]
    if cluster == 'cluster_4':
        corr_best = cluster4_corr_df[best_features_list]

    # important_features = ['PC','percent_creak','F0_mean','spectral_S_cog','spectral_S_skew','F0_10','F0_90','F0_std','rando']
    # corr_df = corr_df[important_features]

    features = list(corr_best.columns)
    components = list(corr_df['PC'])

    corr_df_array = np.round(np.array(corr_best), 2)

    fig, ax = plt.subplots(figsize = (100,8))
    im = ax.imshow(corr_df_array, cmap='viridis', interpolation='nearest')

    ax.set_xticks(np.arange(len(features)), labels=features, fontsize=20)
    ax.set_yticks(np.arange(len(components)), labels=components, fontsize=20)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(components)):
        for j in range(len(features)):
            text = ax.text(j, i, corr_df_array[i, j],
                           ha="center", va="center", color="w")

    if cluster == 'cluster_0':
        ax.set_title('Correlation between Important Features and PCs for QM', fontsize=28)
    if cluster == 'cluster_1':
        ax.set_title('Correlation between Important Features and PCs for SW', fontsize=28)
    if cluster == 'cluster_2':
        ax.set_title('Correlation between Important Features and PCs for SM', fontsize=28)
    if cluster == 'cluster_3':
        ax.set_title('Correlation between Important Features and PCs for QW', fontsize=28)
    if cluster == 'cluster_4':
        ax.set_title('Correlation between Important Features and PCs for QE', fontsize=28)
    # ax.legend()
    # fig.tight_layout()
    plt.colorbar(im)
    plt.savefig(os.path.join(fig_dir, 'heatmap_important_features_PC_corr_%s.png'%cluster), bbox_inches='tight', dpi=300)
    plt.close()

### specific feature heatmaps ###
feature_list = ['F0_mean','IH_sF1_mean','IH_sF2_mean','IH_sF3_mean','IH_sF4_mean','IH_sF1_mean_dist','IH_sF2_mean_dist','IH_sF3_mean_dist','IH_sF4_mean_dist',
                'AY_sF1_mean_first', 'AY_sF2_mean_first', 'AY_sF3_mean_first', 'AY_sF4_mean_first', 'AY_sF1_mean_third', 'AY_sF2_mean_third', 'AY_sF3_mean_third', 'AY_sF4_mean_third',
                'vowel_avg_dur', 'percent_creak', 'spectral_S_duration','spectral_S_cog','spectral_S_skew']

features = feature_list
components = list(corr_df['PC'])

corr_specific = corr_df[features]

corr_df_array = np.round(np.array(corr_specific), 2)

fig, ax = plt.subplots(figsize = (100,8))
im = ax.imshow(corr_df_array, cmap='viridis', interpolation='nearest')

ax.set_xticks(np.arange(len(features)), labels=features, fontsize=20)
ax.set_yticks(np.arange(len(components)), labels=components, fontsize=20)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(len(components)):
    for j in range(len(features)):
        text = ax.text(j, i, corr_df_array[i, j],
                       ha="center", va="center", color="w")

ax.set_title('Correlation between Specific Features and PCs for Grand Features', fontsize=28)
# ax.legend()
# fig.tight_layout()
plt.colorbar(im)
plt.savefig(os.path.join(fig_dir, 'heatmap_grand_corr_specific.png'), bbox_inches='tight', dpi=300)
plt.close()

### cluster heatmaps ###

cluster0_ablated_df = cluster_ablated_df[cluster_ablated_df['kind'] == 'cluster_0']
cluster1_ablated_df = cluster_ablated_df[cluster_ablated_df['kind'] == 'cluster_1']
cluster2_ablated_df = cluster_ablated_df[cluster_ablated_df['kind'] == 'cluster_2']
cluster3_ablated_df = cluster_ablated_df[cluster_ablated_df['kind'] == 'cluster_3']
cluster4_ablated_df = cluster_ablated_df[cluster_ablated_df['kind'] == 'cluster_4']

cluster_dict  = {'cluster_0':cluster0_ablated_df,'cluster_1':cluster1_ablated_df,'cluster_2':cluster2_ablated_df,'cluster_3':cluster3_ablated_df,'cluster_4':cluster4_ablated_df}
# loop_number = loop_number + 1

feature_list = ['F0_mean','IH_sF1_mean','IH_sF2_mean','IH_sF3_mean','IH_sF4_mean','IH_sF1_mean_dist','IH_sF2_mean_dist','IH_sF3_mean_dist','IH_sF4_mean_dist',
                'AY_sF1_mean_first', 'AY_sF2_mean_first', 'AY_sF3_mean_first', 'AY_sF4_mean_first', 'AY_sF1_mean_third', 'AY_sF2_mean_third', 'AY_sF3_mean_third', 'AY_sF4_mean_third',
                'vowel_avg_dur', 'percent_creak', 'spectral_S_duration','spectral_S_cog','spectral_S_skew']

features = feature_list

for cluster, value in cluster_dict.items():

    ablated_reduced = value[value['accuracy'] == 1]
    best_features_list = feature_list # using the specified feature list from above
    if cluster == 'cluster_0':
        corr_best = cluster0_corr_df[best_features_list]
    if cluster == 'cluster_1':
        corr_best = cluster1_corr_df[best_features_list]
    if cluster == 'cluster_2':
        corr_best = cluster2_corr_df[best_features_list]
    if cluster == 'cluster_3':
        corr_best = cluster3_corr_df[best_features_list]
    if cluster == 'cluster_4':
        corr_best = cluster4_corr_df[best_features_list]

    # important_features = ['PC','percent_creak','F0_mean','spectral_S_cog','spectral_S_skew','F0_10','F0_90','F0_std','rando']
    # corr_df = corr_df[important_features]

    features = list(corr_best.columns)
    components = list(corr_df['PC'])

    corr_df_array = np.round(np.array(corr_best), 2)

    fig, ax = plt.subplots(figsize = (100,8))
    im = ax.imshow(corr_df_array, cmap='viridis', interpolation='nearest')

    ax.set_xticks(np.arange(len(features)), labels=features, fontsize=20)
    ax.set_yticks(np.arange(len(components)), labels=components, fontsize=20)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(components)):
        for j in range(len(features)):
            text = ax.text(j, i, corr_df_array[i, j],
                           ha="center", va="center", color="w")
    if cluster == 'cluster_0':
        ax.set_title('Correlation between Specific Features and PCs for QM', fontsize=28)
    if cluster == 'cluster_1':
        ax.set_title('Correlation between Specific Features and PCs for SW', fontsize=28)
    if cluster == 'cluster_2':
        ax.set_title('Correlation between Specific Features and PCs for SM', fontsize=28)
    if cluster == 'cluster_3':
        ax.set_title('Correlation between Specific Features and PCs for QW', fontsize=28)
    if cluster == 'cluster_4':
        ax.set_title('Correlation between Specific Features and PCs for QE', fontsize=28)
    # ax.legend()
    # fig.tight_layout()
    plt.colorbar(im)
    plt.savefig(os.path.join(fig_dir, 'heatmap_%s_corr_specific.png'%cluster), bbox_inches='tight', dpi=300)
    plt.close()


#### proximity ###############################
# social
# gender_id_avg_rating = gender_id.groupby('Participant', as_index=False)['Rating'].mean()
# gender_id_prox_social = pd.DataFrame({'Participant':gender_id['Participant'], 'prox_social': gender_id['participant_prox_social']}).drop_duplicates()
# gender_id_social = pd.merge(gender_id_avg_rating, gender_id_prox_social, on='Participant')
#
# sexual_orientation_avg_rating = sexual_orientation.groupby('Participant', as_index=False)['Rating'].mean()
# sexual_orientation_prox_social = pd.DataFrame({'Participant':sexual_orientation['Participant'], 'prox_social': sexual_orientation['participant_prox_social']}).drop_duplicates()
# sexual_orientation_social = pd.merge(sexual_orientation_avg_rating, sexual_orientation_prox_social, on='Participant')
#
# voice_id_avg_rating = voice_id.groupby('Participant', as_index=False)['Rating'].mean()
# voice_id_prox_social = pd.DataFrame({'Participant':voice_id['Participant'], 'prox_social': voice_id['participant_prox_social']}).drop_duplicates()
# voice_id_social = pd.merge(voice_id_avg_rating, voice_id_prox_social, on='Participant')
#
# # affiliation
# gender_id_prox_affiliation = pd.DataFrame({'Participant':gender_id['Participant'], 'prox_affiliation': gender_id['participant_prox_affiliation']}).drop_duplicates()
# sexual_orientation_prox_affiliation = pd.DataFrame({'Participant':sexual_orientation['Participant'], 'prox_affiliation': sexual_orientation['participant_prox_affiliation']}).drop_duplicates()
# voice_id_prox_affiliation = pd.DataFrame({'Participant':voice_id['Participant'], 'prox_affiliation': voice_id['participant_prox_affiliation']}).drop_duplicates()
#
# gender_id_affiliation = pd.merge(gender_id_avg_rating, gender_id_prox_affiliation, on='Participant')
# sexual_orientation_affiliation = pd.merge(sexual_orientation_avg_rating, sexual_orientation_prox_affiliation, on='Participant')
# voice_id_affiliation = pd.merge(voice_id_avg_rating, voice_id_prox_affiliation, on='Participant')
#
# # media
# gender_id_prox_media = pd.DataFrame({'Participant':gender_id['Participant'], 'prox_media': gender_id['participant_prox_media']}).drop_duplicates()
# sexual_orientation_prox_media = pd.DataFrame({'Participant':sexual_orientation['Participant'], 'prox_media': sexual_orientation['participant_prox_media']}).drop_duplicates()
# voice_id_prox_media = pd.DataFrame({'Participant':voice_id['Participant'], 'prox_media': voice_id['participant_prox_media']}).drop_duplicates()
#
# gender_id_media = pd.merge(gender_id_avg_rating, gender_id_prox_media, on='Participant')
# sexual_orientation_media = pd.merge(sexual_orientation_avg_rating, sexual_orientation_prox_media, on='Participant')
# voice_id_media = pd.merge(voice_id_avg_rating, voice_id_prox_media, on='Participant')
#
# # plot
# fig, axes = plt.subplots(3, 3)
# fig.subplots_adjust(hspace=0.5)
# fig.set_size_inches(16, 6)
# fig.suptitle("Rating by Proximity to LGBTQ+ Community, %s Participants"%number_participants, fontsize=20, fontweight='bold')
# fig.subplots_adjust( top = 0.85 )
#
# # social
# axes[0,0].set_title('Gender Identity')
# axes[0,0].set_xlim(0,100)
# axes[0,0].set_ylim(1,7)
# sns.regplot(data=gender_id_social, x='prox_social', y='Rating', ax=axes[0,0], color='#d55e00')
# axes[0,0].set_xlabel('Percent of Social Circle')
# axes[0,0].set_ylabel('')
#
# axes[0,1].set_title('Sexual Orientation')
# axes[0,1].set_xlim(0,100)
# axes[0,1].set_ylim(1,7)
# sns.regplot(data=sexual_orientation_social, x='prox_social', y='Rating', ax=axes[0,1], color='#0072b2')
# axes[0,1].set_xlabel('Percent of Social Circle')
# axes[0,1].set_ylabel('')
#
# axes[0,2].set_title('Voice Identity')
# axes[0,2].set_xlim(0,100)
# axes[0,2].set_ylim(1,7)
# sns.regplot(data=voice_id_social, x='prox_social', y='Rating', ax=axes[0,2], color='#009e73')
# axes[0,2].set_xlabel('Percent of Social Circle')
# axes[0,2].set_ylabel('')
#
# # affiliation
# # axes[1,0].set_title('Gender Identity')
# axes[1,0].set_xlim(0,100)
# axes[1,0].set_ylim(1,7)
# sns.regplot(data=gender_id_affiliation, x='prox_affiliation', y='Rating', ax=axes[1,0], color='#d55e00')
# axes[1,0].set_xlabel('Percent of Affiliation')
# axes[1,0].set_ylabel('Rating (1-Male, 7-Female)')
#
# # axes[1,1].set_title('Sexual Orientation')
# axes[1,1].set_xlim(0,100)
# axes[1,1].set_ylim(1,7)
# sns.regplot(data=sexual_orientation_affiliation, x='prox_affiliation', y='Rating', ax=axes[1,1], color='#0072b2')
# axes[1,1].set_xlabel('Percent of Affiliation')
# axes[1,1].set_ylabel('Rating (1-Homo, 7-Het)')
#
# # axes[1,2].set_title('Voice Identity')
# axes[1,2].set_xlim(0,100)
# axes[1,2].set_ylim(1,7)
# sns.regplot(data=voice_id_affiliation, x='prox_affiliation', y='Rating', ax=axes[1,2], color='#009e73')
# axes[1,2].set_xlabel('Percent of Affiliation')
# axes[1,2].set_ylabel('Rating (1-Masc, 7-Femme)')
#
# # media
# # axes[2,0].set_title('Gender Identity')
# axes[2,0].set_xlim(0,100)
# axes[2,0].set_ylim(1,7)
# sns.regplot(data=gender_id_media, x='prox_media', y='Rating', ax=axes[2,0], color='#d55e00')
# axes[2,0].set_xlabel('Percent of Media Consumed')
# axes[2,0].set_ylabel('')
#
# # axes[2,1].set_title('Sexual Orientation')
# axes[2,1].set_xlim(0,100)
# axes[2,1].set_ylim(1,7)
# sns.regplot(data=sexual_orientation_media, x='prox_media', y='Rating', ax=axes[2,1], color='#0072b2')
# axes[2,1].set_xlabel('Percent of Media Consumed')
# axes[2,1].set_ylabel('')
#
# # axes[2,2].set_title('Voice Identity')
# axes[2,2].set_xlim(0,100)
# axes[2,2].set_ylim(1,7)
# sns.regplot(data=voice_id_media, x='prox_media', y='Rating', ax=axes[2,2], color='#009e73')
# axes[2,2].set_xlabel('Percent of Media Consumed')
# axes[2,2].set_ylabel('')
#
# # plt.show()
# plt.savefig(os.path.join(dir,'figs', 'proximity_social.png'), bbox_inches='tight', dpi=300)
# plt.close()


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
# gender_id_avg_rating = gender_id.groupby('WAV', as_index=False)['Rating_z_score'].mean()
# gender_id_avg_F0 = gender_id.groupby('WAV', as_index=False)['F0_mean'].mean()
# gender_id_color = gender_id[['WAV','color']].drop_duplicates().reset_index().drop(columns='index')
# gender_id_F0 = pd.merge(gender_id_avg_rating, gender_id_avg_F0, on='WAV')
# gender_id_F0 = pd.merge(gender_id_F0, gender_id_color, on='WAV')
#
# sexual_orientation_avg_rating = sexual_orientation.groupby('WAV', as_index=False)['Rating_z_score'].mean()
# sexual_orientation_avg_F0 = sexual_orientation.groupby('WAV', as_index=False)['F0_mean'].mean()
# sexual_orientation_color = sexual_orientation[['WAV','color']].drop_duplicates().reset_index().drop(columns='index')
# sexual_orientation_F0 = pd.merge(sexual_orientation_avg_rating, sexual_orientation_avg_F0, on='WAV')
# sexual_orientation_F0 = pd.merge(sexual_orientation_F0, sexual_orientation_color, on='WAV')
#
# voice_id_rating = voice_id.groupby('WAV', as_index=False)['Rating_z_score'].mean()
# voice_id_avg_F0 = voice_id.groupby('WAV', as_index=False)['F0_mean'].mean()
# voice_id_color = voice_id[['WAV','color']].drop_duplicates().reset_index().drop(columns='index')
# voice_id_F0 = pd.merge(voice_id_rating, voice_id_avg_F0, on='WAV')
# voice_id_F0 = pd.merge(voice_id_F0, voice_id_color, on='WAV')
#
# # plot
# fig, axes = plt.subplots(1, 3)
# fig.subplots_adjust(hspace=0.5)
# fig.set_size_inches(16, 6)
# fig.suptitle("Avg F0 by Rating, %s Participants"%number_participants, fontsize=20, fontweight='bold')
# fig.subplots_adjust( top = 0.85 )
#
# axes[0].set_title('Gender Identity')
# axes[0].set_xlim(-1.5,1.5)
# sns.regplot(data=gender_id_F0, x='Rating_z_score', y='F0_mean', ax=axes[0], marker = '+', scatter_kws={'facecolors':gender_id_F0['color']}, line_kws={"color": "#648FFF"})
# axes[0].set_xlabel('Rating (-: Male, +: Female)')
# axes[0].set_ylabel('Avg F0')
#
# axes[1].set_title('Sexual Orientation')
# axes[1].set_xlim(-1.5,1.5)
# sns.regplot(data=sexual_orientation_F0, x='Rating_z_score', y='F0_mean', ax=axes[1], marker = '+', scatter_kws={'facecolors':sexual_orientation_F0['color']}, line_kws={"color": "#785EF0"})
# axes[1].set_xlabel('Rating (-: Homo, +: Het)')
# axes[1].set_ylabel('')
#
# axes[2].set_title('Voice Identity')
# axes[2].set_xlim(-1.5,1.5)
# sns.regplot(data=voice_id_F0, x='Rating_z_score', y='F0_mean', ax=axes[2], marker = '+', scatter_kws={'facecolors':voice_id_F0['color']}, line_kws={"color": "#DC267F"})
# axes[2].set_xlabel('Rating (-: Masc, +: Femme)')
# axes[2].set_ylabel('')
#
# scatter1_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c=colors['color'][0], marker = 'o')
# scatter2_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c=colors['color'][1], marker = 'o')
# scatter3_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c=colors['color'][2], marker = 'o')
# scatter4_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c=colors['color'][3], marker = 'o')
# scatter5_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c=colors['color'][4], marker = 'o')
# axes[0].legend([scatter1_proxy, scatter5_proxy, scatter3_proxy, scatter4_proxy, scatter2_proxy], ['straight men','queer men','queer NB, men and women','queer women','straight women'], numpoints = 1, loc = 'upper left')
#
#
# # plt.show()
# plt.savefig(os.path.join(dir,'figs', 'F0_avgbycondition.png'), bbox_inches='tight', dpi=300)
# plt.close()

# overlay
# fig, axes = plt.subplots()
# fig.set_size_inches(18, 10)
# axes.set_title('Avg F0 by Condition Rating, %s Participants'%number_participants, fontsize=20, fontweight='bold')
# axes.set_xlim(-1.5,1.5)
# sns.regplot(data=gender_id_F0, x='Rating_z_score', y='F0_mean', marker = '+', scatter_kws={'facecolors':gender_id_F0['color']}, line_kws={"color": "#648FFF"})
# sns.regplot(data=sexual_orientation_F0, x='Rating_z_score', y='F0_mean', marker = '+', scatter_kws={'facecolors':sexual_orientation_F0['color']}, line_kws={"color": "#785EF0"})
# sns.regplot(data=voice_id_F0, x='Rating_z_score', y='F0_mean', marker = '+', scatter_kws={'facecolors':voice_id_F0['color']}, line_kws={"color": "#DC267F"})
# axes.set_xlabel('Rating (-1.5-1.5)')
# axes.set_ylabel('Avg F0')
# scatter1_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c=colors['color'][0], marker = 'o')
# scatter2_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c=colors['color'][1], marker = 'o')
# scatter3_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c=colors['color'][2], marker = 'o')
# scatter4_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c=colors['color'][3], marker = 'o')
# scatter5_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c=colors['color'][4], marker = 'o')
# axes.legend([scatter1_proxy, scatter5_proxy, scatter3_proxy, scatter4_proxy, scatter2_proxy], ['straight men','queer men','queer NB, men and women','queer women','straight women'], numpoints = 1, loc = 'upper left')
# plt.savefig(os.path.join(dir,'figs', 'F0_avgbycondition_overlay.png'), bbox_inches='tight', dpi=300)
# plt.close()
# plt.show()
