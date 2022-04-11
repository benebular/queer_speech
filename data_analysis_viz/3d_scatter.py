## script for 3D dummy data for queer speech project
# author: Ben Lang
# blang@ucsd.edu

import pandas as pd
import numpy as np
import os
import sklearn
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import string
#import mpld3

dir = '/Users/bcl/Documents/GitHub/queer_speech'
os.chdir(dir)
ratings_fname = os.path.join(dir,'feature_extraction/queer_data.csv')
ratings = pd.read_csv(ratings_fname)
ratings = ratings.drop('Unnamed: 0', axis=1) # might need to change column name when it imports in, there's a random addition at the beginning


# establish participant list
participants = pd.Series(range(len(ratings.groupby('Participant').count()))) + 1
print(participants)


## rating for visualization
ratings['Condition'] = ratings['Condition'].astype('category')
ratings['WAV'] = ratings['WAV'].astype('category')
ratings_avg = ratings.groupby(['WAV', 'Condition'], as_index=False)[['Rating_z_score','kmeans_5_cluster']].mean()
# clusters = ratings.groupby(['WAV', 'Condition'], as_index=False)['kmeans_5_cluster'].mean()
ratings_avg_pivot = ratings_avg.pivot(index = 'WAV', columns = 'Condition', values = ['Rating_z_score','kmeans_5_cluster'])
ratings_avg_pivot = pd.DataFrame(ratings_avg_pivot.values, columns = ['gender_id','sexual_orientation','voice_id','cluster','delete1','delete2'], index=ratings_avg_pivot.index)
ratings_avg_pivot = ratings_avg_pivot.drop(['delete1','delete2'], axis = 1)
ratings_avg_pivot['cluster'] = ratings_avg_pivot['cluster'].astype('int64')
print(ratings_avg_pivot)

# clusters_avg_pivot = clusters.pivot(index = 'WAV', columns = 'Condition', values = 'kmeans_5_cluster')
# clusters_avg_pivot = clusters_avg_pivot.drop(['sexual_orientation','voice_id'], axis=1)
# clusters_avg_pivot = clusters_avg_pivot.rename(columns = {'gender_id':'kmeans_5_cluster'})
# ratings_avg_pivot.set_index(['Condition']).VALUE.unstack().reset_index()

# ratings_avg_pivot.index.name = None
# ratings_avg_pivot.index = pd.Series(ratings_avg_pivot.index).astype(str)
# clusters_avg_pivot.index.name = None
# clusters_avg_pivot.index = pd.Series(clusters_avg_pivot.index).astype(str)
# categories = pd.Series(ratings_avg_pivot.index).astype(str)

# import random
# get_colors = lambda n: list(map(lambda i: "#" + "%06x" % random.randint(0, 0xFFFFFF),range(n)))
# colors = pd.DataFrame(get_colors(5), columns={'color'}) # sample return:  ['#8af5da', '#fbc08c', '#b741d0', '#e599f1', '#bbcb59', '#a2a6c0']
# colors['cluster'] = [0,1,2,3,4]
# labels = ratings_avg_pivot.groupby(['cluster'], as_index=False)[['gender_id','sexual_orientation','voice_id']].mean()

colors = pd.DataFrame({"color": ['#648FFF','#785EF0','#DC267F','#FE6100','#FFB000']})
colors['cluster'] = [0,1,2,3,4]
ratings_avg_pivot = pd.merge(ratings_avg_pivot, colors, on = 'cluster', how = "outer")


## plot averages
fig = plt.figure(figsize = (10,7))
ax = plt.axes(projection='3d')
# for i, txt in enumerate(categories): # plots ewach point in red and then plots a text from a separate list to label the dots
#     ax.scatter(ratings_avg_pivot['gender_id'][i],ratings_avg_pivot['sexual_orientation'][i],ratings_avg_pivot['voice_id'][i], alpha=0.8, s=30, color=colors[i])
#     ax.text(ratings_avg_pivot['gender_id'][i],ratings_avg_pivot['sexual_orientation'][i],ratings_avg_pivot['voice_id'][i],  '%s' % (str(txt)), size=10, zorder=1, color='k')
ax.scatter(ratings_avg_pivot['gender_id'], ratings_avg_pivot['sexual_orientation'], ratings_avg_pivot['voice_id'], alpha=0.8, s=50, c=ratings_avg_pivot['color']) # just the scatter, no text
plt.title('Gender Identity, Sexual Orientation, and Voice Identity Ratings (Avg)', fontweight='bold')
ax.set_xlabel('Gender ID', fontweight='bold')
ax.set_ylabel('PSO', fontweight='bold')
ax.set_zlabel('Voice (Masc-N-Femme)', fontweight='bold')
scatter1_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c=colors['color'][0], marker = 'o')
scatter2_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c=colors['color'][1], marker = 'o')
scatter3_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c=colors['color'][2], marker = 'o')
scatter4_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c=colors['color'][3], marker = 'o')
scatter5_proxy = matplotlib.lines.Line2D([0],[0], linestyle="none", c=colors['color'][4], marker = 'o')
ax.legend([scatter1_proxy, scatter5_proxy, scatter3_proxy, scatter4_proxy, scatter2_proxy], ['straight masc men','queer masc men','queer NB men and women','queer femme women','straight femme women'], numpoints = 1, loc = 'lower right')
# plt.legend()
plt.show()

# plot individual ratings
fig = plt.figure(figsize = (10,7))
ax = plt.axes(projection='3d')
for participant in participants:
    ratings_indiv = ratings[ratings['Participant'] == participant]
    ratings_indiv_pivot = ratings_indiv.pivot(index = 'WAV', columns = 'Condition', values = 'Rating')
    ax.scatter(ratings_indiv_pivot['gender_id'], ratings_indiv_pivot['sexual_orientation'], ratings_indiv_pivot['voice_id'], alpha=0.8, s=30)
plt.title('Gender Identity, Sexual Orientation, and Vocie Identity Ratings (Indiv)', fontweight='bold')
ax.set_xlabel('Gender ID', fontweight='bold')
ax.set_ylabel('PSO', fontweight='bold')
ax.set_zlabel('Voice (Masc-N-Femme)', fontweight='bold')
plt.show()

# ## check if ratings already made and load
# if os.path.isfile(ratings_fname):
#     ratings = pd.read_csv(ratings_fname)
#
# else:
# ## create fake ratings data for 100 participants, random responses
#     col_names = ['q1','q2','q3']
#     ratings = pd.DataFrame()
#     for i in col_names:
#         rando = np.random.randint(1,10, size=100)
#         rando_list = rando.tolist()
#         ratings[i] = rando_list
#     # ratings['SO'] = ratings.q1.apply(lambda x: np.random.choice(range(1,5)))
#     # ratings['gender'] = ratings.q1.apply(lambda x: np.random.choice(range(1,7)))
#     ratings.to_csv("rando_ratings.csv", index=True, encoding='utf-8')
#     # di_SO={"L":"1","G":"2","B":"3","Q":"4"}
#     # di_gender={"M":"1","F":"2","NB":"3","Q":"4","C":"5","T":"6","Q":"7"}
#
# q1 = ratings['q1'].tolist()
# q2 = ratings['q2'].tolist()
# q3 = ratings['q3'].tolist()
# data= [q1, q2, q3]
# x, y, z = data
# # colors = ['red','green','blue']
# # groups = ['Voice','PSO','Gender']
#
# # plot random ratings, means no correlations between 3 independent measures
# fig = plt.figure(figsize = (10,7))
# ax = plt.axes(projection='3d')
# ax.scatter(x, y, z, alpha = 0.8, s=30)
# plt.title('LGBTQ+ PSO Clouds', fontweight='bold')
# ax.set_xlabel('Voice (Masc-NB-Femme)', fontweight='bold')
# ax.set_ylabel('PSO', fontweight='bold')
# ax.set_zlabel('Gender (Speaker)', fontweight='bold')
# plt.show()




## make fake data with skew towards certain groupings
# N = 60
# g1 = (0.6 + 0.6 * np.random.rand(N), np.random.rand(N),0.4+0.1*np.random.rand(N))
# g2 = (0.4+0.3 * np.random.rand(N), 0.5*np.random.rand(N),0.1*np.random.rand(N))
# g3 = (0.3*np.random.rand(N),0.3*np.random.rand(N),0.3*np.random.rand(N))

N = 60
gay = (0.6 + 0.6 * np.random.randint(1,10,N), np.random.randint(1,10,N),0.4+0.1*np.random.randint(1,10,N))
lesbian = (0.4+0.3 * np.random.randint(1,10,N), 0.5*np.random.randint(1,10,N),0.1*np.random.randint(1,10,N))
bisexual = (0.3*np.random.randint(1,10,N),0.3*np.random.randint(1,10,N),0.3*np.random.randint(1,10,N))
queer = (0.1*np.random.randint(1,10,N),0.1*np.random.randint(1,10,N),0.1*np.random.randint(1,10,N))

# data = (gay,lesbian,bisexual)
# colors = ("red", "green", "blue")
# groups = ("coffee", "tea", "water")

g1 = gay[0].tolist()
g2 = gay[1].tolist()
g3 = gay[2].tolist()
l1 = lesbian[0].tolist()
l2 = lesbian[1].tolist()
l3 = lesbian[2].tolist()
b1 = bisexual[0].tolist()
b2 = bisexual[1].tolist()
b3 = bisexual[2].tolist()
q1 = queer[0].tolist()
q2 = queer[1].tolist()
q3 = queer[2].tolist()
custom_lines = [Line2D([0], [0], marker='o', color='w', markerfacecolor='r', markersize=15),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='k', markersize=15),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='b', markersize=15),
                Line2D([0], [0], marker='o', color='w', markerfacecolor='g', markersize=15)]

# plot random ratings, means no correlations between 3 independent measures
fig = plt.figure(figsize = (10,7))
ax = plt.axes(projection='3d')
ax.scatter(g1, g2, g3, alpha=0.8, s=30, c='r')
ax.scatter(l1, l2, l3, alpha=0.8, s=30, c='k')
ax.scatter(b1, b2, b3, alpha=0.8, s=30, c='b')
ax.scatter(q1, q2, q3, alpha=0.8, s=30, c='g')
plt.title('LGBTQ+ Clouds PCA', fontweight='bold')
ax.set_xlabel('Voice (Masc-NB-Femme)', fontweight='bold')
ax.set_ylabel('PSO', fontweight='bold')
ax.set_zlabel('Gender (Speaker)', fontweight='bold')
ax.legend(handles=custom_lines)
plt.show()





# import matplotlib.pyplot as plt
# import numpy as np
#
# # Fixing random state for reproducibility
# np.random.seed(19680801)
# def randrange(n, vmin, vmax):
#     return (vmax - vmin)*np.random.rand(n) + vmin
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# n = 100
#
# # For each set of style and range settings, plot n random points in the box
# # defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
# for m, zlow, zhigh in [('o', -50, -25), ('^', -30, -5), ('x', -40, -15)]:
#     xs = randrange(n, 23, 32)
#     ys = randrange(n, 0, 100)
#     zs = randrange(n, zlow, zhigh)
#     ax.scatter(xs, ys, zs, marker=m)
#
# plt.title('LGBTQ+ PSO Clouds', fontweight='bold')
# ax.set_xlabel('Voice (Masc-NB-Femme)', fontweight='bold')
# ax.set_ylabel('PSO', fontweight='bold')
# ax.set_zlabel('Gender (Speaker)', fontweight='bold')
#
# plt.show()
