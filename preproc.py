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
vs_fname = os.path.join(dir,'feature_extraction','vs_output.csv')
vs = pd.read_csv(vs_fname)
# view all labels in each file
# with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
#     print(vs[['Filename','Label']])

# make a list of all the features in the VS output
vs_colnames = pd.Series(vs.columns.values.tolist())
vs_colnames = vs_colnames.drop(labels=[0, 1, 272], axis=0)
vs_features = ['strF0_mean','H1H2c_mean','CPP_mean']

# make lists of relevant segments for certain features
vowels = ['IY','AA','IH','AX','AH','EH','EY','UW','AE','AO','OW','AY','AW','OY','IX']
sonorants = ['IY','AA','IH','AX','AH','EH','EY','UW','AE','AO','OW','AY','AW','ER','EL','N','NG','M','R','W','L','OY','Y']

vs_features_avg = pd.DataFrame()

for i in vs_features:
    vs_pivot = vs.pivot_table(index = 'Filename', columns = 'Label', values = i)
    vs_append = pd.DataFrame(vs_pivot.mean(axis=1))
    pd.concat([vs_features_avg,vs_append], axis=1)


for i in vs_colnames:
    vs_pivot = vs.pivot_table(index = 'Filename', columns = 'Label', values = i)
    vs_pivot.index.name = None
    vs_pivot.index = pd.Series(vs_pivot.index).astype(str)
    vs_append = pd.Series(vs_pivot.index).astype(str)

    print(vs_pivot)
