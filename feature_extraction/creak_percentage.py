## raw data organization for creaky voice from COVAREP
# author: Ben lang
# e: blang@ucsd.edu

import numpy as np
import pandas as pd
import os
import os.path as op
import sys
import re
import glob

# creak
print ('Extracting creak...')
dir = '/Users/bcl/Documents/MATLAB/covarep/output'
os.chdir(dir)
creak_WAV_list = []
creak_percent_list =[]
for name in glob.glob(os.path.join(dir,'*')):
    # get corresponding WAV name
    creak_fname = str(name)
    creak_WAV_name = re.search(r'creak_(.*?).csv', creak_fname).group(1) # using a regular expression to take whichever output file has the word "creak" in the folder
    creak_WAV_list.append(creak_WAV_name) # append that file to a list of all the output files with creak
    # read in the creak csv for a WAV
    creak = pd.read_csv(name, header=None, nrows=1) # this takes time per file because they are ms-by-ms and so it's like 40000 lines, be patient
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

## now you can use creak_data in any additional analysis