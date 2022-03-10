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
# np.set_printoptions(threshold=sys.maxsize)

dir = '/Users/bcl/Documents/GitHub/queer_speech'
os.chdir(dir)
vs_fname = os.path.join(dir,'feature_extraction','vs_output.csv')
vs = pd.read_csv(vs_fname)
